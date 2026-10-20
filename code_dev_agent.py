"""
code_dev_agent.py - multi-file / multi-folder code generation agent (LangGraph)

Flow:
    planner -> writer -> security_check -> executor -> reviewer
                  ^            |               |
                  |            v               v
                  +-------- debugger <---------+      (hard-capped, see MAX_ITERATIONS)

Key guarantees
- Output is a project (dict of relative_path -> content), not one string.
- Writer generates ONE file per LLM call (far more reliable for small local models).
- Every retry loop is capped: max iterations + same-error detection.
- Executor never "runs a server forever": server/library projects get an import smoke test.
- Missing third-party packages are an environment issue, NOT sent to the debugger loop.
- Reviewer is deterministic (no LLM), so the report can never hallucinate.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

load_dotenv()
log = logging.getLogger("code_dev_agent")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MAX_ITERATIONS = int(os.getenv("CODE_DEV_MAX_ITER", "4"))     # total attempts
MAX_SAME_ERROR = 3                                            # same error N times -> stop
MAX_FILES = 25
MAX_FILE_CHARS = 60_000
RUN_TIMEOUT = int(os.getenv("CODE_DEV_RUN_TIMEOUT", "30"))
SMOKE_TIMEOUT = int(os.getenv("CODE_DEV_SMOKE_TIMEOUT", "90"))
INSTALL_TIMEOUT = int(os.getenv("CODE_DEV_INSTALL_TIMEOUT", "900"))
LOCAL_SYNTAX_RETRIES = 2
TAIL = 3000

EXT = {"python": ".py", "javascript": ".js", "bash": ".sh"}
DEFAULT_ENTRY = {"python": "main.py", "javascript": "main.js", "bash": "main.sh"}

# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
def _init_llm():
    try:
        from local_llm_service import get_local_chat_llm
        return get_local_chat_llm(temperature=0.1)
    except Exception as e:  # noqa: BLE001
        log.warning("local_llm_service unavailable (%s), trying OpenAI-compatible server", e)
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url="http://127.0.0.1:8088/v1",
            api_key="x-estal-local",
            model_name="X-Estal-4B",
            temperature=0.1,  # was 0.9: too random for code
        )
    except Exception as e:  # noqa: BLE001
        log.error("No LLM backend could be initialised: %s", e)
        return None


llm_base = _init_llm()

try:  # optional, kept only for backwards compatibility with other modules
    from tools import AI_ML_TOOLS, CODE_DEV_TOOLS, FILE_OPS_TOOLS, SECURITY_PRIVACY_TOOLS
    DEV_TOOLS = CODE_DEV_TOOLS + FILE_OPS_TOOLS + AI_ML_TOOLS + SECURITY_PRIVACY_TOOLS
except Exception:  # noqa: BLE001
    DEV_TOOLS = []
try:
    llm_dev = llm_base.bind_tools(DEV_TOOLS) if (llm_base and DEV_TOOLS) else llm_base
except Exception:  # noqa: BLE001
    llm_dev = llm_base


def _response_text(response) -> str:
    c = getattr(response, "content", "") or getattr(response, "text", "")
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return c or ""


def _llm_text(system: str, user: str, retries: int = 2) -> str:
    if llm_base is None:
        raise RuntimeError("LLM backend is not available")
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            text = _response_text(llm_base.invoke([SystemMessage(content=system), HumanMessage(content=user)]))
            if text.strip():
                return text
            last = RuntimeError("empty response")
        except Exception as e:  # noqa: BLE001
            last = e
        if attempt < retries:
            time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last}")


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class CodeDevState(TypedDict, total=False):
    task: str
    language: Literal["python", "javascript", "bash"]
    user_plan: str                      # spec given by the user (never overwritten)
    plan_context: str                   # planner's file layout / notes
    project_name: str
    entry_point: str
    run_mode: Literal["script", "server", "library"]
    planned_files: List[Dict[str, str]]
    dependencies: List[str]
    files: Dict[str, str]               # relative path -> content
    failed_files: List[str]
    file_path: str                      # entry file (compat)
    code: str                           # entry file content (compat)
    output_dir: str
    workdir: str
    install_deps: bool
    security_passed: bool
    security_feedback: Optional[str]
    execution_output: str
    test_result: str
    error: Optional[str]
    fix_instructions: str
    missing_modules: List[str]
    warnings: List[str]
    review_feedback: str
    status: str                         # success | needs_debugging | security_violation | deps_missing | partial | failed
    stop_reason: Optional[str]
    iteration_count: int
    max_iterations: int
    error_signatures: List[str]
    final_output: str


# --------------------------------------------------------------------------- #
# Helpers: paths, parsing, validation
# --------------------------------------------------------------------------- #
def safe_rel_path(p: Optional[str]) -> Optional[str]:
    """Return a clean relative posix path, or None if it is unsafe."""
    p = (p or "").strip().strip("`'\"").replace("\\", "/")
    if not p or p.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", p):
        return None
    parts = [x for x in PurePosixPath(p).parts if x not in ("", ".")]
    if not parts or any(x == ".." for x in parts):
        return None
    if any(re.search(r'[<>:"|?*\x00-\x1f]', x) for x in parts):
        return None
    return "/".join(parts)


def parse_json_safely(raw_text: str) -> dict:
    clean = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.DOTALL).strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
    if m:
        clean = m.group(1).strip()
    try:
        data = json.loads(clean)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        s, e = clean.find("{"), clean.rfind("}")
        if s != -1 and e > s:
            try:
                data = json.loads(clean[s:e + 1])
                return data if isinstance(data, dict) else {}
            except Exception:  # noqa: BLE001
                pass
    return {}


def extract_code(text: str, path: str = "main.py") -> str:
    """Strip <think> blocks, markdown fences and chatter; return raw file content."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    if path.endswith((".md", ".rst", ".txt")):          # docs may legitimately contain fences
        m = re.fullmatch(r"```(?:markdown|md|text|txt)?[ \t]*\r?\n(.*)\r?\n```", text, re.DOTALL)
        return (m.group(1) if m else text).strip()
    if path.endswith(".py"):
        try:
            ast.parse(text)
            return text                                    # already clean python
        except SyntaxError:
            pass
    blocks = re.findall(r"```[\w+.-]*[ \t]*\r?\n(.*?)```", text, re.DOTALL)
    if blocks:
        code = max(blocks, key=len)
    else:
        m = re.match(r"```[\w+.-]*[ \t]*\r?\n(.*)", text, re.DOTALL)   # truncated: no closing fence
        code = m.group(1) if m else text
    code = re.sub(r"^[ \t]*```[\w+.-]*[ \t]*$", "", code, flags=re.MULTILINE)
    return code.strip()

def extract_code_block(raw_text: str, default_lang: str = "python") -> str:
    return extract_code(raw_text, "main" + EXT.get(default_lang, ".py"))


def _stub_functions(tree: ast.AST) -> List[str]:
    bad = []
    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if n.name.startswith("__") and n.name.endswith("__"):
            continue
        decos = " ".join(ast.unparse(d) for d in n.decorator_list)
        if "abstractmethod" in decos or "overload" in decos:
            continue
        body = [s for s in n.body
                if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant)
                        and isinstance(s.value.value, str))]
        def _is_stub(s):
            if isinstance(s, ast.Pass):
                return True
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis:
                return True
            if isinstance(s, ast.Raise) and "NotImplementedError" in ast.unparse(s):
                return True
            return False
        if not body or all(_is_stub(s) for s in body):
            bad.append(n.name)
    return bad


def validate_content(path: str, content: str) -> Optional[str]:
    """Return an error string if the file is not acceptable, else None."""
    if not content or not content.strip():
        return "file is empty"
    if len(content) > MAX_FILE_CHARS:
        return f"file too large ({len(content)} chars)"
    if path.endswith(".py"):
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return f"SyntaxError line {e.lineno}: {e.msg}"
        stubs = _stub_functions(tree)
        if stubs:
            return f"stub/unimplemented functions (implement fully): {', '.join(stubs[:8])}"
    elif path.endswith(".json"):
        try:
            json.loads(content)
        except Exception as e:  # noqa: BLE001
            return f"invalid JSON: {e}"
    return None


def _mod_name(path: str) -> str:
    p = path[:-3] if path.endswith(".py") else path
    parts = p.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_test(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py") or path.startswith("tests/")


def _api_summary(files: Dict[str, str], exclude: str, limit: int = 3500) -> str:
    """Public signatures of already-written modules so later files import correctly."""
    out = []
    for path, src in files.items():
        if path == exclude or not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        sigs = []
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sigs.append(f"def {n.name}({ast.unparse(n.args)})")
            elif isinstance(n, ast.ClassDef):
                meths = [m.name for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                sigs.append(f"class {n.name}  # methods: {', '.join(meths[:10])}")
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        sigs.append(f"{t.id} = ...")
        if sigs:
            out.append(f"# {path}  (import as: {_mod_name(path)})\n" + "\n".join(sigs))
    return "\n\n".join(out)[:limit]


_PIP_NAMES = {
    "fitz": "pymupdf", "PIL": "pillow", "cv2": "opencv-python", "sklearn": "scikit-learn",
    "yaml": "pyyaml", "dotenv": "python-dotenv", "bs4": "beautifulsoup4",
    "faster_whisper": "faster-whisper", "sentence_transformers": "sentence-transformers",
    "multipart": "python-multipart", "ffmpeg": "ffmpeg-python",
}


def _requirements(deps: List[str]) -> str:
    stdlib = getattr(sys, "stdlib_module_names", set())
    seen, out = set(), []
    for d in deps:
        d = d.strip().strip("`'\"")
        if not d:
            continue
        base = re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0]
        if base in stdlib:
            continue
        d = _PIP_NAMES.get(base, d)
        if d.lower() not in seen:
            seen.add(d.lower())
            out.append(d)
    return "\n".join(out) + ("\n" if out else "")


def _files_from_text(text: str) -> List[str]:
    found = re.findall(r"`([\w./-]+\.(?:py|js|ts|sh|txt|md|json|toml|ya?ml|html|css|cfg|ini))`", text or "")
    out = []
    for f in found:
        p = safe_rel_path(f)
        if p and p not in out:
            out.append(p)
    return out


def _deps_from_text(text: str) -> List[str]:
    m = re.search(r"requirements\.txt`?\s*with:?\s*([^\n]+)", text or "")
    if not m:
        return []
    return [x.strip().strip("`.") for x in m.group(1).split(",") if x.strip()]


def _tail(s: Optional[str], n: int = TAIL) -> str:
    s = s or ""
    return s if len(s) <= n else "...[truncated]...\n" + s[-n:]


# --------------------------------------------------------------------------- #
# Node 1: planner
# --------------------------------------------------------------------------- #
def planner_node(state: CodeDevState) -> dict:
    task = state.get("task", "")
    language = state.get("language", "python")
    user_plan = state.get("user_plan") or state.get("plan_context") or ""   # never lose the user's spec

    prompt = f"""Task: {task}
Language: {language}
User spec (follow it exactly when present):
{user_plan[:8000] or '(none)'}

Design the project file layout. Return ONLY valid JSON:
{{
  "project_name": "<short_snake_case>",
  "entry_point": "<relative path of main file>",
  "run_mode": "script" | "server" | "library",
  "files": [{{"path": "<relative/path.ext>", "purpose": "<one line>"}}],
  "dependencies": ["<third-party pip packages only>"],
  "notes": "<constraints>"
}}
Rules:
- Relative paths only. Folders are allowed (e.g. "parsers/pdf_parser.py").
- Include EVERY file the project needs. Max {MAX_FILES} files. A small task may be a single file.
- Imports between project modules must be absolute from the project root
  (e.g. `from parsers.pdf_parser import parse_pdf`).
- run_mode "script" = runs and exits; "server" = web app / UI that never exits; "library" = importable modules."""

    data: dict = {}
    try:
        data = parse_json_safely(_llm_text("You are a Principal Software Architect. Output JSON only.", prompt))
    except Exception as e:  # noqa: BLE001
        log.warning("planner failed, using fallback plan: %s", e)

    planned: List[Dict[str, str]] = []
    seen = set()
    for item in data.get("files") or []:
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            continue
        p = safe_rel_path(str(item.get("path", "")))
        if p and p not in seen:
            seen.add(p)
            planned.append({"path": p, "purpose": str(item.get("purpose", ""))[:200]})
    if not planned:
        for p in _files_from_text(user_plan):
            if p not in seen:
                seen.add(p)
                planned.append({"path": p, "purpose": ""})
    if not planned:
        planned = [{"path": safe_rel_path(state.get("file_path")) or DEFAULT_ENTRY[language], "purpose": task[:200]}]
    planned = planned[:MAX_FILES]

    given = safe_rel_path(state.get("file_path"))
    if given and len(planned) == 1:                       # honour caller's file name for single-file tasks
        planned[0]["path"] = given

    paths = [p["path"] for p in planned]
    ext = EXT[language]
    entry = safe_rel_path(data.get("entry_point"))
    if entry not in paths:
        entry = next((c for c in ("main" + ext, "app" + ext, "index" + ext) if c in paths), None) \
            or next((p for p in paths if p.endswith(ext) and not _is_test(p)), paths[-1])

    deps = [d for d in (data.get("dependencies") or []) if isinstance(d, str)] + _deps_from_text(user_plan)
    files: Dict[str, str] = dict(state.get("files") or {})
    if language == "python":
        req = _requirements(deps)
        if req:
            files["requirements.txt"] = req
            if "requirements.txt" not in paths:
                planned.insert(0, {"path": "requirements.txt", "purpose": "dependency list"})

    run_mode = data.get("run_mode")
    if run_mode not in ("script", "server", "library"):
        blob = (" ".join(deps) + " " + user_plan).lower()
        run_mode = "server" if re.search(r"fastapi|flask|streamlit|uvicorn|django", blob) else "script"

    # write order: requirements first, entry point last (it can import everything else)
    planned.sort(key=lambda x: (x["path"] != "requirements.txt", x["path"] == entry))

    project = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(data.get("project_name") or "project")).strip("_")[:40] or "project"
    tree = "\n".join(f"- {p['path']}: {p['purpose']}" for p in planned)
    plan_context = f"Entry point: {entry}\nRun mode: {run_mode}\nFiles:\n{tree}\nNotes: {data.get('notes', '')}"

    return {
        "project_name": project,
        "entry_point": entry,
        "file_path": entry,
        "run_mode": run_mode,
        "planned_files": planned,
        "dependencies": deps,
        "files": files,
        "plan_context": plan_context,
        "user_plan": user_plan,
        "iteration_count": 0,
        "error_signatures": [],
        "failed_files": [],
        "stop_reason": None,
    }


# --------------------------------------------------------------------------- #
# Node 2: writer (one LLM call per file)
# --------------------------------------------------------------------------- #
WRITER_SYSTEM = "You are a Senior Staff Software Engineer. You output the raw content of ONE file only."


def _write_one(state: CodeDevState, path: str, purpose: str, files: Dict[str, str], fix: bool) -> str:
    language = state.get("language", "python")
    planned = state.get("planned_files") or []
    tree = "\n".join(f"- {p['path']}: {p['purpose']}" for p in planned)
    api = _api_summary(files, exclude=path)
    deps = ", ".join(state.get("dependencies") or []) or "standard library only"

    fix_block = ""
    if fix:
        fix_block += f"\nCURRENT CONTENT OF {path} (edit it, keep what already works):\n{files.get(path, '')[:12000]}\n"
        if state.get("error"):
            fix_block += f"\nERROR TO FIX:\n{_tail(state['error'], 2500)}\n"
        if state.get("fix_instructions"):
            fix_block += f"\nFIX INSTRUCTIONS:\n{state['fix_instructions']}\n"
        if state.get("security_feedback"):
            fix_block += f"\nSECURITY PROBLEM TO FIX:\n{state['security_feedback']}\n"

    base = f"""Project task: {state.get('task', '')}
Language: {language}
Run mode: {state.get('run_mode', 'script')}
Third-party packages allowed: {deps}

Project files:
{tree}

Spec from user:
{(state.get('user_plan') or '')[:6000] or '(none)'}

Already written modules (use these exact names when importing):
{api or '(none yet)'}

YOU ARE WRITING ONLY THIS FILE: {path}
Purpose: {purpose or 'see spec'}
{fix_block}
Rules:
1. Complete, working content. No stubs, TODOs, `pass`-only or NotImplementedError functions.
2. Imports of project modules must be absolute from the project root (e.g. `from parsers.pdf_parser import parse_pdf`).
3. Do not run servers, UIs or long loops at import time; put them under `if __name__ == "__main__":`.
4. Never hard-code secrets; read them from environment variables.
5. Python scripts that run and exit: add small self-tests under `if __name__ == "__main__":`.
6. Output ONLY the raw content of {path}. No markdown fences, no explanation. First character must be file content."""

    problem = None
    content = ""
    for attempt in range(LOCAL_SYNTAX_RETRIES + 1):
        user = base if not problem else base + (
            f"\n\nYour previous output was rejected: {problem}\nReturn the complete corrected file."
        )
        content = extract_code(_llm_text(WRITER_SYSTEM, user), path)
        problem = validate_content(path, content)
        if not problem:
            return content
        log.info("writer: %s attempt %d rejected: %s", path, attempt + 1, problem)
    return content  # last attempt; executor's static validation will report it


def code_writer_node(state: CodeDevState) -> dict:
    files: Dict[str, str] = dict(state.get("files") or {})
    planned = list(state.get("planned_files") or [])
    known = {p["path"] for p in planned}
    entry = state.get("entry_point") or state.get("file_path") or DEFAULT_ENTRY[state.get("language", "python")]
    warnings = list(state.get("warnings") or [])

    fix_mode = bool(state.get("error") or state.get("security_feedback")) and bool(files)
    if fix_mode:
        targets = [p for p in (state.get("failed_files") or []) if safe_rel_path(p)] or [entry]
        for p in targets:
            if p not in known:
                planned.append({"path": p, "purpose": "added during fix"})
                known.add(p)
    else:
        targets = [p["path"] for p in planned if p["path"] not in files]

    purposes = {p["path"]: p["purpose"] for p in planned}
    for path in targets:
        try:
            files[path] = _write_one({**state, "planned_files": planned}, path, purposes.get(path, ""), files, fix_mode)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"writer failed for {path}: {e}")
            log.warning("writer failed for %s: %s", path, e)

    return {
        "files": files,
        "planned_files": planned,
        "code": files.get(entry, ""),
        "error": None,
        "security_feedback": None,
        "failed_files": [],
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Node 3: security audit
# --------------------------------------------------------------------------- #
_DANGEROUS = [
    (r"rm\s+-rf?\s+(/|~|\$HOME|\*)(\s|\"|'|$)", "destructive rm -rf on root/home"),
    (r"shutil\.rmtree\s*\(\s*['\"](/|~)", "rmtree on root/home"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb"),
    (r"\bmkfs\.\w+", "filesystem formatting"),
    (r"\bdd\s+if=.*\bof=/dev/", "raw device write"),
    (r"(curl|wget)[^\n|]*\|\s*(sudo\s+)?(ba)?sh\b", "pipe remote script to shell"),
    (r"__import__\s*\(\s*['\"]pty['\"]\s*\)\s*\.spawn", "pty spawn"),
    (r"\bos\.system\s*\(\s*['\"]\s*(sudo|chmod\s+777\s+/)", "privileged os.system"),
    (r"(?i)\b(api[_-]?key|secret|token|password)\s*=\s*['\"](?!your|example|change|xxx|placeholder)[A-Za-z0-9_\-/+=]{24,}['\"]",
     "hard-coded secret (use environment variables)"),
]


def security_audit_node(state: CodeDevState) -> dict:
    files = state.get("files") or {}
    violations: List[tuple] = []
    for path, src in files.items():
        for pattern, reason in _DANGEROUS:
            if re.search(pattern, src):
                violations.append((path, reason))
    if not violations:
        return {"security_passed": True, "security_feedback": None}

    it = state.get("iteration_count", 0) + 1
    limit = state.get("max_iterations", MAX_ITERATIONS)
    feedback = "Security violations:\n" + "\n".join(f"- {p}: {r}" for p, r in violations)
    out = {
        "security_passed": False,
        "security_feedback": feedback,
        "failed_files": sorted({p for p, _ in violations}),
        "status": "security_violation",
        "iteration_count": it,
    }
    if it >= limit:
        out["stop_reason"] = f"Security violations remain after {it} attempts."
    return out


# --------------------------------------------------------------------------- #
# Node 4: executor (sandboxed)
# --------------------------------------------------------------------------- #
_SMOKE_CODE = r'''
import sys, os, json, importlib, traceback
sys.path.insert(0, os.getcwd())
failures = []
for mod in json.loads(sys.argv[1]):
    try:
        importlib.import_module(mod)
    except BaseException:
        failures.append({"module": mod, "tb": traceback.format_exc()})
print("__SMOKE__" + json.dumps(failures))
'''


def _clean_env() -> Dict[str, str]:
    """Pass the environment through but never leak API keys/tokens into generated code."""
    env = {k: v for k, v in os.environ.items()
           if not re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", k, re.I)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _dec(x) -> str:
    if x is None:
        return ""
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else x


def _posix_limits(timeout: int):
    def apply():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (timeout + 5, timeout + 5))
            resource.setrlimit(resource.RLIMIT_FSIZE, (200 * 1024 * 1024, 200 * 1024 * 1024))
        except Exception:  # noqa: BLE001
            pass
    return apply


def _run(cmd: List[str], cwd, timeout: int, env: Dict[str, str], limits: bool = True):
    kw: Dict[str, Any] = {}
    if limits and os.name == "posix":
        kw["preexec_fn"] = _posix_limits(timeout)
    try:
        p = subprocess.run(cmd, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, **kw)
        return p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        return -1, _dec(e.stdout), _dec(e.stderr), True
    except FileNotFoundError as e:
        return 127, "", f"Command not found: {cmd[0]} ({e})", False


def _materialize(files: Dict[str, str], root: Path) -> None:
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    base = root.resolve()
    for rel, content in files.items():
        p = (root / rel).resolve()
        if base not in p.parents:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _failed_files_from_trace(text: str, root: Path, files: Dict[str, str]) -> List[str]:
    out: List[str] = []
    base = root.resolve()
    for m in re.finditer(r'File "([^"]+)", line \d+', text or ""):
        try:
            rel = Path(m.group(1)).resolve().relative_to(base).as_posix()
        except Exception:  # noqa: BLE001
            continue
        if rel in files and rel not in out:
            out.append(rel)
    return list(reversed(out))[:3]          # innermost frame first


def _missing_module(text: str) -> Optional[str]:
    m = re.search(r"No module named ['\"]([\w.\-]+)['\"]", text or "")
    return m.group(1).split(".")[0] if m else None


def _ensure_venv(workdir: Path, files: Dict[str, str], env: Dict[str, str]):
    """Create (once) a venv inside workdir and install requirements.txt. Returns (python, error)."""
    venv = workdir / "venv"
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    req = files.get("requirements.txt", "")
    h = hashlib.sha1(req.encode()).hexdigest()
    marker = venv / ".req_hash"
    if py.exists() and marker.exists() and marker.read_text() == h:
        return str(py), None
    if not py.exists():
        rc, _, err, to = _run([sys.executable, "-m", "venv", str(venv)], workdir, 300, env, limits=False)
        if rc != 0 or to:
            return None, f"venv creation failed: {_tail(err, 500)}"
    if req.strip():
        rf = workdir / "requirements.txt"
        rf.write_text(req, encoding="utf-8")
        rc, out, err, to = _run([str(py), "-m", "pip", "install", "-q", "-r", str(rf)],
                                workdir, INSTALL_TIMEOUT, env, limits=False)
        if to:
            return None, f"pip install timed out after {INSTALL_TIMEOUT}s"
        if rc != 0:
            return None, f"pip install failed (check requirements.txt):\n{_tail(err or out, 1500)}"
    marker.write_text(h)
    return str(py), None


def _fail(error: str, failed: List[str], out: str = "", result: str = "Execution failed", **extra) -> dict:
    return {"status": "needs_debugging", "error": error, "failed_files": failed,
            "execution_output": out, "test_result": result, **extra}


def real_executor_node(state: CodeDevState) -> dict:
    files = state.get("files") or {}
    language = state.get("language", "python")
    entry = state.get("entry_point") or state.get("file_path") or DEFAULT_ENTRY[language]
    run_mode = state.get("run_mode", "script")
    install = bool(state.get("install_deps"))
    planned = [p["path"] for p in state.get("planned_files") or []]
    problems = []
    for path in dict.fromkeys(planned + list(files)):
        if path not in files:
            problems.append((path, "planned file was never written"))
            continue
        err = validate_content(path, files[path])
        if err:
            problems.append((path, err))
    if entry not in files:
        problems.append((entry, "entry point missing"))
    if problems:
        return _fail("Static validation failed:\n" + "\n".join(f"- {p}: {m}" for p, m in problems),
                     list(dict.fromkeys(p for p, _ in problems)), result="Static validation failed")

    # 2) materialise project in a sandbox dir
    workdir = Path(state.get("workdir") or tempfile.mkdtemp(prefix="agent_run_"))
    root = workdir / "project"
    _materialize(files, root)
    env = _clean_env()
    base_out = {"workdir": str(workdir)}

    # non-python languages: simple run
    if language != "python":
        exe = ["node", entry] if language == "javascript" else ["bash", entry]
        rc, out, err, to = _run(exe, root, RUN_TIMEOUT, env)
        if to:
            return _fail(f"Timed out after {RUN_TIMEOUT}s", [entry], out, "Timeout", **base_out)
        if rc != 0:
            return _fail(_tail(err) or f"exit code {rc}", [entry], out, **base_out)
        return {"status": "success", "error": None, "execution_output": _tail(out) or "exit code 0",
                "test_result": "Run OK", **base_out}

    py = sys.executable
    if install:
        py, venv_err = _ensure_venv(workdir, files, env)
        if venv_err:
            return _fail(venv_err, ["requirements.txt"], result="Dependency install failed", **base_out)

    local_tops = {p.split("/")[0].removesuffix(".py") for p in files}
    missing: List[str] = []
    log_parts: List[str] = []

    def classify(text: str) -> Optional[str]:
        """Return third-party module name if this failure is only a missing package."""
        mod = _missing_module(text)
        return mod if mod and mod not in local_tops else None

    # 3) run according to mode
    if run_mode == "script":
        rc, out, err, to = _run([py, entry], root, RUN_TIMEOUT, env)
        log_parts.append(out)
        if to:
            return _fail(f"Timed out after {RUN_TIMEOUT}s. A script must terminate "
                         f"(servers/UIs must be started only under `if __name__ == '__main__'` "
                         f"and the project run_mode should be 'server').",
                         [entry], out, "Timeout", **base_out)
        if rc != 0:
            mod = classify(err)
            if mod:
                missing.append(mod)
            else:
                return _fail(_tail(err) or f"exit code {rc}",
                             _failed_files_from_trace(err, root, files) or [entry], out, **base_out)
    else:
        mods = [_mod_name(p) for p in files if p.endswith(".py") and not _is_test(p) and _mod_name(p)]
        rc, out, err, to = _run([py, "-c", _SMOKE_CODE, json.dumps(mods)], root, SMOKE_TIMEOUT, env)
        m = re.search(r"__SMOKE__(.*)$", out or "", re.M)
        if to or not m:
            return _fail(("Import smoke test timed out" if to else "Import smoke test crashed") + ":\n"
                         + _tail(err or out, 1500), _failed_files_from_trace(err, root, files) or [entry],
                         "", "Smoke test failed", **base_out)
        failures = json.loads(m.group(1))
        real = []
        for f in failures:
            mod = classify(f["tb"])
            if mod:
                missing.append(mod)
            else:
                real.append(f)
        log_parts.append(f"Import smoke test: {len(mods) - len(failures)}/{len(mods)} modules imported OK")
        if real:
            tb = "\n\n".join(f"[{f['module']}]\n{_tail(f['tb'], 1200)}" for f in real[:2])
            ff: List[str] = []
            for f in real:
                for p in _failed_files_from_trace(f["tb"], root, files)[:1]:   # innermost frame only
                    if p not in ff:
                        ff.append(p)
            return _fail(tb, ff[:3] or [entry], "\n".join(log_parts), "Import errors", **base_out)

    missing = sorted(set(missing))
    if missing and install:   # packages should be installed but are not -> requirements.txt is wrong
        return _fail(f"Packages still missing after install: {', '.join(missing)}. "
                     f"Fix requirements.txt (pip package names).", ["requirements.txt"],
                     "\n".join(log_parts), "Dependency mismatch", **base_out)
    if missing:
        return {"status": "deps_missing", "error": None, "missing_modules": missing,
                "execution_output": "\n".join(log_parts),
                "test_result": f"Static + partial checks passed; not fully run (missing packages: {', '.join(missing)})",
                **base_out}

    # 4) tests (only if pytest exists in the interpreter used)
    tests = [p for p in files if p.endswith(".py") and _is_test(p)]
    result = "Run OK" if run_mode == "script" else "All modules import OK"
    if tests:
        rc, out, err, to = _run([py, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider"] + tests,
                                root, SMOKE_TIMEOUT, env)
        blob = (out or "") + (err or "")
        if "No module named pytest" in blob:
            log_parts.append("pytest not installed: tests skipped")
        elif to:
            return _fail("Tests timed out", tests[:2], "\n".join(log_parts), "Tests timeout", **base_out)
        elif rc not in (0, 5):
            return _fail(_tail(blob), (_failed_files_from_trace(blob, root, files) or tests)[:3],
                         "\n".join(log_parts), "Tests failed", **base_out)
        else:
            log_parts.append(_tail(out, 800))
            result = "Tests passed"

    return {"status": "success", "error": None, "missing_modules": [],
            "execution_output": _tail("\n".join(x for x in log_parts if x) or "exit code 0"),
            "test_result": result, **base_out}


def _error_signature(err: str) -> str:
    lines = [l.strip() for l in (err or "").splitlines() if l.strip()]
    last = lines[-1] if lines else "unknown"
    return re.sub(r"\d+", "N", last)[:200]


def debugger_node(state: CodeDevState) -> dict:
    err = state.get("error") or "Unknown failure"
    files = state.get("files") or {}
    entry = state.get("entry_point") or state.get("file_path") or ""
    it = state.get("iteration_count", 0) + 1
    limit = state.get("max_iterations", MAX_ITERATIONS)
    failed = [p for p in (state.get("failed_files") or []) if p in files or safe_rel_path(p)] or [entry]

    sig = _error_signature(err)
    sigs = list(state.get("error_signatures") or []) + [sig]
    repeats = sigs.count(sig)

    # First attempt at a runtime traceback: fix only the innermost failing file (callers usually work).
    # If the same error comes back, widen to every file in the traceback.
    if repeats == 1 and "Traceback" in err and state.get("test_result") != "Import errors":
        failed = failed[:1]

    stop = None
    if repeats >= MAX_SAME_ERROR:
        stop = f"Same error repeated {repeats} times - automatic fixing is not converging: {sig}"
    elif it >= limit:
        stop = f"Reached max attempts ({limit})."

    fix = ""
    if not stop:
        ctx = "\n\n".join(f"### {p}\n{files[p][:8000]}" for p in failed[:3] if p in files)
        prompt = f"""The project failed. Explain the bug and tell the file writer exactly what to change.

Failing file(s):
{ctx}

Error:
{_tail(err, 2500)}

Return ONLY JSON: {{"root_cause": "<why>", "fix_instructions": "<precise instructions>"}}"""
        try:
            data = parse_json_safely(_llm_text("You are a Systems Debugger. Output JSON only.", prompt, retries=1))
            fix = f"Cause: {data.get('root_cause', '')}\nDo: {data.get('fix_instructions', '')}".strip()
        except Exception as e:  # noqa: BLE001
            log.warning("debugger LLM failed: %s", e)
        if not fix or fix == "Cause: \nDo:":
            fix = f"Fix exactly this error: {sig}"
        if "No module named" in err:
            fix += "\nUse absolute imports from the project root and make sure the imported module/file exists."
        if repeats > 1:
            fix += f"\nThis same error already occurred {repeats - 1} time(s) after earlier fixes. Use a DIFFERENT approach."

    return {
        "error": err,
        "failed_files": failed,
        "fix_instructions": fix,
        "iteration_count": it,
        "error_signatures": sigs,
        "stop_reason": stop,
    }

def _resolve_output_dir(project: str, output_dir: Optional[str]) -> Path:
    if output_dir:
        return Path(output_dir)
    try:
        from tools._common import safe_output_path
        return Path(str(safe_output_path("code", project)))
    except Exception: 
        return Path("generated_code") / project


def _run_hints(state: CodeDevState) -> List[str]:
    files = state.get("files") or {}
    entry = state.get("entry_point", "")
    hints = []
    if "requirements.txt" in files:
        hints.append("pip install -r requirements.txt")
    for p, src in files.items():
        if not p.endswith(".py"):
            continue
        if re.search(r"\bFastAPI\s*\(", src):
            hints.append(f"uvicorn {_mod_name(p)}:app --reload")
        elif re.search(r"^\s*(import|from)\s+streamlit", src, re.M):
            hints.append(f"streamlit run {p}")
    if state.get("language") == "python" and state.get("run_mode") == "script" and entry:
        hints.append(f"python {entry}")
    return hints


def reviewer_node(state: CodeDevState) -> dict:
    files = state.get("files") or {}
    status = state.get("status", "needs_debugging")
    final_status = {"success": "success", "deps_missing": "partial"}.get(status, "failed")

    out_dir = _resolve_output_dir(state.get("project_name", "project"), state.get("output_dir"))
    saved: List[str] = []
    if files:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir.resolve()
        for rel, content in files.items():
            dest = (out_dir / rel).resolve()
            if base not in dest.parents:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            saved.append(rel)

    icon = {"success": "OK", "partial": "PARTIAL", "failed": "FAILED"}[final_status]
    lines = [f"## Overview & Files Saved [{icon}]",
             f"- Task: {state.get('task', '')[:300]}",
             f"- Location: `{out_dir}`",
             f"- Entry point: `{state.get('entry_point', '')}` (run mode: {state.get('run_mode', 'script')})",
             f"- Attempts used: {state.get('iteration_count', 0) + 1}/{state.get('max_iterations', MAX_ITERATIONS)}",
             "", "## Files"]
    lines += [f"- `{p}` ({len(files[p].splitlines())} lines)" for p in sorted(saved)] or ["- (no files produced)"]
    lines += ["", "## Execution & Test Results", f"- Result: {state.get('test_result') or 'not executed'}"]
    if state.get("execution_output"):
        lines += ["```", _tail(state["execution_output"], 1500), "```"]
    if final_status == "partial":
        lines.append(f"- Not fully run because these packages are not installed: "
                     f"{', '.join(state.get('missing_modules') or [])}. "
                     f"Install them (or call with install_deps=True) and re-run.")
    if final_status == "failed":
        lines.append(f"- Stopped: {state.get('stop_reason') or 'unresolved error'}")
        if state.get("error"):
            lines += ["- Last error:", "```", _tail(state["error"], 1500), "```"]
        lines.append("- Files above are the best attempt, NOT verified.")
    hints = _run_hints(state)
    if hints:
        lines += ["", "## How to run"] + [f"- `{h}`" for h in dict.fromkeys(hints)]
    if state.get("warnings"):
        lines += ["", "## Warnings"] + [f"- {w}" for w in state["warnings"]]

    workdir = state.get("workdir")
    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)

    return {
        "status": final_status,
        "output_dir": str(out_dir),
        "final_output": "\n".join(lines),
        "review_feedback": f"{final_status}: {len(saved)} file(s) written to {out_dir}",
        "workdir": "",
    }


# --------------------------------------------------------------------------- #
# Routing + graph
# --------------------------------------------------------------------------- #
def route_after_security(state: CodeDevState) -> str:
    if state.get("security_passed", True):
        return "executor"
    return "reviewer" if state.get("stop_reason") else "writer"

def route_after_execution(state: CodeDevState) -> str:
    return "debugger" if state.get("status") == "needs_debugging" else "reviewer"

def route_after_debugger(state: CodeDevState) -> str:
    return "reviewer" if state.get("stop_reason") else "writer"

workflow = StateGraph(CodeDevState)
workflow.add_node("planner", planner_node)
workflow.add_node("writer", code_writer_node)
workflow.add_node("security_check", security_audit_node)
workflow.add_node("executor", real_executor_node)
workflow.add_node("debugger", debugger_node)
workflow.add_node("reviewer", reviewer_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "writer")
workflow.add_edge("writer", "security_check")
workflow.add_conditional_edges("security_check", route_after_security,
                               {"executor": "executor", "writer": "writer", "reviewer": "reviewer"})
workflow.add_conditional_edges("executor", route_after_execution,
                               {"debugger": "debugger", "reviewer": "reviewer"})
workflow.add_conditional_edges("debugger", route_after_debugger,
                               {"writer": "writer", "reviewer": "reviewer"})
workflow.add_edge("reviewer", END)
code_dev_agent = workflow.compile()

GRAPH_CONFIG = {"recursion_limit": 100}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _initial_state(task: str, language: str = "python", file_path: Optional[str] = None,
                   plan: Optional[str] = None, output_dir: Optional[str] = None,
                   install_deps: bool = False, max_iterations: int = MAX_ITERATIONS) -> CodeDevState:
    return {
        "task": task,
        "language": language,
        "user_plan": plan or "",
        "plan_context": "",
        "files": {},
        "file_path": safe_rel_path(file_path) or "",
        "output_dir": output_dir or "",
        "install_deps": install_deps,
        "security_passed": True,
        "security_feedback": None,
        "execution_output": "",
        "test_result": "",
        "error": None,
        "status": "needs_debugging",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "error_signatures": [],
        "failed_files": [],
        "warnings": [],
        "stop_reason": None,
        "final_output": "",
    }


def run_code_dev_task(task: str, language: str = "python", file_path: Optional[str] = None,
                      plan: Optional[str] = None, output_dir: Optional[str] = None,
                      install_deps: bool = False, max_iterations: int = MAX_ITERATIONS) -> Dict[str, Any]:
    """
    Run an end-to-end (multi-file) coding task.
    status: "success" (verified) | "partial" (written, deps missing so not fully run)
            | "failed" (best attempt saved, NOT verified) | "error" (agent itself crashed)
    """
    agent = "Code & Dev Engineer"
    if llm_base is None:
        return {"status": "error", "agent": agent, "error": "llm_unavailable",
                "reply": "No LLM backend is available. Start the local model server and retry."}
    try:
        final = code_dev_agent.invoke(
            _initial_state(task, language, file_path, plan, output_dir, install_deps, max_iterations),
            config=GRAPH_CONFIG)
    except Exception as e: 
        log.exception("code dev agent crashed")
        return {"status": "error", "agent": agent, "error": str(e), "reply": f"Code Dev agent crashed: {e}"}

    return {
        "status": final.get("status", "failed"),
        "agent": agent,
        "reply": final.get("final_output", ""),
        "files": final.get("files", {}),
        "code": final.get("code", ""),
        "file_path": final.get("entry_point", ""),
        "output_dir": final.get("output_dir", ""),
        "execution_output": final.get("execution_output", ""),
        "error": final.get("error"),
        "stop_reason": final.get("stop_reason"),
    }


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    state = _initial_state("create a notebook_light a end to end produxtion ready rag system where user upload pdf txt video and get answers for the uploded file use spark 2.5 1.7b open weight local model as llm ",
                            install_deps=False)
    final_state: Dict[str, Any] = dict(state)
    print("\nRunning Code Dev Agent...\n")
    for step in code_dev_agent.stream(state, config=GRAPH_CONFIG, stream_mode="updates"):
        for node, update in step.items():
            final_state.update(update or {})
            print(f"[{node.upper()}] done", end="")
            if update and update.get("status"):
                print(f" | status={update['status']}", end="")
            if update and update.get("error"):
                print(f" | error={str(update['error'])[:120]!r}", end="")
            print()
    print("\n" + "=" * 60)
    print(final_state.get("final_output", ""))
    print("=" * 60)
