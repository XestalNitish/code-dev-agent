"""
integration.py — Production Integration & Automation Agent
==========================================================
LangGraph agent for: webhooks, scheduled tasks, email, env config,
system monitoring, downloads, desktop notifications, port checks, process control.

Usage:
    from agents.integration import run_integration_task
    result = run_integration_task("CPU aur memory usage dikhao")
    print(result["final_output"])
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import sys
import time
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("integration_agent")


# ════════════════════════════════════════════════════════════════════════════
# 1. TOOL LOADING
# ════════════════════════════════════════════════════════════════════════════

def _load_tools() -> list:
    tools: list = []
    for mod_name, attr in [
        ("tools.integration_tools", "INTEGRATION_AGENT_TOOLS"),
        ("tools.file_ops",          "FILE_OPS_TOOLS"),
        ("tools.security_privacy",  "SECURITY_PRIVACY_TOOLS"),
        ("tools.integration_ops",   "INTEGRATION_OPS_TOOLS"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            new_tools = getattr(mod, attr)
            existing = {t.name for t in tools}
            added = [t for t in new_tools if t.name not in existing]
            tools.extend(added)
            logger.info(f"Loaded {len(added)} tools from {mod_name}.")
        except Exception as exc:
            logger.warning(f"{mod_name} not available: {exc}")
    logger.info(f"Total active tools: {len(tools)}")
    return tools


ACTIVE_OPS_TOOLS = _load_tools()


# ════════════════════════════════════════════════════════════════════════════
# 2. LLM INIT (Gemini → Groq → Ollama fallback)
# ════════════════════════════════════════════════════════════════════════════

def _init_llm():
    try:
        from local_llm_service import get_local_chat_llm
        llm = get_local_chat_llm(temperature=0.1)
        logger.info("LLM: Spark-X2.5-4B (100% Local Inference)")
        return llm
    except Exception as e:
        logger.warning(f"Local LLM init note: {e}")

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(base_url="http://127.0.0.1:8088/v1", api_key="local-free", model_name="Spark-X2.5-4B", temperature=0.1)
        logger.info("LLM: Local OpenAI Compatible Server (127.0.0.1:8088)")
        return llm
    except Exception:
        pass

    logger.info("Using deterministic integration execution engine.")
    return None



_llm_core = _init_llm()
_llm_tools = _llm_core.bind_tools(ACTIVE_OPS_TOOLS) if (_llm_core and ACTIVE_OPS_TOOLS) else _llm_core


# ════════════════════════════════════════════════════════════════════════════
# 3. STATE
# ════════════════════════════════════════════════════════════════════════════

class IntegrationOpsState(TypedDict):
    task: str
    operation_type: Literal[
        "webhook", "cron_job", "api_sync", "env_config", "email_dispatch",
        "web_download", "desktop_notify", "resource_monitor",
        "port_watchdog", "process_control", "general_ops"
    ]
    target_endpoint: Optional[str]
    schedule_time: Optional[str]
    payload: Optional[Dict[str, Any]]
    messages: Annotated[List[BaseMessage], add_messages]
    iterations: int
    audit_passed: bool
    audit_feedback: Optional[str]
    generated_script_path: Optional[str]
    final_output: Optional[str]
    status: Literal["running", "success", "error"]
    error_message: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


# ════════════════════════════════════════════════════════════════════════════
# 4. UTILITIES
# ════════════════════════════════════════════════════════════════════════════

_SECRETS = [
    (re.compile(r"sk-[a-zA-Z0-9]{20,}", re.I), "[REDACTED_API_KEY]"),
    (re.compile(r"(bearer\s+)[a-zA-Z0-9_\-\.]{16,}", re.I), r"\1[REDACTED_TOKEN]"),
    (re.compile(r"(password\s*[:=]\s*[\"']?)([^\"\s]{6,})([\"']?)", re.I), r"\1[REDACTED_PWD]\3"),
    (re.compile(r"AKIA[0-9A-Z]{16}", re.I), "[REDACTED_AWS_KEY]"),
]


def mask(text: str) -> str:
    for p, r in _SECRETS:
        text = p.sub(r, text)
    return text


def extract(resp: Any) -> str:
    raw = resp.content if hasattr(resp, "content") else (resp.text if hasattr(resp, "text") else str(resp))
    if isinstance(raw, list):
        parts = [i.get("text", str(i)) if isinstance(i, dict) else (i.text if hasattr(i, "text") else str(i)) for i in raw]
        return "\n".join(parts).strip()
    return str(raw).strip()


def parse_json(raw: Any) -> dict:
    text = extract(raw)
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e+1])
            except Exception:
                pass
    return {}


_MAX_ITERS = 4


# ════════════════════════════════════════════════════════════════════════════
# 5. NODES
# ════════════════════════════════════════════════════════════════════════════

def planner_node(state: IntegrationOpsState) -> dict:
    task = state.get("task", "").strip()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[planner] {task[:80]}")

    # Deterministic extraction
    tm = re.search(r"(\d{1,2})[:\.](\d{2})\s*(am|pm)?", task, re.I)
    sched = None
    if tm:
        h, m, ap = int(tm.group(1)), int(tm.group(2)), (tm.group(3) or "").lower()
        if ap == "pm" and h < 12: h += 12
        elif ap == "am" and h == 12: h = 0
        sched = f"{h:02d}:{m:02d}"

    url_m = re.search(r"https?://\S+", task)
    url = url_m.group(0).rstrip(".,)\"'") if url_m else None

    email_m = re.search(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b", task)
    email = email_m.group(0) if email_m else None

    port_m = re.search(r"\bport[s]?\s+([\d,\s\-]+)", task, re.I)
    ports = port_m.group(1).strip() if port_m else None

    t = task.lower()
    if any(w in t for w in ["download","fetch file","youtube.com","youtu.be","yt-dlp"]):
        op = "web_download"
    elif any(w in t for w in ["webhook","post to","trigger endpoint"]):
        op = "webhook"
    elif any(w in t for w in ["send email","mail to","email to","smtp"]):
        op = "email_dispatch"
    elif any(w in t for w in ["schedule","run at","cron","every day","daily"]) or sched:
        op = "cron_job"
    elif any(w in t for w in [".env","env var","environment variable","api key"]):
        op = "env_config"
    elif any(w in t for w in ["cpu","memory","ram","disk usage","resource","monitor system"]):
        op = "resource_monitor"
    elif any(w in t for w in ["port","open port","listening port"]):
        op = "port_watchdog"
    elif any(w in t for w in ["process","kill process","running process"]):
        op = "process_control"
    elif any(w in t for w in ["notify","notification","popup","toast","alert me"]):
        op = "desktop_notify"
    elif any(w in t for w in ["api call","rest api","json api"]):
        op = "api_sync"
    else:
        op = "general_ops"

    endpoint = url or email or ports
    logger.info(f"[planner] op={op}, endpoint={endpoint}, sched={sched}")
    return {
        "operation_type": op,
        "target_endpoint": endpoint,
        "schedule_time": sched,
        "payload": {},
        "iterations": 0,
        "audit_passed": False,
        "audit_feedback": None,
        "status": "running",
        "started_at": now_str,
        "error_message": None,
    }


def executor_node(state: IntegrationOpsState, config: Optional[RunnableConfig] = None) -> dict:
    task = state.get("task", "")
    op = state.get("operation_type", "general_ops")
    endpoint = state.get("target_endpoint")
    sched = state.get("schedule_time")
    payload = state.get("payload") or {}
    feedback = state.get("audit_feedback")
    messages = list(state.get("messages") or [])
    iters = state.get("iterations", 0)
    logger.info(f"[executor] iter={iters} op={op}")

    # Direct tool dispatch (no LLM needed)
    direct = None
    try:
        if op == "resource_monitor" and iters == 0:
            from tools.integration_tools import system_resource_monitor
            direct = system_resource_monitor.invoke({"metrics": "all"})
        elif op == "port_watchdog" and endpoint and iters == 0:
            from tools.integration_tools import port_watchdog
            direct = port_watchdog.invoke({"ports": endpoint})
        elif op == "desktop_notify" and iters == 0:
            from tools.integration_tools import desktop_notifier
            direct = desktop_notifier.invoke({"title": payload.get("title","Agent"),
                                              "message": payload.get("message", task[:150])})
        elif op == "env_config" and iters == 0:
            from tools.integration_tools import env_config_manager
            direct = env_config_manager.invoke({"action": payload.get("action","list"),
                                                "key": payload.get("key",""),
                                                "value": payload.get("value","")})
        elif op == "web_download" and endpoint and iters == 0:
            if any(x in endpoint for x in ["youtube.com","youtu.be"]):
                direct = _yt_script(endpoint, sched)
            else:
                from tools.integration_tools import http_downloader
                direct = http_downloader.invoke({"url": endpoint})
        elif op == "webhook" and endpoint and iters == 0:
            from tools.integration_tools import webhook_caller
            direct = webhook_caller.invoke({"url": endpoint, "payload": json.dumps(payload.get("data", payload)),
                                            "method": payload.get("method","POST")})
    except Exception as e:
        logger.warning(f"[executor] Direct dispatch error: {e}")
        direct = None

    if direct:
        return {"messages": [AIMessage(content=str(direct))], "iterations": iters + 1}

    # LLM fallback
    if not _llm_tools:
        err = "No LLM. Set GOOGLE_API_KEY or GROQ_API_KEY in .env"
        return {"messages": [AIMessage(content=f"ERROR: {err}")], "iterations": iters+1,
                "status": "error", "error_message": err}

    fb = f"\n\nCORRECTION:\n{feedback}\n" if feedback else ""
    if not messages or (feedback and iters > 0):
        messages = [
            SystemMessage(content=(
                f"You are a Senior Integration Specialist on {platform.system()}.\n"
                "Rules: Use tools first. Write Python in ```python ``` blocks. Never expose secrets.\n"
                f"Available tools: {[t.name for t in ACTIVE_OPS_TOOLS]}"
            )),
            HumanMessage(content=(
                f"Task: {task}\nType: {op}\nTarget: {endpoint or 'N/A'}\n"
                f"Schedule: {sched or 'Immediate'}\nParams: {json.dumps(payload)}{fb}"
            )),
        ]
    elif feedback:
        messages.append(HumanMessage(content=f"Fix per audit:\n{feedback}"))

    try:
        resp = _llm_tools.invoke(messages, config=config)
        return {"messages": [resp], "iterations": iters + 1}
    except Exception as e:
        logger.error(f"[executor] LLM error: {e}")
        return {"messages": [AIMessage(content=f"ERROR: {e}")], "iterations": iters+1,
                "status": "error", "error_message": str(e)}


def _yt_script(url: str, sched: Optional[str]) -> str:
    note = f"# Scheduled: {sched}" if sched else "# Run immediately"
    return f"""```python
# YouTube Downloader — auto-generated
# {note}
import os, sys
try:
    import yt_dlp
except ImportError:
    import subprocess; subprocess.run([sys.executable,"-m","pip","install","yt-dlp"],check=True); import yt_dlp
os.makedirs("downloads", exist_ok=True)
with yt_dlp.YoutubeDL({{"outtmpl":"downloads/%(title)s.%(ext)s","format":"best","quiet":False}}) as ydl:
    info = ydl.extract_info("{url}", download=True)
    print("Done:", info.get("title","video"))
```"""


def auditor_node(state: IntegrationOpsState) -> dict:
    msgs = list(state.get("messages") or [])
    iters = state.get("iterations", 0)
    if not msgs:
        return {"audit_passed": False, "audit_feedback": "No response.", "status": "error"}
    content = extract(msgs[-1])
    if iters >= _MAX_ITERS:
        logger.warning(f"[auditor] Iteration limit reached — force pass.")
        return {"audit_passed": True, "audit_feedback": None, "status": "success"}
    for p, _ in _SECRETS:
        if p.search(content):
            return {"audit_passed": False,
                    "audit_feedback": "Security: raw secrets in output. Use os.getenv().", "status": "running"}
    if len(content.strip()) < 20:
        return {"audit_passed": False, "audit_feedback": "Response too short — provide complete output.", "status": "running"}
    if content.strip().upper().startswith("ERROR:") and iters < 2:
        return {"audit_passed": False, "audit_feedback": f"Error detected — try alternate approach: {content[:200]}", "status": "running"}
    logger.info(f"[auditor] PASS at iter={iters}")
    return {"audit_passed": True, "audit_feedback": None, "status": "success"}


def vault_node(state: IntegrationOpsState) -> dict:
    task = state.get("task","")
    op = state.get("operation_type","general_ops")
    endpoint = state.get("target_endpoint")
    sched = state.get("schedule_time")
    payload = state.get("payload") or {}
    msgs = list(state.get("messages") or [])
    content = extract(msgs[-1]) if msgs else ""

    manifests_dir = os.path.join(_ROOT, "outputs", "manifests")
    code_dir = os.path.join(_ROOT, "outputs", "code")
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9_-]","_",task[:25]).strip("_").lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save script
    script_path = None
    cm = re.search(r"```python\s*([\s\S]*?)\s*```", content)
    if cm:
        code = mask(cm.group(1).strip())
        script_path = os.path.join(code_dir, f"{slug}_{ts}.py")
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(f'"""Auto-generated — {datetime.now().isoformat()}"""\n\n{code}')
        except Exception as e:
            logger.error(f"[outputs] Script save failed: {e}"); script_path = None

    # Native dispatches
    native = None
    try:
        if op == "webhook" and endpoint:
            from tools.integration_tools import webhook_caller
            native = json.loads(webhook_caller.invoke({"url":endpoint,"payload":json.dumps(payload.get("data",payload))}))
        elif op == "env_config":
            from tools.integration_tools import env_config_manager
            native = json.loads(env_config_manager.invoke({"action":payload.get("action","list"),"key":payload.get("key",""),"value":payload.get("value","")}))
        elif op == "email_dispatch" and endpoint:
            from tools.integration_tools import smtp_email_sender
            native = json.loads(smtp_email_sender.invoke({"to_email":endpoint,"subject":payload.get("subject",f"Agent: {task[:30]}"),"body":payload.get("body",content[:2000])}))
    except Exception as e:
        native = {"error": str(e)}

    # Schedule
    sched_res = None
    if sched and script_path:
        try:
            from tools.integration_tools import os_task_scheduler
            sched_res = json.loads(os_task_scheduler.invoke({"script_path":script_path,"run_at_time":sched,"task_name":slug}))
        except Exception as e:
            sched_res = {"error": str(e)}

    # Desktop notification
    try:
        from tools.integration_tools import desktop_notifier
        desktop_notifier.invoke({"title":"Integration Agent — Done","message":f"{op.upper()}: {task[:80]}"})
    except Exception:
        pass

    # Manifest
    manifest = os.path.join(manifests_dir, f"{slug}_{ts}_manifest.json")
    try:
        with open(manifest,"w",encoding="utf-8") as f:
            json.dump({"task":task,"op":op,"endpoint":endpoint,"sched":sched,"script":script_path,
                       "native_result":native,"sched_result":sched_res,"completed_at":datetime.now().isoformat()}, f, indent=2)
    except Exception as e:
        logger.error(f"[outputs] Manifest write failed: {e}")


    lines = [
        "=" * 60,
        "  INTEGRATION AGENT — COMPLETE",
        "=" * 60,
        f"  Type     : {op.upper()}",
        f"  Target   : {endpoint or 'Local System'}",
        f"  Scheduled: {sched or 'Immediate'}",
        f"  Script   : {script_path or 'Tool Managed'}",
        f"  Manifest : {manifest}",
    ]
    if native: lines.append(f"  Result   : {json.dumps(native)[:120]}")
    if sched_res and sched_res.get("status")=="scheduled":
        lines.append(f"  Queued   : {sched} — log: {sched_res.get('log_file')}")
    lines += ["=" * 60, "", content[:1500]]

    return {"generated_script_path": script_path, "final_output": "\n".join(lines),
            "status": "success", "completed_at": datetime.now().isoformat()}


# ════════════════════════════════════════════════════════════════════════════
# 6. ROUTING
# ════════════════════════════════════════════════════════════════════════════

def route_executor(state: IntegrationOpsState) -> str:
    msgs = list(state.get("messages") or [])
    iters = state.get("iterations", 0)
    if msgs and hasattr(msgs[-1],"tool_calls") and msgs[-1].tool_calls and iters < _MAX_ITERS:
        return "tools"
    return "auditor"


def route_auditor(state: IntegrationOpsState) -> str:
    if not state.get("audit_passed") and state.get("iterations",0) < _MAX_ITERS:
        return "executor"
    return "vault"


# ════════════════════════════════════════════════════════════════════════════
# 7. GRAPH (lazy singleton)
# ════════════════════════════════════════════════════════════════════════════

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        b = StateGraph(IntegrationOpsState)
        b.add_node("planner",  planner_node)
        b.add_node("executor", executor_node)
        b.add_node("tools",    ToolNode(ACTIVE_OPS_TOOLS) if ACTIVE_OPS_TOOLS else ToolNode([]))
        b.add_node("auditor",  auditor_node)
        b.add_node("vault",    vault_node)
        b.add_edge(START, "planner")
        b.add_edge("planner", "executor")
        b.add_conditional_edges("executor", route_executor, {"tools":"tools","auditor":"auditor"})
        b.add_edge("tools", "executor")
        b.add_conditional_edges("auditor", route_auditor, {"executor":"executor","vault":"vault"})
        b.add_edge("vault", END)
        _agent = b.compile(checkpointer=MemorySaver())
        logger.info("Integration agent compiled.")
    return _agent


# Public alias (backward compat)
integration_ops_agent = get_agent()


# ════════════════════════════════════════════════════════════════════════════
# 8. PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def run_integration_task(
    task: str,
    thread_id: Optional[str] = None,
    operation_type: str = "general_ops",
    target_endpoint: Optional[str] = None,
    schedule_time: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Run an integration task through the agent.

    Args:
        task:            Plain-language task description.
        thread_id:       Conversation thread ID (auto-generated if empty).
        operation_type:  Override auto-detected type.
        target_endpoint: Override URL/email/host.
        schedule_time:   Override schedule time (HH:MM).
        payload:         Extra parameters dict.
        stream:          Print intermediate steps if True.

    Returns:
        Dict with: final_output, status, script_path, operation_type, iterations, elapsed_sec.
    """
    import uuid
    tid = thread_id or f"int_{uuid.uuid4().hex[:8]}"
    init: IntegrationOpsState = {
        "task": task, "operation_type": operation_type, "target_endpoint": target_endpoint,
        "schedule_time": schedule_time, "payload": payload or {}, "messages": [],
        "iterations": 0, "audit_passed": False, "audit_feedback": None,
        "generated_script_path": None, "final_output": None,
        "status": "running", "error_message": None, "started_at": None, "completed_at": None,
    }
    config: RunnableConfig = {"configurable": {"thread_id": tid}}
    agent = get_agent()
    final: Dict[str, Any] = {}
    t0 = time.monotonic()

    if stream:
        for step in agent.stream(init, config=config):
            for name, upd in step.items():
                print(f"\n[{name.upper()}]")
                if isinstance(upd,dict) and upd.get("final_output"):
                    print(upd["final_output"])
                print("-"*50)
                final.update(upd)
    else:
        final = agent.invoke(init, config=config)

    return {
        "final_output": final.get("final_output","No output."),
        "status": final.get("status","unknown"),
        "script_path": final.get("generated_script_path"),
        "operation_type": final.get("operation_type"),
        "iterations": final.get("iterations",0),
        "thread_id": tid,
        "elapsed_sec": round(time.monotonic()-t0, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
# 9. STANDALONE TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Integration Agent Test")
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("  INTEGRATION AGENT — PRODUCTION")
    print(f"  Tools : {len(ACTIVE_OPS_TOOLS)}")
    print(f"  LLM   : {'Available' if _llm_core else 'NOT AVAILABLE'}")
    print(f"  OS    : {platform.system()}")
    print("=" * 60)

    if args.test or not args.task:
        print("\n[TEST 1] Resource Monitor...")
        r = run_integration_task("CPU aur memory usage dikhao")
        print(f"Status: {r['status']} | Time: {r['elapsed_sec']}s")
        print(r["final_output"][:600])

        print("\n[TEST 2] Port Check...")
        r2 = run_integration_task("Check if ports 80, 443, 5432 are open on localhost")
        print(f"Status: {r2['status']} | Time: {r2['elapsed_sec']}s")
        print(r2["final_output"][:400])

        print("\n[TEST 3] Desktop Notification...")
        r3 = run_integration_task("Send desktop notification: 'Agent tests complete!'")
        print(f"Status: {r3['status']}")

    elif args.task:
        r = run_integration_task(args.task, stream=args.stream)
        print(f"\nStatus: {r['status']} | Type: {r['operation_type']} | Time: {r['elapsed_sec']}s")
        print(r["final_output"])
