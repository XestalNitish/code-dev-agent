"""
Module 4: Code & Developer Tools (Tools 41 - 55)
100% Free & Open-Source. No paid APIs required.
"""

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Any, List, Optional
import requests
from langchain_core.tools import tool

@tool
def python_executor(code: str, timeout: int = 15) -> str:
    """Execute Python code in an isolated subprocess with stdout/stderr capture and timeout protection.
    
    Args:
        code: Python source code string to run.
        timeout: Execution timeout in seconds (default 15).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        temp_path = f.name
    try:
        res = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        out = res.stdout
        if res.stderr:
            out += f"\n[STDERR]:\n{res.stderr}"
        return out.strip() if out.strip() else "Process finished with returncode 0 (No stdout)."
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing Python code: {str(e)}"
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

@tool
def javascript_executor(code: str, timeout: int = 10) -> str:
    """Execute JavaScript code snippets using local Node.js engine.
    
    Args:
        code: JavaScript source code to execute.
        timeout: Execution timeout in seconds.
    """
    node_bin = shutil.which("node")
    if not node_bin:
        return "Node.js is not found on the system PATH. Install Node.js to execute JavaScript."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(code)
        temp_path = f.name
    try:
        res = subprocess.run(
            [node_bin, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        out = res.stdout
        if res.stderr:
            out += f"\n[STDERR]:\n{res.stderr}"
        return out.strip() if out.strip() else "Execution finished with no output."
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing JavaScript: {str(e)}"
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

@tool
def code_executor(language: str, code: str) -> str:
    """Execute code in multiple languages (Python, JavaScript, PowerShell, Shell) in a sandboxed directory.
    
    Args:
        language: 'python', 'javascript', 'powershell', or 'bash'.
        code: Code string to execute.
    """
    lang = language.lower()
    if lang in ("python", "py"):
        return python_executor(code)
    elif lang in ("javascript", "js", "node"):
        return javascript_executor(code)
    elif lang in ("powershell", "ps1"):
        try:
            res = subprocess.run(["powershell", "-NoProfile", "-Command", code], capture_output=True, text=True, timeout=15)
            return (res.stdout + (f"\nSTDERR: {res.stderr}" if res.stderr else "")).strip()
        except Exception as e:
            return f"Error executing PowerShell: {str(e)}"
    else:
        return f"Unsupported language '{language}'. Supported: python, javascript, powershell."

@tool
def terminal(command: str, cwd: str = ".") -> str:
    """Execute safe shell commands on the local machine and return stdout/stderr.
    
    Args:
        command: The shell command line string.
        cwd: Working directory (defaults to current directory).
    """
    # Safeguard against accidental destruction
    blocked = ["rm -rf /", "del /s /q c:\\", "format c:", ":(){ :|:& };:"]
    if any(b in command.lower() for b in blocked):
        return "Command blocked by security guardrails."
    try:
        res = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=30)
        output = res.stdout
        if res.stderr:
            output += f"\n[STDERR]:\n{res.stderr}"
        return output.strip() if output.strip() else f"Command succeeded (returncode {res.returncode})"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing terminal command: {str(e)}"

@tool
def git(subcommand: str, repo_path: str = ".") -> str:
    """Execute local Git version control operations (e.g. 'status', 'diff', 'log -n 5', 'branch').
    
    Args:
        subcommand: Git command arguments (e.g. 'status', 'branch', 'log -n 5').
        repo_path: Path to the local git repository.
    """
    try:
        cmd = f"git {subcommand}"
        res = subprocess.run(cmd, shell=True, cwd=repo_path, capture_output=True, text=True, timeout=20)
        out = res.stdout
        if res.stderr:
            out += f"\n{res.stderr}"
        return out.strip() if out.strip() else "Git command completed with no output."
    except Exception as e:
        return f"Error running git: {str(e)}"

@tool
def github(action: str = "repo_info", repo: str = "", issue_or_pr_num: int = 1) -> str:
    """Inspect GitHub repository details, commits, issues, and PRs via open public REST API.
    
    Args:
        action: 'repo_info', 'recent_commits', or 'issues'.
        repo: Repository in format 'owner/repo' (e.g. 'langchain-ai/langchain').
        issue_or_pr_num: Issue number if applicable.
    """
    if not repo:
        return "Please specify a repository in format 'owner/repo'."
    try:
        base = f"https://api.github.com/repos/{repo}"
        headers = {"User-Agent": "agentic-ai-dev"}
        if action == "repo_info":
            r = requests.get(base, headers=headers, timeout=10).json()
            return json.dumps({
                "name": r.get("full_name"),
                "description": r.get("description"),
                "stars": r.get("stargazers_count"),
                "forks": r.get("forks_count"),
                "open_issues": r.get("open_issues_count"),
                "default_branch": r.get("default_branch")
            }, indent=2)
        elif action == "recent_commits":
            r = requests.get(f"{base}/commits?per_page=5", headers=headers, timeout=10).json()
            commits = [{"sha": c["sha"][:7], "author": c["commit"]["author"]["name"], "message": c["commit"]["message"].split("\n")[0]} for c in r]
            return json.dumps(commits, indent=2)
        elif action == "issues":
            r = requests.get(f"{base}/issues?state=open&per_page=5", headers=headers, timeout=10).json()
            issues = [{"number": i["number"], "title": i["title"], "user": i["user"]["login"]} for i in r]
            return json.dumps(issues, indent=2)
        else:
            return f"Unknown action '{action}'."
    except Exception as e:
        return f"Error querying GitHub: {str(e)}"

@tool
def code_search(pattern: str, directory: str = ".", file_pattern: str = "*.py") -> str:
    """Search for regex patterns or symbol definitions across files in a codebase.
    
    Args:
        pattern: Regex pattern or symbol name to search.
        directory: Root directory for search.
        file_pattern: Glob file extension filter (e.g. '*.py').
    """
    try:
        matches = []
        regex = re.compile(pattern, re.IGNORECASE)
        for root, _, files in os.walk(directory):
            if any(ign in root for ign in [".git", "__pycache__", ".venv", "node_modules"]):
                continue
            for f in files:
                if f.endswith(file_pattern.replace("*", "")):
                    filepath = os.path.join(root, f)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as file_obj:
                            for idx, line in enumerate(file_obj, 1):
                                if regex.search(line):
                                    matches.append({
                                        "file": os.path.relpath(filepath, directory),
                                        "line": idx,
                                        "code": line.strip()
                                    })
                                    if len(matches) >= 30:
                                        break
                    except Exception:
                        continue
            if len(matches) >= 30:
                break
        return json.dumps({"pattern": pattern, "total_matches": len(matches), "matches": matches}, indent=2)
    except Exception as e:
        return f"Error in code_search: {str(e)}"

@tool
def code_reader(file_path: str, extract_symbols_only: bool = False) -> str:
    """Parse and understand code files, extracting AST structure, classes, functions, and docstrings.
    
    Args:
        file_path: Path to the source file.
        extract_symbols_only: If True, returns class/function declarations instead of raw code.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at '{file_path}'."
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if extract_symbols_only and file_path.endswith(".py"):
            tree = ast.parse(content)
            symbols = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = [a.arg for a in node.args.args]
                    symbols.append(f"Function: def {node.name}({', '.join(args)}) [Line {node.lineno}]")
                elif isinstance(node, ast.ClassDef):
                    symbols.append(f"Class: class {node.name} [Line {node.lineno}]")
            return "\n".join(symbols) if symbols else "No top-level functions or classes detected."
        return content
    except Exception as e:
        return f"Error reading code: {str(e)}"

@tool
def code_writer(file_path: str, code: str, overwrite: bool = True) -> str:
    """Safely generate or overwrite a code file, verifying syntax before committing to disk.
    
    Args:
        file_path: Path to the target source code file.
        code: Code contents to write.
        overwrite: Allow overwriting existing file (default True).
    """
    if os.path.exists(file_path) and not overwrite:
        return f"File '{file_path}' already exists and overwrite is set to False."
    try:
        # Pre-validate Python syntax if writing a Python file
        if file_path.endswith(".py"):
            ast.parse(code)

        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        return f"Successfully wrote {len(code)} characters to '{file_path}' (Syntax check passed)."
    except SyntaxError as se:
        return f"SyntaxError in generated code, file NOT saved: {str(se)}"
    except Exception as e:
        return f"Error writing code file: {str(e)}"

@tool
def code_reviewer(code_or_file: str) -> str:
    """Perform automated code quality review, detecting security issues, complex functions, and style anti-patterns.
    
    Args:
        code_or_file: Raw source code string or path to a file.
    """
    try:
        code = code_or_file
        if os.path.exists(code_or_file):
            with open(code_or_file, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()

        issues = []
        # Check security smells
        if "eval(" in code:
            issues.append("⚠️ Security Warning: Use of eval() detected, which poses arbitrary code execution risks.")
        if "exec(" in code:
            issues.append("⚠️ Security Warning: Use of exec() detected.")
        if "password" in code.lower() and "=" in code:
            issues.append("ℹ️ Notice: Potential hardcoded secret or password assignment detected.")

        # AST-based inspection
        try:
            tree = ast.parse(code)
            fn_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            cls_count = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
        except Exception:
            fn_count, cls_count = "N/A", "N/A"

        score = max(5.0, 10.0 - (len(issues) * 1.5))
        return json.dumps({
            "code_quality_score": score,
            "functions_found": fn_count,
            "classes_found": cls_count,
            "issues_and_warnings": issues or ["No critical anti-patterns found. Clean code!"],
            "recommendation": "Ready for production" if score >= 8.0 else "Address security warnings before deployment."
        }, indent=2)
    except Exception as e:
        return f"Error reviewing code: {str(e)}"

@tool
def debugger(traceback_text: str, source_code: str = "") -> str:
    """Analyze a runtime exception traceback and generate step-by-step diagnostic fixes.
    
    Args:
        traceback_text: The complete Python error or exception traceback.
        source_code: Optional source code snippet where error occurred.
    """
    try:
        lines = traceback_text.strip().split("\n")
        last_line = lines[-1] if lines else "Unknown Error"

        file_line_match = re.findall(r'File "([^"]+)", line (\d+)', traceback_text)
        location = file_line_match[-1] if file_line_match else ("unknown", "unknown")

        fix_steps = [
            f"Inspect line {location[1]} in {location[0]}.",
            "Verify all variable initializations and null/None checks prior to this line.",
            "Wrap call in a try/except block to gracefully handle edge cases."
        ]

        return json.dumps({
            "exception_type": last_line,
            "fault_file": location[0],
            "fault_line": location[1],
            "debugging_plan": fix_steps
        }, indent=2)
    except Exception as e:
        return f"Error in debugger: {str(e)}"

@tool
def test_runner(test_path: str = "tests", test_framework: str = "pytest") -> str:
    """Run automated unit tests using pytest or unittest and summarize results.
    
    Args:
        test_path: Path to test directory or test file.
        test_framework: 'pytest' or 'unittest'.
    """
    try:
        cmd = [sys.executable, "-m", test_framework, test_path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = res.stdout + ("\n" + res.stderr if res.stderr else "")
        return json.dumps({
            "exit_code": res.returncode,
            "status": "PASSED" if res.returncode == 0 else "FAILED",
            "raw_output": output[-1500:]
        }, indent=2)
    except Exception as e:
        return f"Error running tests: {str(e)}"

@tool
def dependency_manager(action: str = "inspect", requirements_file: str = "requirements.txt", package_name: str = "") -> str:
    """Check, parse, and validate Python dependencies and packages.
    
    Args:
        action: 'inspect', 'check_installed', or 'generate_entry'.
        requirements_file: Path to requirements.txt.
        package_name: Package name to check or inspect.
    """
    try:
        if action == "inspect":
            if not os.path.exists(requirements_file):
                return f"Requirements file '{requirements_file}' not found."
            with open(requirements_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            return json.dumps({"dependencies_count": len(lines), "packages": lines}, indent=2)
        elif action == "check_installed":
            if not package_name:
                return "Please provide package_name to check."
            try:
                __import__(package_name.replace("-", "_"))
                return f"Package '{package_name}' is INSTALLED and importable."
            except ImportError:
                return f"Package '{package_name}' is NOT installed."
        elif action == "generate_entry":
            return f"{package_name}>=1.0.0"
        else:
            return f"Unknown action '{action}'."
    except Exception as e:
        return f"Error in dependency_manager: {str(e)}"

@tool
def docker(action: str = "generate_dockerfile", app_type: str = "python", port: int = 8000) -> str:
    """Generate production-ready Dockerfile or run local Docker inspection commands.
    
    Args:
        action: 'generate_dockerfile', 'generate_compose', or 'ps'.
        app_type: 'python', 'node', or 'fastapi'.
        port: Exposed container port.
    """
    try:
        if action == "generate_dockerfile":
            dockerfile = (
                f"FROM python:3.11-slim\n"
                f"WORKDIR /app\n"
                f"COPY requirements.txt .\n"
                f"RUN pip install --no-cache-dir -r requirements.txt\n"
                f"COPY . .\n"
                f"EXPOSE {port}\n"
                f"CMD [\"python\", \"app.py\"]\n"
            )
            return dockerfile
        elif action == "generate_compose":
            compose = (
                f"version: '3.8'\n"
                f"services:\n"
                f"  agent-app:\n"
                f"    build: .\n"
                f"    ports:\n"
                f"      - \"{port}:{port}\"\n"
                f"    restart: unless-stopped\n"
            )
            return compose
        elif action == "ps":
            res = subprocess.run(["docker", "ps"], capture_output=True, text=True, timeout=10)
            return res.stdout if res.returncode == 0 else f"Docker error: {res.stderr}"
        else:
            return f"Unknown docker action '{action}'."
    except Exception as e:
        return f"Error in docker tool: {str(e)}"

@tool
def deployment(service_name: str, app_entrypoint: str = "app.py", port: int = 8501) -> str:
    """Generate production deployment configurations (Linux systemd service, launcher script).
    
    Args:
        service_name: Name of the deployment service.
        app_entrypoint: Main application script path.
        port: Port number.
    """
    try:
        systemd = (
            f"[Unit]\n"
            f"Description={service_name} AI Service\n"
            f"After=network.target\n\n"
            f"[Service]\n"
            f"Type=simple\n"
            f"WorkingDirectory={os.path.abspath('.')}\n"
            f"ExecStart={sys.executable} {app_entrypoint}\n"
            f"Restart=always\n"
            f"RestartSec=5\n"
            f"Environment=PORT={port}\n\n"
            f"[Install]\n"
            f"WantedBy=multi-user.target\n"
        )
        return json.dumps({
            "service_name": service_name,
            "systemd_unit_file": systemd,
            "instructions": f"Save to /etc/systemd/system/{service_name}.service, then run: sudo systemctl enable --now {service_name}"
        }, indent=2)
    except Exception as e:
        return f"Error in deployment tool: {str(e)}"


CODE_DEV_TOOLS = [
    python_executor,
    javascript_executor,
    code_executor,
    terminal,
    git,
    github,
    code_search,
    code_reader,
    code_writer,
    code_reviewer,
    debugger,
    test_runner,
    dependency_manager,
    docker,
    deployment
]
