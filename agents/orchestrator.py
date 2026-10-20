"""
orchestrator.py — Master Orchestration & Unified Multi-Agent Chatbot
====================================================================
Seamlessly routes user requests to 7 Autonomous Specialist Agents & 150 Tools:
- Code & Dev Engineer (code_dev)
- Integration & OS Ops (integration)
- Deep Web Researcher (research_web)
- Database & SQL Analyst (database)
- Media & Content Producer (media_content)
- Memory & Context Vault (memory_context)
- AI/ML Engineering Agent (ai_ml_engineering)

100% Offline, Local X-Estal-4B Inference Engine via llama-server (Port 8088).
Zero Thinking Tags. Zero External APIs.
"""

from __future__ import annotations

import os
import sys
import re
import json
import logging
from typing import Dict, Any, List, Optional, Generator

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from tools import ALL_TOOLS, TOOLS_BY_CATEGORY, get_tool_by_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("orchestrator")


def get_llm(temperature: float = 0.2):
    """
    100% Offline & Free Local X-Estal-4B Inference Engine.
    No paid APIs required.
    """
    try:
        from local_llm_service import get_local_chat_llm
        return get_local_chat_llm(temperature=temperature)
    except Exception as e:
        logger.warning(f"Local LLM service hook: {e}")

    local_base = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8088/v1")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=local_base,
        api_key="x-estal-local",
        model_name="X-Estal-4B",
        temperature=temperature,
    )


SYSTEM_PROMPT = """You are X-Estal, a high-performance autonomous Agentic AI system running 100% locally on the user's system.
You have access to 150 production-grade open-source tools across 18 domains and 7 autonomous specialist agents (Coding, Web Research, Databases, OS Operations, Media, Memory, and AI/ML).

CRITICAL INSTRUCTIONS:
1. NEVER output internal thinking tags like `<think>` or reasoning monologues. Answer directly, concisely, and cleanly.
2. Provide direct, authoritative, and helpful answers in the user's language (Hindi / Hinglish / English).
3. When providing code, use proper markdown fences with language tags (e.g. ```python).
4. If an action or tool execution is requested, state the result clearly.
"""


def detect_specialist_agent(prompt: str) -> Optional[str]:
    """
    Determine if a user prompt should be dispatched to a specialist agent.
    Returns agent key (e.g. 'code_dev', 'integration', 'research_web', 'database', etc.) or None.
    """
    lower = prompt.lower().strip()

    # Explicit prefix tags from UI
    if lower.startswith("[code & dev engineer") or lower.startswith("[code task") or lower.startswith("[code"):
        return "code_dev"
    if lower.startswith("[deep research") or lower.startswith("[research task") or lower.startswith("[research"):
        return "research_web"
    if lower.startswith("[database") or lower.startswith("[sql analyst") or lower.startswith("[db"):
        return "database"
    if lower.startswith("[integration") or lower.startswith("[os ops") or lower.startswith("[ops"):
        return "integration"
    if lower.startswith("[media") or lower.startswith("[content agent") or lower.startswith("[video"):
        return "media_content"
    if lower.startswith("[memory") or lower.startswith("[vault"):
        return "memory_context"
    if lower.startswith("[ai/ml") or lower.startswith("[machine learning"):
        return "ai_ml_engineering"

    # Natural Language Keyword Routing
    # 1. Integration & OS Ops
    ops_patterns = [
        "cpu", "ram", "disk usage", "system resource", "cron expression",
        "webhook", "kill process", "list process", "port watchdog",
        "resource monitor", "system monitor", "notify desktop", "desktop notification"
    ]
    if any(p in lower for p in ops_patterns):
        return "integration"

    # 2. Database & SQL
    db_patterns = [
        "sql query", "select *", "database schema", "table schema",
        "sqlite database", "postgres query", "mysql query", "mongodb collection",
        "schema analyzer", "generate sql", "sql migration"
    ]
    if any(p in lower for p in db_patterns):
        return "database"

    # 3. Code Development & Engineering
    code_patterns = [
        "write python script", "python script banao", "code likho", "script banao",
        "debug this code", "code review", "unit test likho", "run pytest",
        "javascript function", "create script", "write code for"
    ]
    if any(p in lower for p in code_patterns):
        return "code_dev"

    # 4. Web Research
    research_patterns = [
        "search the web", "duckduckgo", "latest news on", "arxiv paper",
        "google trends", "research about", "summarize webpage", "web search for"
    ]
    if any(p in lower for p in research_patterns):
        return "research_web"

    # 5. Media & Content
    media_patterns = [
        "youtube shorts script", "video script", "tiktok script", "reel script",
        "generate voiceover", "tts audio", "podcast script", "storyboard"
    ]
    if any(p in lower for p in media_patterns):
        return "media_content"

    # 6. Memory & Recall
    memory_patterns = [
        "save to long term memory", "remember this fact", "store in memory",
        "recall from memory", "vector search memory"
    ]
    if any(p in lower for p in memory_patterns):
        return "memory_context"

    # 7. AI / ML
    ml_patterns = [
        "train model", "train classifier", "churn dataset", "scikit-learn pipeline",
        "dataset inspector", "huggingface model search", "count tokens in"
    ]
    if any(p in lower for p in ml_patterns):
        return "ai_ml_engineering"

    return None


def execute_specialist_agent(agent_key: str, task: str) -> Dict[str, Any]:
    """Execute the designated specialist agent and format output."""
    clean_task = re.sub(r"^\[.*?\]:\s*", "", task).strip()

    try:
        if agent_key == "integration":
            from agents.integration import run_integration_task
            res = run_integration_task(clean_task)
            output = res.get("final_output") or "Integration & OS Ops completed."
            return {
                "status": "success",
                "agent": "⚙️ Integration & OS Ops Agent",
                "reply": f"⚙️ **[Executed by: Integration & OS Ops Agent]**\n\n{output}",
                "data": res,
            }

        elif agent_key == "code_dev":
            from agents.code_dev import run_code_dev_task
            res = run_code_dev_task(clean_task)
            reply = res.get("reply") or res.get("code") or "Code task executed."
            return {
                "status": "success",
                "agent": "💻 Code & Dev Engineer",
                "reply": f"💻 **[Executed by: Code & Dev Engineer]**\n\n{reply}",
                "data": res,
            }

        elif agent_key == "database":
            from agents.database import run_database_task
            res = run_database_task(clean_task)
            reply = res.get("reply") or "Database analysis completed."
            return {
                "status": "success",
                "agent": "🗄️ Database & SQL Analyst",
                "reply": f"🗄️ **[Executed by: Database & SQL Analyst]**\n\n{reply}",
                "data": res,
            }

        elif agent_key == "research_web":
            from agents.research_web import run_research_task
            res = run_research_task(clean_task)
            reply = res.get("reply") or "Research completed."
            return {
                "status": "success",
                "agent": "🔍 Deep Web Researcher",
                "reply": f"🔍 **[Executed by: Deep Web Researcher]**\n\n{reply}",
                "data": res,
            }

        elif agent_key == "media_content":
            from agents.media_content import run_media_task
            res = run_media_task(clean_task)
            reply = res.get("reply") or "Media production completed."
            return {
                "status": "success",
                "agent": "🎨 Media & Content Agent",
                "reply": f"🎨 **[Executed by: Media & Content Agent]**\n\n{reply}",
                "data": res,
            }

        elif agent_key == "memory_context":
            from agents.memory_context import run_memory_task
            res = run_memory_task(clean_task)
            reply = res.get("reply") or "Memory operation completed."
            return {
                "status": "success",
                "agent": "🧠 Memory & Context Vault",
                "reply": f"🧠 **[Executed by: Memory & Context Vault]**\n\n{reply}",
                "data": res,
            }

        elif agent_key == "ai_ml_engineering":
            from agents.ai_ml_engineering import run_ai_ml_task
            res = run_ai_ml_task(clean_task)
            reply = res.get("reply") or "AI/ML modeling completed."
            return {
                "status": "success",
                "agent": "🔬 AI/ML Engineering Agent",
                "reply": f"🔬 **[Executed by: AI/ML Engineering Agent]**\n\n{reply}",
                "data": res,
            }

    except Exception as e:
        logger.error(f"Error dispatching to {agent_key}: {e}", exc_info=True)
        return {
            "status": "error",
            "agent": agent_key,
            "reply": f"⚠️ Agent {agent_key} encountered an issue: {str(e)}",
        }

    return {"status": "skipped", "reply": "No matching specialist agent found."}


def run_orchestrator_chat(
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    forced_agent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run conversation through the master orchestrator.
    Routes to specialist agents when an agentic task is recognized,
    or directly generates clean responses using the local X-Estal model.
    """
    # 1. Check for specialist agent dispatch
    target_agent = forced_agent or detect_specialist_agent(prompt)
    if target_agent:
        logger.info(f"Orchestrator routed prompt to specialist agent: '{target_agent}'")
        agent_res = execute_specialist_agent(target_agent, prompt)
        if agent_res.get("status") == "success":
            return agent_res

    # 2. Direct Conversational Inference with Local X-Estal Model
    try:
        from local_llm_service import query_local_model
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for turn in history[-8:]:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })
        messages.append({"role": "user", "content": prompt})

        res = query_local_model(messages, max_tokens=1024)
        return {
            "status": "success",
            "agent": "🧭 Master Orchestrator",
            "reply": res["display"],
            "model": "X-Estal-4B-Local",
        }
    except Exception as e:
        logger.warning(f"Direct local query fallback: {e}")
        llm = get_llm()
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if history:
            for turn in history[-8:]:
                r = turn.get("role", "user")
                c = turn.get("content", "")
                messages.append(HumanMessage(content=c) if r == "user" else AIMessage(content=c))
        messages.append(HumanMessage(content=prompt))
        try:
            resp = llm.invoke(messages)
            content = resp.content if hasattr(resp, "content") else str(resp)
            return {
                "status": "success",
                "agent": "🧭 Master Orchestrator",
                "reply": str(content),
                "model": "X-Estal-4B-Local",
            }
        except Exception as e2:
            return {
                "status": "error",
                "agent": "🧭 Master Orchestrator",
                "reply": f"Local model execution error: {str(e2)}",
                "model": "X-Estal-4B-Local",
            }


def stream_orchestrator_chat(
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    forced_agent: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Stream chat responses chunk by chunk from the local offline model,
    or yield specialist agent execution output.
    """
    target_agent = forced_agent or detect_specialist_agent(prompt)
    if target_agent:
        logger.info(f"Stream orchestrator routed prompt to specialist agent: '{target_agent}'")
        res = execute_specialist_agent(target_agent, prompt)
        reply = res.get("reply", "Done.")
        yield reply
        return

    # Direct streaming from local X-Estal model
    try:
        from local_llm_service import stream_local_model
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for turn in history[-8:]:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })
        messages.append({"role": "user", "content": prompt})
        for token in stream_local_model(messages, max_tokens=1024):
            yield token
    except Exception:
        llm = get_llm()
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if history:
            for turn in history[-8:]:
                r = turn.get("role", "user")
                c = turn.get("content", "")
                messages.append(HumanMessage(content=c) if r == "user" else AIMessage(content=c))
        messages.append(HumanMessage(content=prompt))
        for chunk in llm.stream(messages):
            t = chunk.content if hasattr(chunk, "content") else str(chunk)
            if t:
                yield str(t)
