"""
agents package — 8 Specialist & Master Autonomous AI Agents.
All free, open-source, and runnable locally with zero required paid APIs.
"""

from typing import Dict, Any, Callable

from .orchestrator import run_orchestrator_chat, stream_orchestrator_chat, get_llm
from .integration import run_integration_task, get_agent as get_integration_agent, integration_ops_agent
from .code_dev import code_dev_agent, run_code_dev_task
from .database import database_data_agent, run_database_task
from .media_content import media_content_agent, run_media_task
from .memory_context import memory_context_agent, run_memory_task
from .research_web import research_web_agent, run_research_task
from .ai_ml_engineering import ai_ml_agent, run_ai_ml_task

# Master dictionary mapping agent identifiers to their runnable graphs / instances
ALL_AGENTS: Dict[str, Any] = {
    "orchestrator": run_orchestrator_chat,
    "integration": integration_ops_agent,
    "code_dev": code_dev_agent,
    "database": database_data_agent,
    "media_content": media_content_agent,
    "memory_context": memory_context_agent,
    "research_web": research_web_agent,
    "ai_ml_engineering": ai_ml_agent,
}

# Standardized direct execution runner functions for every agent
AGENT_RUNNERS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    "orchestrator": lambda t: run_orchestrator_chat(t),
    "integration": lambda t: run_integration_task(t),
    "code_dev": lambda t: run_code_dev_task(t),
    "database": lambda t: run_database_task(t),
    "media_content": lambda t: run_media_task(t),
    "memory_context": lambda t: run_memory_task(t),
    "research_web": lambda t: run_research_task(t),
    "ai_ml_engineering": lambda t: run_ai_ml_task(t),
}

__all__ = [
    # Master Chatbot / Orchestrator
    "run_orchestrator_chat",
    "stream_orchestrator_chat",
    "get_llm",

    # Specialist Agent Task Runners (Callable)
    "run_integration_task",
    "run_code_dev_task",
    "run_database_task",
    "run_media_task",
    "run_memory_task",
    "run_research_task",
    "run_ai_ml_task",

    # Specialist Agent Graphs
    "get_integration_agent",
    "integration_ops_agent",
    "code_dev_agent",
    "database_data_agent",
    "media_content_agent",
    "memory_context_agent",
    "research_web_agent",
    "ai_ml_agent",

    # Registries
    "ALL_AGENTS",
    "AGENT_RUNNERS",
]
