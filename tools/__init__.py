"""
tools package — Agentic AI Tools Suite
======================================
Modular, production-grade tools for LangChain & LangGraph agents.
Engineered by X-Estal Nitish (Nitish Kumar).
"""

from typing import Dict, List, Any, Optional

from .code_dev import CODE_DEV_TOOLS
from .file_ops import FILE_OPS_TOOLS
from .ai_ml_engineering import AI_ML_TOOLS
from .security_privacy import SECURITY_PRIVACY_TOOLS
from .integration_ops import INTEGRATION_OPS_TOOLS
from .doc_generation import DOC_GENERATION_TOOLS

# Category fallbacks for specialist agent compatibility
DATABASE_DATA_TOOLS: List[Any] = []
MEDIA_CONTENT_TOOLS: List[Any] = []
MEMORY_CONTEXT_TOOLS: List[Any] = []
RESEARCH_WEB_TOOLS: List[Any] = []
ALL_RESEARCH_TOOLS: List[Any] = []
SOCIAL_GROWTH_TOOLS: List[Any] = []
TEXT_NLP_TOOLS: List[Any] = []
GEO_WEATHER_TOOLS: List[Any] = []
MATH_ANALYSIS_TOOLS: List[Any] = []
EDUCATION_TOOLS: List[Any] = []
FINANCE_BUSINESS_TOOLS: List[Any] = []
TIME_PRODUCTIVITY_TOOLS: List[Any] = []
INTEGRATION_AGENT_TOOLS: List[Any] = []

# Combined catalog
ALL_TOOLS = (
    CODE_DEV_TOOLS +
    FILE_OPS_TOOLS +
    AI_ML_TOOLS +
    SECURITY_PRIVACY_TOOLS +
    INTEGRATION_OPS_TOOLS +
    DOC_GENERATION_TOOLS
)

TOOLS_BY_CATEGORY: Dict[str, List[Any]] = {
    "Code & Dev Engineering": CODE_DEV_TOOLS,
    "File & Archive Operations": FILE_OPS_TOOLS,
    "AI & ML Tools": AI_ML_TOOLS,
    "Security & Privacy Audit": SECURITY_PRIVACY_TOOLS,
    "System & Integration Ops": INTEGRATION_OPS_TOOLS,
    "Document Generation": DOC_GENERATION_TOOLS,
}

TOOLS_BY_NAME: Dict[str, Any] = {
    getattr(t, "name", str(t)): t for t in ALL_TOOLS
}

def get_tool_by_name(name: str) -> Optional[Any]:
    return TOOLS_BY_NAME.get(name)

def list_all_tools() -> List[Dict[str, str]]:
    catalog = []
    for category, tool_list in TOOLS_BY_CATEGORY.items():
        for t in tool_list:
            catalog.append({
                "name": getattr(t, "name", str(t)),
                "description": getattr(t, "description", ""),
                "category": category,
            })
    return catalog

__all__ = [
    "ALL_TOOLS",
    "TOOLS_BY_CATEGORY",
    "TOOLS_BY_NAME",
    "CODE_DEV_TOOLS",
    "FILE_OPS_TOOLS",
    "AI_ML_TOOLS",
    "SECURITY_PRIVACY_TOOLS",
    "INTEGRATION_OPS_TOOLS",
    "DOC_GENERATION_TOOLS",
    "DATABASE_DATA_TOOLS",
    "MEDIA_CONTENT_TOOLS",
    "MEMORY_CONTEXT_TOOLS",
    "RESEARCH_WEB_TOOLS",
    "ALL_RESEARCH_TOOLS",
    "SOCIAL_GROWTH_TOOLS",
    "TEXT_NLP_TOOLS",
    "GEO_WEATHER_TOOLS",
    "MATH_ANALYSIS_TOOLS",
    "EDUCATION_TOOLS",
    "FINANCE_BUSINESS_TOOLS",
    "TIME_PRODUCTIVITY_TOOLS",
    "INTEGRATION_AGENT_TOOLS",
    "get_tool_by_name",
    "list_all_tools",
]
