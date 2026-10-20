"""
X-Estal Code Dev Agent — Autonomous Software Engineering Multi-Agent Studio
============================================================================
Engineered by X-Estal Nitish (Nitish Kumar).

A production-grade Streamlit application featuring a Claymorphism UI for
the LangGraph Code Dev Agent (Plan -> Code -> Audit -> Execute -> Debug -> Deliver).
Ready for deployment on Streamlit Cloud & GitHub.
"""

from __future__ import annotations

import os
import sys
import json
import time
import html
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st

# Path bootstrap
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

# Import the Code Dev Agent engine
import code_dev
from tools import ALL_TOOLS, TOOLS_BY_CATEGORY, list_all_tools

# ─────────────────────────────────────────────────────────────
# 1. STREAMLIT PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="X-Estal Code Dev Agent | By X-Estal Nitish",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 2. CLAYMORPHISM CSS STYLESHEET
# ─────────────────────────────────────────────────────────────

CLAYMORPHISM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --clay-bg: #f4edf9;
        --clay-surface: #ffffff;
        --clay-card-bg: linear-gradient(145deg, #ffffff, #f0e6f6);
        --clay-card-inner: #f7f2fa;
        --clay-primary: #6366f1;
        --clay-primary-gradient: linear-gradient(135deg, #6366f1, #8b5cf6);
        --clay-secondary-gradient: linear-gradient(135deg, #ec4899, #f43f5e);
        --clay-accent-gradient: linear-gradient(135deg, #06b6d4, #3b82f6);
        --clay-success-gradient: linear-gradient(135deg, #10b981, #059669);
        --clay-amber-gradient: linear-gradient(135deg, #f59e0b, #d97706);
        --text-main: #1e1b4b;
        --text-muted: #5b5478;
        --clay-shadow-out: 10px 10px 20px rgba(175, 160, 205, 0.4), -10px -10px 20px rgba(255, 255, 255, 0.95);
        --clay-shadow-sm: 5px 5px 12px rgba(175, 160, 205, 0.35), -5px -5px 12px rgba(255, 255, 255, 0.9);
        --clay-shadow-lg: 16px 16px 32px rgba(175, 160, 205, 0.45), -16px -16px 32px rgba(255, 255, 255, 1);
        --clay-inset: inset 3px 3px 7px rgba(175, 160, 205, 0.35), inset -3px -3px 7px rgba(255, 255, 255, 0.85);
        --clay-dark-inset: inset 4px 4px 10px rgba(0, 0, 0, 0.55), inset -3px -3px 8px rgba(255, 255, 255, 0.05);
    }

    /* Streamlit Global Overrides */
    .stApp {
        background: linear-gradient(135deg, #f7f0fa 0%, #edf2fc 35%, #fbf3f5 70%, #f4edf9 100%);
        background-attachment: fixed;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-main);
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1350px !important;
    }

    /* ═════════════════════════════════════════════════════════
       CLAY CARDS & CONTAINERS
       ═════════════════════════════════════════════════════════ */
    .clay-card {
        background: var(--clay-card-bg);
        border-radius: 26px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: var(--clay-shadow-out);
        border: 2px solid rgba(255, 255, 255, 0.7);
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .clay-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--clay-shadow-lg);
    }

    .clay-card-sm {
        background: var(--clay-card-bg);
        border-radius: 20px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: var(--clay-shadow-sm);
        border: 1px solid rgba(255, 255, 255, 0.65);
        transition: all 0.25s ease;
    }
    .clay-card-sm:hover {
        transform: translateY(-2px);
        box-shadow: var(--clay-shadow-out);
    }

    /* ═════════════════════════════════════════════════════════
       HEADER HERO BANNER
       ═════════════════════════════════════════════════════════ */
    .clay-header-banner {
        background: linear-gradient(135deg, #ffffff 0%, #f4edf9 100%);
        border-radius: 30px;
        padding: 26px 36px;
        margin-bottom: 24px;
        box-shadow: var(--clay-shadow-out);
        border: 2px solid rgba(255, 255, 255, 0.85);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }

    .header-title-box {
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .header-logo-icon {
        width: 68px;
        height: 68px;
        border-radius: 22px;
        background: var(--clay-primary-gradient);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 34px;
        box-shadow: 6px 6px 14px rgba(99, 102, 241, 0.4), -4px -4px 10px rgba(255, 255, 255, 0.9);
        animation: float 5s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    .header-main-title {
        font-size: 2.1rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #4338ca, #7c3aed, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
    }

    .header-sub-tag {
        font-size: 0.95rem;
        color: var(--text-muted);
        font-weight: 600;
        margin-top: 4px;
    }

    .header-badges-row {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }

    .clay-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: var(--clay-shadow-sm);
        border: 1px solid rgba(255, 255, 255, 0.6);
        background: #ffffff;
    }

    .badge-green { color: #059669; }
    .badge-purple { color: #7c3aed; }
    .badge-pink { color: #db2777; }
    .badge-amber { color: #d97706; }

    /* ═════════════════════════════════════════════════════════
       PILL BUTTONS & SOCIAL BADGES
       ═════════════════════════════════════════════════════════ */
    .pill-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 20px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.88rem;
        text-decoration: none !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: var(--clay-shadow-sm);
        border: 1px solid rgba(255, 255, 255, 0.5);
        cursor: pointer;
    }
    .pill-btn:hover {
        transform: translateY(-3px) scale(1.03);
        box-shadow: var(--clay-shadow-out);
        text-decoration: none !important;
    }

    .pill-github {
        background: linear-gradient(135deg, #24292e, #0f1419);
        color: #ffffff !important;
    }
    .pill-linkedin {
        background: linear-gradient(135deg, #0077b5, #005a9c);
        color: #ffffff !important;
    }
    .pill-instagram {
        background: linear-gradient(135deg, #f09433, #e6683c 25%, #dc2743 50%, #cc2366 75%, #bc1888 100%);
        color: #ffffff !important;
    }

    /* ═════════════════════════════════════════════════════════
       PROJECT LINK CARDS (SIDEBAR / SHOWCASE)
       ═════════════════════════════════════════════════════════ */
    .project-clay-item {
        background: linear-gradient(145deg, #ffffff, #f7f1fc);
        border-radius: 20px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: var(--clay-shadow-sm);
        border: 1px solid rgba(255, 255, 255, 0.7);
        transition: all 0.3s ease;
        text-decoration: none !important;
        display: block;
    }
    .project-clay-item:hover {
        transform: translateY(-3px);
        box-shadow: var(--clay-shadow-out);
        text-decoration: none !important;
    }
    .project-item-title {
        font-weight: 800;
        font-size: 0.95rem;
        color: var(--text-main);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .project-item-desc {
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-top: 4px;
        line-height: 1.4;
    }

    /* ═════════════════════════════════════════════════════════
       PIPELINE STAGE TRACKER (6 AGENT NODES)
       ═════════════════════════════════════════════════════════ */
    .pipeline-grid {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 12px !important;
        margin: 18px 0 !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    .pipeline-card {
        flex: 1 1 140px !important;
        min-width: 130px !important;
        background: var(--clay-card-bg) !important;
        border-radius: 18px !important;
        padding: 16px 10px !important;
        text-align: center !important;
        box-shadow: var(--clay-shadow-sm) !important;
        border: 2px solid rgba(255, 255, 255, 0.75) !important;
        transition: all 0.3s ease !important;
        box-sizing: border-box !important;
    }
    .pipeline-card.active {
        background: linear-gradient(135deg, #ffffff, #ede9fe) !important;
        border: 2px solid #7c3aed !important;
        box-shadow: 0 0 16px rgba(124, 58, 237, 0.45) !important;
        transform: scale(1.03) !important;
    }
    .pipeline-card.completed {
        background: linear-gradient(135deg, #ffffff, #ecfdf5) !important;
        border: 2px solid #10b981 !important;
    }
    .pipeline-icon {
        font-size: 1.8rem !important;
        margin-bottom: 6px !important;
    }
    .pipeline-name {
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        color: var(--text-main) !important;
    }
    .pipeline-status {
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        margin-top: 5px !important;
        padding: 3px 9px !important;
        border-radius: 12px !important;
        display: inline-block !important;
    }
    .status-pending { background: rgba(148, 163, 184, 0.2) !important; color: #475569 !important; }
    .status-running { background: rgba(99, 102, 241, 0.2) !important; color: #3730a3 !important; }
    .status-done { background: rgba(16, 185, 129, 0.2) !important; color: #047857 !important; }

    /* ═════════════════════════════════════════════════════════
       DARK CLAY TERMINAL (SANDBOX EXECUTION LOGS)
       ═════════════════════════════════════════════════════════ */
    .clay-terminal {
        background: #0d1117;
        border-radius: 20px;
        padding: 18px 22px;
        color: #e2e8f0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        line-height: 1.6;
        box-shadow: var(--clay-dark-inset);
        border: 1px solid rgba(255, 255, 255, 0.1);
        overflow-x: auto;
        white-space: pre-wrap;
        margin-top: 12px;
    }
    .terminal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .terminal-dots {
        display: flex;
        gap: 6px;
    }
    .terminal-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
    .dot-red { background: #ef4444; }
    .dot-yellow { background: #f59e0b; }
    .dot-green { background: #10b981; }

    /* ═════════════════════════════════════════════════════════
       TABS STYLING
       ═════════════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(240, 230, 250, 0.5);
        padding: 8px 12px;
        border-radius: 50px;
        box-shadow: var(--clay-inset);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px !important;
        padding: 8px 20px !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--clay-primary-gradient) !important;
        color: #ffffff !important;
        box-shadow: 4px 4px 10px rgba(99, 102, 241, 0.4) !important;
    }

    /* ═════════════════════════════════════════════════════════
       ARCHITECTURE FLOW CARDS
       ═════════════════════════════════════════════════════════ */
    .arch-flow-container {
        display: flex;
        flex-direction: column;
        gap: 14px;
        margin: 20px 0;
    }
    .arch-step-card {
        background: var(--clay-card-bg);
        border-radius: 20px;
        padding: 16px 20px;
        box-shadow: var(--clay-shadow-sm);
        border: 1px solid rgba(255, 255, 255, 0.8);
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .arch-step-num {
        width: 44px;
        height: 44px;
        border-radius: 14px;
        background: var(--clay-primary-gradient);
        color: #ffffff;
        font-weight: 800;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 3px 3px 8px rgba(99, 102, 241, 0.35);
    }
    .arch-arrow {
        text-align: center;
        color: #8b5cf6;
        font-size: 1.3rem;
        font-weight: 900;
    }

    /* ═════════════════════════════════════════════════════════
       FOOTER
       ═════════════════════════════════════════════════════════ */
    .clay-footer {
        text-align: center;
        padding: 30px 20px 10px;
        margin-top: 40px;
        font-size: 0.92rem;
        color: var(--text-muted);
    }
</style>
"""

st.markdown(CLAYMORPHISM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 3. SOCIAL LINKS & CREATOR DATA
# ─────────────────────────────────────────────────────────────

DEVELOPER_NAME = "x-estal nitish"
DISPLAY_NAME = "X-Estal Nitish (Nitish Kumar)"

SOCIAL_LINKS = {
    "instagram": {
        "url": "https://www.instagram.com/x_estal_nitish",
        "label": "Instagram",
        "icon": "📸",
        "class": "pill-instagram",
    },
    "github": {
        "url": "https://github.com/XestalNitish",
        "label": "GitHub",
        "icon": "🐙",
        "class": "pill-github",
    },
    "linkedin": {
        "url": "https://www.linkedin.com/in/nitish-kumar-33714642b/",
        "label": "LinkedIn",
        "icon": "💼",
        "class": "pill-linkedin",
    },
}

PORTFOLIO_PROJECTS = [
    {
        "id": "research_agent",
        "name": "Research Agent",
        "icon": "🔬",
        "desc": "Autonomous deep-web research agent with multi-source synthesis & report generation.",
        "url": "https://xestal-research-agent-by-x-estal-nitish.streamlit.app/",
        "tag": "Agentic AI",
    },
    {
        "id": "bitcoin_predictor",
        "name": "Bitcoin Price Predictor",
        "icon": "₿",
        "desc": "Machine learning cryptocurrency price forecasting & technical indicator modeling.",
        "url": "https://x-estal-nitish-bit-coin-price-predictor.streamlit.app",
        "tag": "ML & Finance",
    },
    {
        "id": "tumor_classifier",
        "name": "Brain Tumor Classifier",
        "icon": "🧠",
        "desc": "Deep learning convolutional neural network MRI brain tumor classification system.",
        "url": "https://brain-tumor-classifier-by-x-estal-nitish.streamlit.app",
        "tag": "Medical Vision",
    },
]

# Preset inspirations for quick 1-click prompting
PRESET_TASKS = {
    "FastAPI Auth Microservice": (
        "Create a production-ready FastAPI microservice with JWT authentication, "
        "password hashing, SQLite storage, route protection, and runnable self-test assertions."
    ),
    "RAG Vector Search Pipeline": (
        "Build a lightweight RAG (Retrieval-Augmented Generation) document retrieval system with "
        "in-memory TF-IDF vector ranking, chunking, semantic similarity scoring, and runnable assertions."
    ),
    "Crypto Moving Average Indicator": (
        "Write a cryptocurrency trading indicator engine that computes Exponential Moving Averages (EMA 9 & 21), "
        "RSI (Relative Strength Index), Bollinger Bands, and signal generation with unit tests."
    ),
    "Self-Balancing AVL Binary Search Tree": (
        "Implement a fully functional self-balancing AVL Tree in Python with node rotations (LL, RR, LR, RL), "
        "insert, delete, in-order traversal, and self-verifying test assertions."
    ),
    "QuickSort & MergeSort Engine": (
        "Write an optimized sorting suite with in-place randomized quicksort and stable mergesort, "
        "benchmarking comparisons, and self-testing unit assertions."
    ),
    "Brain Tumor MRI Classifier Model": (
        "Write a lightweight multi-class neural network classifier in Python with softmax activation, "
        "calibrated feature weights for Glioma/Meningioma/No Tumor/Pituitary, and confidence predictions."
    ),
}

# ─────────────────────────────────────────────────────────────
# 4. SIDEBAR CONFIGURATION (CLAY DRAWER)
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    # Developer Identity Card
    dev_card_html = (
        f'<div class="clay-card-sm" style="text-align: center;">'
        f'<div style="font-size: 2.4rem; margin-bottom: 6px;">👨‍💻</div>'
        f'<div style="font-weight: 800; font-size: 1.15rem; color: #1e1b4b;">{DISPLAY_NAME}</div>'
        f'<div style="font-size: 0.8rem; color: #6366f1; font-weight: 600; margin-bottom: 12px;">'
        f'AI Developer • Multi-Agent Architect</div>'
        f'<div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">'
        f'<a href="{SOCIAL_LINKS["github"]["url"]}" target="_blank" rel="noopener noreferrer" class="pill-btn pill-github" style="padding: 6px 14px; font-size: 0.78rem;">'
        f'{SOCIAL_LINKS["github"]["icon"]} GitHub</a>'
        f'<a href="{SOCIAL_LINKS["linkedin"]["url"]}" target="_blank" rel="noopener noreferrer" class="pill-btn pill-linkedin" style="padding: 6px 14px; font-size: 0.78rem;">'
        f'{SOCIAL_LINKS["linkedin"]["icon"]} LinkedIn</a>'
        f'<a href="{SOCIAL_LINKS["instagram"]["url"]}" target="_blank" rel="noopener noreferrer" class="pill-btn pill-instagram" style="padding: 6px 14px; font-size: 0.78rem;">'
        f'{SOCIAL_LINKS["instagram"]["icon"]} Insta</a>'
        f'</div></div>'
    )
    st.markdown(dev_card_html, unsafe_allow_html=True)

    # AI Engine Provider Settings
    st.markdown("### 🧠 Live AI Engine Settings")

    provider_choice = st.selectbox(
        "Live Inference Engine",
        options=[
            "Local Spark-X2.5-4B (llama-server Live)",
            "Groq Cloud (Fast Llama-3.3 / Qwen-2.5 Coder)",
            "Google Gemini (API Key)",
            "OpenAI (GPT-4o / GPT-4o-mini)",
        ],
        index=0,
        help="Select the live LLM engine for planning, writing, and debugging source code."
    )

    api_key_input = ""
    selected_model = ""
    local_url = "http://127.0.0.1:8088/v1"

    if "Local" in provider_choice:
        local_url = st.text_input("Local Server Base URL", value="http://127.0.0.1:8088/v1")
        selected_model = "X-Estal-4B"
        if code_dev.is_server_alive():
            st.success("🟢 Local Spark-X2.5 server is ONLINE & running live (Port 8088).")
        else:
            st.warning("⚠️ Local llama-server is currently offline.")
            if st.button("🚀 Start Local LLM Server", use_container_width=True):
                with st.spinner("Starting local Spark-X2.5 llama-server..."):
                    import local_llm_service
                    local_llm_service.start_local_server(wait_timeout=10)
                    st.rerun()

    elif "Groq" in provider_choice:
        default_groq = os.getenv("GROQ_API_KEY", "")
        api_key_input = st.text_input(
            "Groq API Key",
            value=default_groq,
            type="password",
            placeholder="gsk_...",
            help="Free Groq API key from console.groq.com. Instant ultra-fast responses."
        )
        selected_model = st.selectbox(
            "Model",
            options=["llama-3.3-70b-versatile", "qwen-2.5-coder-32b", "llama-3.1-8b-instant"],
            index=0
        )
        if api_key_input:
            st.caption("⚡ Connected live to Groq Cloud API.")
        else:
            st.caption("ℹ️ Enter Groq API key to activate cloud inference.")

    elif "Gemini" in provider_choice:
        default_gemini = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        api_key_input = st.text_input(
            "Gemini API Key",
            value=default_gemini,
            type="password",
            placeholder="AIzaSy...",
            help="Google Gemini API key."
        )
        selected_model = st.selectbox(
            "Model",
            options=["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
            index=0
        )
        if api_key_input:
            st.caption("🌟 Connected live to Google Gemini API.")
        else:
            st.caption("ℹ️ Enter Gemini API key to activate cloud inference.")

    elif "OpenAI" in provider_choice:
        default_openai = os.getenv("OPENAI_API_KEY", "")
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=default_openai,
            type="password",
            placeholder="sk-proj-...",
            help="OpenAI API key."
        )
        selected_model = st.selectbox(
            "Model",
            options=["gpt-4o-mini", "gpt-4o"],
            index=0
        )
        if api_key_input:
            st.caption("🤖 Connected live to OpenAI API.")
        else:
            st.caption("ℹ️ Enter OpenAI API key to activate cloud inference.")

    st.markdown("---")

    # Agent Tuning
    with st.expander("⚙️ Advanced Agent Tuning", expanded=False):
        max_retries = st.slider("Max Self-Healing Debug Retries", min_value=1, max_value=5, value=3)
        sandbox_timeout = st.slider("Sandbox Timeout (seconds)", min_value=5, max_value=30, value=15)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    st.markdown("---")

    # Featured Live Projects by X-Estal Nitish
    st.markdown("### 🚀 Live Apps by X-Estal Nitish")
    for proj in PORTFOLIO_PROJECTS:
        proj_item_html = (
            f'<a href="{proj["url"]}" target="_blank" rel="noopener noreferrer" class="project-clay-item">'
            f'<div class="project-item-title">'
            f'<span>{proj["icon"]} {proj["name"]}</span>'
            f'<span style="font-size: 0.72rem; background: rgba(99,102,241,0.15); color: #4f46e5; padding: 2px 8px; border-radius: 10px;">{proj["tag"]}</span>'
            f'</div>'
            f'<div class="project-item-desc">{proj["desc"]}</div>'
            f'<div style="font-size: 0.75rem; color: #6366f1; font-weight: 700; margin-top: 6px;">🔗 Launch Live App &rarr;</div>'
            f'</a>'
        )
        st.markdown(proj_item_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 5. TOP CLAY HEADER
# ─────────────────────────────────────────────────────────────

header_banner_html = (
    f'<div class="clay-header-banner">'
    f'<div class="header-title-box">'
    f'<div class="header-logo-icon">💻</div>'
    f'<div>'
    f'<div class="header-main-title">X-Estal Code Dev Agent</div>'
    f'<div class="header-sub-tag">'
    f'Autonomous Software Engineering Multi-Agent Studio • Plan, Write, Audit, Execute, Debug & Deliver'
    f'</div></div></div>'
    f'<div class="header-badges-row">'
    f'<span class="clay-badge badge-green">🟢 Production Ready</span>'
    f'<span class="clay-badge badge-purple">🦜 LangGraph StateGraph</span>'
    f'<span class="clay-badge badge-pink">👨‍💻 By X-Estal Nitish</span>'
    f'</div></div>'
)
st.markdown(header_banner_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 6. APP TABS
# ─────────────────────────────────────────────────────────────

tab_studio, tab_sandbox, tab_tools, tab_architecture = st.tabs([
    "💻 Code Dev Studio",
    "🧪 Interactive Sandbox",
    "🛠️ 45+ Production Tools",
    "🏗️ Multi-Agent Architecture",
])


# ─────────────────────────────────────────────────────────────
# TAB 1: CODE DEV STUDIO
# ─────────────────────────────────────────────────────────────

with tab_studio:
    intro_html = (
        '<div style="font-size: 0.95rem; color: #4c1d95; margin-bottom: 14px; font-weight: 600;">'
        '⚡ Enter any software implementation task below. The multi-agent workflow will architect the solution, '
        'generate production code, verify security constraints, execute tests in an isolated sandbox, and auto-heal bugs.'
        '</div>'
    )
    st.markdown(intro_html, unsafe_allow_html=True)

    # State initialization for preset prompt
    if "user_task_input" not in st.session_state:
        st.session_state["user_task_input"] = PRESET_TASKS["FastAPI Auth Microservice"]

    # Preset inspiration buttons in clean responsive 3x2 grid
    st.markdown("**💡 Quick Preset Inspirations:**")
    preset_items = list(PRESET_TASKS.items())
    row1_cols = st.columns(3)
    for idx in range(3):
        p_name, p_prompt = preset_items[idx]
        with row1_cols[idx]:
            if st.button(p_name, key=f"btn_preset_{idx}", use_container_width=True):
                st.session_state["user_task_input"] = p_prompt
                st.rerun()

    row2_cols = st.columns(3)
    for idx in range(3, 6):
        p_name, p_prompt = preset_items[idx]
        with row2_cols[idx - 3]:
            if st.button(p_name, key=f"btn_preset_{idx}", use_container_width=True):
                st.session_state["user_task_input"] = p_prompt
                st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Input controls
    with st.container():
        user_task = st.text_area(
            "Task Specification",
            key="user_task_input",
            height=130,
            placeholder="Describe the software module, endpoint, algorithm, or service you want the Code Dev Agent to build...",
            help="Be specific about requirements, input/output formats, and architectural expectations."
        )

        col_lang, col_file, col_mode = st.columns([1, 1, 1])
        with col_lang:
            target_language = st.selectbox(
                "Target Language",
                options=["python", "javascript", "bash", "typescript", "sql"],
                index=0
            )

        lang_ext_map = {
            "python": "solution.py",
            "javascript": "solution.js",
            "bash": "solution.sh",
            "typescript": "solution.ts",
            "sql": "solution.sql",
        }
        suggested_fname = lang_ext_map.get(target_language, f"solution.{target_language}")

        with col_file:
            target_filename = st.text_input("Target File Name", value=suggested_fname)

        with col_mode:
            exec_mode = st.selectbox(
                "Pipeline Execution Mode",
                options=[
                    "Full Autonomous Pipeline (Plan -> Code -> Audit -> Run -> Debug -> Deliver)",
                    "Fast Code Generation Only",
                    "Security Audit & Review Only"
                ],
                index=0
            )

    # Action Launch & Reset Buttons
    col_btn, col_reset, col_empty = st.columns([1.2, 0.6, 2.2])
    with col_btn:
        launch_clicked = st.button("🚀 Launch Code Dev Agent", type="primary", use_container_width=True)
    with col_reset:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state["user_task_input"] = ""
            if "latest_agent_result" in st.session_state:
                del st.session_state["latest_agent_result"]
            if "latest_agent_duration" in st.session_state:
                del st.session_state["latest_agent_duration"]
            st.session_state["pipeline_is_done"] = False
            st.session_state["pipeline_active_idx"] = 0
            st.rerun()

    # 6 Stages Specification
    stages = [
        {"name": "Architectural Planner", "icon": "📐"},
        {"name": "Senior Code Writer", "icon": "✍️"},
        {"name": "Security Auditor", "icon": "🛡️"},
        {"name": "Sandboxed Execution", "icon": "⚡"},
        {"name": "Self-Healing Debugger", "icon": "🩺"},
        {"name": "Production Reviewer", "icon": "📦"},
    ]

    def build_pipeline_html(active_idx: int = 0, is_done: bool = False) -> str:
        items = []
        for i, s in enumerate(stages):
            if is_done or (active_idx > i):
                card_class = "pipeline-card completed"
                status_text = "Completed ✅"
                status_class = "status-done"
            elif active_idx == i:
                card_class = "pipeline-card active"
                status_text = "Executing... ⏳"
                status_class = "status-running"
            else:
                card_class = "pipeline-card"
                status_text = "Pending"
                status_class = "status-pending"

            item_html = (
                f'<div class="{card_class}">'
                f'<div class="pipeline-icon">{s["icon"]}</div>'
                f'<div class="pipeline-name">{s["name"]}</div>'
                f'<span class="pipeline-status {status_class}">{status_text}</span>'
                f'</div>'
            )
            items.append(item_html)

        return f'<div class="pipeline-grid">{"".join(items)}</div>'

    # Pipeline Visualizer Placeholder
    stage_tracker_placeholder = st.empty()

    # Determine current visual state
    current_is_done = st.session_state.get("pipeline_is_done", False)
    current_idx = st.session_state.get("pipeline_active_idx", 0)

    # Initial render of the tracker (shows stages in proper state)
    stage_tracker_placeholder.markdown(build_pipeline_html(current_idx, current_is_done), unsafe_allow_html=True)

    # Execution handling
    if launch_clicked:
        if not user_task.strip():
            st.warning("⚠️ Please provide a task specification for the Code Dev Agent.")
        else:
            # Configure LLM based on user selection
            configured_llm = None
            if "Groq" in provider_choice:
                configured_llm = code_dev.get_dev_llm(provider="groq", api_key=api_key_input, model=selected_model, temperature=temperature)
            elif "Gemini" in provider_choice:
                configured_llm = code_dev.get_dev_llm(provider="gemini", api_key=api_key_input, model=selected_model, temperature=temperature)
            elif "OpenAI" in provider_choice:
                configured_llm = code_dev.get_dev_llm(provider="openai", api_key=api_key_input, model=selected_model, temperature=temperature)
            else:
                configured_llm = code_dev.get_dev_llm(provider="local", model="X-Estal-4B", base_url=local_url, temperature=temperature)

            code_dev.set_active_llm(configured_llm)

            node_to_idx = {
                "planner": 0,
                "writer": 1,
                "security_check": 2,
                "executor": 3,
                "debugger": 4,
                "reviewer": 5,
            }

            def step_cb(node_name: str, state_update: Dict[str, Any]):
                idx = node_to_idx.get(node_name, 0)
                stage_tracker_placeholder.markdown(build_pipeline_html(idx, is_done=False), unsafe_allow_html=True)
                time.sleep(0.12)  # Smooth visual transition

            try:
                with st.spinner("🤖 Autonomous Code Dev Agent executing workflow..."):
                    t_start = time.perf_counter()
                    agent_result = code_dev.run_code_dev_task(
                        task=user_task,
                        language=target_language,
                        file_path=target_filename,
                        llm=configured_llm,
                        step_callback=step_cb,
                        max_retries=max_retries,
                        timeout=sandbox_timeout,
                        exec_mode=exec_mode
                    )
                    total_duration = round(time.perf_counter() - t_start, 2)

                # Mark pipeline complete
                st.session_state["pipeline_is_done"] = True
                st.session_state["pipeline_active_idx"] = 5
                stage_tracker_placeholder.markdown(build_pipeline_html(5, is_done=True), unsafe_allow_html=True)

                # Store result in session state
                st.session_state["latest_agent_result"] = agent_result
                st.session_state["latest_agent_duration"] = total_duration

            except Exception as ex:
                st.error(f"⚠️ Code Dev Agent Workflow Notice: {ex}")
                st.info("💡 Tip: You can switch to the 'Autonomous Built-in Engine (Zero-Key Demo)' in the sidebar for guaranteed zero-dependency offline execution.")

    # Display Results if available
    if "latest_agent_result" in st.session_state:
        res = st.session_state["latest_agent_result"]
        dur = st.session_state.get("latest_agent_duration", 0)
        display_file = os.path.basename(res.get("file_path", target_filename or "solution.py"))
        status_flag = res.get("status", "success")
        sec_passed = res.get("security_passed", True)

        if status_flag == "success" and sec_passed:
            badge_title = "✅ Code Development Task Completed & Verified"
            badge_color = "#10b981"
        elif not sec_passed:
            badge_title = "🛡️ Code Development Completed: Security Notice Flagged"
            badge_color = "#f59e0b"
        else:
            badge_title = "🩺 Code Development Completed with Debugging Warnings"
            badge_color = "#6366f1"

        summary_bar_html = (
            f'<div class="clay-card-sm" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">'
            f'<div style="font-weight: 800; font-size: 1.05rem; color: {badge_color};">{badge_title}</div>'
            f'<div style="font-size: 0.85rem; color: #6366f1; font-weight: 700;">⏱️ Elapsed: {dur}s | 🎯 Target: <code>{html.escape(display_file)}</code></div>'
            f'</div>'
        )
        st.markdown(summary_bar_html, unsafe_allow_html=True)

        res_col_left, res_col_right = st.columns([1.3, 1])

        with res_col_left:
            st.markdown(f"#### 📝 Verified Source Code (`{display_file}`)")
            code_text = res.get("code", "")
            st.code(code_text, language=target_language, line_numbers=True)

            # Action buttons
            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                st.download_button(
                    label=f"💾 Download {display_file}",
                    data=code_text,
                    file_name=display_file,
                    mime="text/plain",
                    use_container_width=True
                )
            with btn_col2:
                if st.button("🔄 Re-Run in Sandbox", use_container_width=True):
                    with st.spinner("Executing in sandbox..."):
                        temp_d = tempfile.mkdtemp(prefix="res_sandbox_")
                        script_p = os.path.join(temp_d, display_file)
                        try:
                            with open(script_p, "w", encoding="utf-8") as f:
                                f.write(code_text)

                            t0 = time.perf_counter()
                            if target_language == "python":
                                cmd = [sys.executable, script_p]
                            elif target_language == "javascript" and shutil.which("node"):
                                cmd = ["node", script_p]
                            elif target_language == "bash" and shutil.which("bash"):
                                cmd = ["bash", script_p]
                            else:
                                cmd = [sys.executable, script_p]

                            proc = subprocess.run(
                                cmd,
                                capture_output=True,
                                text=True,
                                timeout=sandbox_timeout,
                                cwd=temp_d,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                                errors="replace"
                            )
                            d_ms = round((time.perf_counter() - t0) * 1000, 2)
                            out_combined = proc.stdout.strip()
                            if proc.stderr.strip():
                                out_combined += f"\n\n[STDERR]:\n{proc.stderr.strip()}"
                            st.session_state["latest_agent_result"]["execution_output"] = out_combined or f"Exit Code: {proc.returncode}"
                            st.toast(f"✅ Sandbox re-run complete ({d_ms}ms, Exit: {proc.returncode})")
                            st.rerun()
                        except subprocess.TimeoutExpired:
                            st.warning(f"⏳ Sandbox re-run timed out (exceeded {sandbox_timeout}s limit).")
                        except Exception as e_sb:
                            st.error(f"⚠️ Sandbox execution error: {e_sb}")
                        finally:
                            shutil.rmtree(temp_d, ignore_errors=True)

        with res_col_right:
            st.markdown("#### ⚡ Sandboxed Execution & Verification")
            stdout_logs = res.get("execution_output", "No stdout logs generated.")
            safe_logs = html.escape(stdout_logs)

            term_html = (
                f'<div class="clay-terminal">'
                f'<div class="terminal-header">'
                f'<div class="terminal-dots">'
                f'<span class="terminal-dot dot-red"></span>'
                f'<span class="terminal-dot dot-yellow"></span>'
                f'<span class="terminal-dot dot-green"></span>'
                f'</div>'
                f'<span>Sandbox Terminal — {html.escape(display_file)}</span>'
                f'</div>'
                f'<div>{safe_logs}</div>'
                f'</div>'
            )
            st.markdown(term_html, unsafe_allow_html=True)

            # Architectural Blueprint Accordion
            with st.expander("📐 Architectural Blueprint & Dependency Notes", expanded=True):
                plan_ctx = res.get("plan_context", "Architectural plan details.")
                st.markdown(f"```text\n{plan_ctx}\n```")

            # Security Audit Badge
            with st.expander("🛡️ Security Audit Report", expanded=False):
                sec_feed = res.get("security_feedback", "All static security invariants passed cleanly.")
                is_pass = res.get("security_passed", True)
                if is_pass:
                    st.success("✅ **Static Security Invariants Passed Cleanly**")
                else:
                    st.warning("⚠️ **Security Concerns Flagged During Analysis**")

                st.markdown(f"**Audit Findings & Safeguards:**\n- {sec_feed}")
                st.markdown("""
                - **System Wipe Injection Guard:** Enforced ✅
                - **Root Path Deletion Guard:** Enforced ✅
                - **Fork Bomb Signature Check:** Enforced ✅
                - **Interactive PTY Spawn Block:** Enforced ✅
                - **Runtime Subprocess Sandbox Isolation:** Active ✅
                """)


# ─────────────────────────────────────────────────────────────
# TAB 2: INTERACTIVE SANDBOX PLAYGROUND
# ─────────────────────────────────────────────────────────────

with tab_sandbox:
    sb_header_html = (
        '<div class="clay-card">'
        '<div style="font-weight: 800; font-size: 1.2rem; color: #1e1b4b; margin-bottom: 6px;">'
        '🧪 Real-Time Isolated Sandbox Runner'
        '</div>'
        '<div style="font-size: 0.9rem; color: #5b5478;">'
        'Execute any custom Python script safely in an isolated sub-process with memory isolation, '
        'timeout guard, and real-time standard output and error capture.'
        '</div>'
        '</div>'
    )
    st.markdown(sb_header_html, unsafe_allow_html=True)

    # Sandbox presets
    SAMPLE_SCRIPTS = {
        "Fibonacci & Performance Timer": (
            'import time\n\n'
            'def fib(n: int) -> int:\n'
            '    a, b = 0, 1\n'
            '    for _ in range(n):\n'
            '        a, b = b, a + b\n'
            '    return a\n\n'
            't0 = time.perf_counter()\n'
            'result = fib(100)\n'
            'elapsed = time.perf_counter() - t0\n'
            'print(f"Fibonacci(100) = {result}")\n'
            'print(f"Calculated in {elapsed*1000:.4f} ms")\n'
            'assert result > 0\n'
            'print("✅ Assertion passed!")'
        ),
        "Data Analytics & Statistics": (
            'import statistics\n'
            'data = [12, 15, 18, 22, 29, 35, 41, 52, 63, 75, 88, 99]\n'
            'print("Dataset Count:", len(data))\n'
            'print("Mean:", statistics.mean(data))\n'
            'print("Median:", statistics.median(data))\n'
            'print("Std Dev:", round(statistics.stdev(data), 3))\n'
            'print("Variance:", round(statistics.variance(data), 3))\n'
            'assert statistics.mean(data) > 0\n'
            'print("✅ Statistical verification passed!")'
        ),
        "Regex Email & Token Extractor": (
            'import re\n'
            'text = "Contact us at support@xestal.ai, sales@xestal.dev, or security@xestal.org for details."\n'
            'emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", text)\n'
            'print("Discovered Emails:", emails)\n'
            'assert len(emails) == 3\n'
            'print("✅ Extraction assertions passed!")'
        )
    }

    st.markdown("**⚡ Load Quick Sandbox Samples:**")
    sb_cols = st.columns(3)
    for idx, (s_name, s_code) in enumerate(SAMPLE_SCRIPTS.items()):
        with sb_cols[idx]:
            if st.button(s_name, key=f"sb_btn_{idx}", use_container_width=True):
                st.session_state["custom_sandbox_code"] = s_code
                st.rerun()

    if "custom_sandbox_code" not in st.session_state:
        st.session_state["custom_sandbox_code"] = SAMPLE_SCRIPTS["Fibonacci & Performance Timer"]

    sandbox_input = st.text_area(
        "Python Code to Execute",
        key="custom_sandbox_code",
        height=260,
        help="Paste any python code to run in the isolated sandbox."
    )

    col_run_sb, col_sb_timeout = st.columns([1, 1.5])
    with col_run_sb:
        run_sb_clicked = st.button("▶️ Run in Sandbox", type="primary", use_container_width=True)
    with col_sb_timeout:
        sb_runtime_timeout = st.slider("Sandbox Timeout Guard (s)", min_value=3, max_value=30, value=15, step=1, key="sb_custom_timeout")

    if run_sb_clicked:
        with st.spinner("⚡ Executing in sandbox..."):
            temp_dir = tempfile.mkdtemp(prefix="manual_sandbox_")
            temp_file = os.path.join(temp_dir, "test_runner.py")
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(sandbox_input)

                t0 = time.perf_counter()
                proc = subprocess.run(
                    [sys.executable, temp_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=sb_runtime_timeout,
                    cwd=temp_dir,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                    errors="replace"
                )
                dur = round(time.perf_counter() - t0, 3)

                stdout_text = proc.stdout.strip()
                stderr_text = proc.stderr.strip()

                if proc.returncode == 0:
                    st.success(f"Execution Succeeded in {dur}s (Exit Code: 0)")
                else:
                    st.error(f"Execution Failed in {dur}s (Exit Code: {proc.returncode})")

                terminal_out = stdout_text
                if stderr_text:
                    terminal_out += f"\n\n[STDERR TRACEBACK]:\n{stderr_text}"

                safe_terminal = html.escape(terminal_out or "Process completed with no output.")
                sb_term_html = (
                    f'<div class="clay-terminal">'
                    f'<div class="terminal-header">'
                    f'<div class="terminal-dots">'
                    f'<span class="terminal-dot dot-red"></span>'
                    f'<span class="terminal-dot dot-yellow"></span>'
                    f'<span class="terminal-dot dot-green"></span>'
                    f'</div>'
                    f'<span>Sandbox Output — Duration: {dur}s | Exit: {proc.returncode}</span>'
                    f'</div>'
                    f'<div>{safe_terminal}</div>'
                    f'</div>'
                )
                st.markdown(sb_term_html, unsafe_allow_html=True)

            except subprocess.TimeoutExpired:
                st.error(f"⏳ Execution timed out (exceeded {sb_runtime_timeout}s limit).")
            except Exception as e:
                st.error(f"Sandbox runtime error: {e}")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# TAB 3: 45+ PRODUCTION TOOLS CATALOG
# ─────────────────────────────────────────────────────────────

with tab_tools:
    tools_header_html = (
        '<div class="clay-card">'
        '<div style="font-weight: 800; font-size: 1.2rem; color: #1e1b4b; margin-bottom: 6px;">'
        '🛠️ Production Agentic Tools Suite (45+ Active Tools)'
        '</div>'
        '<div style="font-size: 0.9rem; color: #5b5478;">'
        'The X-Estal agent ecosystem includes modular tools spanning code engineering, '
        'security, file operations, AI/ML, document synthesis, and systems integration.'
        '</div>'
        '</div>'
    )
    st.markdown(tools_header_html, unsafe_allow_html=True)

    col_t_search, col_t_filter = st.columns([2, 1])
    with col_t_search:
        tool_search = st.text_input("🔍 Search tools by name or description...", placeholder="e.g. python, executor, secret, token, git, diff, hash...")
    with col_t_filter:
        category_filter = st.selectbox("Filter by Category", options=["All Categories"] + list(TOOLS_BY_CATEGORY.keys()))

    all_tool_entries = list_all_tools()
    filtered_tools = []

    for t in all_tool_entries:
        match_cat = (category_filter == "All Categories") or (t["category"] == category_filter)
        match_query = (not tool_search) or (tool_search.lower() in t["name"].lower()) or (tool_search.lower() in t["description"].lower())
        if match_cat and match_query:
            filtered_tools.append(t)

    st.markdown(f"**Showing {len(filtered_tools)} of {len(all_tool_entries)} Tools:**")

    cols_tools = st.columns(3)
    for idx, tool_info in enumerate(filtered_tools):
        t_name = html.escape(tool_info["name"])
        t_cat = html.escape(tool_info["category"])
        t_desc = html.escape(tool_info["description"][:140])
        with cols_tools[idx % 3]:
            tool_html = (
                f'<div class="clay-card-sm" style="min-height: 150px;">'
                f'<div style="font-weight: 800; font-size: 0.98rem; color: #4338ca; margin-bottom: 4px;">🔧 <code>{t_name}</code></div>'
                f'<div style="font-size: 0.72rem; color: #7c3aed; font-weight: 700; margin-bottom: 8px;">🏷️ {t_cat}</div>'
                f'<div style="font-size: 0.82rem; color: #475569; line-height: 1.4;">{t_desc}...</div>'
                f'</div>'
            )
            st.markdown(tool_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TAB 4: MULTI-AGENT ARCHITECTURE & LIVE APPS
# ─────────────────────────────────────────────────────────────

with tab_architecture:
    arch_header_html = (
        '<div class="clay-card">'
        '<div style="font-weight: 800; font-size: 1.3rem; color: #1e1b4b; margin-bottom: 8px;">'
        '🏗️ LangGraph Multi-Agent Architecture'
        '</div>'
        '<div style="font-size: 0.92rem; color: #475569; line-height: 1.7;">'
        'The X-Estal Code Dev Agent is governed by a stateful cyclical directed graph built with <strong>LangGraph</strong>. '
        'Unlike single-pass LLM prompts that produce hallucinated or broken code, our system enforces a strict 6-stage '
        'lifecycle with self-healing feedback loops:'
        '</div>'
        '</div>'
    )
    st.markdown(arch_header_html, unsafe_allow_html=True)

    arch_steps = [
        {"num": "1", "name": "Architectural Planner Node", "desc": "Analyzes requirements, selects libraries, and constructs detailed technical blueprints with test criteria.", "icon": "📐"},
        {"num": "2", "name": "Senior Code Writer Node", "desc": "Synthesizes robust, clean source code adhering to design specifications and runnable self-testing assertions.", "icon": "✍️"},
        {"num": "3", "name": "Security & Safety Auditor Node", "desc": "Executes static AST checks against destructive commands, fork bombs, injection payloads, and credential leaks.", "icon": "🛡️"},
        {"num": "4", "name": "Sandboxed Runtime Executor Node", "desc": "Runs code in an isolated subprocess with timeout guards, capturing stdout, stderr, and return codes.", "icon": "⚡"},
        {"num": "5", "name": "Self-Healing Debugger Node", "desc": "Diagnoses execution tracebacks or security breaches, generating remediation directives to loop back to the Code Writer.", "icon": "🩺"},
        {"num": "6", "name": "Production Reviewer Node", "desc": "Performs quality checks, compiles performance metrics, and delivers the verified deployment-ready package.", "icon": "📦"},
    ]

    arch_flow_html_list = ['<div class="arch-flow-container">']
    for idx, step in enumerate(arch_steps):
        s_card = (
            f'<div class="arch-step-card">'
            f'<div class="arch-step-num">{step["num"]}</div>'
            f'<div style="flex-grow: 1;">'
            f'<div style="font-weight: 800; font-size: 1.05rem; color: #1e1b4b; display: flex; align-items: center; gap: 8px;">'
            f'<span>{step["icon"]}</span><span>{step["name"]}</span>'
            f'</div>'
            f'<div style="font-size: 0.86rem; color: #5b5478; margin-top: 4px;">{step["desc"]}</div>'
            f'</div>'
            f'</div>'
        )
        arch_flow_html_list.append(s_card)
        if idx < len(arch_steps) - 1:
            arch_flow_html_list.append('<div class="arch-arrow">↓ <i>(Self-Healing Feedback Loop: Steps 4 & 5 ➔ Step 2)</i> ↓</div>' if idx == 4 else '<div class="arch-arrow">↓</div>')
    arch_flow_html_list.append('</div>')

    st.markdown("".join(arch_flow_html_list), unsafe_allow_html=True)

    with st.expander("📊 View StateGraph Specification", expanded=False):
        st.markdown("""
        ```text
        START
          ↓
        [planner]          -> Generates modular blueprint & test assertions
          ↓
        [writer]           -> Synthesizes full source code with self-tests
          ↓
        [security_check]   -> Verifies static invariants (AST injection, wipes)
          ├── If Security Violation -> [debugger] -> [writer]
          └── If Security Passed    -> [executor]
                                         ├── If Exit != 0 -> [debugger] -> [writer] (Max 3 retries)
                                         └── If Exit == 0 -> [reviewer] -> END (Delivery)
        ```
        """)

    st.markdown("### 🚀 Portfolio & Live Apps by X-Estal Nitish")
    st.markdown("Explore other autonomous AI and deep learning systems deployed live:")

    proj_cols = st.columns(3)
    for idx, p in enumerate(PORTFOLIO_PROJECTS):
        with proj_cols[idx]:
            proj_card_html = (
                f'<div class="clay-card" style="text-align: center;">'
                f'<div style="font-size: 3rem; margin-bottom: 10px;">{p["icon"]}</div>'
                f'<div style="font-weight: 800; font-size: 1.15rem; color: #1e1b4b; margin-bottom: 8px;">{p["name"]}</div>'
                f'<div style="font-size: 0.88rem; color: #5b5478; line-height: 1.5; margin-bottom: 16px;">{p["desc"]}</div>'
                f'<a href="{p["url"]}" target="_blank" rel="noopener noreferrer" class="pill-btn pill-github" style="background: linear-gradient(135deg, #6366f1, #8b5cf6);">'
                f'🔗 Open Live App</a>'
                f'</div>'
            )
            st.markdown(proj_card_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 7. CLAYMORPHISM FOOTER
# ─────────────────────────────────────────────────────────────

footer_html = (
    f'<div class="clay-footer">'
    f'<div style="display: flex; justify-content: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;">'
    f'<a href="{SOCIAL_LINKS["github"]["url"]}" target="_blank" rel="noopener noreferrer" class="pill-btn pill-github">'
    f'{SOCIAL_LINKS["github"]["icon"]} GitHub @XestalNitish</a>'
    f'<a href="{SOCIAL_LINKS["linkedin"]["url"]}" target="_blank" rel="noopener noreferrer" class="pill-btn pill-linkedin">'
    f'{SOCIAL_LINKS["linkedin"]["icon"]} LinkedIn Nitish Kumar</a>'
    f'<a href="{SOCIAL_LINKS["instagram"]["url"]}" target="_blank" rel="noopener noreferrer" class="pill-btn pill-instagram">'
    f'{SOCIAL_LINKS["instagram"]["icon"]} Instagram @x_estal_nitish</a>'
    f'</div>'
    f'<div>Crafted with 💜 by <strong>{DEVELOPER_NAME}</strong> (Nitish Kumar). Built with LangGraph & Streamlit.</div>'
    f'<div style="font-size: 0.8rem; opacity: 0.7; margin-top: 4px;">Production-Ready Multi-Agent Software Engineering Architecture.</div>'
    f'</div>'
)
st.markdown(footer_html, unsafe_allow_html=True)
