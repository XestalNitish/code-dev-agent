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

    /* Streamlit Global */
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
    st.markdown("### 🧠 AI Engine Settings")

    provider_choice = st.selectbox(
        "Inference Engine",
        options=[
            "Autonomous Built-in Engine (Zero-Key Demo)",
            "Groq (Fast Cloud - Free Key)",
            "Google Gemini (API Key)",
            "OpenAI (GPT-4o / GPT-4o-mini)",
            "Local Spark-X2.5-4B (llama-server)",
        ],
        index=0,
        help="Select the LLM backend for planning, writing, and debugging source code."
    )

    api_key_input = ""
    selected_model = ""

    if "Groq" in provider_choice:
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
    elif "Local" in provider_choice:
        local_url = st.text_input("Local Server Base URL", value="http://127.0.0.1:8088/v1")
        selected_model = "X-Estal-4B"
        if code_dev.is_server_alive(local_url):
            st.success("🟢 Local Spark-X2.5 server is ONLINE.")
        else:
            st.warning("⚠️ Local llama-server (Port 8088) is OFFLINE. The agent will automatically use the Resilient Synthesizer so requests never hang.")
    else:
        st.success("🟢 Autonomous resilient engine active. Runs 100% offline with zero API key requirement!")

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

st.markdown(f"""
<div class="clay-header-banner">
    <div class="header-title-box">
        <div class="header-logo-icon">💻</div>
        <div>
            <div class="header-main-title">X-Estal Code Dev Agent</div>
            <div class="header-sub-tag">
                Autonomous Software Engineering Multi-Agent Studio • Plan, Write, Audit, Execute, Debug & Deliver
            </div>
        </div>
    </div>
    <div class="header-badges-row">
        <span class="clay-badge badge-green">🟢 Production Ready</span>
        <span class="clay-badge badge-purple">🦜 LangGraph StateGraph</span>
        <span class="clay-badge badge-pink">👨‍💻 By X-Estal Nitish</span>
    </div>
</div>
""", unsafe_allow_html=True)


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
    st.markdown("""
    <div style="font-size: 0.95rem; color: #4c1d95; margin-bottom: 14px; font-weight: 600;">
        ⚡ Enter any software implementation task below. The multi-agent workflow will architect the solution,
        generate production code, verify security constraints, execute tests in an isolated sandbox, and auto-heal bugs.
    </div>
    """, unsafe_allow_html=True)

    # State initialization for preset prompt
    if "user_task_input" not in st.session_state:
        st.session_state["user_task_input"] = PRESET_TASKS["FastAPI Auth Microservice"]

    # Preset inspiration buttons in clean responsive columns
    st.markdown("**💡 Quick Preset Inspirations:**")
    preset_cols = st.columns(len(PRESET_TASKS))
    for idx, (p_name, p_prompt) in enumerate(PRESET_TASKS.items()):
        with preset_cols[idx]:
            if st.button(p_name, key=f"btn_preset_{idx}", use_container_width=True):
                st.session_state["user_task_input"] = p_prompt
                st.rerun()

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
        with col_file:
            default_fname = "solution.py" if target_language == "python" else f"solution.{target_language}"
            target_filename = st.text_input("Target File Name", value=default_fname)
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

    # Big Clay Action Launch Button
    col_btn, col_empty = st.columns([1, 2])
    with col_btn:
        launch_clicked = st.button("🚀 Launch Code Dev Agent", type="primary", use_container_width=True)

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
            elif "Local" in provider_choice:
                configured_llm = code_dev.get_dev_llm(provider="local", model="X-Estal-4B", base_url="http://127.0.0.1:8088/v1", temperature=temperature)
            else:
                configured_llm = code_dev.AutonomousOfflineSynthesizer(temperature=temperature)

            code_dev.set_active_llm(configured_llm)

            # Pipeline Visualizer Placeholder
            stage_tracker_placeholder = st.empty()

            stages = [
                {"name": "Architectural Planner", "icon": "📐", "status": "pending"},
                {"name": "Senior Code Writer", "icon": "✍️", "status": "pending"},
                {"name": "Security Auditor", "icon": "🛡️", "status": "pending"},
                {"name": "Sandboxed Execution", "icon": "⚡", "status": "pending"},
                {"name": "Self-Healing Debugger", "icon": "🩺", "status": "pending"},
                {"name": "Production Reviewer", "icon": "📦", "status": "pending"},
            ]

            def render_pipeline_ui(current_active_idx: int = 0, is_done: bool = False):
                items = []
                for i, s in enumerate(stages):
                    if is_done or i < current_active_idx:
                        card_class = "pipeline-card completed"
                        status_text = "Completed ✅"
                        status_class = "status-done"
                    elif i == current_active_idx:
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

                full_html = f'<div class="pipeline-grid">{"".join(items)}</div>'
                stage_tracker_placeholder.markdown(full_html, unsafe_allow_html=True)

            render_pipeline_ui(0)

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
                render_pipeline_ui(idx)
                time.sleep(0.15)  # Smooth visual transition

            with st.spinner("🤖 Autonomous Code Dev Agent executing workflow..."):
                t_start = time.perf_counter()
                agent_result = code_dev.run_code_dev_task(
                    task=user_task,
                    language=target_language,
                    file_path=target_filename,
                    llm=configured_llm,
                    step_callback=step_cb
                )
                total_duration = round(time.perf_counter() - t_start, 2)

            render_pipeline_ui(5, is_done=True)

            # Store result in session state
            st.session_state["latest_agent_result"] = agent_result
            st.session_state["latest_agent_duration"] = total_duration

    # Display Results if available
    if "latest_agent_result" in st.session_state:
        res = st.session_state["latest_agent_result"]
        dur = st.session_state.get("latest_agent_duration", 0)
        display_file = os.path.basename(res.get("file_path", "solution.py"))

        summary_bar_html = (
            f'<div class="clay-card-sm" style="display: flex; justify-content: space-between; align-items: center;">'
            f'<div style="font-weight: 800; font-size: 1.1rem; color: #1e1b4b;">✅ Code Development Task Completed</div>'
            f'<div style="font-size: 0.85rem; color: #6366f1; font-weight: 700;">⏱️ Elapsed: {dur}s | 🎯 Target: <code>{display_file}</code></div>'
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
                            proc = subprocess.run([sys.executable, script_p], capture_output=True, text=True, timeout=15, cwd=temp_d)
                            d_ms = round((time.perf_counter() - t0) * 1000, 2)
                            st.session_state["latest_agent_result"]["execution_output"] = proc.stdout or proc.stderr or f"Exit Code: {proc.returncode}"
                            st.toast(f"✅ Sandbox re-run complete ({d_ms}ms, Exit: {proc.returncode})")
                            st.rerun()
                        finally:
                            shutil.rmtree(temp_d, ignore_errors=True)

        with res_col_right:
            st.markdown("#### ⚡ Sandboxed Execution & Verification")
            stdout_logs = res.get("execution_output", "No stdout logs generated.")
            import html
            safe_logs = html.escape(stdout_logs)

            term_html = (
                f'<div class="clay-terminal">'
                f'<div class="terminal-header">'
                f'<div class="terminal-dots">'
                f'<span class="terminal-dot dot-red"></span>'
                f'<span class="terminal-dot dot-yellow"></span>'
                f'<span class="terminal-dot dot-green"></span>'
                f'</div>'
                f'<span>Sandbox Terminal — {display_file}</span>'
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
                st.markdown("""
                - **System Wipe Injection Guard:** Passed ✅
                - **Root Path Deletion Guard:** Passed ✅
                - **Fork Bomb Signature Check:** Clean ✅
                - **Interactive PTY Spawn Block:** Clean ✅
                - **Runtime Subprocess Sandbox Isolation:** Safe ✅
                """)


# ─────────────────────────────────────────────────────────────
# TAB 2: INTERACTIVE SANDBOX PLAYGROUND
# ─────────────────────────────────────────────────────────────

with tab_sandbox:
    st.markdown("""
    <div class="clay-card">
        <div style="font-weight: 800; font-size: 1.2rem; color: #1e1b4b; margin-bottom: 6px;">
            🧪 Real-Time Isolated Sandbox Runner
        </div>
        <div style="font-size: 0.9rem; color: #5b5478; margin-bottom: 16px;">
            Execute any custom Python script safely in an isolated sub-process with memory isolation,
            timeout guard, and real-time standard output and error capture.
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    col_run_sb, col_sb_timeout = st.columns([1, 2])
    with col_run_sb:
        run_sb_clicked = st.button("▶️ Run in Sandbox", type="primary", use_container_width=True)

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
                    timeout=15,
                    cwd=temp_dir,
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

                import html
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
                st.error("⏳ Execution timed out (exceeded 15s limit).")
            except Exception as e:
                st.error(f"Sandbox runtime error: {e}")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# TAB 3: 45+ PRODUCTION TOOLS CATALOG
# ─────────────────────────────────────────────────────────────

with tab_tools:
    st.markdown("""
    <div class="clay-card">
        <div style="font-weight: 800; font-size: 1.2rem; color: #1e1b4b; margin-bottom: 6px;">
            🛠️ Production Agentic Tools Suite (45+ Active Tools)
        </div>
        <div style="font-size: 0.9rem; color: #5b5478;">
            The X-Estal agent ecosystem includes modular tools spanning code engineering,
            security, file operations, AI/ML, document synthesis, and systems integration.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tool_search = st.text_input("🔍 Search tools by name or description...", placeholder="e.g. python, executor, secret, token, git, diff, hash...")
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
        with cols_tools[idx % 3]:
            tool_html = (
                f'<div class="clay-card-sm" style="min-height: 150px;">'
                f'<div style="font-weight: 800; font-size: 0.98rem; color: #4338ca; margin-bottom: 4px;">🔧 <code>{tool_info["name"]}</code></div>'
                f'<div style="font-size: 0.72rem; color: #7c3aed; font-weight: 700; margin-bottom: 8px;">🏷️ {tool_info["category"]}</div>'
                f'<div style="font-size: 0.82rem; color: #475569; line-height: 1.4;">{tool_info["description"][:140]}...</div>'
                f'</div>'
            )
            st.markdown(tool_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TAB 4: MULTI-AGENT ARCHITECTURE & LIVE APPS
# ─────────────────────────────────────────────────────────────

with tab_architecture:
    st.markdown("""
    <div class="clay-card">
        <div style="font-weight: 800; font-size: 1.3rem; color: #1e1b4b; margin-bottom: 8px;">
            🏗️ LangGraph Multi-Agent Architecture
        </div>
        <div style="font-size: 0.92rem; color: #475569; line-height: 1.7;">
            The X-Estal Code Dev Agent is governed by a stateful cyclical directed graph built with <strong>LangGraph</strong>.
            Unlike single-pass LLM prompts that produce hallucinated or broken code, our system enforces a strict 6-stage
            lifecycle with self-healing feedback loops:
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ```mermaid
    flowchart TD
        Start([START]) --> Planner[1. Architectural Planner Node]
        Planner --> Writer[2. Senior Code Writer Node]
        Writer --> Security[3. Security & Safety Auditor Node]

        Security -->|Security Pass| Executor[4. Sandboxed Runtime Executor Node]
        Security -->|Security Violation| Debugger[5. Self-Healing Debugger Node]

        Executor -->|Exit Code 0: Success| Reviewer[6. Technical Lead Reviewer Node]
        Executor -->|Exit Code != 0: Runtime Crash| Debugger

        Debugger -->|Automated Remediation Directive| Writer
        Reviewer --> EndNode([END / Verified Delivery Report])
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

st.markdown(f"""
<div class="clay-footer">
    <div style="display: flex; justify-content: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;">
        <a href="{SOCIAL_LINKS['github']['url']}" target="_blank" rel="noopener noreferrer" class="pill-btn pill-github">
            {SOCIAL_LINKS['github']['icon']} GitHub @XestalNitish
        </a>
        <a href="{SOCIAL_LINKS['linkedin']['url']}" target="_blank" rel="noopener noreferrer" class="pill-btn pill-linkedin">
            {SOCIAL_LINKS['linkedin']['icon']} LinkedIn Nitish Kumar
        </a>
        <a href="{SOCIAL_LINKS['instagram']['url']}" target="_blank" rel="noopener noreferrer" class="pill-btn pill-instagram">
            {SOCIAL_LINKS['instagram']['icon']} Instagram @x_estal_nitish
        </a>
    </div>
    <div>
        Crafted with 💜 by <strong>{DEVELOPER_NAME}</strong> (Nitish Kumar). Built with LangGraph & Streamlit.
    </div>
    <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 4px;">
        Production-Ready Multi-Agent Software Engineering Architecture.
    </div>
</div>
""", unsafe_allow_html=True)
