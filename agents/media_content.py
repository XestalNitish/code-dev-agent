import os
import json
import re
import time
from datetime import datetime
from typing import TypedDict, Literal, Optional, List, Dict, Any, Annotated
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

# --- 1. Tool Ingestion from tools package ---
from tools import (
    RESEARCH_WEB_TOOLS,
    TEXT_NLP_TOOLS,
    FILE_OPS_TOOLS,
    MEDIA_CONTENT_TOOLS,
    SOCIAL_GROWTH_TOOLS,
    DOC_GENERATION_TOOLS
)

load_dotenv()

# Staged Tool Scopes
RESEARCH_STAGE_TOOLS = list(RESEARCH_WEB_TOOLS) + list(FILE_OPS_TOOLS) + list(TEXT_NLP_TOOLS)
MEDIA_STAGE_TOOLS = list(MEDIA_CONTENT_TOOLS) + list(SOCIAL_GROWTH_TOOLS)
DELIVERY_STAGE_TOOLS = list(DOC_GENERATION_TOOLS) + list(FILE_OPS_TOOLS)

def _init_media_llm():
    try:
        from local_llm_service import get_local_chat_llm
        return get_local_chat_llm(temperature=0.2)
    except Exception:
        pass
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url="http://127.0.0.1:8088/v1",
            api_key="x-estal-local",
            model_name="X-Estal-4B",
            temperature=0.2,
        )
    except Exception:
        return None

llm_base = _init_media_llm()
llm_researcher = llm_base.bind_tools(RESEARCH_STAGE_TOOLS) if (llm_base and RESEARCH_STAGE_TOOLS) else llm_base
llm_media_engine = llm_base.bind_tools(MEDIA_STAGE_TOOLS) if (llm_base and MEDIA_STAGE_TOOLS) else llm_base
llm_deliverer = llm_base.bind_tools(DELIVERY_STAGE_TOOLS) if (llm_base and DELIVERY_STAGE_TOOLS) else llm_base
llm_auditor = llm_base


# --- 3. Production State Schema ---
class SceneBlueprint(TypedDict):
    scene_number: int
    timestamp: str              # e.g., "0:00 - 0:03"
    voiceover_script: str       # Spoken narration
    on_screen_text: str         # Kinetic text / subtitles
    visual_direction: str       # Framing, transition, lighting
    image_prompt: str           # Prompt for AI B-roll generation
    sound_sfx: str              # Audio cues (whoosh, hit, riser)

class MediaContentState(TypedDict):
    topic: str
    target_platform: Literal["youtube_shorts", "instagram_reels", "long_form_vlog", "kinetic_story"]
    target_duration_sec: int
    tone_style: str
    attached_files: Optional[List[str]]
    research_messages: Annotated[List[BaseMessage], add_messages]
    research_iterations: int
    grounded_context: Optional[str]
    media_messages: Annotated[List[BaseMessage], add_messages]
    media_iterations: int
    social_growth_meta: Optional[Dict[str, Any]]
    hook_options: Optional[List[str]]
    selected_hook: Optional[str]
    scenes: Optional[List[SceneBlueprint]]
    raw_production_blueprint: Optional[Dict[str, Any]]
    audit_passed: bool
    audit_feedback: Optional[str]
    retry_count: int
    delivery_messages: Annotated[List[BaseMessage], add_messages]
    delivery_iterations: int
    saved_script_path: Optional[str]
    saved_manifest_path: Optional[str]
    final_summary: Optional[str]


def clamp_text(text: str, max_chars: int = 2500) -> str:
    return text[:max_chars] if len(text) > max_chars else text

def parse_json_safely(raw_text: str) -> dict:
    clean = raw_text.strip()
    if "```" in clean:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if match:
            clean = match.group(1).strip()
    try:
        return json.loads(clean)
    except Exception:
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(clean[start:end + 1])
            except Exception:
                pass
    return {}

def robust_invoke_with_retry(llm_instance, messages, max_attempts: int = 3):
    """Handles rate limits and API disruptions with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return llm_instance.invoke(messages)
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err or "quota" in err:
                backoff = (attempt + 1) * 7
                print(f"⚠️ [Backoff] Rate limit reached. Sleeping {backoff}s (Attempt {attempt+1}/{max_attempts})...")
                time.sleep(backoff)
            else:
                if attempt == max_attempts - 1:
                    raise e
                time.sleep(2)
    return llm_instance.invoke(messages)

def sanitize_and_clamp_messages(messages: List[BaseMessage], max_history: int = 6) -> List[BaseMessage]:
    """Sliding window keeps token usage under budget."""
    clamped = []
    for m in messages[-max_history:]:
        if isinstance(m, ToolMessage) and isinstance(m.content, str):
            clamped.append(ToolMessage(content=clamp_text(m.content), tool_call_id=m.tool_call_id))
        else:
            clamped.append(m)
    return clamped

def research_and_context_agent(state: MediaContentState) -> dict:
    """Stage 1: Gathers grounding data using RESEARCH_STAGE_TOOLS."""
    messages = state.get("research_messages") or []
    iters = state.get("research_iterations", 0)

    if not messages:
        topic = state["topic"]
        attached = state.get("attached_files") or []
        init_prompt = f"""You are a Lead Viral Media Researcher.
Topic: {topic}
Attached Files: {attached}

Directives:
1. If the topic involves news, tech frameworks, or statistics, use RESEARCH_WEB_TOOLS.
2. If files are provided, read them using FILE_OPS_TOOLS.
3. If readability or NLP metrics are requested, use TEXT_NLP_TOOLS.
4. If no search is needed, return a clean factual brief."""

        messages = [
            SystemMessage(content="You verify facts and extract research context using your bound tools."),
            HumanMessage(content=init_prompt)
        ]

    prepared_msgs = sanitize_and_clamp_messages(messages)
    response = robust_invoke_with_retry(llm_researcher, prepared_msgs)
    
    return {
        "research_messages": [response],
        "research_iterations": iters + 1
    }

def media_strategy_agent(state: MediaContentState) -> dict:
    """Stage 2: Formulates growth strategy using MEDIA_STAGE_TOOLS."""
    messages = state.get("media_messages") or []
    iters = state.get("media_iterations", 0)

    if not messages:
        topic = state["topic"]
        platform = state.get("target_platform", "youtube_shorts")
        tone = state.get("tone_style", "fast-paced dramatic")
        research_notes = [str(m.content) for m in state.get("research_messages", []) if isinstance(m, ToolMessage)]
        context_str = "\n".join(research_notes)[:2500] if research_notes else "Self-contained topic."

        init_prompt = f"""You are an Audience Retention & Growth Strategist.
Topic: {topic}
Platform: {platform}
Tone: {tone}
Research Grounding:
{context_str}

Directives:
1. Use SOCIAL_GROWTH_TOOLS to pull trending tags, audience angles, and optimal formats.
2. Use MEDIA_CONTENT_TOOLS to check format constraints.
3. Once finished, output structured retention angles and growth metadata."""

        messages = [
            SystemMessage(content="You optimize video distribution and formats using MEDIA_CONTENT_TOOLS and SOCIAL_GROWTH_TOOLS."),
            HumanMessage(content=init_prompt)
        ]

    prepared_msgs = sanitize_and_clamp_messages(messages)
    response = robust_invoke_with_retry(llm_media_engine, prepared_msgs)

    return {
        "media_messages": [response],
        "media_iterations": iters + 1
    }

def hook_synthesizer_node(state: MediaContentState) -> dict:
    """Synthesizes strategy into structured hook variants and growth metadata."""
    topic = state["topic"]
    media_msgs = state.get("media_messages") or []
    
    last_text = ""
    for m in reversed(media_msgs):
        if isinstance(m, AIMessage) and m.content:
            last_text = str(m.content)
            break

    prompt = f"""Synthesize exactly 3 high-CTR opening hooks from this data:
{last_text}

Return ONLY valid JSON matching this schema:
{{
  "hooks": ["<Hook 1: Negative bias/Contrarian>", "<Hook 2: Open Curiosity Gap>", "<Hook 3: High stakes value>"],
  "recommended_hook_index": 0,
  "social_meta": {{
    "target_audience": "<Target audience segment>",
    "suggested_tags": ["#tag1", "#tag2", "#tag3"]
  }}
}}"""

    response = robust_invoke_with_retry(llm_base, prompt)
    data = parse_json_safely(getattr(response, "text", "") or getattr(response, "content", ""))
    
    hooks = data.get("hooks", [f"Stop scrolling if you care about {topic}."])
    rec_idx = data.get("recommended_hook_index", 0)
    selected = hooks[rec_idx] if 0 <= rec_idx < len(hooks) else hooks[0]

    return {
        "hook_options": hooks,
        "selected_hook": selected,
        "social_growth_meta": data.get("social_meta", {})
    }

def storyboard_writer_node(state: MediaContentState) -> dict:
    """Generates the scene-by-scene production storyboard within strict pacing limits."""
    topic = state["topic"]
    hook = state["selected_hook"]
    duration = state.get("target_duration_sec", 30)
    platform = state.get("target_platform", "youtube_shorts")
    tone = state.get("tone_style", "fast-paced dramatic")
    feedback = state.get("audit_feedback")

    feedback_prompt = f"\nCRITICAL CORRECTION FROM AUDITOR (MUST RESOLVE):\n{feedback}\n" if feedback else ""
    target_scenes = max(4, duration // 5)
    max_words = int(duration * 2.5)

    prompt = f"""You are an elite Video Director and Screenwriter.

Topic: {topic}
Selected Hook: {hook}
Target Duration: {duration} seconds (Hard speech limit: max {max_words} words across entire voiceover)
Target Scenes: ~{target_scenes}
Platform: {platform}
Tone: {tone}
{feedback_prompt}

Generate the production storyboard. Every scene MUST include:
- `scene_number`: Integer
- `timestamp`: Time interval (e.g., "0:00 - 0:03")
- `voiceover_script`: High-impact spoken dialogue
- `on_screen_text`: Kinetic text cues (1-4 words)
- `visual_direction`: Framing, transitions, and camera movement
- `image_prompt`: Detailed visual prompt for AI B-roll generation
- `sound_sfx`: Audio cues (whoosh, hit, riser, vinyl stop)

Return ONLY valid JSON:
{{
  "title": "<Catchy Video Title>",
  "target_bpm_music": "<Recommended Music Genre / BPM>",
  "scenes": [
    {{
      "scene_number": 1,
      "timestamp": "0:00 - 0:03",
      "voiceover_script": "{hook}",
      "on_screen_text": "HOOK KEYWORDS IN BOLD",
      "visual_direction": "Aggressive punch-in on subject with glitch transition",
      "image_prompt": "Cinematic 8k portrait of an engineer looking shocked in a dark server room, volumetric blue lighting --ar 9:16",
      "sound_sfx": "Heavy bass impact + tape stop"
    }}
  ]
}}"""

    response = robust_invoke_with_retry(llm_base, prompt)
    data = parse_json_safely(getattr(response, "text", "") or getattr(response, "content", ""))
    scenes = data.get("scenes", [])

    return {
        "scenes": scenes,
        "raw_production_blueprint": data
    }

def pacing_and_retention_auditor(state: MediaContentState) -> dict:
    """Deterministic Pacing + Retention Audit (Prevents rushed narration)."""
    scenes = state.get("scenes") or []
    target_duration = state.get("target_duration_sec", 30)
    retries = state.get("retry_count", 0)
    total_words = sum(len(s.get("voiceover_script", "").split()) for s in scenes)
    max_safe_words = int(target_duration * 2.8)
    if total_words > max_safe_words:
        return {
            "audit_passed": False,
            "audit_feedback": (
                f"Pacing Alert: Script has {total_words} words for a {target_duration}s video. "
                f"Hard ceiling is {max_safe_words} words. Trim dialogue sentences directly."
            ),
            "retry_count": retries + 1
        }
    if not scenes or not scenes[0].get("voiceover_script"):
        return {
            "audit_passed": False,
            "audit_feedback": "Scene 1 voiceover is empty. Hook must be present.",
            "retry_count": retries + 1
        }
    return {
        "audit_passed": True,
        "audit_feedback": None,
        "retry_count": retries + 1
    }

def delivery_initiator_node(state: MediaContentState) -> dict:
    """Prepares structured storyboard export for DELIVERY_STAGE_TOOLS."""
    topic = state["topic"]
    scenes = state.get("scenes") or []
    blueprint = state.get("raw_production_blueprint") or {}
    platform = state.get("target_platform", "youtube_shorts")
    duration = state.get("target_duration_sec", 30)
    social_meta = state.get("social_growth_meta") or {}

    clean_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', topic[:30]).strip().replace(' ', '_').lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = f"media_vault/{clean_title}_{timestamp}_storyboard.md"
    os.makedirs("media_vault", exist_ok=True)

    formatted_md = f"""# 🎬 Production Storyboard: {blueprint.get('title', topic.title())}
**Platform:** `{platform.upper()}` | **Duration:** `{duration}s`
**Music/Audio Style:** {blueprint.get('target_bpm_music', 'Dynamic modern beat')}
**Tags:** {' '.join(social_meta.get('suggested_tags', []))}

---
## 🎯 Shot List
"""
    for sc in scenes:
        formatted_md += f"\n### Scene {sc.get('scene_number')} ({sc.get('timestamp')})\n"
        formatted_md += f"- **Voiceover:** {sc.get('voiceover_script')}\n"
        formatted_md += f"- **On-Screen Text:** `{sc.get('on_screen_text')}`\n"
        formatted_md += f"- **Visual Direction:** {sc.get('visual_direction')}\n"
        formatted_md += f"- **AI Prompt:** *\"{sc.get('image_prompt')}\"*\n"
        formatted_md += f"- **SFX:** {sc.get('sound_sfx')}\n"
    delivery_prompt = [
        SystemMessage(content="You are a Production Document Deliverer with DOC_GENERATION_TOOLS and FILE_OPS_TOOLS."),
        HumanMessage(content=f"Invoke your document saving or file tools to save this storyboard to `{target_path}`:\n\n{formatted_md}")
    ]
    return {
        "delivery_messages": delivery_prompt,
        "delivery_iterations": 0,
        "saved_script_path": target_path
    }

def delivery_agent_node(state: MediaContentState) -> dict:
    """Executes DOC_GENERATION_TOOLS and FILE_OPS_TOOLS."""
    messages = state.get("delivery_messages") or []
    iters = state.get("delivery_iterations", 0)
    prepared_msgs = sanitize_and_clamp_messages(messages)
    response = robust_invoke_with_retry(llm_deliverer, prepared_msgs)
    return {
        "delivery_messages": [response],
        "delivery_iterations": iters + 1
    }

def deterministic_vault_exporter(state: MediaContentState) -> dict:
    """Guaranteed Python File I/O: Deterministic fallback writes files safely to media_vault/."""
    topic = state["topic"]
    scenes = state.get("scenes") or []
    blueprint = state.get("raw_production_blueprint") or {}
    platform = state.get("target_platform", "youtube_shorts")
    duration = state.get("target_duration_sec", 30)
    social_meta = state.get("social_growth_meta") or {}
    script_path = state.get("saved_script_path")

    clean_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', topic[:30]).strip().replace(' ', '_').lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("media_vault", exist_ok=True)
    manifest_filename = f"media_vault/{clean_title}_{timestamp}_manifest.json"
    manifest_payload = {
        "project": {
            "title": blueprint.get("title", topic.title()),
            "platform": platform,
            "target_duration_sec": duration,
            "audio_style": blueprint.get("target_bpm_music", "Dynamic / Modern"),
            "social_metadata": social_meta,
            "created_at": datetime.now().isoformat()
        },
        "hook_options": state.get("hook_options", []),
        "selected_hook": state.get("selected_hook", ""),
        "scenes": scenes
    }
    with open(manifest_filename, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, ensure_ascii=False)
    if not script_path or not os.path.exists(script_path):
        script_path = f"media_vault/{clean_title}_{timestamp}_storyboard.md"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(f"# Storyboard: {blueprint.get('title', topic.title())}\n\n")
            for sc in scenes:
                f.write(f"### Scene {sc.get('scene_number')} ({sc.get('timestamp')})\n")
                f.write(f"- Voiceover: {sc.get('voiceover_script')}\n")
                f.write(f"- Visual: {sc.get('visual_direction')}\n")
                f.write(f"- Prompt: {sc.get('image_prompt')}\n\n")

    total_words = sum(len(s.get("voiceover_script", "").split()) for s in scenes)

    summary = (
        f"✨ Production Blueprint Complete!\n\n"
        f"📁 **Storyboard Markdown:** `{script_path}`\n"
        f"⚙️ **Asset Manifest JSON:** `{manifest_filename}`\n"
        f"📊 **Metrics:** {len(scenes)} scenes | {total_words} words | Duration: {duration}s"
    )

    return {
        "saved_script_path": script_path,
        "saved_manifest_path": manifest_filename,
        "final_summary": summary
    }

def route_research_loop(state: MediaContentState) -> str:
    messages = state.get("research_messages") or []
    if not messages:
        return "media_strategist"
    last_msg = messages[-1]
    iters = state.get("research_iterations", 0)

    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0 and iters < 2:
        return "research_tools"
    return "media_strategist"

def route_media_loop(state: MediaContentState) -> str:
    messages = state.get("media_messages") or []
    if not messages:
        return "hook_synthesizer"
    last_msg = messages[-1]
    iters = state.get("media_iterations", 0)

    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0 and iters < 2:
        return "media_tools"
    return "hook_synthesizer"

def route_audit_loop(state: MediaContentState) -> str:
    passed = state.get("audit_passed", True)
    retries = state.get("retry_count", 0)
    if not passed and retries < 2:
        return "storyboard_writer"
    return "delivery_initiator"

def route_delivery_loop(state: MediaContentState) -> str:
    messages = state.get("delivery_messages") or []
    if not messages:
        return "vault_exporter"
    last_msg = messages[-1]
    iters = state.get("delivery_iterations", 0)

    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0 and iters < 2:
        return "delivery_tools"
    return "vault_exporter"

builder = StateGraph(MediaContentState)
builder.add_node("researcher", research_and_context_agent)
builder.add_node("research_tools", ToolNode(RESEARCH_STAGE_TOOLS))
builder.add_node("media_strategist", media_strategy_agent)
builder.add_node("media_tools", ToolNode(MEDIA_STAGE_TOOLS))
builder.add_node("hook_synthesizer", hook_synthesizer_node)
builder.add_node("storyboard_writer", storyboard_writer_node)
builder.add_node("auditor", pacing_and_retention_auditor)
builder.add_node("delivery_initiator", delivery_initiator_node)
builder.add_node("delivery_agent", delivery_agent_node)
builder.add_node("delivery_tools", ToolNode(DELIVERY_STAGE_TOOLS))
builder.add_node("vault_exporter", deterministic_vault_exporter)

builder.add_edge(START, "researcher")
builder.add_conditional_edges(
    "researcher",
    route_research_loop,
    {
        "research_tools": "research_tools",
        "media_strategist": "media_strategist"
    }
)
builder.add_edge("research_tools", "researcher")
builder.add_conditional_edges(
    "media_strategist",
    route_media_loop,
    {
        "media_tools": "media_tools",
        "hook_synthesizer": "hook_synthesizer"
    }
)
builder.add_edge("media_tools", "media_strategist")
builder.add_edge("hook_synthesizer", "storyboard_writer")
builder.add_edge("storyboard_writer", "auditor")
builder.add_conditional_edges(
    "auditor",
    route_audit_loop,
    {
        "storyboard_writer": "storyboard_writer",
        "delivery_initiator": "delivery_initiator"
    }
)
builder.add_edge("delivery_initiator", "delivery_agent")
builder.add_conditional_edges(
    "delivery_agent",
    route_delivery_loop,
    {
        "delivery_tools": "delivery_tools",
        "vault_exporter": "vault_exporter"
    }
)
builder.add_edge("delivery_tools", "delivery_agent")
builder.add_edge("vault_exporter", END)
media_content_agent = builder.compile()


def run_media_task(
    topic: str,
    target_platform: str = "youtube_shorts",
    target_duration_sec: int = 30,
    tone_style: str = "engaging and professional"
) -> Dict[str, Any]:
    """
    Run an end-to-end media and content production task.
    Produces scripts, scene blueprints, and voiceovers 100% locally.
    """
    init_state = {
        "topic": topic,
        "target_platform": target_platform,
        "target_duration_sec": target_duration_sec,
        "tone_style": tone_style,
        "attached_files": [],
        "research_messages": [],
        "research_iterations": 0,
        "grounded_context": None,
        "media_messages": [],
        "media_iterations": 0,
        "social_growth_meta": None,
        "hook_options": None,
        "selected_hook": None,
        "scenes": None,
        "raw_production_blueprint": None,
        "audit_passed": False,
        "audit_feedback": None,
        "retry_count": 0,
        "delivery_messages": [],
        "delivery_iterations": 0,
        "saved_script_path": None,
        "saved_manifest_path": None,
        "final_summary": None,
    }
    try:
        final_state = media_content_agent.invoke(init_state)
        summary = final_state.get("final_summary") or ""
        hook = final_state.get("selected_hook", "")
        if not summary and hook:
            summary = f"### Media Script Plan\n\n**Hook:** {hook}\n**Platform:** {target_platform}\n**Duration:** {target_duration_sec}s"
        return {
            "status": "success",
            "agent": "Media & Content Agent",
            "reply": summary or "Media content production complete.",
            "hook": hook,
            "script_path": final_state.get("saved_script_path"),
            "manifest_path": final_state.get("saved_manifest_path"),
        }
    except Exception as e:
        # Fallback to local media tools
        try:
            from tools.media_content import script_generator, title_generator
            titles = title_generator.invoke({"topic": topic})
            script = script_generator.invoke({"topic": topic, "platform": target_platform})
            rep = f"### Generated Media Package\n\n**Suggested Titles:**\n{titles}\n\n**Script:**\n{script}"
            return {
                "status": "success",
                "agent": "Media & Content Agent",
                "reply": rep,
            }
        except Exception as fb_err:
            return {
                "status": "error",
                "agent": "Media & Content Agent",
                "reply": f"Media execution error: {e} ({fb_err})",
                "error": str(e),
            }


if __name__ == "__main__":
    sample_request = {
        "topic": "Why writing raw SQL inside Python loops destroys database performance",
        "target_platform": "youtube_shorts",
        "target_duration_sec": 30,
        "tone_style": "high energy developer breakdown with sharp glitch transitions",
        "attached_files": [],
        "research_messages": [],
        "research_iterations": 0,
        "grounded_context": None,
        "media_messages": [],
        "media_iterations": 0,
        "social_growth_meta": None,
        "hook_options": None,
        "selected_hook": None,
        "scenes": None,
        "raw_production_blueprint": None,
        "audit_passed": False,
        "audit_feedback": None,
        "retry_count": 0,
        "delivery_messages": [],
        "delivery_iterations": 0,
        "saved_script_path": None,
        "saved_manifest_path": None,
        "final_summary": None
    }

    print("\n🎬 Launching Production Media Pipeline...\n")

    for step in media_content_agent.stream(sample_request):
        for node_name, state_update in step.items():
            print(f"[NODE COMPLETED: {node_name.upper()}]")
            if "selected_hook" in state_update and state_update["selected_hook"]:
                print(f"-> Selected Hook: \"{state_update['selected_hook']}\"")
            if "audit_passed" in state_update:
                print(f"-> Pacing Audit Passed: {state_update['audit_passed']}")
            if "final_summary" in state_update and state_update["final_summary"]:
                print("\n" + "=" * 50)
                print(state_update["final_summary"])
                print("=" * 50)
            print("-" * 50)
