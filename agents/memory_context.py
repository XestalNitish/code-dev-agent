import os
import json
import re
import time
import uuid
import threading
from datetime import datetime
from typing import TypedDict, Literal, Optional, List, Dict, Any, Annotated
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# --- 1. Import Exact Tools from tools Package ---
from tools import (
    MEMORY_CONTEXT_TOOLS,
    FILE_OPS_TOOLS,
    SECURITY_PRIVACY_TOOLS
)

load_dotenv()

# Active Memory Operations & Governance Toolset
MEMORY_GOVERNANCE_TOOLS = list(MEMORY_CONTEXT_TOOLS) + list(FILE_OPS_TOOLS) + list(SECURITY_PRIVACY_TOOLS)


# --- 2. Thread-Safe ChromaDB Singleton Provider ---
_CHROMA_LOCK = threading.Lock()
_CHROMA_CLIENT: Optional[chromadb.PersistentClient] = None
_MEMORY_COLLECTION = None

def get_memory_collection():
    """Thread-safe persistent Chroma vector store."""
    global _CHROMA_CLIENT, _MEMORY_COLLECTION
    if _MEMORY_COLLECTION is None:
        with _CHROMA_LOCK:
            if _MEMORY_COLLECTION is None:
                db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db_store")
                os.makedirs(db_path, exist_ok=True)
                _CHROMA_CLIENT = chromadb.PersistentClient(path=db_path)
                embed_fn = embedding_functions.DefaultEmbeddingFunction()
                _MEMORY_COLLECTION = _CHROMA_CLIENT.get_or_create_collection(
                    name="agent_long_term_memory",
                    embedding_function=embed_fn,
                    metadata={"hnsw:space": "cosine"}
                )
    return _MEMORY_COLLECTION



def get_universal_llm(temperature: float = 0.0):
    try:
        from local_llm_service import get_local_chat_llm
        return get_local_chat_llm(temperature=temperature)
    except Exception:
        pass
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url="http://127.0.0.1:8088/v1",
            api_key="x-estal-local",
            model_name="X-Estal-4B",
            temperature=temperature,
        )
    except Exception:
        return None

llm_gate = get_universal_llm(temperature=0.0)
llm_writer = get_universal_llm(temperature=0.0)
llm_compiler = get_universal_llm(temperature=0.2)


# --- 4. State Schema ---
class ExtractedFact(TypedDict):
    statement: str
    category: Literal["user_preference", "tech_stack", "project_rule", "research_finding", "general"]

class RetrievedMemory(TypedDict):
    content: str
    category: str
    score: float
    id: str

class MemoryContextState(TypedDict):
    task: str
    user_id: Optional[str]
    action: Literal["auto", "read", "write", "both", "none"]
    search_query: Optional[str]
    extracted_facts: Optional[List[ExtractedFact]]
    retrieved_memories: Optional[List[RetrievedMemory]]
    token_budget: Optional[int]
    status: Literal["success", "empty", "error", "bypassed"]
    error: Optional[str]
    formatted_context: Optional[str]
    final_output: Optional[str]


# --- 5. Utilities ---
def safe_parse_json(raw_text: str) -> dict:
    clean = raw_text.strip()
    if "```" in clean:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if match:
            clean = match.group(1).strip()
    try:
        return json.loads(clean)
    except Exception:
        start, end = clean.find("{"), clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(clean[start:end + 1])
            except Exception:
                pass
    return {}

def robust_invoke_with_retry(llm_instance, prompt_payload, max_attempts: int = 3):
    """Backoff handler to prevent rate limits from crashing agent execution."""
    for attempt in range(max_attempts):
        try:
            return llm_instance.invoke(prompt_payload)
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err or "quota" in err or "resource_exhausted" in err:
                wait_sec = (attempt + 1) * 6
                print(f"⚠️ [Memory Agent Backoff] Sleeping {wait_sec}s (Attempt {attempt+1}/{max_attempts})...")
                time.sleep(wait_sec)
            else:
                if attempt == max_attempts - 1:
                    raise e
                time.sleep(1.5)
    return llm_instance.invoke(prompt_payload)

def sanitize_sensitive_data(text: str) -> str:
    """Security Guard: Masks API tokens, private keys, and passwords before vector indexing."""
    sanitized = re.sub(r'(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*["\']?([a-zA-Z0-9_\-\.]{8,})["\']?', r'\1: [REDACTED_SECRET]', text)
    sanitized = re.sub(r'sk-[a-zA-Z0-9]{20,}', '[REDACTED_API_KEY]', sanitized)
    sanitized = re.sub(r'AIzaSy[a-zA-Z0-9_-]{33}', '[REDACTED_API_KEY]', sanitized)
    return sanitized


# --- 6. Pipeline Nodes ---

def context_gatekeeper(state: MemoryContextState) -> dict:
    """Intent Router & Classification."""
    task = state.get("task", "")[:2500]

    prompt = f"""You are the Executive Memory Router & Intent Classifier.
Analyze this input payload:

Payload:
{task}

Classification Rules:
- "read": The user is querying, asking instructions, or referencing past setup rules.
- "write": The payload explicitly provides new rules, tech stack decisions, architectural preferences, or facts.
- "both": The payload gives new facts to store AND simultaneously asks a question needing past knowledge.
- "none": Ephemeral greetings, small talk, or transient conversational filler.

Return ONLY a valid JSON object:
{{
  "action": "read" | "write" | "both" | "none",
  "search_query": "<concise semantic query if read/both, else null>",
  "reasoning": "<concise 1-line rationale>"
}}"""

    try:
        response = robust_invoke_with_retry(llm_gate, prompt)
        raw_text = getattr(response, "text", "") or getattr(response, "content", "")
        parsed = safe_parse_json(raw_text)
    except Exception:
        parsed = {"action": "read", "search_query": task}

    action = parsed.get("action", "read")
    search_query = parsed.get("search_query") or task

    if action not in ["read", "write", "both", "none"]:
        action = "read"

    return {
        "action": action,
        "search_query": search_query,
        "status": "success" if action != "none" else "bypassed"
    }


def memory_writer(state: MemoryContextState) -> dict:
    """Knowledge Extraction + Security Sanitization + Cosine Deduplication."""
    content = state.get("task", "")[:3500]
    user_id = state.get("user_id") or "global_system"
    collection = get_memory_collection()

    prompt = f"""You are a Knowledge Extraction Engine.
Extract atomic, high-signal, persistent facts from this content:

Source:
{content}

Directives:
1. Extract ONLY durable facts (user preferences, architectural rules, tech stack decisions, verified constraints).
2. Omit ephemeral chatter, greetings, and transient error traces.
3. Every fact must be an independent atomic sentence.

Return ONLY valid JSON:
{{
  "facts": [
    {{
      "statement": "<atomic factual statement>",
      "category": "user_preference" | "tech_stack" | "project_rule" | "research_finding" | "general"
    }}
  ]
}}"""

    try:
        response = robust_invoke_with_retry(llm_writer, prompt)
        raw_text = getattr(response, "text", "") or getattr(response, "content", "")
        parsed_data = safe_parse_json(raw_text)
        extracted_facts: List[ExtractedFact] = parsed_data.get("facts", [])
    except Exception as e:
        return {
            "extracted_facts": [],
            "status": "error",
            "error": f"Extraction failure: {str(e)}"
        }

    if not extracted_facts:
        return {
            "extracted_facts": [],
            "status": "empty",
            "final_output": "No durable facts found to write."
        }

    docs_to_add = []
    metas_to_add = []
    ids_to_add = []
    skipped_count = 0

    with _CHROMA_LOCK:
        has_records = collection.count() > 0

    for item in extracted_facts:
        raw_statement = item.get("statement", "").strip()
        if not raw_statement:
            continue

        # Security & Privacy Filter Check
        safe_statement = sanitize_sensitive_data(raw_statement)

        # Cosine Deduplication Check
        is_dup = False
        if has_records:
            try:
                with _CHROMA_LOCK:
                    existing = collection.query(query_texts=[safe_statement], n_results=1)
                if existing and existing.get("distances") and len(existing["distances"][0]) > 0:
                    # Cosine distance < 0.20 = > 80% semantic match
                    if existing["distances"][0][0] < 0.20:
                        is_dup = True
            except Exception:
                is_dup = False

        if not is_dup:
            cat = item.get("category", "general")
            docs_to_add.append(safe_statement)
            metas_to_add.append({
                "category": cat,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            })
            ids_to_add.append(str(uuid.uuid4()))
        else:
            skipped_count += 1

    # Batch Vector Upsert
    if docs_to_add:
        with _CHROMA_LOCK:
            collection.upsert(
                documents=docs_to_add,
                metadatas=metas_to_add,
                ids=ids_to_add
            )

    # Cold Vault Backup (JSON Snapshot for Crash Recovery via FILE_OPS pattern)
    try:
        vault_dir = "memory_vault"
        os.makedirs(vault_dir, exist_ok=True)
        backup_path = os.path.join(vault_dir, f"{user_id}_backup.json")
        existing_backup = []
        if os.path.exists(backup_path):
            with open(backup_path, "r", encoding="utf-8") as f:
                existing_backup = json.load(f)
        
        for d, m, i in zip(docs_to_add, metas_to_add, ids_to_add):
            existing_backup.append({"id": i, "statement": d, "metadata": m})

        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(existing_backup, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Vault backup non-blocking notification: {e}")

    summary = (
        f"✅ Long-Term Memory Synced: {len(docs_to_add)} atomic facts stored "
        f"({skipped_count} semantic duplicates filtered)."
    )

    return {
        "extracted_facts": extracted_facts,
        "status": "success",
        "final_output": summary if state.get("action") == "write" else None,
        "error": None
    }


def memory_retriever_node(state: MemoryContextState) -> dict:
    """Semantic Vector Search with Cosine Distance Filtering."""
    search_query = state.get("search_query") or state.get("task", "")
    collection = get_memory_collection()

    with _CHROMA_LOCK:
        total_docs = collection.count()

    if total_docs == 0:
        return {
            "retrieved_memories": [],
            "status": "empty",
            "formatted_context": "No persistent long-term memory recorded yet."
        }

    try:
        with _CHROMA_LOCK:
            results = collection.query(
                query_texts=[search_query],
                n_results=min(5, total_docs)
            )

        retrieved: List[RetrievedMemory] = []
        if results and results.get("documents") and len(results["documents"][0]) > 0:
            for doc, meta, dist, mem_id in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0]
            ):
                # Cosine distance < 0.75 threshold for strict semantic relevance
                if dist < 0.75:
                    retrieved.append({
                        "id": mem_id,
                        "content": doc,
                        "category": meta.get("category", "general"),
                        "score": round(1.0 - dist, 3)
                    })

        return {
            "retrieved_memories": retrieved,
            "status": "success" if retrieved else "empty"
        }
    except Exception as e:
        return {
            "retrieved_memories": [],
            "status": "error",
            "error": f"Vector search failure: {str(e)}"
        }


def context_compiler(state: MemoryContextState) -> dict:
    """Synthesizes vector entries into an LLM-ready Context Injection Block."""
    task = state.get("task", "")
    retrieved = state.get("retrieved_memories") or []
    extracted = state.get("extracted_facts") or []

    if not retrieved:
        empty_msg = "No prior stored context matched this objective."
        if extracted:
            empty_msg += f" (Indexed {len(extracted)} new facts from current payload)."
        return {
            "formatted_context": empty_msg,
            "final_output": empty_msg
        }

    prompt = f"""You are an Executive Context Synthesizer.
Synthesize these retrieved memory items into a dense, deduplicated Context Block for downstream agents.

Objective:
{task}

Retrieved Memory Records:
{json.dumps(retrieved, indent=2)}

Directives:
1. Group related facts under bold markdown category tags.
2. Filter out items irrelevant to the current objective.
3. Be dense and concise (Max 250 words). Do NOT output JSON."""

    try:
        response = robust_invoke_with_retry(llm_compiler, prompt)
        clean_markdown = (getattr(response, "text", "") or getattr(response, "content", "")).strip()
    except Exception:
        # Deterministic Markdown Fallback if synthesis model is busy
        clean_markdown = "### Stored Project Guidelines:\n"
        for mem in retrieved:
            clean_markdown += f"- [{mem['category'].upper()}] {mem['content']}\n"

    if extracted:
        clean_markdown += f"\n\n*(Note: {len(extracted)} new facts indexed concurrently in this session)*"

    return {
        "formatted_context": clean_markdown,
        "final_output": clean_markdown
    }


# --- 7. Routing Helpers ---

def route_after_gatekeeper(state: MemoryContextState) -> str:
    action = state.get("action", "read")
    if action in ["write", "both"]:
        return "writer"
    elif action == "none":
        return END
    return "retriever"

def route_after_writer(state: MemoryContextState) -> str:
    if state.get("action") == "both":
        return "retriever"
    return END


# --- 8. StateGraph Assembly ---

builder = StateGraph(MemoryContextState)

builder.add_node("gatekeeper", context_gatekeeper)
builder.add_node("writer", memory_writer)
builder.add_node("retriever", memory_retriever_node)
builder.add_node("compiler", context_compiler)

builder.add_edge(START, "gatekeeper")

builder.add_conditional_edges(
    "gatekeeper",
    route_after_gatekeeper,
    {
        "writer": "writer",
        "retriever": "retriever",
        END: END
    }
)

builder.add_conditional_edges(
    "writer",
    route_after_writer,
    {
        "retriever": "retriever",
        END: END
    }
)

builder.add_edge("retriever", "compiler")
builder.add_edge("compiler", END)
memory_context_agent = builder.compile()


def run_memory_task(
    task: str,
    action: str = "auto",
    user_id: str = "default_user"
) -> Dict[str, Any]:
    """
    Run a long-term memory ingestion or retrieval task through the Memory Agent.
    Guarantees privacy sanitization and vector indexing 100% locally.
    """
    init_state = {
        "task": task,
        "user_id": user_id,
        "action": action,
        "search_query": None,
        "extracted_facts": None,
        "retrieved_memories": None,
        "token_budget": 300,
        "status": "success",
        "error": None,
        "formatted_context": None,
        "final_output": None,
    }
    try:
        final_state = memory_context_agent.invoke(init_state)
        out = final_state.get("final_output") or final_state.get("formatted_context") or ""
        return {
            "status": "success",
            "agent": "Memory & Context Vault",
            "reply": out or "Memory state processed.",
            "output": out,
            "facts": final_state.get("extracted_facts"),
        }
    except Exception as e:
        # Fallback to local memory tool
        try:
            from tools.memory_context import conversation_memory, user_memory
            res = conversation_memory.invoke({"action": "read", "session_id": user_id})
            return {
                "status": "success",
                "agent": "Memory & Context Vault",
                "reply": f"Retrieved Context:\n{res}",
            }
        except Exception as fb_err:
            return {
                "status": "error",
                "agent": "Memory & Context Vault",
                "reply": f"Memory execution error: {e} ({fb_err})",
                "error": str(e),
            }


# --- 9. Execution Verification ---
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 PRODUCTION VERIFICATION: MEMORY CONTEXT AGENT")
    print("=" * 60)

    # 1. Test Ingestion & Privacy Sanitization
    test_ingest = {
        "task": (
            "Store architectural rule: All agents must use deterministic vault exporters for file persistence. "
            "Also our internal secret token is sk-1234567890abcdef1234567890 (verify security masking)."
        ),
        "user_id": "nitish_developer",
        "action": "auto",
        "search_query": None,
        "extracted_facts": None,
        "retrieved_memories": None,
        "token_budget": 300,
        "status": "success",
        "error": None,
        "formatted_context": None,
        "final_output": None
    }

    print("\n▶ STEP 1: Ingestion & Vector Storage...")
    res_ingest = memory_context_agent.invoke(test_ingest)
    print(res_ingest.get("final_output"))

    # 2. Test Semantic Retrieval
    test_retrieve = {
        "task": "How do our agents handle file persistence?",
        "user_id": "nitish_developer",
        "action": "auto",
        "search_query": None,
        "extracted_facts": None,
        "retrieved_memories": None,
        "token_budget": 300,
        "status": "success",
        "error": None,
        "formatted_context": None,
        "final_output": None
    }

    print("\n▶ STEP 2: Semantic Vector Retrieval...")
    res_retrieve = memory_context_agent.invoke(test_retrieve)
    print("\n" + res_retrieve.get("formatted_context"))
    print("\n" + "=" * 60)
