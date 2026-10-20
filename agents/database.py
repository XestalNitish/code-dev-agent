from typing import TypedDict, Literal, Optional, List, Dict, Any, Tuple
import os
import sqlite3
import json
import re
import time
from dotenv import load_dotenv
try:
    from langchain_mistralai import ChatMistralAI
except ImportError:
    ChatMistralAI = None
from langgraph.prebuilt import tool_node,ToolNode
from langgraph.graph import StateGraph , START,END
from typing import TypedDict , Literal
from tools import DATABASE_DATA_TOOLS
load_dotenv()

def _init_db_llm():
    try:
        from local_llm_service import get_local_chat_llm
        return get_local_chat_llm(temperature=0.1)
    except Exception:
        pass
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url="http://127.0.0.1:8088/v1",
            api_key="x-estal-local",
            model_name="X-Estal-4B",
            temperature=0.1,
        )
    except Exception:
        return None

llm = _init_db_llm()
llm2 = llm.bind_tools(DATABASE_DATA_TOOLS) if (llm and DATABASE_DATA_TOOLS) else llm
from typing import TypedDict, Literal, Optional, List, Dict, Any
class DatabaseDataState(TypedDict):
    task: str                                                  
    db_type: Literal["postgres", "mysql", "redis", "mongodb", "vector"]
    schema_info: str                                           
    generated_query: str                                       
    query_type: Literal["read", "write", "migration"]
    safety_status: Literal["safe", "unsafe", "needs_fix"]
    safety_notes: Optional[str]                                
    execution_result: Any                                      
    error: Optional[str]                                      
    retry_count: int                                          
    final_output: str       

def Inspector(state:DatabaseDataState):


    task = state["task"]
    schema_catalog=state["schema_info"]
    prompt= f"""You are a Principal Database Architect and Schema Analyst.
    Your task is to analyze the user request, identify the target database dialect, inspect existing tables/collections, and extract relevant schema definitions.
    Input Parameters:
        - Task: {task}
        - Available Schemas / Catalog: {schema_catalog}
    Execution Rules:
    1. Identify the intended database engine strictly as one of: ["postgres", "mysql", "redis", "mongodb", "vector"].
    2. If schema details are missing, invoke `schema_analyzer` or `database_reader` to inspect table columns, keys, and indexes.
    3. Determine whether the requested operation is "read", "write", or "migration".
    4. Extract only the tables and columns strictly relevant to the task.
    Strict Output Constraints:
        - Return ONLY a valid JSON object matching this schema:
    {{
        "db_type": "postgres" | "mysql" | "redis" | "mongodb" | "vector",
        "query_type": "read" | "write" | "migration",
        "target_entities": ["<list of relevant tables or collections>"],
        "schema_context": "<markdown text detailing column names, data types, primary/foreign keys>"
    }}"""
    response= llm2.invoke(prompt)
    raw_text = getattr(response, "text", "") or getattr(response, "content", "")

    clean_text = raw_text.strip()
    if "```" in clean_text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        if match:
            clean_text = match.group(1).strip()

    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError:
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            data = json.loads(clean_text[start_idx : end_idx + 1])
        else:
            data = {}
    return {
        "db_type": data.get("db_type", "postgres"),
        "query_type": data.get("query_type", "read"),
        "schema_info": data.get("schema_context", ""),
        "safety_notes": f"Targeting entities: {', '.join(data.get('target_entities', []))}"
    }

def Query_Migration(state:DatabaseDataState):
    time.sleep(2)
    task=state["task"]
    db_type = state["db_type"]
    query_type= state["query_type"]
    schema_context= state["schema_info"]
    error=state["error"]
    prompt=f"""You are an Expert Database Engineer and Query Optimization Specialist.
    Your task is to generate syntax-accurate, dialect-compliant, and high-performance database statements based strictly on the provided schema.
    Input Parameters:
        - Task: {task}
        - DB Engine: {db_type}
        - Query Type: {query_type}
        - Schema Context: {schema_context}
        - Previous Fix Feedback (if correcting): {error}
    Execution Rules:
    1. Generate queries tailored strictly to {db_type}:
       - For postgres/mysql: Output standard ANSI/dialect SQL with correct table/column identifiers.
       - For mongodb: Output valid JSON/BSON MQL queries (e.g., db.collection.find(...)).
       - For redis: Output standardized Redis commands (e.g., GET, HGETALL, ZREVRANGEBYSCORE).
       - For vector: Output similarity search query payloads with embedding parameters.
       - For migration: Use `migration_manager` patterns (DDL scripts or alter migrations).
    2. Never invent column or table names not present in Schema Context.
    3. If {error} is present, resolve the exact syntax error or schema mismatch identified.
    4. Output raw query string directly. Do NOT wrap inside JSON to prevent string escaping errors.
    Strict Output Constraints:
    Return the generated query inside a standard fenced block:
    ```query
    <your executable query/command here>"""
    response=llm2.invoke(prompt)
    raw_text = getattr(response, "text", "") or getattr(response, "content", "")
    generated_query = raw_text.strip()
    if "```query" in raw_text:
        generated_query = raw_text.split("```query")[1].split("```")[0].strip()
    elif "```sql" in raw_text:
        generated_query = raw_text.split("```sql")[1].split("```")[0].strip()
    elif "```" in raw_text:
        generated_query = raw_text.split("```")[1].split("```")[0].strip()
    return {
        "generated_query": generated_query,
        "error": None
    }

def qury_connector(state:DatabaseDataState):
    time.sleep(2)
    generated_query = state["generated_query"]
    db_type=state["db_type"]
    schema_context=state["schema_info"]
    error=state["error"]
    prompt=f"""You are a Database Reliability and Query Debugging Specialist.
    Your task is to analyze failed database executions, locate syntax or schema mismatches, and formulate actionable correction instructions.
    Input Parameters:
    - Failing Query: {generated_query}
    - DB Engine: {db_type}
    - Schema Context: {schema_context}
    - Execution Error / Traceback: {error}
    Execution Rules:
    1. Isolate the exact point of failure (e.g., column does not exist, type mismatch, syntax error near token, missing index).
    2. If column ambiguity or table naming is the issue, verify against {schema_context}.
    3. Provide concise, direct instructions for the Query Generator node to output the correct syntax.
    Strict Output Constraints:
    - Return ONLY a valid JSON object matching this schema:
    {{
      "root_cause": "<concise explanation of why the query failed>",
      "error_category": "syntax_error" | "schema_mismatch" | "type_mismatch" | "permission_error",
      "correction_instructions": "<exact instructions to regenerate the correct query>"
    }}"""
    response=llm2.invoke(prompt)
    raw_text = getattr(response, "text", "") or getattr(response, "content", "")
    clean_text = raw_text.strip()
    if "```" in clean_text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        if match:
            clean_text = match.group(1).strip()
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError:
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                data = json.loads(clean_text[start_idx : end_idx + 1])
            except Exception:
                data = {}
        else:
            data = {}
    root_cause = data.get("root_cause", "Unknown execution error")
    correction_instructions = data.get("correction_instructions", raw_text)
    return {
        "error": f"Root Cause: {root_cause}\nInstructions: {correction_instructions}",
        "retry_count": state.get("retry_count", 0) + 1,
        "safety_status": "needs_fix"
    }

def reporting(state:DatabaseDataState):
    time.sleep(2)
    task= state["task"]
    generated_query=state["generated_query"]
    execution_result=state["execution_result"]
    prompt=f"""You are a Senior Business Intelligence and Data Delivery Analyst.
    Your task is to parse raw database execution results and translate them into a clear, professional analytical deliverable for the user.
    Input Parameters:
    - Original Request: {task}
    - Executed Query: {generated_query}
    - Execution Output / Records: {execution_result}
    Execution Rules:
    1. If output contains record sets (rows/documents):
       - Present the data using a clean Markdown table with appropriate column headers.
       - Summarize key findings, counts, or anomalies observed in the data.
    2. If output indicates a mutation (INSERT/UPDATE/DELETE/MIGRATION):
       - Confirm affected rows/status clearly.
    3. Never expose raw connection strings, internal system IDs, or unformatted blobs.
    Strict Output Constraints:
    - Return a clean, professional Markdown report structured with:
      ## Execution Summary
      ## Query Executed
      ## Results & Data View
      ## Business Insights / Notes"""
    response=llm2.invoke(prompt)
    raw_text = getattr(response, "text", "") or getattr(response, "content", "")
    final_report = raw_text.strip()
    return {
        "final_output": final_report,
        "safety_status": "safe"
    }

def safety_validator_node(state: DatabaseDataState):
    query = state.get("generated_query", "").upper()

    destructive_keywords = ["DROP DATABASE", "DROP TABLE", "TRUNCATE", "FLUSHALL"]    
    for kw in destructive_keywords:
        if kw in query:
            return {
                "safety_status": "unsafe",
                "error": f"Security Alert: Destructive query detected ({kw}). Execution blocked."
            }       
    return {
        "safety_status": "safe",
        "error": None
    }

def execute_sql(engine: str, query: str) -> Tuple[Any, str | None]:
    """Production executor for SQL dialects (Postgres, MySQL, SQLite)."""
    try:
        if engine == "postgres":
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            conn = psycopg2.connect(
                os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/dbname")
            )
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                if query.strip().upper().startswith("SELECT"):
                    result = [dict(row) for row in cur.fetchall()]
                else:
                    conn.commit()
                    result = f"Rows affected: {cur.rowcount}"
            conn.close()
            return result, None

        elif engine == "mysql":
            import pymysql
            
            conn = pymysql.connect(
                host=os.getenv("MYSQL_HOST", "localhost"),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
                database=os.getenv("MYSQL_DB", "test"),
                cursorclass=pymysql.cursors.DictCursor
            )
            with conn.cursor() as cur:
                cur.execute(query)
                if query.strip().upper().startswith("SELECT"):
                    result = cur.fetchall()
                else:
                    conn.commit()
                    result = f"Rows affected: {cur.rowcount}"
            conn.close()
            return result, None

        else:
            # Fallback to local SQLite file
            conn = sqlite3.connect(os.getenv("SQLITE_PATH", "production.db"))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(query)
            if query.strip().upper().startswith("SELECT"):
                result = [dict(row) for row in cur.fetchall()]
            else:
                conn.commit()
                result = f"Rows affected: {cur.rowcount}"
            conn.close()
            return result, None

    except Exception as e:
        return None, str(e)

def execute_nosql(engine: str, command: str) -> Tuple[Any, str | None]:
    """Production executor for Redis & MongoDB."""
    try:
        if engine == "redis":
            import redis
            r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
            parts = command.strip().split()
            cmd, args = parts[0].upper(), parts[1:]
            result = r.execute_command(cmd, *args)
            return str(result), None
        elif engine == "mongodb":
            from pymongo import MongoClient
            import json
            client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
            db = client[os.getenv("MONGO_DB", "app_data")]
            coll_name, op, raw_payload = command.split("|", 2)
            collection = db[coll_name]
            payload = json.loads(raw_payload) if raw_payload.strip() else {}
            
            if op == "find":
                result = list(collection.find(payload, {"_id": 0}).limit(50))
            elif op == "insert":
                res = collection.insert_one(payload)
                result = f"Inserted ID: {res.inserted_id}"
            else:
                result = f"Operation {op} completed"
            return result, None

        return None, f"Unsupported NoSQL engine: {engine}"

    except Exception as e:
        return None, str(e)

def dynamic_schema_extractor(db_type: str) -> str:
    """Live schema inspection directly from DB metadata."""
    try:
        if db_type == "postgres":
            query = """
            SELECT table_name, column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
            """
            rows, err = execute_sql("postgres", query)
            if err: return f"Schema Error: {err}"
            return "\n".join([f"{r['table_name']}.{r['column_name']} ({r['data_type']})" for r in rows])

        elif db_type == "sqlite":
            conn = sqlite3.connect(os.getenv("SQLITE_PATH", "production.db"))
            cur = conn.cursor()
            cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cur.fetchall() if row[0]]
            conn.close()
            return "\n\n".join(tables) if tables else "No tables found."

        elif db_type == "mongodb":
            from pymongo import MongoClient
            client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
            db = client[os.getenv("MONGO_DB", "app_data")]
            collections = db.list_collection_names()
            return f"Collections: {', '.join(collections)}"

        return "Default schema context."
    except Exception as e:
        return f"Introspection failed: {str(e)}"
    
def database_executor_node(state: DatabaseDataState):
    query = state.get("generated_query", "")
    db_type = state.get("db_type", "sqlite")
    
    if state.get("safety_status") == "unsafe":
        return {"execution_result": None, "error": state.get("error")}

    if db_type in ["postgres", "mysql", "sqlite"]:
        result, error = execute_sql(db_type, query)
    elif db_type in ["redis", "mongodb"]:
        result, error = execute_nosql(db_type, query)
    elif db_type == "vector":
        result, error = None, "Vector DB integration pending client initialization."
    else:
        result, error = None, f"Unknown database dialect: {db_type}"

    return {
        "execution_result": result,
        "error": error
    }

def route_after_execution(state: DatabaseDataState) -> str:
    error = state.get("error")
    retry_count = state.get("retry_count", 0)
    if error and retry_count < 2:
        return "qury_connector"
    return "reporting"

def live_schema_fetcher_node(state: DatabaseDataState):
    """Database se automatically live schema fetch karta hai bina user ke diye."""
    existing_schema = state.get("schema_info", "")
    target_engine = state.get("db_type", "sqlite")
    if not existing_schema or existing_schema == "auto":
        live_metadata = dynamic_schema_extractor(target_engine)
        return {"schema_info": live_metadata}
        
    return {"schema_info": existing_schema}

def route_after_safety(state: DatabaseDataState) -> str:
    """Unsafe queries ko execution se rok kar seedhe reporting par bhejta hai."""
    if state.get("safety_status") == "unsafe":
        return "reporting"
    return "executor"

def route_after_execution(state: DatabaseDataState) -> str:
    """Execution error aane par corrector ko bhejta hai (max 2 retries)."""
    error = state.get("error")
    retry_count = state.get("retry_count", 0)

    if error and retry_count < 2:
        return "qury_connector"
    return "reporting"

tool_node=ToolNode(DATABASE_DATA_TOOLS)
workflow = StateGraph(DatabaseDataState)
workflow.add_node("live_schema_fetcher", live_schema_fetcher_node)
workflow.add_node("inspector", Inspector)
workflow.add_node("query_generator", Query_Migration)
workflow.add_node("safety_validator", safety_validator_node)
workflow.add_node("executor", database_executor_node)
workflow.add_node("qury_connector", qury_connector)
workflow.add_node("reporting", reporting)
workflow.add_edge(START, "live_schema_fetcher")
workflow.add_edge("live_schema_fetcher", "inspector")
workflow.add_edge("inspector", "query_generator")
workflow.add_edge("query_generator", "safety_validator")

workflow.add_conditional_edges(
    "safety_validator",
    route_after_safety,
    {
        "executor": "executor",
        "reporting": "reporting"
    }
)
workflow.add_conditional_edges(
    "executor",
    route_after_execution,
    {
        "qury_connector": "qury_connector",
        "reporting": "reporting"
    }
)
workflow.add_edge("qury_connector", "query_generator") 
workflow.add_edge("reporting", END)
database_data_agent = workflow.compile()


def run_database_task(
    task: str,
    db_type: str = "sqlite"
) -> Dict[str, Any]:
    """
    Run an end-to-end database task through the Database & SQL Agent.
    Generates SQL, validates safety, and executes queries 100% locally.
    """
    init_state = {
        "task": task,
        "schema_info": "auto",
        "db_type": db_type,
        "generated_query": "",
        "query_type": "read",
        "safety_status": "safe",
        "safety_notes": "",
        "execution_result": None,
        "error": None,
        "retry_count": 0,
        "final_output": "",
    }
    try:
        final_state = database_data_agent.invoke(init_state)
        query = final_state.get("generated_query", "")
        out = final_state.get("final_output") or ""
        if not out and query:
            out = f"### Database Query Plan\n\n```sql\n{query}\n```\n\n**Status:** {final_state.get('safety_status', 'Validated')}\n**Notes:** {final_state.get('safety_notes', 'Safe execution')}"
        return {
            "status": "success",
            "agent": "Database & SQL Analyst",
            "reply": out or "Database operation complete.",
            "query": query,
            "result": final_state.get("execution_result"),
        }
    except Exception as e:
        # Fallback to local SQL tool
        try:
            from tools.database_data import sql_generator
            sql_res = sql_generator.invoke({"natural_query": task, "dialect": db_type})
            return {
                "status": "success",
                "agent": "Database & SQL Analyst",
                "reply": f"Generated SQL Query:\n```sql\n{sql_res}\n```",
                "query": sql_res,
            }
        except Exception as fb_err:
            return {
                "status": "error",
                "agent": "Database & SQL Analyst",
                "reply": f"Database execution error: {e} ({fb_err})",
                "error": str(e),
            }


if __name__ == "__main__":
    prod_test_input = {
        "task": "Active users ka count aur unke roles list karo",
        "schema_info": "auto",           # Live schema fetcher auto-detect karega
        "db_type": "sqlite",             # Ya 'postgres', 'mysql', 'mongodb'
        "generated_query": "",
        "query_type": "read",
        "safety_status": "safe",
        "safety_notes": "",
        "execution_result": None,
        "error": None,
        "retry_count": 0,
        "final_output": ""
    }

    for step in database_data_agent.stream(prod_test_input):
        for node_name, state_update in step.items():
            print(f"\n[NODE COMPLETED: {node_name.upper()}]")
            if "schema_info" in state_update:
                print(f"-> Fetched Schema:\n{state_update['schema_info'][:150]}...")
            if "generated_query" in state_update:
                print(f"-> Generated Query: {state_update['generated_query']}")
            if "safety_status" in state_update and state_update["safety_status"] == "unsafe":
                print(f"-> Security Alert: Query Blocked!")
            if "error" in state_update and state_update["error"]:
                print(f"-> Traceback/Error: {state_update['error']}")
    print("\nFinished Graph Execution.")
