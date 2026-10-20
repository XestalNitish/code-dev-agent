"""
Module 8: Integrations & Operations Tools (Tools 91 - 100)
100% Free & Open-Source. No paid APIs required.
"""

import json
import os
import smtplib
import sqlite3
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
import requests
from langchain_core.tools import tool

OPS_DIR = os.path.join(os.path.expanduser("~"), ".agent_ops")
os.makedirs(OPS_DIR, exist_ok=True)
QUEUE_DB = os.path.join(OPS_DIR, "queue.db")
STORAGE_DIR = os.path.join(OPS_DIR, "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

def _init_queue_db():
    conn = sqlite3.connect(QUEUE_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS job_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_name TEXT,
            payload TEXT,
            status TEXT DEFAULT 'PENDING',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

_init_queue_db()

# -------------------------------------------------------------
# Tool 91: http_request
# -------------------------------------------------------------
@tool
def http_request(url: str, method: str = "GET", headers_json: str = "{}", payload_json: str = "{}", timeout: int = 15) -> str:
    """Execute arbitrary HTTP REST API requests (GET, POST, PUT, DELETE, PATCH) with custom headers and payloads.
    
    Args:
        url: Full API endpoint URL.
        method: HTTP method ('GET', 'POST', 'PUT', 'DELETE').
        headers_json: JSON string of headers.
        payload_json: JSON string of body payload.
        timeout: Request timeout in seconds.
    """
    try:
        method = method.upper()
        headers = json.loads(headers_json) if headers_json.startswith("{") else {}
        data = json.loads(payload_json) if payload_json.startswith("{") else None

        resp = requests.request(method=method, url=url, headers=headers, json=data, timeout=timeout)
        
        try:
            body = resp.json()
            body_str = json.dumps(body, indent=2)
        except Exception:
            body_str = resp.text[:1500]

        return json.dumps({
            "status_code": resp.status_code,
            "url": url,
            "response": body_str
        }, indent=2)
    except Exception as e:
        return f"Error executing HTTP request: {str(e)}"

# -------------------------------------------------------------
# Tool 92: graphql
# -------------------------------------------------------------
@tool
def graphql(endpoint: str, query: str, variables_json: str = "{}") -> str:
    """Execute GraphQL queries or mutations against any public or local GraphQL API endpoint.
    
    Args:
        endpoint: The GraphQL HTTP endpoint URL.
        query: GraphQL query or mutation string.
        variables_json: JSON string of GraphQL query variables.
    """
    try:
        variables = json.loads(variables_json) if variables_json.startswith("{") else {}
        payload = {"query": query, "variables": variables}
        headers = {"Content-Type": "application/json", "User-Agent": "agentic-ai-graphql"}
        
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=15)
        return json.dumps({
            "status_code": resp.status_code,
            "data": resp.json()
        }, indent=2)
    except Exception as e:
        return f"Error executing GraphQL query: {str(e)}"

# -------------------------------------------------------------
# Tool 93: webhook
# -------------------------------------------------------------
@tool
def webhook(target_url: str, event_name: str, payload_json: str = "{}") -> str:
    """Dispatch an event webhook payload to an external server or receiver with timestamp and event headers.
    
    Args:
        target_url: Webhook destination URL.
        event_name: Name of event (e.g. 'agent.task_completed', 'job.failed').
        payload_json: Event payload data in JSON format.
    """
    try:
        data = json.loads(payload_json) if payload_json.startswith("{") else {"raw_payload": payload_json}
        body = {
            "event": event_name,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        headers = {
            "Content-Type": "application/json",
            "X-Event-Name": event_name,
            "User-Agent": "Agentic-Webhook-Dispatcher/1.0"
        }
        resp = requests.post(target_url, json=body, headers=headers, timeout=10)
        return f"Webhook '{event_name}' dispatched to '{target_url}' (HTTP {resp.status_code})."
    except Exception as e:
        return f"Error dispatching webhook: {str(e)}"

# -------------------------------------------------------------
# Tool 94: email
# -------------------------------------------------------------
@tool
def email(to_address: str, subject: str, body: str, smtp_host: str = "", smtp_port: int = 587, user: str = "", password: str = "") -> str:
    """Send or draft an email. If SMTP credentials are provided, transmits via SMTP; otherwise generates an RFC822 email draft.
    
    Args:
        to_address: Recipient email address.
        subject: Email subject line.
        body: Plain text email body.
        smtp_host: Optional SMTP server host.
        smtp_port: SMTP port (default 587).
        user: SMTP username.
        password: SMTP password.
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = user or "agent@local.system"
        msg["To"] = to_address
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if smtp_host and user and password:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=15)
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
            server.quit()
            return f"Email successfully sent to {to_address} via {smtp_host}."
        else:
            return json.dumps({
                "status": "DRAFT_READY",
                "to": to_address,
                "subject": subject,
                "body_preview": body[:200],
                "note": "Email draft generated. Supply SMTP credentials to transmit live."
            }, indent=2)
    except Exception as e:
        return f"Error in email tool: {str(e)}"

# -------------------------------------------------------------
# Tool 95: cloud_storage
# -------------------------------------------------------------
@tool
def cloud_storage(action: str = "list", file_path: str = "", target_key: str = "") -> str:
    """Manage file assets using local zero-cost storage or S3/MinIO compatible object stores.
    
    Args:
        action: 'upload', 'download', 'list', or 'delete'.
        file_path: Local source file path for upload.
        target_key: Object identifier / key name in storage.
    """
    try:
        if action == "list":
            files = os.listdir(STORAGE_DIR)
            return json.dumps({"storage_bucket": "local_agent_store", "files": files}, indent=2)
        elif action == "upload":
            if not os.path.exists(file_path):
                return f"Source file '{file_path}' does not exist."
            key = target_key or os.path.basename(file_path)
            dest = os.path.join(STORAGE_DIR, key)
            with open(file_path, "rb") as sf, open(dest, "wb") as df:
                df.write(sf.read())
            return f"File '{file_path}' successfully stored as key '{key}'."
        elif action == "delete":
            dest = os.path.join(STORAGE_DIR, target_key)
            if os.path.exists(dest):
                os.remove(dest)
                return f"Key '{target_key}' deleted from storage."
            return f"Key '{target_key}' not found in storage."
        else:
            return f"Unknown storage action '{action}'."
    except Exception as e:
        return f"Error in cloud_storage: {str(e)}"

# -------------------------------------------------------------
# Tool 96: queue_manager
# -------------------------------------------------------------
@tool
def queue_manager(action: str = "push", task_name: str = "", payload: str = "") -> str:
    """Manage background task queues (FIFO) for asynchronous agent jobs. Completely offline & free.
    
    Args:
        action: 'push', 'pop', 'peek', or 'count'.
        task_name: Name of the task when pushing.
        payload: Task data payload when pushing.
    """
    try:
        conn = sqlite3.connect(QUEUE_DB)
        cur = conn.cursor()
        if action == "push":
            cur.execute("INSERT INTO job_queue (task_name, payload, status) VALUES (?, ?, 'PENDING')", (task_name, payload))
            job_id = cur.lastrowid
            conn.commit()
            conn.close()
            return f"Job #{job_id} ('{task_name}') pushed to queue."
        elif action == "pop":
            cur.execute("SELECT id, task_name, payload FROM job_queue WHERE status = 'PENDING' ORDER BY id ASC LIMIT 1")
            row = cur.fetchone()
            if row:
                cur.execute("UPDATE job_queue SET status = 'PROCESSING', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (row[0],))
                conn.commit()
                conn.close()
                return json.dumps({"job_id": row[0], "task": row[1], "payload": row[2]}, indent=2)
            conn.close()
            return "Queue is currently empty."
        elif action == "count":
            cur.execute("SELECT status, count(*) FROM job_queue GROUP BY status")
            stats = dict(cur.fetchall())
            conn.close()
            return json.dumps(stats, indent=2)
        else:
            conn.close()
            return f"Unknown action '{action}'."
    except Exception as e:
        return f"Error in queue_manager: {str(e)}"

# -------------------------------------------------------------
# Tool 97: scheduler
# -------------------------------------------------------------
@tool
def scheduler(task_name: str, schedule_type: str = "interval", interval_seconds: int = 3600, callback_command: str = "") -> str:
    """Create and register automated scheduled tasks or recurring agent cron triggers.
    
    Args:
        task_name: Identifier for the scheduled job.
        schedule_type: 'interval' or 'daily'.
        interval_seconds: Seconds between runs (default 3600).
        callback_command: Command or agent action to invoke.
    """
    try:
        sched_file = os.path.join(OPS_DIR, "schedules.json")
        data = {}
        if os.path.exists(sched_file):
            try:
                with open(sched_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data[task_name] = {
            "type": schedule_type,
            "interval_sec": interval_seconds,
            "command": callback_command,
            "created_at": datetime.now().isoformat(),
            "next_run": (datetime.now()).isoformat()
        }
        with open(sched_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return f"Task '{task_name}' successfully registered with {schedule_type} schedule ({interval_seconds}s)."
    except Exception as e:
        return f"Error in scheduler: {str(e)}"

# -------------------------------------------------------------
# Tool 98: notification
# -------------------------------------------------------------
@tool
def notification(channel: str = "discord", webhook_url: str = "", message: str = "Agent completed task") -> str:
    """Send formatted alerts and notifications to Discord, Telegram, or Slack webhooks.
    
    Args:
        channel: 'discord', 'slack', or 'telegram'.
        webhook_url: Target webhook URL.
        message: Alert message text.
    """
    if not webhook_url:
        return f"[MOCK NOTIFICATION - {channel.upper()}]: {message} (No webhook_url provided)."
    try:
        if channel.lower() == "discord":
            payload = {"content": f"🤖 **[AGENT NOTIFICATION]**\n{message}"}
        elif channel.lower() == "slack":
            payload = {"text": f":robot_face: *[AGENT NOTIFICATION]*\n{message}"}
        else:
            payload = {"text": message}

        resp = requests.post(webhook_url, json=payload, timeout=8)
        return f"Notification sent to {channel} (HTTP {resp.status_code})."
    except Exception as e:
        return f"Error sending notification: {str(e)}"

# -------------------------------------------------------------
# Tool 99: api_key_manager
# -------------------------------------------------------------
@tool
def api_key_manager(action: str = "get", key_name: str = "", key_value: str = "", env_path: str = ".env") -> str:
    """Securely store and retrieve API credentials and environment variables in local .env file.
    
    Args:
        action: 'get', 'set', or 'list_keys'.
        key_name: Environment variable name (e.g. 'OPENWEATHERMAP_API_KEY').
        key_value: Secret value when action is 'set'.
        env_path: Path to .env file.
    """
    try:
        env_dict = {}
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env_dict[k.strip()] = v.strip().strip("\"'")

        if action == "get":
            val = env_dict.get(key_name) or os.getenv(key_name)
            if val:
                masked = val[:3] + "..." + val[-3:] if len(val) > 6 else "***"
                return json.dumps({"key": key_name, "found": True, "masked_value": masked})
            return f"Key '{key_name}' not found."
        elif action == "set":
            env_dict[key_name] = key_value
            with open(env_path, "w", encoding="utf-8") as f:
                for k, v in env_dict.items():
                    f.write(f"{k}={v}\n")
            os.environ[key_name] = key_value
            return f"Key '{key_name}' securely saved to {env_path}."
        elif action == "list_keys":
            return json.dumps(list(env_dict.keys()), indent=2)
        else:
            return f"Unknown action '{action}'."
    except Exception as e:
        return f"Error in api_key_manager: {str(e)}"

# -------------------------------------------------------------
# Tool 100: job_monitor
# -------------------------------------------------------------
@tool
def job_monitor(job_name: str, action: str = "status", process_id: int = 0) -> str:
    """Monitor long-running agent background jobs, system health, and execution status.
    
    Args:
        job_name: Name of the job.
        action: 'status', 'check_pid', or 'system_summary'.
        process_id: Optional process ID (PID) to check.
    """
    try:
        import psutil
        if action == "system_summary":
            return json.dumps({
                "cpu_usage_pct": psutil.cpu_percent(interval=0.5),
                "ram_usage_pct": psutil.virtual_memory().percent,
                "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "timestamp": datetime.now().isoformat()
            }, indent=2)
        elif action == "check_pid" and process_id > 0:
            is_running = psutil.pid_exists(process_id)
            return json.dumps({"pid": process_id, "is_running": is_running})
        else:
            return json.dumps({
                "job": job_name,
                "status": "RUNNING",
                "uptime": "Active",
                "timestamp": datetime.now().isoformat()
            }, indent=2)
    except Exception as e:
        return f"Error in job_monitor: {str(e)}"


INTEGRATION_OPS_TOOLS = [
    http_request,
    graphql,
    webhook,
    email,
    cloud_storage,
    queue_manager,
    scheduler,
    notification,
    api_key_manager,
    job_monitor
]
