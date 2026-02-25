"""Check for sub-agents that finished but didn't report back."""
from __future__ import annotations
import json
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from ..models import Alert, Severity

NAME = "dead_agents"
logger = logging.getLogger(__name__)


def check(db) -> list[Alert]:
    """Find tasks with running status but whose agent session is no longer active."""
    alerts = []

    # Get running tasks that have a session/agent associated
    rows = db.execute(
        """
        SELECT id, description, status, updated_at, source_chat
        FROM tasks
        WHERE status = 'running'
          AND updated_at < %s
        """,
        (datetime.now(timezone.utc) - timedelta(minutes=3),)
    )

    # Get active sessions
    active_sessions = set()
    try:
        proc = subprocess.run(
            ["openclaw", "sessions", "--active", "10", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            sessions = data.get("sessions", data) if isinstance(data, dict) else data
            for s in sessions:
                key = s.get("key", s.get("sessionKey", ""))
                active_sessions.add(key)
    except Exception as e:
        logger.warning(f"Failed to list sessions: {e}")
        return []  # Can't check without session data

    for row in rows:
        task_id = row["id"]
        # Check if there's an active session for this task
        has_session = any(task_id in s for s in active_sessions)
        if not has_session:
            mins = int((datetime.now(timezone.utc) - row["updated_at"]).total_seconds() / 60)
            desc = (row["description"] or "")[:60]
            alerts.append(Alert(
                check_name=NAME,
                key=f"dead_agent:{task_id}",
                message=f"任务 {task_id} 状态为 running 但无活跃 session ({mins} 分钟): {desc}",
                severity=Severity.CRITICAL,
                context={"task_id": task_id, "chat_id": row.get("source_chat")},
            ))

    return alerts
