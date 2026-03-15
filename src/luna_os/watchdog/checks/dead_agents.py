"""Check for sub-agents that finished but didn't report back.

Fallback mechanism: if a task is 'running' but its subagent session
has ended, automatically mark it as failed (step_fail).
"""
from __future__ import annotations
import json
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from ..models import Alert, Severity

NAME = "dead_agents"
logger = logging.getLogger(__name__)

# Path to emit_event.py for auto-failing dead tasks
EMIT_EVENT_SCRIPT = "/home/ubuntu/.openclaw/workspace/scripts/emit_event.py"


def _auto_fail_task(task_id: str) -> bool:
    """Auto-fail a task whose subagent session has ended without reporting.

    This is the fallback/safety-net mechanism. It calls step_fail so the
    planner can advance (or surface the error) rather than stalling.

    Returns True if the fail was triggered successfully.
    """
    try:
        proc = subprocess.run(
            [
                "python3", EMIT_EVENT_SCRIPT,
                "step.failed",
                "--task-id", task_id,
                "--result", "subagent session ended without reporting",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0:
            logger.info(f"Auto-failed task {task_id} (dead agent fallback)")
            return True
        else:
            logger.warning(
                f"Auto-fail call returned rc={proc.returncode} for {task_id}: "
                f"{proc.stderr.strip()[:200]}"
            )
            return False
    except Exception as e:
        logger.warning(f"Auto-fail failed for {task_id}: {e}")
        return False


def check(db) -> list[Alert]:
    """Find tasks with running status but whose agent session is no longer active.

    For each dead task found:
    1. Automatically call step_fail (fallback mechanism).
    2. Generate an Alert for monitoring visibility.
    """
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

            # --- Fallback mechanism: auto-fail the task ---
            auto_failed = _auto_fail_task(task_id)
            status_note = "已自动标记为失败" if auto_failed else "自动标记失败（调用出错，请手动处理）"

            alerts.append(Alert(
                check_name=NAME,
                key=f"dead_agent:{task_id}",
                message=(
                    f"任务 {task_id} 状态为 running 但无活跃 session ({mins} 分钟): {desc} "
                    f"— {status_note}"
                ),
                severity=Severity.CRITICAL,
                context={"task_id": task_id, "chat_id": row.get("source_chat")},
            ))

    return alerts
