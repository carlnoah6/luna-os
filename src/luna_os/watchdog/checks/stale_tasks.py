"""Check for tasks stuck in 'running' state too long."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from ..models import Alert, Severity

NAME = "stale_tasks"
STALE_MINUTES = 15


def check(db) -> list[Alert]:
    """Find tasks that have been running for too long without updates."""
    alerts = []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_MINUTES)

    rows = db.execute(
        """
        SELECT id, description, status, updated_at, source_chat
        FROM tasks
        WHERE status = 'running'
          AND updated_at < %s
        ORDER BY updated_at ASC
        """,
        (cutoff,)
    )

    for row in rows:
        task_id = row["id"]
        mins = int((datetime.now(timezone.utc) - row["updated_at"]).total_seconds() / 60)
        desc = (row["description"] or "")[:60]
        alerts.append(Alert(
            check_name=NAME,
            key=f"stale_task:{task_id}",
            message=f"任务 {task_id} 已运行 {mins} 分钟无更新: {desc}",
            severity=Severity.WARN,
            context={"task_id": task_id, "chat_id": row.get("source_chat")},
        ))

    return alerts
