"""Check waiting tasks with Lark todo integration.

Polls the task store for tasks in 'waiting' state with wait_type='todo'.
If the corresponding Lark todo is completed, auto-responds the task.

Intended to run via cron every 2 minutes:
    luna-os check-waiting-todos
"""

from __future__ import annotations

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def check_waiting_todos() -> dict[str, int]:
    """Check all waiting todo tasks and respond completed ones.

    Returns dict with 'checked' and 'responded' counts.
    """
    try:
        from lark_toolkit import LarkClient
        from lark_toolkit.todo import get_task as get_lark_task
    except ImportError:
        logger.warning("lark-toolkit not installed, skipping todo check")
        return {"checked": 0, "responded": 0, "error": "lark-toolkit not installed"}

    from luna_os.store.postgres import PostgresBackend

    store = PostgresBackend()
    waiting = store.waiting_tasks()
    if not waiting:
        return {"checked": 0, "responded": 0}

    client = LarkClient()
    checked = 0
    responded = 0

    for task in waiting:
        if task.get("wait_type") != "todo":
            continue
        todo_id = task.get("wait_todo_id")
        if not todo_id:
            continue

        checked += 1
        try:
            todo = get_lark_task(client, todo_id, token=client.get_user_token())
            task_data = todo.get("task", todo)
            completed_at = task_data.get("completed_at")

            if completed_at and completed_at != "0":
                task_id = task["id"]
                logger.info("Task %s todo completed, responding...", task_id)
                result = subprocess.run(
                    [
                        sys.executable, "-m", "luna_os.cli",
                        "task", "respond", task_id, "用户已完成待办",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    responded += 1
                    logger.info("Task %s responded OK", task_id)
                else:
                    logger.error(
                        "Task %s respond failed: %s",
                        task_id,
                        result.stderr[:200],
                    )
        except Exception:
            logger.exception("Task %s check failed", task.get("id", "?"))

    if checked:
        logger.info("Checked %d todos, responded %d", checked, responded)

    return {"checked": checked, "responded": responded}
