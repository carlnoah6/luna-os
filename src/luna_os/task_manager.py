"""Task management: add, start, complete, fail, cancel, list.

Orchestrates task lifecycle using pluggable StorageBackend and
optional NotificationProvider and AgentRunner.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from luna_os.agents.base import AgentRunner
from luna_os.notifications.base import NotificationProvider
from luna_os.store.base import StorageBackend
from luna_os.types import Task

logger = logging.getLogger(__name__)

MAX_CONCURRENT = 6
HEALTH_CHECK_TIMEOUT_MINUTES = 45
CLEANUP_DAYS = 7

# Routine task patterns — skip chat creation
ROUTINE_TASK_PATTERNS = [
    "periodic",
    "dailyReport",
    "morningGreeting",
    "wikiSync",
]


def _serialize(obj: Any) -> Any:
    """JSON serializer for datetime and Decimal."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _elapsed_minutes(task: Task) -> float | None:
    """Compute minutes since a task started."""
    if not task.started_at:
        return None
    started = task.started_at
    if isinstance(started, str):
        started = datetime.fromisoformat(started)
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    return (now - started).total_seconds() / 60


def _is_routine_task(description: str) -> bool:
    desc_lower = description.lower()
    return any(p.lower() in desc_lower for p in ROUTINE_TASK_PATTERNS)


def _detect_cycle(store: StorageBackend, new_task_id: str, new_deps: list[str]) -> bool:
    """Check if adding *new_task_id* with *new_deps* would create a dependency cycle."""
    all_tasks = store.list_tasks()
    adj: dict[str, list[str]] = {}
    for t in all_tasks:
        adj[t.id] = list(t.depends_on or [])
    adj[new_task_id] = list(new_deps)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {node: WHITE for node in adj}

    def dfs(u: str) -> bool:
        color[u] = GRAY
        for v in adj.get(u, []):
            if v not in color:
                color[v] = WHITE
            if color.get(v) == GRAY:
                return True
            if color.get(v) == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(color.get(node) == WHITE and dfs(node) for node in list(adj.keys()))


class TaskManager:
    """High-level task lifecycle management.

    Parameters
    ----------
    store:
        Storage backend for persistence.
    notifications:
        Optional notification provider for sending messages.
    agent_runner:
        Optional agent runner for spawning/checking agent sessions.
    max_concurrent:
        Maximum number of concurrently running tasks.
    """

    def __init__(
        self,
        store: StorageBackend,
        notifications: NotificationProvider | None = None,
        agent_runner: AgentRunner | None = None,
        max_concurrent: int = MAX_CONCURRENT,
    ) -> None:
        self.store = store
        self.notifications = notifications
        self.agent_runner = agent_runner
        self.max_concurrent = max_concurrent

    # -- Notifications helper --------------------------------------------------

    def _notify(self, chat_id: str, text: str) -> None:
        """Send a notification if a provider is configured."""
        if self.notifications and chat_id:
            try:
                self.notifications.send_message(chat_id, text)
            except Exception as exc:
                logger.warning("Notification failed: %s", exc)

    # -- Commands --------------------------------------------------------------

    def add(
        self,
        description: str,
        source_chat: str | None = None,
        depends_on: list[str] | None = None,
        priority: str = "normal",
        create_chat: bool = True,
    ) -> Task:
        """Create a new task.

        Raises ``ValueError`` on duplicate or dependency cycle.
        """
        if not description:
            raise ValueError("description is required")

        # Duplicate check
        dup = self.store.find_duplicate(description)
        if dup:
            raise ValueError(f"Duplicate task: {dup}")

        task_id = self.store.next_task_id()

        # Cycle check
        if depends_on and _detect_cycle(self.store, task_id, depends_on):
            raise ValueError("Dependency cycle detected")

        task = self.store.add_task(
            task_id, description, source_chat=source_chat,
            priority=priority, depends_on=depends_on,
        )

        # Create chat if needed
        if (
            create_chat
            and not _is_routine_task(description)
            and self.notifications
        ):
            running = self.store.active_tasks()
            if len(running) < self.max_concurrent:
                try:
                    chat_id = self.notifications.create_chat(
                        f"Task {task_id} {description[:30]}",
                        f"Luna OS task: {description}",
                        [],  # members managed by caller
                    )
                    if chat_id:
                        self.store.update_task(task_id, task_chat_id=chat_id)
                        task = self.store.get_task(task_id) or task
                except Exception as exc:
                    logger.warning("Chat creation failed: %s", exc)

        return task

    def show(self, task_id: str) -> dict[str, Any]:
        """Return task details as a dict, with elapsed_min if running."""
        task = self.store.get_task(task_id)
        if not task:
            raise KeyError(f"Task {task_id} not found")
        d = task.to_dict()
        elapsed = _elapsed_minutes(task)
        if elapsed is not None:
            d["elapsed_min"] = round(elapsed, 1)
        return d

    def start(self, task_id: str, session_key: str = "") -> Task:
        """Mark a task as running."""
        running = self.store.active_tasks()
        if len(running) >= self.max_concurrent:
            raise RuntimeError(
                f"Queue full: {len(running)}/{self.max_concurrent}"
            )
        self.store.start_task(task_id, session_key)
        return self.store.get_task(task_id)  # type: ignore[return-value]

    def complete(
        self,
        task_id: str,
        result: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0,
    ) -> dict[str, Any]:
        """Mark a task as done and emit a step.done event."""
        self.store.complete_task(task_id, result)
        if input_tokens or output_tokens or cost_usd:
            self.store.update_task_cost(task_id, input_tokens, output_tokens, cost_usd)

        # Emit step.done so planner can auto-advance
        self.store.emit_event("step.done", task_id=task_id, payload={"result": result})

        # Notify source chat
        task = self.store.get_task(task_id)
        if task and task.source_chat:
            self._notify(task.source_chat, f"Task {task_id} completed: {result}")
        if task and task.task_chat_id:
            self._notify(task.task_chat_id, f"Task {task_id} completed\n\n{result}")

        # Check for unblocked tasks
        ready = self.store.ready_tasks()
        out: dict[str, Any] = {"id": task_id, "status": "done"}
        if ready:
            out["unblocked"] = [t.id for t in ready]
        return out

    def fail(self, task_id: str, error: str = "") -> dict[str, Any]:
        """Mark a task as failed."""
        self.store.fail_task(task_id, error)
        task = self.store.get_task(task_id)
        if task and task.source_chat:
            self._notify(task.source_chat, f"Task {task_id} failed: {error}")
        return {"id": task_id, "status": "failed"}

    def cancel(self, task_id: str) -> dict[str, Any]:
        """Cancel a task."""
        self.store.cancel_task(task_id)
        return {"id": task_id, "status": "cancelled"}

    def list_tasks(self, status_filter: str | None = None) -> list[dict[str, Any]]:
        """List tasks with optional status filter."""
        tasks = self.store.list_tasks(status=status_filter)
        result = []
        for t in tasks:
            d = t.to_dict()
            elapsed = _elapsed_minutes(t)
            if elapsed is not None:
                d["elapsed_min"] = round(elapsed, 1)
            result.append(d)
        return result

    def ready(self) -> list[dict[str, Any]]:
        """Return tasks ready to spawn."""
        return [t.to_dict() for t in self.store.ready_tasks()]

    def active(self) -> list[dict[str, Any]]:
        """Return running tasks."""
        tasks = self.store.active_tasks()
        result = []
        for t in tasks:
            d = t.to_dict()
            elapsed = _elapsed_minutes(t)
            if elapsed is not None:
                d["elapsed_min"] = round(elapsed, 1)
            result.append(d)
        return result

    def status(self) -> dict[str, Any]:
        """Quick overview of task counts."""
        tasks = self.store.list_tasks()
        counts: dict[str, int] = {
            "queued": 0, "running": 0, "done": 0, "failed": 0, "cancelled": 0,
        }
        for t in tasks:
            s = t.status.value if hasattr(t.status, "value") else str(t.status)
            counts[s] = counts.get(s, 0) + 1
        ready = self.store.ready_tasks()
        return {
            "total": len(tasks),
            "counts": counts,
            "ready_to_spawn": len(ready),
            "max_concurrent": self.max_concurrent,
        }

    def set_session(self, task_id: str, session_key: str) -> dict[str, Any]:
        """Set session key on a task."""
        self.store.update_task(task_id, session_key=session_key)
        return {"id": task_id, "session_key": session_key}

    def cleanup(self, days: int = CLEANUP_DAYS) -> dict[str, Any]:
        """Remove old completed/cancelled/failed tasks."""
        count = self.store.cleanup_tasks(days)
        return {"cleaned": count, "days": days}

    def health_check(self) -> dict[str, Any]:
        """Check for stuck tasks and auto-fail those exceeding timeout."""
        running = self.store.active_tasks()
        datetime.now(UTC)
        result: dict[str, Any] = {"stuck_failed": [], "active": []}

        for t in running:
            elapsed = _elapsed_minutes(t)
            if elapsed is None:
                continue
            if elapsed > HEALTH_CHECK_TIMEOUT_MINUTES:
                self.store.fail_task(
                    t.id,
                    f"Auto-failed: stuck for {elapsed:.0f} minutes "
                    f"(timeout={HEALTH_CHECK_TIMEOUT_MINUTES}min)",
                )
                result["stuck_failed"].append({
                    "id": t.id,
                    "description": t.description[:80],
                    "elapsed_min": round(elapsed, 1),
                })
            else:
                result["active"].append({
                    "id": t.id,
                    "elapsed_min": round(elapsed, 1),
                })

        return result

    def wait(
        self,
        task_id: str,
        wait_type: str,
        prompt: str,
        options: list[str] | None = None,
    ) -> dict[str, Any]:
        """Put a task into waiting state."""
        task = self.store.get_task(task_id)
        if not task:
            raise KeyError(f"Task {task_id} not found")

        update: dict[str, Any] = {
            "status": "waiting",
            "wait_type": wait_type,
            "wait_prompt": prompt,
            "waited_at": datetime.now(UTC),
        }
        if options:
            update["wait_options"] = json.dumps(options)
        self.store.update_task(task_id, **update)
        self.store.emit_event(
            "task.waiting",
            task_id=task_id,
            payload={"wait_type": wait_type, "prompt": prompt},
        )

        if task.source_chat:
            self._notify(
                task.source_chat,
                f"Task {task_id} waiting for input: {prompt}",
            )

        return {"id": task_id, "status": "waiting", "wait_type": wait_type}

    def respond(self, task_id: str, response: str) -> dict[str, Any]:
        """Resume a waiting task with user response."""
        task = self.store.get_task(task_id)
        if not task:
            raise KeyError(f"Task {task_id} not found")
        if task.status.value != "waiting":
            raise ValueError(f"Task {task_id} is {task.status.value}, not waiting")

        waited_minutes = 0.0
        if task.waited_at:
            waited_at = task.waited_at
            if isinstance(waited_at, str):
                waited_at = datetime.fromisoformat(waited_at)
            if waited_at.tzinfo is None:
                waited_at = waited_at.replace(tzinfo=UTC)
            waited_minutes = (
                datetime.now(UTC) - waited_at
            ).total_seconds() / 60

        self.store.update_task(
            task_id,
            status="running",
            wait_response=response,
            started_at=datetime.now(UTC),
        )
        self.store.emit_event(
            "task.responded",
            task_id=task_id,
            payload={"response": response, "waited_minutes": round(waited_minutes, 1)},
        )

        # Re-spawn agent if runner is configured
        if self.agent_runner:
            prompt = (
                f"Previously executing task {task_id}: {task.description}\n"
                f"User input was requested: {task.wait_prompt}\n"
                f"User responded: {response}\n"
                "Please continue executing the task."
            )
            try:
                self.agent_runner.spawn(task_id, prompt)
            except Exception as exc:
                logger.warning("Agent respawn failed: %s", exc)

        return {
            "id": task_id,
            "status": "running",
            "response": response,
            "waited_minutes": round(waited_minutes, 1),
        }

    def cost_report(self, days: int = 30) -> list[dict[str, Any]]:
        """Get cost summary for recent tasks."""
        return self.store.cost_summary(days=days)
