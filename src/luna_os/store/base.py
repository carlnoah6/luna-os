"""Abstract storage backend interface for tasks, plans, steps, and events."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from luna_os.types import Event, Plan, Step, Task


class StorageBackend(ABC):
    """Abstract interface for task/plan/step/event CRUD operations."""

    # -- Tasks -----------------------------------------------------------------

    @abstractmethod
    def add_task(
        self,
        task_id: str,
        description: str,
        source_chat: str | None = None,
        priority: str = "normal",
        depends_on: list[str] | None = None,
    ) -> Task:
        """Create a new task and return it."""

    @abstractmethod
    def get_task(self, task_id: str) -> Task | None:
        """Retrieve a task by ID, or *None* if not found."""

    @abstractmethod
    def list_tasks(self, status: str | None = None) -> list[Task]:
        """List tasks, optionally filtered by status."""

    @abstractmethod
    def update_task(self, task_id: str, **fields: Any) -> None:
        """Update arbitrary fields on a task."""

    @abstractmethod
    def start_task(self, task_id: str, session_key: str = "") -> None:
        """Transition a task to *running*."""

    @abstractmethod
    def complete_task(self, task_id: str, result: str = "") -> None:
        """Transition a task to *done*."""

    @abstractmethod
    def fail_task(self, task_id: str, error: str = "") -> None:
        """Transition a task to *failed*."""

    @abstractmethod
    def cancel_task(self, task_id: str) -> None:
        """Cancel a task."""

    @abstractmethod
    def ready_tasks(self) -> list[Task]:
        """Return queued tasks whose dependencies are all satisfied."""

    @abstractmethod
    def active_tasks(self) -> list[Task]:
        """Return currently running tasks."""

    @abstractmethod
    def cleanup_tasks(self, days: int = 7) -> int:
        """Remove old completed/cancelled/failed tasks. Return count deleted."""

    @abstractmethod
    def next_task_id(self) -> str:
        """Generate the next sequential task ID."""

    @abstractmethod
    def update_task_cost(
        self,
        task_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0,
    ) -> None:
        """Increment token/cost counters for a task."""

    @abstractmethod
    def find_duplicate(self, description: str) -> str | None:
        """Return the ID of a running/queued task with the same description, or *None*."""

    @abstractmethod
    def task_count_by_status(self) -> dict[str, int]:
        """Return ``{status: count}`` mapping."""

    @abstractmethod
    def boost_priority(self, task_id: str) -> None:
        """Boost a queued task's priority by one level."""

    @abstractmethod
    def cost_summary(self, days: int = 30) -> list[dict[str, Any]]:
        """Return daily cost summary for recent completed tasks."""

    # -- Plans -----------------------------------------------------------------

    @abstractmethod
    def create_plan(
        self,
        plan_id: str,
        chat_id: str,
        goal: str,
        steps: list[dict[str, Any]],
        status: str = "draft",
    ) -> Plan:
        """Create a new plan with its steps."""

    @abstractmethod
    def get_plan(self, plan_id: str) -> Plan | None:
        """Return a plan (with steps populated) or *None*."""

    @abstractmethod
    def get_plan_by_chat(
        self, chat_id: str, status_filter: str | None = None
    ) -> Plan | None:
        """Find the most recent plan for a chat, optionally filtered by status."""

    @abstractmethod
    def list_plans(self, status: str | None = None) -> list[Plan]:
        """List plans, optionally filtered by status."""

    @abstractmethod
    def update_plan_status(self, plan_id: str, status: str) -> None:
        """Change plan status."""

    @abstractmethod
    def cancel_plan(self, plan_id: str) -> None:
        """Cancel a plan and its pending steps."""

    @abstractmethod
    def next_plan_id(self) -> str:
        """Generate the next sequential plan ID."""

    # -- Steps -----------------------------------------------------------------

    @abstractmethod
    def get_step(self, plan_id: str, step_num: int) -> Step | None:
        """Return a single step, or *None*."""

    @abstractmethod
    def update_step(self, plan_id: str, step_num: int, **fields: Any) -> None:
        """Update arbitrary fields on a step."""

    @abstractmethod
    def start_step(self, plan_id: str, step_num: int, task_id: str) -> None:
        """Mark a step as running and associate it with a task."""

    @abstractmethod
    def complete_step(self, plan_id: str, step_num: int, result: str) -> None:
        """Mark a step as done."""

    @abstractmethod
    def fail_step(self, plan_id: str, step_num: int, error: str) -> None:
        """Mark a step as failed."""

    @abstractmethod
    def ready_steps(self, plan_id: str) -> list[Step]:
        """Return pending steps whose dependencies are all done."""

    @abstractmethod
    def next_pending_step(self, plan_id: str) -> Step | None:
        """Return the first ready step (convenience wrapper)."""

    @abstractmethod
    def delete_pending_steps(self, plan_id: str) -> None:
        """Delete all pending steps for a plan (used during replan)."""

    @abstractmethod
    def insert_step(
        self,
        plan_id: str,
        step_num: int,
        title: str,
        prompt: str,
        depends_on: list[int] | None = None,
    ) -> None:
        """Insert a single pending step."""

    # -- Events ----------------------------------------------------------------

    @abstractmethod
    def emit_event(
        self,
        event_type: str,
        task_id: str | None = None,
        plan_id: str | None = None,
        step_num: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Write an event to the event queue."""

    @abstractmethod
    def poll_events(self, limit: int = 10) -> list[Event]:
        """Return up to *limit* unprocessed events (oldest first)."""

    @abstractmethod
    def ack_event(self, event_id: int) -> None:
        """Mark a single event as processed."""

    @abstractmethod
    def ack_events(self, event_ids: list[int]) -> None:
        """Mark multiple events as processed."""
