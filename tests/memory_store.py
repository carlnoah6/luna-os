"""In-memory StorageBackend for testing — no database required."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from luna_os.store.base import StorageBackend
from luna_os.types import (
    PRIORITY_MAP,
    PRIORITY_NAMES,
    Event,
    Plan,
    PlanStatus,
    Step,
    StepStatus,
    Task,
    TaskStatus,
    now_utc,
)

SGT = timezone(timedelta(hours=8))


class MemoryBackend(StorageBackend):
    """Pure in-memory StorageBackend for unit tests."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._plans: dict[str, Plan] = {}
        self._events: list[Event] = []
        self._event_seq = 0
        self._task_seq: dict[str, int] = {}
        self._plan_seq: dict[str, int] = {}

    # -- Tasks -----------------------------------------------------------------

    def add_task(
        self,
        task_id: str,
        description: str,
        source_chat: str | None = None,
        priority: str = "normal",
        depends_on: list[str] | None = None,
    ) -> Task:
        now = now_utc()
        pv = PRIORITY_MAP.get(priority, 3)
        task = Task(
            id=task_id,
            description=description,
            status=TaskStatus.QUEUED,
            source_chat=source_chat,
            priority=priority,
            priority_value=pv,
            depends_on=depends_on or [],
            created_at=now,
            updated_at=now,
        )
        self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(self, status: str | None = None) -> list[Task]:
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status.value == status]
        tasks.sort(
            key=lambda t: (
                t.priority_value,
                t.created_at or datetime.min.replace(tzinfo=UTC),
            )
        )
        return tasks

    def update_task(self, task_id: str, **fields: Any) -> None:
        task = self._tasks.get(task_id)
        if not task:
            return
        for k, v in fields.items():
            if hasattr(task, k):
                if k == "status" and isinstance(v, str):
                    v = TaskStatus(v)
                setattr(task, k, v)
        task.updated_at = now_utc()

    def start_task(self, task_id: str, session_key: str = "") -> None:
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.RUNNING
            task.session_key = session_key
            task.started_at = now_utc()
            task.updated_at = now_utc()

    def complete_task(self, task_id: str, result: str = "") -> None:
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.DONE
            task.result = result
            task.completed_at = now_utc()
            task.updated_at = now_utc()

    def fail_task(self, task_id: str, error: str = "") -> None:
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.result = error
            task.completed_at = now_utc()
            task.updated_at = now_utc()

    def cancel_task(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.CANCELLED
            task.updated_at = now_utc()

    def ready_tasks(self) -> list[Task]:
        queued = [t for t in self._tasks.values() if t.status == TaskStatus.QUEUED]
        done_ids = {t.id for t in self._tasks.values() if t.status == TaskStatus.DONE}
        return [t for t in queued if all(d in done_ids for d in (t.depends_on or []))]

    def active_tasks(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]

    def cleanup_tasks(self, days: int = 7) -> int:
        cutoff = now_utc() - timedelta(days=days)
        to_remove = [
            tid
            for tid, t in self._tasks.items()
            if t.status.value in ("done", "cancelled", "failed")
            and t.updated_at
            and t.updated_at < cutoff
        ]
        for tid in to_remove:
            del self._tasks[tid]
        return len(to_remove)

    def next_task_id(self) -> str:
        today = datetime.now(SGT)
        prefix = f"tid-{today.strftime('%m%d')}-"
        max_seq = 0
        for tid in self._tasks:
            if tid.startswith(prefix):
                try:
                    seq = int(tid.split("-")[-1])
                    if seq > max_seq:
                        max_seq = seq
                except (ValueError, IndexError):
                    pass
        return f"{prefix}{max_seq + 1}"

    def update_task_cost(
        self,
        task_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0,
    ) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.input_tokens += input_tokens
            task.output_tokens += output_tokens
            task.cost_usd += cost_usd

    def find_duplicate(self, description: str) -> str | None:
        desc_lower = description.strip().lower()
        for t in self._tasks.values():
            if (
                t.status.value in ("queued", "running")
                and t.description.strip().lower() == desc_lower
            ):
                return t.id
        return None

    def task_count_by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self._tasks.values():
            s = t.status.value
            counts[s] = counts.get(s, 0) + 1
        return counts

    def boost_priority(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.QUEUED:
            return
        if task.priority_value > 1:
            task.priority_value -= 1
            task.priority = PRIORITY_NAMES.get(task.priority_value, "normal")
            task.priority_boosted = True

    def cost_summary(self, days: int = 30) -> list[dict[str, Any]]:
        return []

    # -- Plans -----------------------------------------------------------------

    def create_plan(
        self,
        plan_id: str,
        chat_id: str,
        goal: str,
        steps: list[dict[str, Any]],
        status: str = "draft",
    ) -> Plan:
        now = now_utc()
        plan_steps: list[Step] = []
        for i, s in enumerate(steps, 1):
            raw_deps = s.get("depends_on") or []
            deps = [d + 1 for d in raw_deps]
            plan_steps.append(
                Step(
                    plan_id=plan_id,
                    step_num=i,
                    title=s.get("title", ""),
                    prompt=s.get("prompt", ""),
                    status=StepStatus.PENDING,
                    depends_on=deps,
                )
            )
        plan = Plan(
            id=plan_id,
            chat_id=chat_id,
            goal=goal,
            status=PlanStatus(status),
            steps=plan_steps,
            created_at=now,
            updated_at=now,
        )
        self._plans[plan_id] = plan
        return plan

    def get_plan(self, plan_id: str) -> Plan | None:
        return self._plans.get(plan_id)

    def get_plan_by_chat(self, chat_id: str, status_filter: str | None = None) -> Plan | None:
        candidates = []
        for p in self._plans.values():
            if p.chat_id == chat_id or p.id == chat_id:
                if status_filter:
                    if p.status.value == status_filter:
                        candidates.append(p)
                else:
                    candidates.append(p)
        if not candidates:
            return None
        candidates.sort(
            key=lambda p: p.created_at or datetime.min.replace(tzinfo=UTC), reverse=True
        )
        return candidates[0]

    def list_plans(self, status: str | None = None) -> list[Plan]:
        plans = list(self._plans.values())
        if status:
            plans = [p for p in plans if p.status.value == status]
        plans.sort(key=lambda p: p.created_at or datetime.min.replace(tzinfo=UTC), reverse=True)
        return plans

    def update_plan_status(self, plan_id: str, status: str) -> None:
        plan = self._plans.get(plan_id)
        if plan:
            plan.status = PlanStatus(status)
            plan.updated_at = now_utc()

    def cancel_plan(self, plan_id: str) -> None:
        plan = self._plans.get(plan_id)
        if plan:
            plan.status = PlanStatus.CANCELLED
            plan.updated_at = now_utc()
            for s in plan.steps:
                if s.status == StepStatus.PENDING:
                    s.status = StepStatus.CANCELLED

    def next_plan_id(self) -> str:
        today = datetime.now(SGT)
        prefix = f"plan-{today.strftime('%m%d')}-"
        max_seq = 0
        for pid in self._plans:
            if pid.startswith(prefix):
                try:
                    seq = int(pid.split("-")[-1])
                    if seq > max_seq:
                        max_seq = seq
                except (ValueError, IndexError):
                    pass
        return f"{prefix}{max_seq + 1}"

    # -- Steps -----------------------------------------------------------------

    def get_step(self, plan_id: str, step_num: int) -> Step | None:
        plan = self._plans.get(plan_id)
        if not plan:
            return None
        for s in plan.steps:
            if s.step_num == step_num:
                return s
        return None

    def update_step(self, plan_id: str, step_num: int, **fields: Any) -> None:
        step = self.get_step(plan_id, step_num)
        if not step:
            return
        for k, v in fields.items():
            if hasattr(step, k):
                if k == "status" and isinstance(v, str):
                    v = StepStatus(v)
                setattr(step, k, v)

    def start_step(self, plan_id: str, step_num: int, task_id: str) -> None:
        step = self.get_step(plan_id, step_num)
        if step:
            step.status = StepStatus.RUNNING
            step.task_id = task_id
            step.started_at = now_utc()

    def complete_step(self, plan_id: str, step_num: int, result: str) -> None:
        step = self.get_step(plan_id, step_num)
        if step:
            step.status = StepStatus.DONE
            step.result = result
            step.completed_at = now_utc()

    def fail_step(self, plan_id: str, step_num: int, error: str) -> None:
        step = self.get_step(plan_id, step_num)
        if step:
            step.status = StepStatus.FAILED
            step.result = error
            step.completed_at = now_utc()

    def ready_steps(self, plan_id: str) -> list[Step]:
        plan = self._plans.get(plan_id)
        if not plan:
            return []
        done_nums = {s.step_num for s in plan.steps if s.status == StepStatus.DONE}
        return [
            s
            for s in plan.steps
            if s.status == StepStatus.PENDING and all(d in done_nums for d in (s.depends_on or []))
        ]

    def next_pending_step(self, plan_id: str) -> Step | None:
        steps = self.ready_steps(plan_id)
        return steps[0] if steps else None

    def delete_pending_steps(self, plan_id: str) -> None:
        plan = self._plans.get(plan_id)
        if plan:
            plan.steps = [s for s in plan.steps if s.status != StepStatus.PENDING]

    def insert_step(
        self,
        plan_id: str,
        step_num: int,
        title: str,
        prompt: str,
        depends_on: list[int] | None = None,
    ) -> None:
        plan = self._plans.get(plan_id)
        if plan:
            plan.steps.append(
                Step(
                    plan_id=plan_id,
                    step_num=step_num,
                    title=title,
                    prompt=prompt,
                    status=StepStatus.PENDING,
                    depends_on=depends_on or [],
                )
            )

    # -- Events ----------------------------------------------------------------

    def emit_event(
        self,
        event_type: str,
        task_id: str | None = None,
        plan_id: str | None = None,
        step_num: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._event_seq += 1
        self._events.append(
            Event(
                id=self._event_seq,
                event_type=event_type,
                task_id=task_id,
                plan_id=plan_id,
                step_num=step_num,
                payload=payload,
                processed=False,
                created_at=now_utc(),
            )
        )

    def poll_events(self, limit: int = 10) -> list[Event]:
        unprocessed = [e for e in self._events if not e.processed]
        return unprocessed[:limit]

    def ack_event(self, event_id: int) -> None:
        for e in self._events:
            if e.id == event_id:
                e.processed = True
                e.processed_at = now_utc()
                break

    def ack_events(self, event_ids: list[int]) -> None:
        ids_set = set(event_ids)
        for e in self._events:
            if e.id in ids_set:
                e.processed = True
                e.processed_at = now_utc()
