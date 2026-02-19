"""Shared data types for the luna-os package."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    """Possible states for a task."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"


class PlanStatus(StrEnum):
    """Possible states for a plan."""

    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    """Possible states for a plan step."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(StrEnum):
    """Task priority levels (lower value = higher priority)."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


PRIORITY_MAP: dict[str, int] = {
    "critical": 1,
    "high": 2,
    "normal": 3,
    "low": 4,
}

PRIORITY_NAMES: dict[int, str] = {v: k for k, v in PRIORITY_MAP.items()}

PRIORITY_ICONS: dict[str, str] = {
    "critical": "\U0001f534",
    "high": "\U0001f7e1",
    "normal": "\U0001f7e2",
    "low": "\U0001f535",
}


@dataclass
class Task:
    """Represents a single task in the system."""

    id: str
    description: str
    status: TaskStatus = TaskStatus.QUEUED
    source_chat: str | None = None
    task_chat_id: str | None = None
    priority: str = "normal"
    priority_value: int = 3
    depends_on: list[str] = field(default_factory=list)
    session_key: str = ""
    result: str = ""
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # Wait-related fields
    wait_type: str | None = None
    wait_prompt: str | None = None
    waited_at: datetime | None = None
    wait_response: str | None = None
    wait_card_id: str | None = None
    wait_todo_id: str | None = None
    wait_options: str | None = None
    priority_boosted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        d: dict[str, Any] = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, (TaskStatus, Priority)):
                v = v.value
            if isinstance(v, datetime):
                v = v.isoformat()
            d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Create a Task from a plain dictionary."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        if "status" in filtered and isinstance(filtered["status"], str):
            with contextlib.suppress(ValueError):
                filtered["status"] = TaskStatus(filtered["status"])
        return cls(**filtered)


@dataclass
class Step:
    """Represents a single step within a plan."""

    plan_id: str
    step_num: int
    title: str
    prompt: str = ""
    status: StepStatus = StepStatus.PENDING
    task_id: str | None = None
    result: str | None = None
    depends_on: list[int] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, StepStatus):
                v = v.value
            if isinstance(v, datetime):
                v = v.isoformat()
            d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        if "status" in filtered and isinstance(filtered["status"], str):
            with contextlib.suppress(ValueError):
                filtered["status"] = StepStatus(filtered["status"])
        return cls(**filtered)


@dataclass
class Plan:
    """Represents a multi-step plan."""

    id: str
    chat_id: str
    goal: str
    status: PlanStatus = PlanStatus.DRAFT
    steps: list[Step] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, PlanStatus):
                v = v.value
            if isinstance(v, list) and k == "steps":
                v = [s.to_dict() for s in v]
            if isinstance(v, datetime):
                v = v.isoformat()
            d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        steps_raw = data.pop("steps", []) if "steps" in data else []
        filtered = {k: v for k, v in data.items() if k in known}
        if "status" in filtered and isinstance(filtered["status"], str):
            with contextlib.suppress(ValueError):
                filtered["status"] = PlanStatus(filtered["status"])
        plan = cls(**filtered)
        plan.steps = [Step.from_dict(s) if isinstance(s, dict) else s for s in steps_raw]
        return plan


@dataclass
class Event:
    """Represents an event in the event queue."""

    id: int | None = None
    event_type: str = ""
    task_id: str | None = None
    plan_id: str | None = None
    step_num: int | None = None
    payload: dict[str, Any] | None = None
    processed: bool = False
    created_at: datetime | None = None
    processed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, datetime):
                v = v.isoformat()
            d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def now_utc() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(UTC)
