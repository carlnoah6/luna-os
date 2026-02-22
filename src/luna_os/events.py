"""Event emission and contract checking utilities.

Provides the ContractHelper class for detached scripts to report progress
via the event queue, and utility functions for event processing.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from luna_os.store.base import StorageBackend
from luna_os.types import Event

logger = logging.getLogger(__name__)


def _extract_session_cost(
    session_key: str,
) -> tuple[int, int, float]:
    """Extract total token usage from an agent session's jsonl file.

    Supports both direct filename keys (``task-xxx``) and Gateway
    session keys (``agent:main:subagent:task-xxx``). For Gateway keys,
    resolves the UUID via ``openclaw sessions --json``.

    Returns ``(input_tokens, output_tokens, cost_usd)``.
    Returns ``(0, 0, 0.0)`` if the file cannot be read.
    """
    sessions_dir = os.environ.get(
        "OPENCLAW_SESSIONS_DIR",
        str(Path.home() / ".openclaw" / "agents" / "main" / "sessions"),
    )

    jsonl_path = _resolve_session_file(session_key, sessions_dir)
    if not jsonl_path or not jsonl_path.exists():
        return 0, 0, 0.0

    total_in = total_out = 0
    total_cost = 0.0
    try:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    usage = obj.get("message", {}).get("usage", {})
                    if usage:
                        total_in += usage.get("input", 0)
                        total_out += usage.get("output", 0)
                        cost = usage.get("cost", {})
                        total_cost += cost.get("total", 0)
                except (json.JSONDecodeError, AttributeError):
                    continue
    except OSError as exc:
        logger.debug("Could not read session file %s: %s", jsonl_path, exc)

    return total_in, total_out, total_cost


def _resolve_session_file(
    session_key: str, sessions_dir: str,
) -> Path | None:
    """Resolve a session key to its .jsonl file path.

    Tries: (1) direct filename, (2) short key from colon-separated key,
    (3) ``openclaw sessions --json`` lookup for UUID.
    """
    base_dir = Path(sessions_dir)

    # Method 1: direct filename (e.g. "task-xxx")
    safe_key = Path(session_key).name
    if safe_key == session_key and ".." not in session_key:
        candidate = base_dir / f"{safe_key}.jsonl"
        if candidate.exists():
            return candidate

    # Method 2: extract short key from Gateway format
    # "agent:main:subagent:task-xxx" → "task-xxx"
    if ":" in session_key:
        short_key = session_key.rsplit(":", 1)[-1]
        candidate = base_dir / f"{short_key}.jsonl"
        if candidate.exists():
            return candidate

    # Method 3: resolve via openclaw sessions list
    try:
        result = subprocess.run(
            ["openclaw", "sessions", "--active", "60", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            sessions = data if isinstance(data, list) else data.get("sessions", [])
            for s in sessions:
                key = s.get("key", "")
                sid = s.get("sessionId", s.get("id", ""))
                if session_key in key and sid:
                    candidate = base_dir / f"{sid}.jsonl"
                    if candidate.exists():
                        return candidate
    except Exception:
        pass

    return None


class ContractHelper:
    """Event-based contract for detached long-running tasks.

    Usage::

        ch = ContractHelper(store, task_id="tid-0217-2", step_id=5)
        ch.start("Transcribing 6 remaining files")
        ch.progress("3/6 done")
        ch.done("All 6 files transcribed", succeeded=6, failed=0)
    """

    def __init__(
        self,
        store: StorageBackend,
        task_id: str,
        chat_id: str = "",
        step_id: int = 0,
        expected_outputs: list[str] | None = None,
    ) -> None:
        self.store = store
        self.task_id = task_id
        self.chat_id = chat_id
        self.step_id = step_id
        self.expected_outputs = expected_outputs or []

    def start(self, detail: str = "") -> None:
        """Emit ``contract.start`` event."""
        self.store.emit_event(
            "contract.start",
            task_id=self.task_id,
            step_num=self.step_id,
            payload={
                "chat_id": self.chat_id,
                "detail": detail,
                "expected_outputs": self.expected_outputs,
            },
        )

    def progress(self, message: str) -> None:
        """Emit ``contract.progress`` event."""
        self.store.emit_event(
            "contract.progress",
            task_id=self.task_id,
            step_num=self.step_id,
            payload={"message": message},
        )

    def done(self, result: str, succeeded: int = 0, failed: int = 0) -> None:
        """Emit ``contract.done`` event. Planner will auto-advance."""
        self._update_task_cost()
        self.store.emit_event(
            "contract.done",
            task_id=self.task_id,
            step_num=self.step_id,
            payload={"result": result, "succeeded": succeeded, "failed": failed},
        )

    def partial(self, result: str, succeeded: int = 0, failed: int = 0) -> None:
        """Emit ``contract.partial`` event. Planner treats as failure."""
        self.store.emit_event(
            "contract.partial",
            task_id=self.task_id,
            step_num=self.step_id,
            payload={"result": result, "succeeded": succeeded, "failed": failed},
        )

    def fail(self, error: str) -> None:
        """Emit ``contract.fail`` event."""
        self._update_task_cost()
        self.store.emit_event(
            "contract.fail",
            task_id=self.task_id,
            step_num=self.step_id,
            payload={"error": error},
        )

    def _update_task_cost(self) -> None:
        """Extract token usage from the agent session and update the task."""
        task = self.store.get_task(self.task_id)
        if not task or not task.session_key:
            return
        inp, out, cost = _extract_session_cost(task.session_key)
        if inp or out or cost:
            self.store.update_task_cost(self.task_id, inp, out, cost)


def resolve_plan_for_task(store: StorageBackend, task_id: str) -> tuple[str | None, int | None]:
    """Find plan_id and step_num for a given task_id by scanning active plans."""
    plans = store.list_plans(status="active")
    for p in plans:
        full = store.get_plan(p.id)
        if full is None:
            continue
        for s in full.steps:
            if s.task_id == task_id:
                return p.id, s.step_num
    return None, None


def process_events(
    store: StorageBackend,
    on_step_done: Any = None,
    on_step_fail: Any = None,
    on_step_waiting: Any = None,
    limit: int = 20,
) -> int:
    """Poll and process events from the queue.

    *on_step_done(plan, step_num, result)* and *on_step_fail(plan, step_num, error)*
    are optional callbacks invoked for step/contract completion and failure events.
    *on_step_waiting(plan, step_num, question)* is called when a step needs user input.

    Returns the number of events processed.
    """
    events = store.poll_events(limit=limit)
    if not events:
        return 0

    processed = 0
    for evt in events:
        try:
            _process_single_event(store, evt, on_step_done, on_step_fail, on_step_waiting)
            store.ack_event(evt.id)
            processed += 1
        except Exception:
            logging.getLogger(__name__).exception("Error processing event %s", evt.id)
            store.ack_event(evt.id)  # ack to prevent infinite loop

    return processed


def _process_single_event(
    store: StorageBackend,
    evt: Event,
    on_step_done: Any,
    on_step_fail: Any,
    on_step_waiting: Any = None,
) -> None:
    """Dispatch a single event to appropriate handler."""
    etype = evt.event_type
    task_id = evt.task_id
    plan_id = evt.plan_id
    step_num = evt.step_num
    payload = evt.payload or {}

    # Resolve plan from task_id if not provided
    if task_id and not plan_id:
        plan_id, step_num = resolve_plan_for_task(store, task_id)

    if etype in ("step.done", "contract.done") and plan_id and step_num is not None:
        result = payload.get("result", "completed")
        plan = store.get_plan(plan_id)
        if plan and on_step_done:
            on_step_done(plan, step_num, result)

    elif etype in ("step.failed", "contract.fail") and plan_id and step_num is not None:
        error = payload.get("error", payload.get("result", "unknown error"))
        plan = store.get_plan(plan_id)
        if plan and on_step_fail:
            on_step_fail(plan, step_num, error)

    elif etype == "contract.partial" and plan_id and step_num is not None:
        result = payload.get("result", "partial completion")
        succeeded = payload.get("succeeded", 0)
        failed = payload.get("failed", 0)
        error = f"Partial: {result} ({succeeded} ok, {failed} failed)"
        plan = store.get_plan(plan_id)
        if plan and on_step_fail:
            on_step_fail(plan, step_num, error)

    elif etype == "step.waiting" and plan_id and step_num is not None:
        question = payload.get("result", payload.get("question", "waiting for input"))
        plan = store.get_plan(plan_id)
        if plan and on_step_waiting:
            on_step_waiting(plan, step_num, question)
    # contract.progress and contract.start are informational — just ack
