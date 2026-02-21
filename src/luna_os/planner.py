"""Planner: multi-step orchestration engine.

Manages plan lifecycle: init, start, step_done, step_fail, replan, cancel,
advance, check_contracts. Uses pluggable StorageBackend, NotificationProvider,
and AgentRunner.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from luna_os.agents.base import AgentRunner
from luna_os.events import process_events, resolve_plan_for_task
from luna_os.notifications.base import NotificationProvider
from luna_os.store.base import StorageBackend
from luna_os.timeline import generate_html, render_png, steps_to_graph_data
from luna_os.types import Plan, Step

logger = logging.getLogger(__name__)

SGT = timezone(timedelta(hours=8))
MAX_CONCURRENT = 6
DRAFT_TIMEOUT_MINUTES = 30
STALE_HOURS = 24


# -- Formatting helpers --------------------------------------------------------


def _format_duration(start: datetime | str | None, end: datetime | str | None = None) -> str:
    """Format duration between two timestamps as a human-readable string."""
    if not start:
        return ""
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    if end:
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
    else:
        end = datetime.now(SGT)
    if start.tzinfo is None:
        start = start.replace(tzinfo=SGT)
    if end.tzinfo is None:
        end = end.replace(tzinfo=SGT)
    delta = end - start
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    hours = secs / 3600
    return f"{hours:.1f}h"


def _short_desc(text: str, max_len: int = 60) -> str:
    """Truncate text for display."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def normalize_step(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize step input to ``{title, prompt, depends_on}`` format."""
    title = raw.get("title") or raw.get("desc") or raw.get("name") or "Untitled"
    prompt = raw.get("prompt") or raw.get("detail") or ""
    depends_on = raw.get("depends_on")
    if depends_on is None:
        depends_on = []
    if isinstance(depends_on, int):
        depends_on = [depends_on]
    return {"title": title, "prompt": prompt, "depends_on": depends_on}


def validate_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and fix step dependencies.

    ``depends_on`` values are **0-indexed** offsets (matching the user/LLM
    convention).  Validation uses the same 0-indexed space; the store layer
    converts to 1-indexed step numbers on persist.

    - Remove self-references (step depending on itself)
    - Remove references to non-existent steps
    - Detect circular dependencies and break them
    """
    total = len(steps)
    valid_ids = set(range(total))  # 0-indexed

    for i, s in enumerate(steps):
        deps = s.get("depends_on") or []
        cleaned = [d for d in deps if d != i and d in valid_ids]
        s["depends_on"] = cleaned

    # Detect cycles via topological sort (0-indexed)
    in_degree = {i: 0 for i in range(total)}
    adj: dict[int, list[int]] = {i: [] for i in range(total)}
    for i, s in enumerate(steps):
        for d in s.get("depends_on") or []:
            adj[d].append(i)
            in_degree[i] += 1
    queue = [n for n, deg in in_degree.items() if deg == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited < total:
        # Break cycles by only allowing deps on earlier steps
        for i, s in enumerate(steps):
            s["depends_on"] = [d for d in (s.get("depends_on") or []) if d < i]

    return steps


def format_plan(plan: Plan) -> str:
    """Format a plan for display as a text message."""
    status_label = {
        "active": "",
        "completed": " [COMPLETED]",
        "cancelled": " [CANCELLED]",
        "draft": " [DRAFT]",
        "paused": " [PAUSED]",
    }.get(plan.status.value if hasattr(plan.status, "value") else str(plan.status), "")
    lines = [f"Plan {plan.id} -- {plan.goal}{status_label}\n"]

    for s in plan.steps:
        num = s.step_num
        title = s.title or ""
        st = s.status.value if hasattr(s.status, "value") else str(s.status)

        if st == "done":
            dur = _format_duration(s.started_at, s.completed_at)
            dur_str = f" [{dur}]" if dur else ""
            result = _short_desc(s.result or "", 60)
            lines.append(f"  [done] {num}. {title} {s.task_id or ''}{dur_str} -- {result}")
        elif st == "running":
            dur = _format_duration(s.started_at)
            dur_str = f" [{dur}]" if dur else ""
            lines.append(f"  [running] {num}. {title} {s.task_id or ''}{dur_str}")
        elif st == "failed":
            result = _short_desc(s.result or "", 60)
            lines.append(f"  [failed] {num}. {title} {s.task_id or ''} -- {result}")
        elif st == "waiting":
            reason = _short_desc(s.result or "等待输入", 60)
            dur = _format_duration(s.started_at)
            dur_str = f" [{dur}]" if dur else ""
            lines.append(f"  [⏸️ waiting] {num}. {title} {s.task_id or ''}{dur_str} -- {reason}")
        elif st == "cancelled":
            lines.append(f"  [cancelled] {num}. {title}")
        else:
            deps = s.depends_on or []
            dep_str = f" (after {','.join(str(d) for d in deps)})" if deps else ""
            lines.append(f"  [pending] {num}. {title}{dep_str}")

    return "\n".join(lines)


def build_plan_summary(plan: Plan, store: StorageBackend) -> str:
    """Build a summary message for a completed plan with cost/time breakdown."""
    done_steps = [s for s in plan.steps if s.status.value == "done"]
    if not done_steps:
        return ""

    plan_start = plan.created_at
    plan_end = max(
        (s.completed_at for s in done_steps if s.completed_at),
        default=None,
    )
    plan_dur = _format_duration(plan_start, plan_end) if plan_start and plan_end else ""

    total_input = 0
    total_output = 0
    total_cost = 0.0
    step_costs: list[tuple[Step, int, int, float]] = []
    for s in done_steps:
        inp, out, cost = 0, 0, 0.0
        if s.task_id:
            task = store.get_task(s.task_id)
            if task:
                inp = task.input_tokens or 0
                out = task.output_tokens or 0
                cost = float(task.cost_usd or 0)
        total_input += inp
        total_output += out
        total_cost += cost
        step_costs.append((s, inp, out, cost))

    lines = [f"Plan completed: {plan.goal or ''}"]
    if plan_dur:
        lines[0] += f"  Duration: {plan_dur}"
    if total_cost > 0:
        lines[0] += f"  Cost: ${total_cost:.4f}"
    lines.append("")

    for s, inp, out, cost in step_costs:
        dur = _format_duration(s.started_at, s.completed_at)
        dur_str = f"[{dur}]" if dur else ""
        tokens_str = f"{(inp + out):,} tok" if (inp + out) > 0 else ""
        pct = f"({cost / total_cost * 100:.0f}%)" if total_cost > 0 and cost > 0 else ""
        cost_str = f"${cost:.4f} {pct}" if cost > 0 else ""

        parts = [f"  {s.step_num}. {s.title or ''}"]
        detail = " | ".join(filter(None, [dur_str, tokens_str, cost_str]))
        if detail:
            parts.append(f"    {detail}")
        result = _short_desc(s.result or "", 150)
        if result:
            parts.append(f"    -> {result}")
        lines.extend(parts)

    if total_cost > 0 or total_input > 0:
        lines.append("")
        lines.append(
            f"  Total: {total_input:,} input + {total_output:,} output "
            f"= {(total_input + total_output):,} tokens | ${total_cost:.4f}"
        )

    return "\n".join(lines)


# -- Planner class -------------------------------------------------------------


class Planner:
    """Multi-step orchestration engine.

    Parameters
    ----------
    store:
        Storage backend for persistence.
    notifications:
        Optional notification provider for sending messages/graphs.
    agent_runner:
        Optional agent runner for spawning step subagents.
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

    # -- Notification helpers --------------------------------------------------

    def _notify(self, chat_id: str, text: str) -> None:
        if self.notifications and chat_id:
            try:
                self.notifications.send_message(chat_id, text)
            except Exception as exc:
                logger.warning("Notification failed: %s", exc)

    def _send_plan_graph(self, plan: Plan, chat_id: str | None = None) -> None:
        """Generate and send a timeline graph for a plan."""
        if not self.notifications:
            return
        target = chat_id or plan.chat_id
        if not target:
            return
        try:
            import shutil
            import tempfile

            steps_data = steps_to_graph_data(plan)
            if not steps_data:
                return
            status_label = {
                "active": "",
                "completed": " [Completed]",
                "cancelled": " [Cancelled]",
                "draft": " [Draft]",
                "paused": " [Paused]",
            }.get(plan.status.value if hasattr(plan.status, "value") else str(plan.status), "")
            title = f"Plan: {(plan.goal or '')[:80]}{status_label}"
            html = generate_html(steps_data, title)
            out_dir = tempfile.mkdtemp(prefix="plan-graph-")
            try:
                import os

                png_path = os.path.join(out_dir, "timeline.png")
                render_png(html, png_path)
                image_key = self.notifications.upload_image(png_path)
                self.notifications.send_image(target, image_key)
            finally:
                shutil.rmtree(out_dir, ignore_errors=True)
        except Exception as exc:
            logger.warning("send_plan_graph failed: %s", exc)

    # -- Spawn helpers ---------------------------------------------------------

    def _build_spawn_prompt(self, plan: Plan, step: Step, task_chat_id: str = "") -> str:
        """Build the prompt for spawning a subagent for a step."""
        prompt_text = step.prompt or step.title or ""
        goal = plan.goal or ""
        chat_id = plan.chat_id or ""
        step_num = step.step_num
        task_id = step.task_id or ""

        completed = [
            s
            for s in plan.steps
            if (s.status.value if hasattr(s.status, "value") else str(s.status)) == "done"
        ]
        context_lines = []
        for cs in completed[-3:]:
            context_lines.append(
                f"- Step {cs.step_num}: {cs.title} -> {_short_desc(cs.result or '', 80)}"
            )
        context = "\n".join(context_lines) if context_lines else "(none)"

        task_chat_section = ""
        if task_chat_id:
            task_chat_section = f"""
## Task Chat
Dedicated chat for progress updates: {task_chat_id}
Report progress at each key milestone.
"""

        return f"""You are executing step {step_num} of a multi-step plan.

## Plan Goal
{goal}

## Current Step
**Step {step_num}: {step.title or ""}**
{prompt_text}

## Context from Completed Steps
{context}

## Task ID
{task_id}

## Reporting
When done, emit: step.done with task-id={task_id}
On failure, emit: step.failed with task-id={task_id}
{task_chat_section}
## Parent Chat
Report results to: {chat_id}
"""

    def _start_ready_steps(self, plan: Plan, ready_steps: list[Step]) -> list[tuple[int, bool]]:
        """Start ready steps up to the concurrency limit.

        Returns list of ``(step_num, spawn_ok)`` tuples.
        """
        running = self.store.active_tasks()
        slots = self.max_concurrent - len(running)
        if slots <= 0:
            return []

        results: list[tuple[int, bool]] = []
        for step in ready_steps[:slots]:
            task_id = self.store.next_task_id()
            self.store.add_task(
                task_id,
                f"[Plan] {plan.goal} -- Step {step.step_num}: {step.title}",
                source_chat=plan.chat_id,
            )
            self.store.start_task(task_id, "cron-pending")
            self.store.start_step(plan.id, step.step_num, task_id)

            spawn_ok = False
            if self.agent_runner:
                # Create task chat if notifications available
                task_chat_id = ""
                if self.notifications:
                    try:
                        task_chat_id = self.notifications.create_chat(
                            f"Task {task_id} Step {step.step_num}: {step.title[:30]}",
                            f"Plan step: {step.title}",
                            [],
                        )
                        if task_chat_id:
                            self.store.update_task(task_id, task_chat_id=task_chat_id)
                    except Exception as exc:
                        logger.warning("Task chat creation failed: %s", exc)

                plan_fresh = self.store.get_plan(plan.id)
                updated_step = self.store.get_step(plan.id, step.step_num)
                if plan_fresh and updated_step:
                    prompt = self._build_spawn_prompt(plan_fresh, updated_step, task_chat_id)
                    try:
                        session_label = f"task-{task_chat_id[-8:]}" if task_chat_id else ""
                        self.agent_runner.spawn(task_id, prompt, session_label)
                        spawn_ok = True
                    except Exception as exc:
                        logger.warning("Spawn failed: %s", exc)

            results.append((step.step_num, spawn_ok))
        return results

    # -- Plan commands ---------------------------------------------------------

    def init(
        self,
        chat_id: str,
        goal: str,
        steps_raw: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a new plan in draft mode.

        Raises ``ValueError`` if an active plan already exists for this chat.
        """
        existing = self.store.get_plan_by_chat(chat_id, status_filter="active")
        if existing:
            raise ValueError(f"Active plan already exists: {existing.id} ({existing.goal})")

        steps = validate_steps([normalize_step(s) for s in steps_raw])
        if not steps:
            raise ValueError("No steps provided")

        plan_id = self.store.next_plan_id()
        plan = self.store.create_plan(plan_id, chat_id, goal, steps)
        self.store.update_plan_status(plan_id, "draft")

        plan_text = format_plan(plan)
        plan_text += "\n\nAwaiting confirmation (30-minute timeout)."
        plan_text += "\nReply 'confirm' to start, or 'modify' to adjust steps."
        self._notify(chat_id, plan_text)
        self._send_plan_graph(plan, chat_id)

        return {
            "plan_id": plan_id,
            "chat_id": chat_id,
            "goal": goal,
            "steps": len(steps),
            "status": "draft",
        }

    def start(self, chat_id: str) -> dict[str, Any]:
        """Activate a draft plan and start all ready steps."""
        plan = self.store.get_plan_by_chat(chat_id, status_filter="draft")
        if not plan:
            raise KeyError(f"No draft plan found for {chat_id}")

        self.store.update_plan_status(plan.id, "active")
        ready = self.store.ready_steps(plan.id)
        results: list[tuple[int, bool]] = []
        if ready:
            results = self._start_ready_steps(plan, ready)
            plan = self.store.get_plan(plan.id)  # type: ignore[assignment]
            if plan:
                self._notify(plan.chat_id, format_plan(plan))
                self._send_plan_graph(plan)

        return {
            "started": True,
            "plan_id": plan.id if plan else "",
            "parallel_steps": [r[0] for r in results],
            "spawned": [r[1] for r in results],
        }

    def show(self, identifier: str) -> str:
        """Display plan status. Accepts plan ID or chat_id."""
        plan = self.store.get_plan(identifier)
        if not plan:
            plan = self.store.get_plan_by_chat(identifier)
        if not plan:
            raise KeyError(f"No plan found for {identifier}")
        return format_plan(plan)

    def step_done(self, chat_id: str, step_num: int, result: str) -> dict[str, Any]:
        """Mark a step as done and advance to next."""
        plan = self.store.get_plan_by_chat(chat_id, status_filter="active")
        if not plan:
            raise KeyError(f"No active plan found for {chat_id}")

        # Idempotency guard
        existing = self.store.get_step(plan.id, step_num)
        if existing and existing.status.value == "done":
            return {"skipped": True, "reason": "step already done", "step": step_num}

        self.store.complete_step(plan.id, step_num, result)

        # Complete associated task
        step = self.store.get_step(plan.id, step_num)
        if step and step.task_id:
            self.store.complete_task(step.task_id, result)
            task = self.store.get_task(step.task_id)
            if task and task.task_chat_id:
                self._notify(
                    task.task_chat_id,
                    f"Step {step_num} completed: {_short_desc(result, 120)}",
                )
            self._notify(
                plan.chat_id,
                f"Step {step_num} ({step.title}) done: {_short_desc(result, 300)}",
            )

        # Try to advance
        ready = self.store.ready_steps(plan.id)
        spawn_results: list[tuple[int, bool]] = []
        plan_completed = False

        if ready:
            spawn_results = self._start_ready_steps(plan, ready)
            plan = self.store.get_plan(plan.id)  # type: ignore[assignment]
            if plan:
                self._notify(plan.chat_id, format_plan(plan))
                self._send_plan_graph(plan)
        else:
            plan = self.store.get_plan(plan.id)  # type: ignore[assignment]
            if plan:
                still_pending = [s for s in plan.steps if s.status.value == "pending"]
                still_running = [s for s in plan.steps if s.status.value == "running"]
                if still_running:
                    self._notify(plan.chat_id, format_plan(plan))
                    self._send_plan_graph(plan)
                elif still_pending:
                    # Pending steps exist but none are ready — deps may include
                    # failed steps or there is a dependency cycle.  Do NOT mark
                    # the plan as completed; log a warning instead.
                    logger.warning(
                        "Plan %s has %d pending steps with unmet deps after step %d done "
                        "(possible failed dependency or cycle)",
                        plan.id,
                        len(still_pending),
                        step_num,
                    )
                    self._notify(
                        plan.chat_id,
                        f"⚠️ {len(still_pending)} pending step(s) cannot proceed "
                        f"(dependencies not met). Check for failed steps.",
                    )
                    self._notify(plan.chat_id, format_plan(plan))
                    self._send_plan_graph(plan)
                else:
                    self.store.update_plan_status(plan.id, "completed")
                    plan = self.store.get_plan(plan.id)  # type: ignore[assignment]
                    if plan:
                        summary = build_plan_summary(plan, self.store)
                        self._notify(plan.chat_id, format_plan(plan))
                        self._send_plan_graph(plan)
                        if summary:
                            self._notify(plan.chat_id, summary)
                        plan_completed = True

        return {
            "step_done": step_num,
            "result": result[:100],
            "next_steps": [r[0] for r in spawn_results],
            "spawned": [r[1] for r in spawn_results],
            "plan_completed": plan_completed,
        }

    def step_fail(self, chat_id: str, step_num: int, error: str) -> dict[str, Any]:
        """Mark a step as failed."""
        plan = self.store.get_plan_by_chat(chat_id, status_filter="active")
        if not plan:
            raise KeyError(f"No active plan found for {chat_id}")

        # Idempotency
        existing = self.store.get_step(plan.id, step_num)
        if existing and existing.status.value == "failed":
            return {"step_already_failed": step_num}

        self.store.fail_step(plan.id, step_num, error)

        step = self.store.get_step(plan.id, step_num)
        if step and step.task_id:
            task = self.store.get_task(step.task_id)
            if task and task.status.value != "failed":
                self.store.fail_task(step.task_id, error)
            if task and task.task_chat_id:
                self._notify(
                    task.task_chat_id,
                    f"Step {step_num} FAILED: {_short_desc(error, 120)}",
                )

        plan = self.store.get_plan(plan.id)  # type: ignore[assignment]
        if plan:
            self._notify(plan.chat_id, format_plan(plan))
            self._send_plan_graph(plan)

        return {"step_failed": step_num, "error": error[:100]}

    def replan(
        self,
        chat_id: str,
        new_steps_raw: list[dict[str, Any]],
        append: bool = False,
    ) -> dict[str, Any]:
        """Replace or append pending steps."""
        plan = self.store.get_plan_by_chat(chat_id, status_filter="active")
        if not plan:
            plan = self.store.get_plan_by_chat(chat_id, status_filter="draft")
        if not plan:
            raise KeyError(f"No plan found for {chat_id}")

        new_steps = validate_steps([normalize_step(s) for s in new_steps_raw])

        if append:
            next_num = max((s.step_num for s in plan.steps), default=0) + 1
            kept_count = len(plan.steps)
        else:
            kept = [s for s in plan.steps if s.status.value in ("done", "running", "failed")]
            next_num = max((s.step_num for s in kept), default=0) + 1
            kept_count = len(kept)
            self.store.delete_pending_steps(plan.id)

        for i, ns in enumerate(new_steps):
            raw_deps = ns.get("depends_on") or []
            step_num = next_num + i
            deps = [d + next_num for d in raw_deps if d + next_num != step_num]
            self.store.insert_step(plan.id, step_num, ns["title"], ns["prompt"], deps)

        # Auto-advance if plan is active
        spawn_ok = False
        original_status = plan.status.value if hasattr(plan.status, "value") else str(plan.status)
        if original_status == "active":
            plan = self.store.get_plan(plan.id)  # type: ignore[assignment]
            if plan:
                has_running = any(s.status.value == "running" for s in plan.steps)
                if not has_running:
                    first_pending = self.store.next_pending_step(plan.id)
                    if first_pending:
                        task_id = self.store.next_task_id()
                        desc = (
                            f"[Plan] {plan.goal} -- "
                            f"Step {first_pending.step_num}: {first_pending.title}"
                        )
                        self.store.add_task(
                            task_id,
                            desc,
                            source_chat=plan.chat_id,
                        )
                        self.store.start_task(task_id, "cron-pending")
                        self.store.start_step(plan.id, first_pending.step_num, task_id)
                        spawn_ok = True

        plan = self.store.get_plan(plan.id)  # type: ignore[assignment]
        if plan:
            self._notify(plan.chat_id, format_plan(plan))
            self._send_plan_graph(plan)

        return {
            "replanned": True,
            "kept_steps": kept_count,
            "new_steps": len(new_steps),
            "spawned": spawn_ok,
        }

    def cancel(self, chat_id: str) -> dict[str, Any]:
        """Cancel an entire plan."""
        plan = self.store.get_plan_by_chat(chat_id, status_filter="active")
        if not plan:
            plan = self.store.get_plan_by_chat(chat_id, status_filter="draft")
        if not plan:
            raise KeyError(f"No plan found for {chat_id}")

        for s in plan.steps:
            st = s.status.value if hasattr(s.status, "value") else str(s.status)
            if st in ("running", "pending"):
                if s.task_id and st == "running":
                    self.store.cancel_task(s.task_id)
                self.store.update_step(plan.id, s.step_num, status="cancelled")

        self.store.cancel_plan(plan.id)
        plan = self.store.get_plan(plan.id)  # type: ignore[assignment]
        if plan:
            self._notify(plan.chat_id, format_plan(plan))
            self._send_plan_graph(plan)

        return {"cancelled": True}

    def advance(self, chat_id: str) -> dict[str, Any]:
        """Check if current step is done and advance."""
        plan = self.store.get_plan_by_chat(chat_id, status_filter="active")
        if not plan or plan.status.value != "active":
            return {"advance": False, "reason": "no active plan"}

        running = [s for s in plan.steps if s.status.value == "running"]
        if not running:
            ready = self.store.ready_steps(plan.id)
            if ready:
                spawn_results = self._start_ready_steps(plan, ready)
                plan = self.store.get_plan(plan.id)  # type: ignore[assignment]
                if plan:
                    self._notify(plan.chat_id, format_plan(plan))
                    self._send_plan_graph(plan)
                started = [s for s, ok in spawn_results if ok]
                return {"advance": True, "started_steps": started}
            else:
                plan = self.store.get_plan(plan.id)  # type: ignore[assignment]
                if plan:
                    still_pending = [s for s in plan.steps if s.status.value == "pending"]
                    still_running = [s for s in plan.steps if s.status.value == "running"]
                    if still_running:
                        self._notify(plan.chat_id, format_plan(plan))
                        return {"advance": False, "still_running": len(still_running)}
                    elif still_pending:
                        logger.warning(
                            "Plan %s has %d pending steps with unmet deps (advance)",
                            plan.id,
                            len(still_pending),
                        )
                        return {
                            "advance": False,
                            "reason": "pending steps with unmet deps",
                            "pending": len(still_pending),
                        }
                    else:
                        self.store.update_plan_status(plan.id, "completed")
                        plan = self.store.get_plan(plan.id)  # type: ignore[assignment]
                        if plan:
                            summary = build_plan_summary(plan, self.store)
                            self._notify(plan.chat_id, format_plan(plan))
                            self._send_plan_graph(plan)
                            if summary:
                                self._notify(plan.chat_id, summary)
                        return {"advance": True, "plan_completed": True}
                return {"advance": False, "reason": "no plan"}

        # Check if running step's task is actually done
        current = running[0]
        if current.task_id:
            task = self.store.get_task(current.task_id)
            if task:
                if task.status.value == "done":
                    return self.step_done(chat_id, current.step_num, task.result or "")
                elif task.status.value == "failed":
                    return self.step_fail(chat_id, current.step_num, task.result or "unknown error")

        return {
            "advance": False,
            "reason": "step still running",
            "step": current.step_num,
        }

    def pause(self, chat_id: str) -> dict[str, Any]:
        """Pause an active plan."""
        plan = self.store.get_plan_by_chat(chat_id, status_filter="active")
        if not plan:
            raise KeyError("No active plan found")
        self.store.update_plan_status(plan.id, "paused")
        self._notify(plan.chat_id, f"Plan paused: {plan.goal[:60]}\nReply 'resume' to continue.")
        return {"paused": True, "plan_id": plan.id}

    def resume(self, plan_id_or_chat: str) -> dict[str, Any]:
        """Resume a paused plan."""
        if plan_id_or_chat == "all":
            paused = self.store.list_plans(status="paused")
            resumed = []
            for p in paused:
                self.store.update_plan_status(p.id, "active")
                full = self.store.get_plan(p.id)
                if full:
                    self._notify(full.chat_id, f"Plan resumed: {full.goal[:60]}")
                    with contextlib.suppress(Exception):
                        self.advance(full.chat_id)
                resumed.append(p.id)
            return {"resumed": resumed, "count": len(resumed)}

        plan = self.store.get_plan(plan_id_or_chat)
        if not plan:
            plan = self.store.get_plan_by_chat(plan_id_or_chat, status_filter="paused")
        if not plan:
            raise KeyError(f"No paused plan found for {plan_id_or_chat}")
        if plan.status.value != "paused":
            raise ValueError(f"Plan {plan.id} is {plan.status.value}, not paused")

        self.store.update_plan_status(plan.id, "active")
        self._notify(plan.chat_id, f"Plan resumed: {plan.goal[:60]}")
        with contextlib.suppress(Exception):
            self.advance(plan.chat_id)
        return {"resumed": True, "plan_id": plan.id}

    def check_advances(self) -> dict[str, Any]:
        """Check all active plans for advancement."""
        plans = self.store.list_plans(status="active")
        advanced = 0
        for p in plans:
            try:
                self.advance(p.chat_id)
                advanced += 1
            except Exception as exc:
                logger.warning("Advance error for %s: %s", p.id, exc)
        return {"checked": len(plans), "advanced": advanced}

    def check_contracts(self) -> dict[str, Any]:
        """Poll event queue and process step/contract completions.

        Also checks for expired drafts and stale plans.
        """
        processed = process_events(
            self.store,
            on_step_done=lambda plan, step_num, result: self.step_done(
                plan.chat_id, step_num, result
            ),
            on_step_fail=lambda plan, step_num, error: self.step_fail(
                plan.chat_id, step_num, error
            ),
        )

        # Check for expired drafts
        expired_drafts = 0
        drafts = self.store.list_plans(status="draft")
        for d in drafts:
            created = d.created_at
            if created:
                if isinstance(created, str):
                    created = datetime.fromisoformat(created)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                age_min = (datetime.now(UTC) - created).total_seconds() / 60
                if age_min > DRAFT_TIMEOUT_MINUTES:
                    self.store.update_plan_status(d.id, "cancelled")
                    full = self.store.get_plan(d.id)
                    if full:
                        msg = f"Draft plan timed out (30 min without confirmation)\n\n{full.goal}"
                        self._notify(full.chat_id, msg)
                    expired_drafts += 1

        # Check for stale plans (no activity in 24h)
        stale_paused = 0
        active_plans = self.store.list_plans(status="active")
        for p in active_plans:
            full = self.store.get_plan(p.id)
            if not full:
                continue
            last_activity = full.created_at
            for s in full.steps:
                for ts in (s.completed_at, s.started_at):
                    if ts:
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts)
                        if isinstance(ts, datetime):
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=UTC)
                            if last_activity is None:
                                last_activity = ts
                            else:
                                la = last_activity
                                if isinstance(la, str):
                                    la = datetime.fromisoformat(la)
                                if isinstance(la, datetime):
                                    if la.tzinfo is None:
                                        la = la.replace(tzinfo=UTC)
                                    if ts > la:
                                        last_activity = ts

            if last_activity:
                if isinstance(last_activity, str):
                    last_activity = datetime.fromisoformat(last_activity)
                if isinstance(last_activity, datetime):
                    if last_activity.tzinfo is None:
                        last_activity = last_activity.replace(tzinfo=UTC)
                    stale_hours = (datetime.now(UTC) - last_activity).total_seconds() / 3600
                    if stale_hours > STALE_HOURS:
                        fresh = self.store.get_plan(p.id)
                        if fresh and fresh.status.value == "active":
                            self.store.update_plan_status(p.id, "paused")
                            self._notify(
                                full.chat_id,
                                f"Plan auto-paused (no progress for 24+ hours)\n\n"
                                f"{full.goal[:60]}\n\nReply 'resume' to continue.",
                            )
                            stale_paused += 1

        # Check stuck running steps
        stuck_steps = self._check_stuck_steps()

        # Advance plans that have ready steps but nothing running
        # (catches cases where _start_ready_steps previously failed due to full slots)
        advanced = self._advance_idle_plans()

        result: dict[str, Any] = {"processed": processed}
        if expired_drafts:
            result["expired_drafts"] = expired_drafts
        if stuck_steps:
            result["stuck_steps_failed"] = stuck_steps
        if stale_paused:
            result["stale_paused"] = stale_paused
        if advanced:
            result["advanced_idle"] = advanced
        return result

    def _advance_idle_plans(self) -> int:
        """Try to start ready steps on active plans with no running steps.

        This catches plans that got stuck because ``_start_ready_steps``
        previously returned empty (e.g. concurrency slots were full).
        """
        advanced = 0
        active_plans = self.store.list_plans(status="active")
        for p in active_plans:
            full = self.store.get_plan(p.id)
            if not full:
                continue
            running = [s for s in full.steps if s.status.value == "running"]
            if running:
                continue  # already has work in progress
            ready = self.store.ready_steps(full.id)
            if not ready:
                continue  # nothing to start
            spawn_results = self._start_ready_steps(full, ready)
            if spawn_results:
                advanced += 1
                full = self.store.get_plan(full.id)  # type: ignore[assignment]
                if full:
                    self._notify(full.chat_id, format_plan(full))
                    self._send_plan_graph(full)
        return advanced

    def _check_stuck_steps(self) -> int:
        """Find and fail stuck running steps whose agent has died."""
        stuck_count = 0
        active_plans = self.store.list_plans(status="active")
        for p in active_plans:
            full = self.store.get_plan(p.id)
            if not full:
                continue
            for step in full.steps:
                if step.status.value != "running":
                    continue
                # Skip waiting tasks
                if step.task_id:
                    task = self.store.get_task(step.task_id)
                    if task and task.status.value == "waiting":
                        continue
                started = step.started_at
                if not started:
                    continue
                if isinstance(started, str):
                    started = datetime.fromisoformat(started)
                if started.tzinfo is None:
                    started = started.replace(tzinfo=UTC)
                age_min = (datetime.now(UTC) - started).total_seconds() / 60

                should_fail = False
                fail_reason = ""

                if age_min > 45:
                    should_fail = True
                    fail_reason = (
                        f"Auto-failed: step stuck for {age_min:.0f} minutes (timeout=45min)"
                    )
                elif age_min > 2 and self.agent_runner:
                    task_id = step.task_id or ""
                    task_obj = self.store.get_task(task_id) if task_id else None
                    task_chat = task_obj.task_chat_id if task_obj else ""
                    if task_chat:
                        session_id = f"task-{task_chat[-8:]}"
                        if not self.agent_runner.is_running(session_id):
                            should_fail = True
                            fail_reason = (
                                f"Auto-failed: agent process dead after {age_min:.0f} minutes"
                            )

                if should_fail:
                    existing = self.store.get_step(full.id, step.step_num)
                    if existing and existing.status.value == "failed":
                        continue
                    self.store.fail_step(full.id, step.step_num, fail_reason)
                    if existing and existing.task_id:
                        task = self.store.get_task(existing.task_id)
                        if task and task.status.value != "failed":
                            self.store.fail_task(existing.task_id, fail_reason)
                    updated_plan = self.store.get_plan(full.id)
                    if updated_plan:
                        self._notify(full.chat_id, format_plan(updated_plan))
                    stuck_count += 1

        return stuck_count

    def list_plans(self) -> list[dict[str, Any]]:
        """List all plans."""
        plans = self.store.list_plans()
        order = {"active": 0, "draft": 0, "paused": 1, "completed": 2, "cancelled": 3}
        plans.sort(
            key=lambda p: (
                order.get(p.status.value if hasattr(p.status, "value") else str(p.status), 9),
                str(p.created_at or ""),
            )
        )
        result = []
        for p in plans:
            full = self.store.get_plan(p.id)
            if not full:
                continue
            done = sum(1 for s in full.steps if s.status.value == "done")
            total = len(full.steps)
            running = [s for s in full.steps if s.status.value == "running"]
            result.append(
                {
                    "id": p.id,
                    "goal": p.goal,
                    "status": p.status.value if hasattr(p.status, "value") else str(p.status),
                    "done": done,
                    "total": total,
                    "running_step": running[0].step_num if running else None,
                }
            )
        return result

    def find_by_task(self, task_id: str) -> dict[str, Any]:
        """Find which plan and step a task_id belongs to."""
        plan_id, step_num = resolve_plan_for_task(self.store, task_id)
        if not plan_id or step_num is None:
            return {"found": False}

        plan = self.store.get_plan(plan_id)
        if not plan:
            return {"found": False}

        step = self.store.get_step(plan_id, step_num)
        next_step = self.store.next_pending_step(plan_id)

        result: dict[str, Any] = {
            "found": True,
            "chat_id": plan.chat_id,
            "goal": plan.goal,
            "planner_status": plan.status.value,
            "step": {
                "id": step_num,
                "desc": step.title if step else "",
                "status": step.status.value if step else "",
            },
            "advance": next_step is not None and step is not None and step.status.value == "done",
        }
        if next_step:
            result["next_step"] = {
                "id": next_step.step_num,
                "desc": next_step.title,
                "detail": next_step.prompt or "",
            }
        return result
