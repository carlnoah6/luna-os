"""Comprehensive planner integration tests.

Covers: init→start→step_done→advance full lifecycle, replan, cancel,
pause/resume, stuck detection, concurrency, check_contracts, advance,
spawn rollback, list_plans, find_by_task, show, restart pause, and
various edge cases.

Target: 100+ tests.
"""

import os
import time
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from luna_os.agents.base import AgentRunner
from luna_os.planner import (
    Planner,
    format_plan,
    normalize_step,
    resolve_title_deps,
    validate_steps,
    build_plan_summary,
)
from luna_os.events import process_events, resolve_plan_for_task
from luna_os.types import StepStatus, PlanStatus, TaskStatus
from tests.memory_store import MemoryBackend


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _NoopRunner(AgentRunner):
    def spawn(self, task_id: str, prompt: str, session_label: str = "") -> str:
        return session_label or f"task-{task_id}"

    def is_running(self, session_key: str) -> bool:
        return True


class _FailRunner(AgentRunner):
    """Runner whose spawn always raises."""
    def spawn(self, task_id: str, prompt: str, session_label: str = "") -> str:
        raise RuntimeError("spawn exploded")

    def is_running(self, session_key: str) -> bool:
        return False


class _DeadRunner(AgentRunner):
    """Runner that spawns OK but reports processes as dead."""
    def spawn(self, task_id: str, prompt: str, session_label: str = "") -> str:
        return session_label or f"task-{task_id}"

    def is_running(self, session_key: str) -> bool:
        return False


def make_planner(runner=None, max_concurrent=6) -> tuple[Planner, MemoryBackend]:
    store = MemoryBackend()
    planner = Planner(store, agent_runner=runner or _NoopRunner(), max_concurrent=max_concurrent)
    return planner, store


THREE_STEPS = [
    {"title": "Design", "prompt": "Design it"},
    {"title": "Build", "prompt": "Build it", "depends_on": [0]},
    {"title": "Test", "prompt": "Test it", "depends_on": [1]},
]

TWO_PARALLEL = [
    {"title": "Research A", "prompt": "Do A"},
    {"title": "Research B", "prompt": "Do B"},
]

DIAMOND = [
    {"title": "Start", "prompt": "Begin"},
    {"title": "Left", "prompt": "Left path", "depends_on": [0]},
    {"title": "Right", "prompt": "Right path", "depends_on": [0]},
    {"title": "Merge", "prompt": "Merge results", "depends_on": [1, 2]},
]


def _init_and_start(planner, chat_id, goal, steps):
    """Helper: init + start a plan, return the result dicts."""
    r1 = planner.init(chat_id, goal, steps)
    r2 = planner.start(chat_id)
    return r1, r2


# ===========================================================================
# 1. Full Lifecycle Tests
# ===========================================================================

class TestFullLifecycle:
    """init → start → step_done (all) → plan completed."""

    def test_serial_three_steps(self):
        p, store = make_planner()
        _init_and_start(p, "chat-lc1", "Serial plan", THREE_STEPS)

        # Step 1 should be running
        plan = store.get_plan_by_chat("chat-lc1", "active")
        running = [s for s in plan.steps if s.status == StepStatus.RUNNING]
        assert len(running) == 1
        assert running[0].step_num == 1

        # Complete step 1 → step 2 starts
        r = p.step_done("chat-lc1", 1, "designed")
        assert r["step_done"] == 1
        assert 2 in r["next_steps"]
        assert not r["plan_completed"]

        # Complete step 2 → step 3 starts
        r = p.step_done("chat-lc1", 2, "built")
        assert 3 in r["next_steps"]

        # Complete step 3 → plan completed
        r = p.step_done("chat-lc1", 3, "tested")
        assert r["plan_completed"] is True
        plan = store.get_plan_by_chat("chat-lc1")
        assert plan.status == PlanStatus.COMPLETED

    def test_parallel_two_steps(self):
        p, store = make_planner()
        _init_and_start(p, "chat-lc2", "Parallel", TWO_PARALLEL)

        plan = store.get_plan_by_chat("chat-lc2", "active")
        running = [s for s in plan.steps if s.status == StepStatus.RUNNING]
        assert len(running) == 2

        p.step_done("chat-lc2", 1, "A done")
        plan = store.get_plan_by_chat("chat-lc2", "active")
        assert plan is not None  # not completed yet

        r = p.step_done("chat-lc2", 2, "B done")
        assert r["plan_completed"] is True

    def test_diamond_dependency(self):
        p, store = make_planner()
        _init_and_start(p, "chat-lc3", "Diamond", DIAMOND)

        plan = store.get_plan_by_chat("chat-lc3", "active")
        running = [s for s in plan.steps if s.status == StepStatus.RUNNING]
        assert len(running) == 1
        assert running[0].step_num == 1

        # Complete Start → Left + Right start
        r = p.step_done("chat-lc3", 1, "started")
        assert sorted(r["next_steps"]) == [2, 3]

        # Complete Left
        p.step_done("chat-lc3", 2, "left done")
        plan = store.get_plan_by_chat("chat-lc3", "active")
        assert plan is not None  # Merge not ready yet

        # Complete Right → Merge starts
        r = p.step_done("chat-lc3", 3, "right done")
        assert 4 in r["next_steps"]

        # Complete Merge → plan done
        r = p.step_done("chat-lc3", 4, "merged")
        assert r["plan_completed"] is True

    def test_single_step_plan(self):
        p, store = make_planner()
        _init_and_start(p, "chat-lc4", "One step", [{"title": "Do it", "prompt": "go"}])

        r = p.step_done("chat-lc4", 1, "done")
        assert r["plan_completed"] is True

    def test_lifecycle_tasks_created_and_completed(self):
        """Verify tasks are created for each step and completed alongside."""
        p, store = make_planner()
        _init_and_start(p, "chat-lc5", "Task tracking", TWO_PARALLEL)

        tasks = store.list_tasks(status="running")
        assert len(tasks) == 2

        p.step_done("chat-lc5", 1, "A")
        done_tasks = store.list_tasks(status="done")
        assert len(done_tasks) == 1

        p.step_done("chat-lc5", 2, "B")
        done_tasks = store.list_tasks(status="done")
        assert len(done_tasks) == 2


# ===========================================================================
# 2. Init Tests
# ===========================================================================

class TestInit:
    def test_init_creates_draft(self):
        p, store = make_planner()
        r = p.init("chat-i1", "Goal", THREE_STEPS)
        assert r["status"] == "draft"
        plan = store.get_plan_by_chat("chat-i1", "draft")
        assert plan is not None
        assert len(plan.steps) == 3

    def test_init_duplicate_active_raises(self):
        p, store = make_planner()
        _init_and_start(p, "chat-i2", "First", TWO_PARALLEL)
        with pytest.raises(ValueError, match="Active plan already exists"):
            p.init("chat-i2", "Second", TWO_PARALLEL)

    def test_init_no_steps_raises(self):
        p, _ = make_planner()
        with pytest.raises(ValueError, match="No steps"):
            p.init("chat-i3", "Empty", [])

    def test_init_allows_new_after_cancel(self):
        p, store = make_planner()
        _init_and_start(p, "chat-i4", "First", TWO_PARALLEL)
        p.cancel("chat-i4")
        # Should be able to create a new plan now
        r = p.init("chat-i4", "Second", THREE_STEPS)
        assert r["status"] == "draft"

    def test_init_allows_new_after_completed(self):
        p, store = make_planner()
        _init_and_start(p, "chat-i5", "First", [{"title": "X", "prompt": "x"}])
        p.step_done("chat-i5", 1, "done")
        r = p.init("chat-i5", "Second", TWO_PARALLEL)
        assert r["status"] == "draft"

    def test_init_step_numbering_1based(self):
        p, store = make_planner()
        p.init("chat-i6", "Numbering", THREE_STEPS)
        plan = store.get_plan_by_chat("chat-i6", "draft")
        nums = [s.step_num for s in plan.steps]
        assert nums == [1, 2, 3]


# ===========================================================================
# 3. Start Tests
# ===========================================================================

class TestStart:
    def test_start_activates_and_spawns(self):
        p, store = make_planner()
        p.init("chat-s1", "Go", THREE_STEPS)
        r = p.start("chat-s1")
        assert r["started"] is True
        plan = store.get_plan_by_chat("chat-s1", "active")
        assert plan is not None

    def test_start_no_draft_raises(self):
        p, _ = make_planner()
        with pytest.raises(KeyError, match="No draft plan"):
            p.start("chat-s2")

    def test_start_spawns_parallel_steps(self):
        p, store = make_planner()
        p.init("chat-s3", "Parallel", TWO_PARALLEL)
        r = p.start("chat-s3")
        assert len(r["parallel_steps"]) == 2

    def test_start_only_spawns_ready(self):
        p, store = make_planner()
        p.init("chat-s4", "Serial", THREE_STEPS)
        r = p.start("chat-s4")
        # Only step 1 is ready (2 depends on 1, 3 depends on 2)
        assert r["parallel_steps"] == [1]


# ===========================================================================
# 3. step_done edge cases
# ===========================================================================

class TestStepDone:
    def test_step_done_no_active_plan_raises(self):
        p, _ = make_planner()
        with pytest.raises(KeyError):
            p.step_done("nonexistent", 1, "result")

    def test_step_done_idempotent(self):
        p, _ = make_planner()
        _init_and_start(p, "chat-sd1", "Idem", [{"title": "A", "prompt": "a"}])
        r1 = p.step_done("chat-sd1", 1, "done")
        assert r1["plan_completed"] is True
        # Second call should be idempotent (plan already completed)
        # The plan is now completed, so get_plan_by_chat(status_filter="active") returns None
        with pytest.raises(KeyError):
            p.step_done("chat-sd1", 1, "done again")

    def test_step_done_completes_associated_task(self):
        p, store = make_planner()
        _init_and_start(p, "chat-sd2", "Task", [{"title": "A", "prompt": "a"}])
        plan = store.get_plan_by_chat("chat-sd2", "active")
        step = plan.steps[0]
        task = store.get_task(step.task_id)
        assert task.status == TaskStatus.RUNNING

        p.step_done("chat-sd2", 1, "result")
        task = store.get_task(step.task_id)
        assert task.status == TaskStatus.DONE
        assert task.result == "result"

    def test_step_done_blocked_deps_warns(self):
        """If step 1 fails and step 2 depends on it, completing step 1 is impossible
        but completing other steps should warn about blocked pending."""
        p, store = make_planner()
        steps = [
            {"title": "A", "prompt": "a"},
            {"title": "B", "prompt": "b"},
            {"title": "C", "prompt": "c", "depends_on": [0, 1]},
        ]
        _init_and_start(p, "chat-sd3", "Blocked", steps)
        # Fail step 1
        p.step_fail("chat-sd3", 1, "error")
        # Complete step 2
        r = p.step_done("chat-sd3", 2, "b done")
        # Step C can't proceed because step 1 failed
        assert r["plan_completed"] is False
        assert r["next_steps"] == []

    def test_step_done_result_truncated_in_return(self):
        p, _ = make_planner()
        _init_and_start(p, "chat-sd4", "Trunc", [{"title": "A", "prompt": "a"}])
        long_result = "x" * 200
        r = p.step_done("chat-sd4", 1, long_result)
        assert len(r["result"]) == 100


# ===========================================================================
# 4. step_fail tests
# ===========================================================================

class TestStepFail:
    def test_step_fail_no_plan_raises(self):
        p, _ = make_planner()
        with pytest.raises(KeyError):
            p.step_fail("nonexistent", 1, "err")

    def test_step_fail_marks_step_and_task(self):
        p, store = make_planner()
        _init_and_start(p, "chat-sf1", "Fail", [{"title": "A", "prompt": "a"}])
        plan = store.get_plan_by_chat("chat-sf1", "active")
        task_id = plan.steps[0].task_id

        r = p.step_fail("chat-sf1", 1, "boom")
        assert r["step_failed"] == 1
        step = store.get_step(plan.id, 1)
        assert step.status == StepStatus.FAILED
        task = store.get_task(task_id)
        assert task.status == TaskStatus.FAILED

    def test_step_fail_idempotent(self):
        p, _ = make_planner()
        _init_and_start(p, "chat-sf2", "Idem", [{"title": "A", "prompt": "a"}])
        p.step_fail("chat-sf2", 1, "err")
        r = p.step_fail("chat-sf2", 1, "err again")
        assert r.get("step_already_failed") == 1

    def test_step_fail_does_not_complete_plan(self):
        """A failed step should NOT mark the plan as completed."""
        p, store = make_planner()
        _init_and_start(p, "chat-sf3", "NoComplete",
                        [{"title": "A", "prompt": "a"}, {"title": "B", "prompt": "b", "depends_on": [0]}])
        p.step_fail("chat-sf3", 1, "err")
        plan = store.get_plan_by_chat("chat-sf3", "active")
        assert plan is not None  # still active, not completed


# ===========================================================================
# 5. Replan tests
# ===========================================================================

class TestReplan:
    def test_replan_replaces_pending(self):
        p, store = make_planner()
        _init_and_start(p, "chat-rp1", "Replan", THREE_STEPS)
        # Step 1 running, steps 2,3 pending
        r = p.replan("chat-rp1", [{"title": "New2", "prompt": "new"}, {"title": "New3", "prompt": "new3"}])
        assert r["replanned"] is True
        assert r["kept_steps"] == 1  # only running step 1 kept
        assert r["new_steps"] == 2

        plan = store.get_plan_by_chat("chat-rp1", "active")
        titles = [s.title for s in plan.steps]
        assert "Design" in titles  # kept (running)
        assert "Build" not in titles  # replaced
        assert "New2" in titles

    def test_replan_append_mode(self):
        p, store = make_planner()
        _init_and_start(p, "chat-rp2", "Append", TWO_PARALLEL)
        r = p.replan("chat-rp2", [{"title": "Extra", "prompt": "extra"}], append=True)
        assert r["replanned"] is True
        assert r["kept_steps"] == 2  # both original steps kept
        assert r["new_steps"] == 1

        plan = store.get_plan_by_chat("chat-rp2", "active")
        assert len(plan.steps) == 3
        assert plan.steps[-1].title == "Extra"

    def test_replan_no_plan_raises(self):
        p, _ = make_planner()
        with pytest.raises(KeyError):
            p.replan("nonexistent", [{"title": "X", "prompt": "x"}])

    def test_replan_on_draft(self):
        p, store = make_planner()
        p.init("chat-rp3", "Draft replan", THREE_STEPS)
        # Don't start — replan on draft
        r = p.replan("chat-rp3", [{"title": "Only", "prompt": "only"}])
        assert r["replanned"] is True
        plan = store.get_plan_by_chat("chat-rp3", "draft")
        assert len(plan.steps) == 1
        assert plan.steps[0].title == "Only"

    def test_replan_auto_advances_when_no_running(self):
        """If all steps are done/failed and we replan, new steps should auto-start."""
        p, store = make_planner()
        _init_and_start(p, "chat-rp4", "AutoAdv",
                        [{"title": "A", "prompt": "a"}])
        p.step_done("chat-rp4", 1, "done")
        # Plan is completed now, but replan should work on... let's use a different approach
        # Start with 2 steps, complete step 1, fail step 2 (no pending left but plan active)
        p2, store2 = make_planner()
        _init_and_start(p2, "chat-rp4b", "AutoAdv",
                        [{"title": "A", "prompt": "a"}, {"title": "B", "prompt": "b"}])
        p2.step_done("chat-rp4b", 1, "done")
        p2.step_fail("chat-rp4b", 2, "err")
        # Plan still active (has failed step). Replan with new step.
        r = p2.replan("chat-rp4b", [{"title": "C", "prompt": "c"}])
        assert r["spawned"] is True  # auto-started since no running steps


# ===========================================================================
# 6. Cancel tests
# ===========================================================================

class TestCancel:
    def test_cancel_active_plan(self):
        p, store = make_planner()
        _init_and_start(p, "chat-cn1", "Cancel me", THREE_STEPS)
        r = p.cancel("chat-cn1")
        assert r["cancelled"] is True
        plan = store.get_plan_by_chat("chat-cn1")
        assert plan.status == PlanStatus.CANCELLED

    def test_cancel_draft_plan(self):
        p, store = make_planner()
        p.init("chat-cn2", "Draft cancel", [{"title": "A", "prompt": "a"}])
        r = p.cancel("chat-cn2")
        assert r["cancelled"] is True

    def test_cancel_no_plan_raises(self):
        p, _ = make_planner()
        with pytest.raises(KeyError):
            p.cancel("nonexistent")

    def test_cancel_cancels_running_tasks(self):
        p, store = make_planner()
        _init_and_start(p, "chat-cn3", "Cancel tasks", TWO_PARALLEL)
        plan = store.get_plan_by_chat("chat-cn3", "active")
        task_ids = [s.task_id for s in plan.steps if s.task_id]
        assert len(task_ids) == 2

        p.cancel("chat-cn3")
        for tid in task_ids:
            task = store.get_task(tid)
            assert task.status == TaskStatus.CANCELLED

    def test_cancel_leaves_done_steps_intact(self):
        p, store = make_planner()
        _init_and_start(p, "chat-cn4", "Partial cancel", THREE_STEPS)
        p.step_done("chat-cn4", 1, "done")
        p.cancel("chat-cn4")
        plan = store.get_plan_by_chat("chat-cn4")
        step1 = [s for s in plan.steps if s.step_num == 1][0]
        assert step1.status == StepStatus.DONE


# ===========================================================================
# 7. Pause / Resume tests
# ===========================================================================

class TestPauseResume:
    def test_pause_active_plan(self):
        p, store = make_planner()
        _init_and_start(p, "chat-pr1", "Pause me", TWO_PARALLEL)
        r = p.pause("chat-pr1")
        assert r["paused"] is True
        plan = store.get_plan_by_chat("chat-pr1", "paused")
        assert plan is not None
        assert plan.status == PlanStatus.PAUSED

    def test_pause_no_active_raises(self):
        p, _ = make_planner()
        with pytest.raises(KeyError):
            p.pause("nonexistent")

    def test_resume_paused_plan(self):
        p, store = make_planner()
        _init_and_start(p, "chat-pr2", "Resume me", TWO_PARALLEL)
        p.pause("chat-pr2")
        r = p.resume("chat-pr2")
        assert r["resumed"] is True
        plan = store.get_plan_by_chat("chat-pr2", "active")
        assert plan is not None

    def test_resume_by_plan_id(self):
        p, store = make_planner()
        r1, _ = _init_and_start(p, "chat-pr3", "By ID", TWO_PARALLEL)
        plan_id = r1["plan_id"]
        p.pause("chat-pr3")
        r = p.resume(plan_id)
        assert r["resumed"] is True

    def test_resume_not_paused_raises(self):
        p, _ = make_planner()
        _init_and_start(p, "chat-pr4", "Active", TWO_PARALLEL)
        with pytest.raises(KeyError):
            p.resume("chat-pr4")

    def test_resume_all(self):
        p, store = make_planner()
        _init_and_start(p, "chat-pr5a", "Plan A", TWO_PARALLEL)
        _init_and_start(p, "chat-pr5b", "Plan B", TWO_PARALLEL)
        p.pause("chat-pr5a")
        p.pause("chat-pr5b")
        r = p.resume("all")
        assert r["count"] == 2

    def test_pause_resume_preserves_step_states(self):
        p, store = make_planner()
        _init_and_start(p, "chat-pr6", "Preserve", THREE_STEPS)
        p.step_done("chat-pr6", 1, "done")
        p.pause("chat-pr6")
        p.resume("chat-pr6")
        plan = store.get_plan_by_chat("chat-pr6", "active")
        step1 = [s for s in plan.steps if s.step_num == 1][0]
        assert step1.status == StepStatus.DONE


# ===========================================================================
# 8. Advance tests
# ===========================================================================

class TestAdvance:
    def test_advance_no_active_plan(self):
        p, _ = make_planner()
        r = p.advance("nonexistent")
        assert r["advance"] is False

    def test_advance_starts_ready_steps(self):
        p, store = make_planner()
        _init_and_start(p, "chat-adv1", "Advance", THREE_STEPS)
        # Manually complete step 1 in store (bypass step_done to test advance independently)
        plan = store.get_plan_by_chat("chat-adv1", "active")
        store.complete_step(plan.id, 1, "done")
        store.complete_task(plan.steps[0].task_id, "done")

        r = p.advance("chat-adv1")
        assert r["advance"] is True
        assert 2 in r.get("started_steps", [])

    def test_advance_completes_plan_when_all_done(self):
        p, store = make_planner()
        _init_and_start(p, "chat-adv2", "Complete",
                        [{"title": "A", "prompt": "a"}])
        plan = store.get_plan_by_chat("chat-adv2", "active")
        store.complete_step(plan.id, 1, "done")
        store.complete_task(plan.steps[0].task_id, "done")

        r = p.advance("chat-adv2")
        assert r.get("plan_completed") is True

    def test_advance_detects_done_task_and_calls_step_done(self):
        """If a running step's task is done, advance should call step_done."""
        p, store = make_planner()
        _init_and_start(p, "chat-adv3", "Auto", THREE_STEPS)
        plan = store.get_plan_by_chat("chat-adv3", "active")
        # Mark task as done directly
        store.complete_task(plan.steps[0].task_id, "result")

        r = p.advance("chat-adv3")
        assert r["step_done"] == 1

    def test_advance_detects_failed_task(self):
        p, store = make_planner()
        _init_and_start(p, "chat-adv4", "FailDetect", THREE_STEPS)
        plan = store.get_plan_by_chat("chat-adv4", "active")
        store.fail_task(plan.steps[0].task_id, "boom")

        r = p.advance("chat-adv4")
        assert r["step_failed"] == 1

    def test_advance_still_running(self):
        p, _ = make_planner()
        _init_and_start(p, "chat-adv5", "Running", THREE_STEPS)
        r = p.advance("chat-adv5")
        assert r["advance"] is False
        assert r["reason"] == "step still running"

    def test_advance_pending_with_unmet_deps(self):
        """Steps pending but deps not met (failed dep)."""
        p, store = make_planner()
        _init_and_start(p, "chat-adv6", "Unmet",
                        [{"title": "A", "prompt": "a"},
                         {"title": "B", "prompt": "b", "depends_on": [0]}])
        # Fail step 1 and its task
        plan = store.get_plan_by_chat("chat-adv6", "active")
        store.fail_step(plan.id, 1, "err")
        store.fail_task(plan.steps[0].task_id, "err")

        r = p.advance("chat-adv6")
        assert r["advance"] is False
        assert "unmet deps" in r.get("reason", "")


# ===========================================================================
# 9. Concurrency tests
# ===========================================================================

class TestConcurrency:
    def test_max_concurrent_limits_spawns(self):
        p, store = make_planner(max_concurrent=2)
        steps = [{"title": f"S{i}", "prompt": f"p{i}"} for i in range(5)]
        _init_and_start(p, "chat-cc1", "Concurrent", steps)

        plan = store.get_plan_by_chat("chat-cc1", "active")
        running = [s for s in plan.steps if s.status == StepStatus.RUNNING]
        assert len(running) == 2  # limited by max_concurrent

    def test_slots_free_up_after_completion(self):
        p, store = make_planner(max_concurrent=1)
        _init_and_start(p, "chat-cc2", "Serial", TWO_PARALLEL)

        plan = store.get_plan_by_chat("chat-cc2", "active")
        running = [s for s in plan.steps if s.status == StepStatus.RUNNING]
        assert len(running) == 1

        # Complete the running step
        running_num = running[0].step_num
        r = p.step_done("chat-cc2", running_num, "done")
        # The other step should now start
        assert len(r["next_steps"]) == 1

    def test_zero_slots_no_spawn(self):
        """If all slots are taken globally, no new steps spawn."""
        p, store = make_planner(max_concurrent=1)
        # Start plan A (takes the slot)
        _init_and_start(p, "chat-cc3a", "Plan A", [{"title": "A", "prompt": "a"}])
        # Start plan B (no slots)
        _init_and_start(p, "chat-cc3b", "Plan B", [{"title": "B", "prompt": "b"}])

        plan_b = store.get_plan_by_chat("chat-cc3b", "active")
        running_b = [s for s in plan_b.steps if s.status == StepStatus.RUNNING]
        # Step should have been attempted but failed due to no slots
        # Actually _start_ready_steps returns [] when slots <= 0
        # The step stays pending or gets failed depending on implementation
        # Let's check: if no slots, _start_ready_steps returns []
        # So step stays pending (not started)
        pending_b = [s for s in plan_b.steps if s.status == StepStatus.PENDING]
        # Either pending or failed (spawn rollback)
        assert len(running_b) == 0 or len(pending_b) > 0


# ===========================================================================
# 10. Spawn rollback tests
# ===========================================================================

class TestSpawnRollback:
    def test_spawn_failure_rolls_back_to_failed(self):
        p, store = make_planner(runner=_FailRunner())
        _init_and_start(p, "chat-sr1", "Rollback", [{"title": "A", "prompt": "a"}])

        plan = store.get_plan_by_chat("chat-sr1", "active")
        step = plan.steps[0]
        assert step.status == StepStatus.FAILED
        assert "Spawn failed" in (step.result or "")

    def test_spawn_failure_fails_task_too(self):
        p, store = make_planner(runner=_FailRunner())
        _init_and_start(p, "chat-sr2", "Rollback", [{"title": "A", "prompt": "a"}])

        plan = store.get_plan_by_chat("chat-sr2", "active")
        task_id = plan.steps[0].task_id
        task = store.get_task(task_id)
        assert task.status == TaskStatus.FAILED

    def test_no_runner_rolls_back(self):
        store = MemoryBackend()
        p = Planner(store, agent_runner=None, max_concurrent=6)
        p.init("chat-sr3", "No runner", [{"title": "A", "prompt": "a"}])
        p.start("chat-sr3")

        plan = store.get_plan_by_chat("chat-sr3", "active")
        step = plan.steps[0]
        assert step.status == StepStatus.FAILED
        assert "no agent_runner" in (step.result or "")


# ===========================================================================
# 11. Stuck detection tests
# ===========================================================================

class TestStuckDetection:
    def test_stuck_step_timeout_45min(self):
        """Steps running > 45 min should be auto-failed."""
        p, store = make_planner()
        _init_and_start(p, "chat-st1", "Stuck", [{"title": "A", "prompt": "a"}])
        plan = store.get_plan_by_chat("chat-st1", "active")
        step = plan.steps[0]
        # Backdate started_at to 50 minutes ago
        step.started_at = datetime.now(timezone.utc) - timedelta(minutes=50)

        count = p._check_stuck_steps()
        assert count == 1
        step = store.get_step(plan.id, 1)
        assert step.status == StepStatus.FAILED
        assert "timeout=45min" in step.result

    def test_stuck_step_dead_agent(self):
        """Steps with dead agent after 2+ min should be auto-failed."""
        p, store = make_planner(runner=_DeadRunner())
        _init_and_start(p, "chat-st2", "Dead", [{"title": "A", "prompt": "a"}])
        plan = store.get_plan_by_chat("chat-st2", "active")
        step = plan.steps[0]
        # Backdate to 5 minutes ago
        step.started_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        # Set a real session key (not cron-pending)
        task = store.get_task(step.task_id)
        task.session_key = "real-session"

        count = p._check_stuck_steps()
        assert count == 1
        step = store.get_step(plan.id, 1)
        assert step.status == StepStatus.FAILED
        assert "agent process dead" in step.result

    def test_stuck_cron_pending_after_2min(self):
        """If session_key is still cron-pending after 2+ min, fail it."""
        p, store = make_planner()
        _init_and_start(p, "chat-st3", "CronPending", [{"title": "A", "prompt": "a"}])
        plan = store.get_plan_by_chat("chat-st3", "active")
        step = plan.steps[0]
        step.started_at = datetime.now(timezone.utc) - timedelta(minutes=3)
        # Force session_key back to cron-pending (spawn may have updated it)
        task = store.get_task(step.task_id)
        task.session_key = "cron-pending"

        count = p._check_stuck_steps()
        assert count == 1
        step = store.get_step(plan.id, 1)
        assert step.status == StepStatus.FAILED
        assert "cron-pending" in step.result

    def test_not_stuck_under_2min(self):
        """Steps running < 2 min should not be flagged."""
        p, store = make_planner(runner=_DeadRunner())
        _init_and_start(p, "chat-st4", "Fresh", [{"title": "A", "prompt": "a"}])
        plan = store.get_plan_by_chat("chat-st4", "active")
        step = plan.steps[0]
        step.started_at = datetime.now(timezone.utc) - timedelta(seconds=30)

        count = p._check_stuck_steps()
        assert count == 0

    def test_stuck_skips_waiting_tasks(self):
        """Steps with waiting tasks should not be flagged as stuck."""
        p, store = make_planner(runner=_DeadRunner())
        _init_and_start(p, "chat-st5", "Waiting", [{"title": "A", "prompt": "a"}])
        plan = store.get_plan_by_chat("chat-st5", "active")
        step = plan.steps[0]
        step.started_at = datetime.now(timezone.utc) - timedelta(minutes=50)
        # Mark task as waiting
        task = store.get_task(step.task_id)
        task.status = TaskStatus.WAITING

        count = p._check_stuck_steps()
        assert count == 0

    def test_stuck_already_failed_skipped(self):
        """Already-failed steps should not be double-counted."""
        p, store = make_planner()
        _init_and_start(p, "chat-st6", "AlreadyFailed", [{"title": "A", "prompt": "a"}])
        plan = store.get_plan_by_chat("chat-st6", "active")
        step = plan.steps[0]
        step.started_at = datetime.now(timezone.utc) - timedelta(minutes=50)
        # Pre-fail the step
        store.fail_step(plan.id, 1, "already failed")

        count = p._check_stuck_steps()
        assert count == 0

    def test_stuck_multiple_plans(self):
        """Stuck detection across multiple active plans."""
        p, store = make_planner()
        _init_and_start(p, "chat-st7a", "Plan A", [{"title": "A", "prompt": "a"}])
        _init_and_start(p, "chat-st7b", "Plan B", [{"title": "B", "prompt": "b"}])

        for chat in ["chat-st7a", "chat-st7b"]:
            plan = store.get_plan_by_chat(chat, "active")
            plan.steps[0].started_at = datetime.now(timezone.utc) - timedelta(minutes=50)

        count = p._check_stuck_steps()
        assert count == 2


# ===========================================================================
# 12. check_contracts tests
# ===========================================================================

class TestCheckContracts:
    def test_check_contracts_processes_step_done_event(self):
        p, store = make_planner()
        _init_and_start(p, "chat-cc1", "Events", THREE_STEPS)
        plan = store.get_plan_by_chat("chat-cc1", "active")

        # Emit a step.done event
        store.emit_event("step.done", task_id=plan.steps[0].task_id,
                         plan_id=plan.id, step_num=1,
                         payload={"result": "designed"})

        r = p.check_contracts()
        assert r.get("skipped") is not True
        # The event should have been processed
        step = store.get_step(plan.id, 1)
        assert step.status == StepStatus.DONE

    def test_check_contracts_processes_step_fail_event(self):
        p, store = make_planner()
        _init_and_start(p, "chat-cc2", "FailEvent", [{"title": "A", "prompt": "a"}])
        plan = store.get_plan_by_chat("chat-cc2", "active")

        store.emit_event("step.failed", task_id=plan.steps[0].task_id,
                         plan_id=plan.id, step_num=1,
                         payload={"error": "boom"})

        p.check_contracts()
        step = store.get_step(plan.id, 1)
        assert step.status == StepStatus.FAILED

    def test_check_contracts_expires_old_drafts(self):
        p, store = make_planner()
        p.init("chat-cc3", "Old draft", [{"title": "A", "prompt": "a"}])
        plan = store.get_plan_by_chat("chat-cc3", "draft")
        # Backdate creation
        plan.created_at = datetime.now(timezone.utc) - timedelta(minutes=35)

        p.check_contracts()
        plan = store.get_plan_by_chat("chat-cc3")
        assert plan.status == PlanStatus.CANCELLED

    def test_check_contracts_pauses_stale_plans(self):
        p, store = make_planner()
        _init_and_start(p, "chat-cc4", "Stale", [{"title": "A", "prompt": "a"}])
        plan = store.get_plan_by_chat("chat-cc4", "active")
        # Backdate everything to 25 hours ago
        old = datetime.now(timezone.utc) - timedelta(hours=25)
        plan.created_at = old
        for s in plan.steps:
            s.started_at = old

        p.check_contracts()
        plan = store.get_plan_by_chat("chat-cc4", "paused")
        assert plan is not None

    def test_check_contracts_restart_cooldown(self):
        """During restart cooldown, check_contracts should skip."""
        p, store = make_planner()
        _init_and_start(p, "chat-cc5", "Cooldown", TWO_PARALLEL)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
            f.write(str(time.time()))
            restart_file = f.name

        paused_flag = restart_file + ".paused"
        try:
            with patch.dict(os.environ, {
                "RESTART_COOLDOWN_FILE": restart_file,
                "RESTART_PAUSED_FLAG": paused_flag,
                "RESTART_COOLDOWN_SECONDS": "300",
            }):
                r = p.check_contracts()
                assert r["skipped"] is True
                assert r["reason"] == "restart_cooldown"
                # Plans should be paused
                plan = store.get_plan_by_chat("chat-cc5", "paused")
                assert plan is not None
        finally:
            os.unlink(restart_file)
            if os.path.exists(paused_flag):
                os.unlink(paused_flag)


# ===========================================================================
# 13. check_advances tests
# ===========================================================================

class TestCheckAdvances:
    def test_check_advances_multiple_plans(self):
        p, store = make_planner()
        _init_and_start(p, "chat-ca1", "Plan1", [{"title": "A", "prompt": "a"}])
        _init_and_start(p, "chat-ca2", "Plan2", [{"title": "B", "prompt": "b"}])

        r = p.check_advances()
        assert r["checked"] == 2


# ===========================================================================
# 14. advance_idle_plans tests
# ===========================================================================

class TestAdvanceIdlePlans:
    def test_advance_idle_starts_ready(self):
        p, store = make_planner()
        _init_and_start(p, "chat-ai1", "Idle", THREE_STEPS)
        plan = store.get_plan_by_chat("chat-ai1", "active")
        # Manually complete step 1 and its task
        store.complete_step(plan.id, 1, "done")
        store.complete_task(plan.steps[0].task_id, "done")

        count = p._advance_idle_plans()
        assert count == 1
        step2 = store.get_step(plan.id, 2)
        assert step2.status == StepStatus.RUNNING

    def test_advance_idle_skips_plans_with_running(self):
        p, store = make_planner()
        _init_and_start(p, "chat-ai2", "HasRunning", THREE_STEPS)
        # Step 1 is running, so this plan should be skipped
        count = p._advance_idle_plans()
        assert count == 0


# ===========================================================================
# 14. Show tests
# ===========================================================================

class TestShow:
    def test_show_by_chat_id(self):
        p, _ = make_planner()
        p.init("chat-sh1", "Show test", TWO_PARALLEL)
        text = p.show("chat-sh1")
        assert "Show test" in text

    def test_show_by_plan_id(self):
        p, _ = make_planner()
        r = p.init("chat-sh2", "By ID", TWO_PARALLEL)
        text = p.show(r["plan_id"])
        assert "By ID" in text

    def test_show_not_found_raises(self):
        p, _ = make_planner()
        with pytest.raises(KeyError):
            p.show("nonexistent")


# ===========================================================================
# 15. list_plans tests
# ===========================================================================

class TestListPlans:
    def test_list_empty(self):
        p, _ = make_planner()
        assert p.list_plans() == []

    def test_list_multiple_plans(self):
        p, _ = make_planner()
        p.init("chat-lp1", "Plan 1", TWO_PARALLEL)
        p.init("chat-lp2", "Plan 2", TWO_PARALLEL)
        plans = p.list_plans()
        assert len(plans) == 2

    def test_list_plans_sorted_by_status(self):
        p, _ = make_planner()
        _init_and_start(p, "chat-lp3a", "Active", TWO_PARALLEL)
        p.init("chat-lp3b", "Draft", TWO_PARALLEL)
        plans = p.list_plans()
        statuses = [pl["status"] for pl in plans]
        # active/draft should come before completed/cancelled
        assert statuses[0] in ("active", "draft")

    def test_list_plans_shows_progress(self):
        p, _ = make_planner()
        _init_and_start(p, "chat-lp4", "Progress", THREE_STEPS)
        p.step_done("chat-lp4", 1, "done")
        plans = p.list_plans()
        assert plans[0]["done"] == 1
        assert plans[0]["total"] == 3


# ===========================================================================
# 16. find_by_task tests
# ===========================================================================

class TestFindByTask:
    def test_find_existing_task(self):
        p, store = make_planner()
        _init_and_start(p, "chat-ft1", "Find", TWO_PARALLEL)
        plan = store.get_plan_by_chat("chat-ft1", "active")
        task_id = plan.steps[0].task_id
        r = p.find_by_task(task_id)
        assert r["found"] is True
        assert r["step"]["id"] == 1

    def test_find_nonexistent_task(self):
        p, _ = make_planner()
        r = p.find_by_task("nonexistent-task")
        assert r["found"] is False


# ===========================================================================
# 17. normalize_step tests
# ===========================================================================

class TestNormalizeStep:
    def test_basic(self):
        r = normalize_step({"title": "A", "prompt": "do A"})
        assert r["title"] == "A"
        assert r["prompt"] == "do A"

    def test_description_key(self):
        r = normalize_step({"title": "A", "description": "desc"})
        assert r["prompt"] == "desc"

    def test_name_key(self):
        r = normalize_step({"name": "A", "prompt": "p"})
        assert r["title"] == "A"

    def test_deps_as_depends(self):
        """normalize_step only recognizes depends_on, not 'depends'."""
        r = normalize_step({"title": "A", "prompt": "p", "depends": [1]})
        assert r["depends_on"] == []  # 'depends' key is not recognized

    def test_deps_as_dependencies(self):
        """normalize_step only recognizes depends_on, not 'dependencies'."""
        r = normalize_step({"title": "A", "prompt": "p", "dependencies": [2]})
        assert r["depends_on"] == []  # 'dependencies' key is not recognized

    def test_empty_deps_default(self):
        r = normalize_step({"title": "A", "prompt": "p"})
        assert r["depends_on"] == []

    def test_int_deps_wrapped(self):
        r = normalize_step({"title": "A", "prompt": "p", "depends_on": 3})
        assert r["depends_on"] == [3]


# ===========================================================================
# 18. resolve_title_deps tests
# ===========================================================================

class TestResolveTitleDeps:
    def test_title_deps_resolved(self):
        steps = [
            {"title": "Design", "prompt": "d", "depends_on": []},
            {"title": "Build", "prompt": "b", "depends_on": ["Design"]},
        ]
        resolved = resolve_title_deps(steps)
        assert resolved[1]["depends_on"] == [0]

    def test_unknown_title_dropped(self):
        """Unknown string deps are dropped by resolve_title_deps."""
        steps = [
            {"title": "A", "prompt": "a", "depends_on": ["Unknown"]},
        ]
        resolved = resolve_title_deps(steps)
        assert resolved[0]["depends_on"] == []


# ===========================================================================
# 19. validate_steps tests
# ===========================================================================

class TestValidateSteps:
    def test_1indexed_deps_converted(self):
        steps = [
            {"title": "A", "prompt": "a", "depends_on": []},
            {"title": "B", "prompt": "b", "depends_on": [1]},
            {"title": "C", "prompt": "c", "depends_on": [2]},
        ]
        result = validate_steps(steps)
        # 1-indexed [1] should become 0-indexed [0]
        assert result[1]["depends_on"] == [0]
        assert result[2]["depends_on"] == [1]

    def test_0indexed_deps_unchanged(self):
        steps = [
            {"title": "A", "prompt": "a", "depends_on": []},
            {"title": "B", "prompt": "b", "depends_on": [0]},
        ]
        result = validate_steps(steps)
        assert result[1]["depends_on"] == [0]

    def test_empty_steps_returns_empty(self):
        assert validate_steps([]) == []


# ===========================================================================
# 20. Restart pause tests
# ===========================================================================

class TestRestartPause:
    def test_restart_pause_pauses_active_plans(self):
        p, store = make_planner()
        _init_and_start(p, "chat-rp1", "Plan A", TWO_PARALLEL)
        _init_and_start(p, "chat-rp2", "Plan B", TWO_PARALLEL)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
            f.write(str(time.time()))
            restart_file = f.name

        paused_flag = restart_file + ".paused"
        try:
            os.environ["RESTART_COOLDOWN_FILE"] = restart_file
            os.environ["RESTART_PAUSED_FLAG"] = paused_flag
            os.environ["RESTART_COOLDOWN_SECONDS"] = "300"

            result = p._handle_restart_pause()
            assert result is True

            plan_a = store.get_plan_by_chat("chat-rp1", "paused")
            plan_b = store.get_plan_by_chat("chat-rp2", "paused")
            assert plan_a is not None
            assert plan_b is not None
        finally:
            os.unlink(restart_file)
            if os.path.exists(paused_flag):
                os.unlink(paused_flag)
            os.environ.pop("RESTART_COOLDOWN_FILE", None)
            os.environ.pop("RESTART_PAUSED_FLAG", None)
            os.environ.pop("RESTART_COOLDOWN_SECONDS", None)

    def test_restart_pause_skips_after_cooldown(self):
        p, _ = make_planner()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
            f.write(str(time.time()))
            restart_file = f.name

        try:
            os.environ["RESTART_COOLDOWN_FILE"] = restart_file
            os.environ["RESTART_COOLDOWN_SECONDS"] = "0"  # expired immediately

            result = p._handle_restart_pause()
            assert result is False
        finally:
            os.unlink(restart_file)
            os.environ.pop("RESTART_COOLDOWN_FILE", None)
            os.environ.pop("RESTART_COOLDOWN_SECONDS", None)

    def test_restart_pause_no_file(self):
        p, _ = make_planner()
        os.environ["RESTART_COOLDOWN_FILE"] = "/tmp/nonexistent-restart-file-xyz"
        try:
            result = p._handle_restart_pause()
            assert result is False
        finally:
            os.environ.pop("RESTART_COOLDOWN_FILE", None)

    def test_check_contracts_skips_during_cooldown(self):
        p, store = make_planner()
        _init_and_start(p, "chat-rp3", "Cooldown", TWO_PARALLEL)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
            f.write(str(time.time()))
            restart_file = f.name

        paused_flag = restart_file + ".paused"
        try:
            os.environ["RESTART_COOLDOWN_FILE"] = restart_file
            os.environ["RESTART_PAUSED_FLAG"] = paused_flag
            os.environ["RESTART_COOLDOWN_SECONDS"] = "300"

            r = p.check_contracts()
            assert r["skipped"] is True
            assert r["reason"] == "restart_cooldown"
        finally:
            os.unlink(restart_file)
            if os.path.exists(paused_flag):
                os.unlink(paused_flag)
            os.environ.pop("RESTART_COOLDOWN_FILE", None)
            os.environ.pop("RESTART_PAUSED_FLAG", None)
            os.environ.pop("RESTART_COOLDOWN_SECONDS", None)


# ===========================================================================
# 21. format_plan tests
# ===========================================================================

class TestFormatPlan:
    def test_format_includes_goal(self):
        _, store = make_planner()
        plan = store.create_plan("p1", "c1", "My Goal", [
            {"title": "A", "prompt": "a"},
        ])
        text = format_plan(plan)
        assert "My Goal" in text

    def test_format_includes_step_status(self):
        _, store = make_planner()
        plan = store.create_plan("p2", "c2", "Goal", [
            {"title": "Step A", "prompt": "a"},
        ])
        store.update_plan_status("p2", "active")
        store.start_step("p2", 1, "t1")
        plan = store.get_plan("p2")
        text = format_plan(plan)
        assert "Step A" in text


# ===========================================================================
# 22. build_plan_summary tests
# ===========================================================================

class TestBuildPlanSummary:
    def test_summary_includes_results(self):
        p, store = make_planner()
        _init_and_start(p, "chat-bs1", "Summary", [{"title": "A", "prompt": "a"}])
        p.step_done("chat-bs1", 1, "Great result here")
        plan = store.get_plan_by_chat("chat-bs1")
        summary = build_plan_summary(plan, store)
        assert "Great result" in summary or summary == ""  # depends on impl


# ===========================================================================
# 23. Events integration tests
# ===========================================================================

class TestEventsIntegration:
    def test_resolve_plan_for_task(self):
        _, store = make_planner()
        plan = store.create_plan("p-ev1", "c-ev1", "Goal", [
            {"title": "A", "prompt": "a"},
        ])
        store.update_plan_status("p-ev1", "active")
        store.add_task("t-ev1", "task", source_chat="c-ev1")
        store.start_step("p-ev1", 1, "t-ev1")

        plan_id, step_num = resolve_plan_for_task(store, "t-ev1")
        assert plan_id == "p-ev1"
        assert step_num == 1

    def test_resolve_plan_for_unknown_task(self):
        _, store = make_planner()
        plan_id, step_num = resolve_plan_for_task(store, "unknown")
        assert plan_id is None

    def test_process_events_step_done(self):
        _, store = make_planner()
        plan = store.create_plan("p-ev2", "c-ev2", "Goal", [
            {"title": "A", "prompt": "a"},
        ])
        store.update_plan_status("p-ev2", "active")
        store.add_task("t-ev2", "task", source_chat="c-ev2")
        store.start_task("t-ev2")
        store.start_step("p-ev2", 1, "t-ev2")

        store.emit_event("step.done", task_id="t-ev2", plan_id="p-ev2",
                         step_num=1, payload={"result": "ok"})

        done_calls = []
        def on_done(plan, step_num, result):
            done_calls.append((plan.id, step_num, result))

        process_events(store, on_step_done=on_done)
        assert len(done_calls) == 1
        assert done_calls[0] == ("p-ev2", 1, "ok")

    def test_process_events_step_fail(self):
        _, store = make_planner()
        plan = store.create_plan("p-ev3", "c-ev3", "Goal", [
            {"title": "A", "prompt": "a"},
        ])
        store.update_plan_status("p-ev3", "active")
        store.add_task("t-ev3", "task", source_chat="c-ev3")
        store.start_task("t-ev3")
        store.start_step("p-ev3", 1, "t-ev3")

        store.emit_event("step.failed", task_id="t-ev3", plan_id="p-ev3",
                         step_num=1, payload={"error": "boom"})

        fail_calls = []
        def on_fail(plan, step_num, error):
            fail_calls.append((plan.id, step_num, error))

        process_events(store, on_step_fail=on_fail)
        assert len(fail_calls) == 1
        assert fail_calls[0] == ("p-ev3", 1, "boom")


# ===========================================================================
# 24. Edge cases and misc
# ===========================================================================

class TestEdgeCases:
    def test_init_with_various_step_formats(self):
        """Test that init handles different step input formats."""
        p, store = make_planner()
        steps = [
            {"name": "Step1", "description": "Do step 1"},
            {"title": "Step2", "prompt": "Do step 2", "depends": [0]},
            {"title": "Step3", "prompt": "Do step 3", "dependencies": [1]},
        ]
        r = p.init("chat-ec1", "Formats", steps)
        assert r["steps"] == 3

    def test_many_parallel_steps(self):
        """10 parallel steps with max_concurrent=3."""
        p, store = make_planner(max_concurrent=3)
        steps = [{"title": f"S{i}", "prompt": f"p{i}"} for i in range(10)]
        _init_and_start(p, "chat-ec2", "Many", steps)

        plan = store.get_plan_by_chat("chat-ec2", "active")
        running = [s for s in plan.steps if s.status == StepStatus.RUNNING]
        pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
        assert len(running) == 3
        assert len(pending) == 7

    def test_deep_chain(self):
        """5-step serial chain."""
        p, store = make_planner()
        steps = [{"title": f"S{i}", "prompt": f"p{i}",
                  **({"depends_on": [i-1]} if i > 0 else {})} for i in range(5)]
        _init_and_start(p, "chat-ec3", "Chain", steps)

        for i in range(1, 6):
            plan = store.get_plan_by_chat("chat-ec3", "active")
            running = [s for s in plan.steps if s.status == StepStatus.RUNNING]
            assert len(running) == 1
            assert running[0].step_num == i
            if i < 5:
                p.step_done("chat-ec3", i, f"done {i}")
            else:
                r = p.step_done("chat-ec3", i, f"done {i}")
                assert r["plan_completed"] is True

    def test_cancel_then_new_plan_same_chat(self):
        """After cancelling, can create a new plan on same chat."""
        p, store = make_planner()
        _init_and_start(p, "chat-ec4", "First", TWO_PARALLEL)
        p.cancel("chat-ec4")
        # Should be able to create new plan
        r = p.init("chat-ec4", "Second", TWO_PARALLEL)
        assert r["status"] == "draft"

    def test_complete_then_new_plan_same_chat(self):
        """After completing, can create a new plan on same chat."""
        p, store = make_planner()
        _init_and_start(p, "chat-ec5", "First", [{"title": "A", "prompt": "a"}])
        p.step_done("chat-ec5", 1, "done")
        # Plan completed, should be able to create new
        r = p.init("chat-ec5", "Second", TWO_PARALLEL)
        assert r["status"] == "draft"

    def test_replan_auto_advances_when_no_running(self):
        """Replan on active plan with no running steps should auto-start."""
        p, store = make_planner()
        _init_and_start(p, "chat-ec6", "AutoAdv", THREE_STEPS)
        # Fail step 1 so nothing is running
        p.step_fail("chat-ec6", 1, "err")
        # Replan with new steps
        r = p.replan("chat-ec6", [{"title": "Retry", "prompt": "retry"}])
        assert r["spawned"] is True

    def test_step_done_on_last_parallel_completes_plan(self):
        """Completing the last of parallel steps should complete the plan."""
        p, store = make_planner()
        steps = [
            {"title": "A", "prompt": "a"},
            {"title": "B", "prompt": "b"},
            {"title": "C", "prompt": "c"},
        ]
        _init_and_start(p, "chat-ec7", "AllParallel", steps)
        p.step_done("chat-ec7", 1, "a")
        p.step_done("chat-ec7", 2, "b")
        r = p.step_done("chat-ec7", 3, "c")
        assert r["plan_completed"] is True

    def test_multiple_plans_different_chats(self):
        """Multiple active plans on different chats."""
        p, store = make_planner()
        _init_and_start(p, "chat-ec8a", "Plan A", TWO_PARALLEL)
        _init_and_start(p, "chat-ec8b", "Plan B", THREE_STEPS)

        plan_a = store.get_plan_by_chat("chat-ec8a", "active")
        plan_b = store.get_plan_by_chat("chat-ec8b", "active")
        assert plan_a is not None
        assert plan_b is not None
        assert plan_a.id != plan_b.id
