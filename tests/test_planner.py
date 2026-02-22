"""Tests for Plan creation, dependencies, ready_steps, and the Planner class."""

import pytest

from luna_os.agents.base import AgentRunner
from luna_os.planner import Planner, format_plan, normalize_step, validate_steps
from tests.memory_store import MemoryBackend


class _NoopRunner(AgentRunner):
    """Minimal agent runner that always succeeds (for tests that don't care)."""

    def spawn(
        self, task_id: str, prompt: str,
        session_label: str = "", reply_chat_id: str = "",
    ) -> str:
        return session_label or f"task-{task_id}"

    def is_running(self, session_key: str) -> bool:
        return True


def make_planner() -> tuple[Planner, MemoryBackend]:
    store = MemoryBackend()
    planner = Planner(store, agent_runner=_NoopRunner(), max_concurrent=6)
    return planner, store


class TestPlanCreation:
    def test_create_plan_with_steps(self):
        store = MemoryBackend()
        plan = store.create_plan(
            "plan-1",
            "chat-1",
            "Build something",
            [
                {"title": "Design", "prompt": "Design the thing"},
                {"title": "Implement", "prompt": "Build it", "depends_on": [0]},
                {"title": "Test", "prompt": "Test it", "depends_on": [1]},
            ],
        )
        assert plan.id == "plan-1"
        assert plan.goal == "Build something"
        assert len(plan.steps) == 3
        # Step numbering is 1-based
        assert plan.steps[0].step_num == 1
        assert plan.steps[1].step_num == 2
        assert plan.steps[2].step_num == 3
        # depends_on gets shifted from 0-based to 1-based
        assert plan.steps[0].depends_on == []
        assert plan.steps[1].depends_on == [1]
        assert plan.steps[2].depends_on == [2]

    def test_get_plan_by_chat(self):
        store = MemoryBackend()
        store.create_plan("plan-1", "chat-1", "Goal", [{"title": "Step 1"}])
        plan = store.get_plan_by_chat("chat-1")
        assert plan is not None
        assert plan.id == "plan-1"

    def test_get_plan_by_chat_with_status(self):
        store = MemoryBackend()
        store.create_plan("plan-1", "chat-1", "Goal", [{"title": "Step 1"}])
        # Draft by default
        plan = store.get_plan_by_chat("chat-1", status_filter="draft")
        assert plan is not None
        plan = store.get_plan_by_chat("chat-1", status_filter="active")
        assert plan is None


class TestReadySteps:
    def test_no_deps_all_ready(self):
        store = MemoryBackend()
        store.create_plan(
            "plan-1",
            "chat-1",
            "Goal",
            [
                {"title": "A"},
                {"title": "B"},
            ],
        )
        ready = store.ready_steps("plan-1")
        assert len(ready) == 2

    def test_deps_block_until_done(self):
        store = MemoryBackend()
        store.create_plan(
            "plan-1",
            "chat-1",
            "Goal",
            [
                {"title": "A"},
                {"title": "B", "depends_on": [0]},
            ],
        )
        ready = store.ready_steps("plan-1")
        assert len(ready) == 1
        assert ready[0].title == "A"

        # Complete step 1 (A)
        store.complete_step("plan-1", 1, "done")
        ready = store.ready_steps("plan-1")
        assert len(ready) == 1
        assert ready[0].title == "B"

    def test_diamond_deps(self):
        store = MemoryBackend()
        # A -> B, A -> C, B+C -> D
        store.create_plan(
            "plan-1",
            "chat-1",
            "Goal",
            [
                {"title": "A"},
                {"title": "B", "depends_on": [0]},
                {"title": "C", "depends_on": [0]},
                {"title": "D", "depends_on": [1, 2]},
            ],
        )
        # Only A is ready
        ready = store.ready_steps("plan-1")
        assert len(ready) == 1
        assert ready[0].title == "A"

        # Complete A -> B and C become ready
        store.complete_step("plan-1", 1, "done")
        ready = store.ready_steps("plan-1")
        assert len(ready) == 2
        titles = {s.title for s in ready}
        assert titles == {"B", "C"}

        # Complete B only -> D still blocked
        store.complete_step("plan-1", 2, "done")
        ready = store.ready_steps("plan-1")
        assert len(ready) == 1
        assert ready[0].title == "C"

        # Complete C -> D ready
        store.complete_step("plan-1", 3, "done")
        ready = store.ready_steps("plan-1")
        assert len(ready) == 1
        assert ready[0].title == "D"


class TestPlannerInit:
    def test_init_creates_draft(self):
        planner, store = make_planner()
        result = planner.init(
            "chat-1",
            "Build it",
            [
                {"title": "Design"},
                {"title": "Build"},
            ],
        )
        assert result["status"] == "draft"
        assert result["steps"] == 2
        plan = store.get_plan(result["plan_id"])
        assert plan is not None
        assert plan.status.value == "draft"

    def test_init_duplicate_active_raises(self):
        planner, store = make_planner()
        planner.init("chat-1", "First goal", [{"title": "Step"}])
        # Activate it
        plan = store.get_plan_by_chat("chat-1", status_filter="draft")
        store.update_plan_status(plan.id, "active")
        with pytest.raises(ValueError, match="Active plan already exists"):
            planner.init("chat-1", "Second goal", [{"title": "Step"}])

    def test_init_no_steps_raises(self):
        planner, store = make_planner()
        with pytest.raises(ValueError, match="No steps"):
            planner.init("chat-1", "Goal", [])


class TestPlannerStart:
    def test_start_activates_plan(self):
        planner, store = make_planner()
        planner.init("chat-1", "Goal", [{"title": "Step 1"}])
        result = planner.start("chat-1")
        assert result["started"] is True
        plan = store.get_plan(result["plan_id"])
        assert plan.status.value == "active"

    def test_start_no_draft_raises(self):
        planner, store = make_planner()
        with pytest.raises(KeyError):
            planner.start("nonexistent-chat")


class TestPlannerStepDone:
    def test_step_done_advances(self):
        planner, store = make_planner()
        planner.init(
            "chat-1",
            "Goal",
            [
                {"title": "A"},
                {"title": "B", "depends_on": [0]},
            ],
        )
        planner.start("chat-1")
        # Find running step
        plan = store.get_plan_by_chat("chat-1", status_filter="active")
        running = [s for s in plan.steps if s.status.value == "running"]
        assert len(running) == 1
        assert running[0].step_num == 1

        result = planner.step_done("chat-1", 1, "Design complete")
        assert result["step_done"] == 1

    def test_step_done_idempotent(self):
        planner, store = make_planner()
        planner.init("chat-1", "Goal", [{"title": "A"}, {"title": "B"}])
        planner.start("chat-1")
        planner.step_done("chat-1", 1, "Done")
        result = planner.step_done("chat-1", 1, "Done again")
        assert result.get("skipped") is True


class TestPlannerStepFail:
    def test_step_fail(self):
        planner, store = make_planner()
        planner.init("chat-1", "Goal", [{"title": "A"}])
        planner.start("chat-1")
        result = planner.step_fail("chat-1", 1, "Broke")
        assert result["step_failed"] == 1

    def test_step_fail_idempotent(self):
        planner, store = make_planner()
        planner.init("chat-1", "Goal", [{"title": "A"}])
        planner.start("chat-1")
        planner.step_fail("chat-1", 1, "Broke")
        result = planner.step_fail("chat-1", 1, "Broke again")
        assert result.get("step_already_failed") == 1


class TestPlannerCancel:
    def test_cancel_plan(self):
        planner, store = make_planner()
        planner.init("chat-1", "Goal", [{"title": "A"}, {"title": "B"}])
        planner.start("chat-1")
        result = planner.cancel("chat-1")
        assert result["cancelled"] is True
        plan = store.get_plan_by_chat("chat-1")
        assert plan.status.value == "cancelled"


class TestPlannerReplan:
    def test_replan_replaces_pending(self):
        planner, store = make_planner()
        planner.init(
            "chat-1",
            "Goal",
            [
                {"title": "A"},
                {"title": "B"},
                {"title": "C"},
            ],
        )
        planner.start("chat-1")
        # Step A is running, B and C are pending
        result = planner.replan("chat-1", [{"title": "New B"}, {"title": "New C"}])
        assert result["replanned"] is True
        assert result["new_steps"] == 2


class TestPlannerPauseResume:
    def test_pause_and_resume(self):
        planner, store = make_planner()
        planner.init("chat-1", "Goal", [{"title": "A"}])
        planner.start("chat-1")
        pause_result = planner.pause("chat-1")
        assert pause_result["paused"] is True

        plan = store.get_plan_by_chat("chat-1", status_filter="paused")
        assert plan is not None

        resume_result = planner.resume(plan.id)
        assert resume_result["resumed"] is True


class TestNormalizeStep:
    def test_normalize_basic(self):
        step = normalize_step({"title": "Do it", "prompt": "Details"})
        assert step["title"] == "Do it"
        assert step["prompt"] == "Details"
        assert step["depends_on"] == []

    def test_normalize_alt_keys(self):
        step = normalize_step({"desc": "Do it", "detail": "Details"})
        assert step["title"] == "Do it"
        assert step["prompt"] == "Details"

    def test_normalize_description_key(self):
        step = normalize_step({"description": "步骤一描述", "depends_on": []})
        assert step["title"] == "步骤一描述"
        assert step["prompt"] == "步骤一描述"

    def test_normalize_description_as_prompt_fallback(self):
        step = normalize_step({"title": "My Step", "description": "Detailed desc"})
        assert step["title"] == "My Step"
        assert step["prompt"] == "Detailed desc"

    def test_normalize_deps_int(self):
        step = normalize_step({"title": "X", "depends_on": 0})
        assert step["depends_on"] == [0]


class TestValidateSteps1Indexed:
    """validate_steps should auto-detect and convert 1-indexed depends_on."""

    def test_1indexed_deps_converted(self):
        steps = [
            {"title": "A", "depends_on": []},
            {"title": "B", "depends_on": [1]},
            {"title": "C", "depends_on": [2]},
            {"title": "D", "depends_on": [2]},
        ]
        result = validate_steps(steps)
        # After conversion: B depends on index 0, C/D depend on index 1
        assert result[0]["depends_on"] == []
        assert result[1]["depends_on"] == [0]
        assert result[2]["depends_on"] == [1]
        assert result[3]["depends_on"] == [1]

    def test_0indexed_deps_unchanged(self):
        steps = [
            {"title": "A", "depends_on": []},
            {"title": "B", "depends_on": [0]},
            {"title": "C", "depends_on": [1]},
        ]
        result = validate_steps(steps)
        assert result[1]["depends_on"] == [0]
        assert result[2]["depends_on"] == [1]

    def test_1indexed_start_only_step1(self):
        """Issue #18 repro: only Step 1 should start, not Step 2/3."""
        planner, store = make_planner()
        planner.init(
            "chat-1",
            "Goal",
            [
                {"title": "Step 1", "depends_on": []},
                {"title": "Step 2", "depends_on": [1]},
                {"title": "Step 3", "depends_on": [2]},
                {"title": "Step 4", "depends_on": [2]},
            ],
        )
        result = planner.start("chat-1")
        # Only Step 1 should be started (no deps)
        assert result["parallel_steps"] == [1]
        assert result["spawned"] == [True]


class TestFormatPlan:
    def test_format_plan_basic(self):
        store = MemoryBackend()
        plan = store.create_plan(
            "p-1",
            "chat-1",
            "My Goal",
            [
                {"title": "Step A"},
                {"title": "Step B"},
            ],
        )
        text = format_plan(plan)
        assert "My Goal" in text
        assert "Step A" in text
        assert "Step B" in text
