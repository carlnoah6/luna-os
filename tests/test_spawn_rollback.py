"""Tests for spawn failure rollback and cron-pending stuck detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from luna_os.agents.base import AgentRunner
from luna_os.planner import Planner
from tests.memory_store import MemoryBackend


class FakeRunner(AgentRunner):
    """Agent runner that records calls and can be told to fail."""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.spawned: list[tuple[str, str]] = []

    def spawn(
        self, task_id: str, prompt: str,
        session_label: str = "", reply_chat_id: str = "",
    ) -> str:
        if self._fail:
            raise RuntimeError("simulated spawn failure")
        session_id = session_label or f"task-{task_id}"
        self.spawned.append((task_id, session_id))
        return session_id

    def is_running(self, session_key: str) -> bool:
        return any(sid == session_key for _, sid in self.spawned)


def _make_planner(
    runner: AgentRunner | None = None,
) -> tuple[Planner, MemoryBackend]:
    store = MemoryBackend()
    planner = Planner(store, agent_runner=runner, max_concurrent=6)
    return planner, store


class TestSpawnRollback:
    """_start_ready_steps must roll back task+step on spawn failure."""

    def test_spawn_success_updates_session_key(self):
        runner = FakeRunner()
        planner, store = _make_planner(runner)

        store.create_plan(
            "p1", "chat-1", "Goal", [{"title": "Step A", "prompt": "do it"}]
        )
        store.update_plan_status("p1", "active")

        ready = store.ready_steps("p1")
        results = planner._start_ready_steps(store.get_plan("p1"), ready)

        assert len(results) == 1
        step_num, ok = results[0]
        assert ok is True

        # Task session_key should be updated from cron-pending to actual value
        step = store.get_step("p1", step_num)
        task = store.get_task(step.task_id)
        assert task.session_key != "cron-pending"
        assert task.status.value == "running"
        assert step.status.value == "running"

    def test_spawn_failure_rolls_back_to_failed(self):
        runner = FakeRunner(fail=True)
        planner, store = _make_planner(runner)

        store.create_plan(
            "p1", "chat-1", "Goal", [{"title": "Step A", "prompt": "do it"}]
        )
        store.update_plan_status("p1", "active")

        ready = store.ready_steps("p1")
        results = planner._start_ready_steps(store.get_plan("p1"), ready)

        assert len(results) == 1
        step_num, ok = results[0]
        assert ok is False

        # Both task and step should be failed, not stuck in running
        step = store.get_step("p1", step_num)
        task = store.get_task(step.task_id)
        assert task.status.value == "failed"
        assert step.status.value == "failed"
        assert "Spawn failed" in (task.result or "")

    def test_no_agent_runner_rolls_back(self):
        planner, store = _make_planner(runner=None)

        store.create_plan(
            "p1", "chat-1", "Goal", [{"title": "Step A", "prompt": "do it"}]
        )
        store.update_plan_status("p1", "active")

        ready = store.ready_steps("p1")
        results = planner._start_ready_steps(store.get_plan("p1"), ready)

        step_num, ok = results[0]
        assert ok is False

        step = store.get_step("p1", step_num)
        task = store.get_task(step.task_id)
        assert task.status.value == "failed"
        assert step.status.value == "failed"
        assert "no agent_runner" in (task.result or "")


class TestStuckDetection:
    """_check_stuck_steps must detect cron-pending as stuck."""

    def test_cron_pending_detected_after_2_min(self):
        runner = FakeRunner()
        planner, store = _make_planner(runner)

        store.create_plan(
            "p1", "chat-1", "Goal", [{"title": "Step A", "prompt": "do it"}]
        )
        store.update_plan_status("p1", "active")

        # Manually create a stuck step (simulating the old bug)
        task_id = store.next_task_id()
        store.add_task(task_id, "test task", source_chat="chat-1")
        store.start_task(task_id, "cron-pending")  # placeholder, never updated
        store.start_step("p1", 1, task_id)

        # Backdate started_at to 3 minutes ago
        step = store.get_step("p1", 1)
        step.started_at = datetime.now(UTC) - timedelta(minutes=3)
        task = store.get_task(task_id)
        task.started_at = datetime.now(UTC) - timedelta(minutes=3)

        stuck = planner._check_stuck_steps()
        assert stuck == 1

        # Step and task should now be failed
        step = store.get_step("p1", 1)
        task = store.get_task(task_id)
        assert step.status.value == "failed"
        assert task.status.value == "failed"
        assert "cron-pending" in (step.result or "")

    def test_real_session_key_not_false_positive(self):
        runner = FakeRunner()
        planner, store = _make_planner(runner)

        store.create_plan(
            "p1", "chat-1", "Goal", [{"title": "Step A", "prompt": "do it"}]
        )
        store.update_plan_status("p1", "active")

        task_id = store.next_task_id()
        store.add_task(task_id, "test task", source_chat="chat-1")
        store.start_task(task_id, "task-abc12345")  # real session key
        store.start_step("p1", 1, task_id)

        # Simulate that the runner knows about this session
        runner.spawned.append((task_id, "task-abc12345"))

        # Backdate to 3 minutes ago
        step = store.get_step("p1", 1)
        step.started_at = datetime.now(UTC) - timedelta(minutes=3)
        task = store.get_task(task_id)
        task.started_at = datetime.now(UTC) - timedelta(minutes=3)

        stuck = planner._check_stuck_steps()
        # Should NOT be detected as stuck because is_running returns True
        assert stuck == 0
        step = store.get_step("p1", 1)
        assert step.status.value == "running"
