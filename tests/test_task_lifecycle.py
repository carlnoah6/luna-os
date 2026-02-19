"""Tests for the TaskManager class."""

import pytest

from luna_os.task_manager import TaskManager
from tests.memory_store import MemoryBackend


def make_tm() -> tuple[TaskManager, MemoryBackend]:
    store = MemoryBackend()
    tm = TaskManager(store, max_concurrent=3)
    return tm, store


class TestTaskManagerAdd:
    def test_add_task(self):
        tm, store = make_tm()
        task = tm.add("Do something")
        assert task.id is not None
        assert task.description == "Do something"
        assert task.status.value == "queued"

    def test_add_duplicate_raises(self):
        tm, store = make_tm()
        tm.add("Do something")
        with pytest.raises(ValueError, match="Duplicate"):
            tm.add("Do something")

    def test_add_with_deps(self):
        tm, store = make_tm()
        t1 = tm.add("First")
        t2 = tm.add("Second", depends_on=[t1.id])
        assert t2.depends_on == [t1.id]

    def test_add_cycle_raises(self):
        tm, store = make_tm()
        t1 = tm.add("First")
        t2 = tm.add("Second", depends_on=[t1.id])
        # Manually adjust deps to create a cycle scenario
        store.update_task(t1.id, depends_on=[t2.id])
        with pytest.raises(ValueError, match="cycle"):
            tm.add("Third", depends_on=[t1.id])

    def test_add_empty_desc_raises(self):
        tm, store = make_tm()
        with pytest.raises(ValueError, match="description"):
            tm.add("")


class TestTaskManagerLifecycle:
    def test_start_and_complete(self):
        tm, store = make_tm()
        task = tm.add("Work")
        started = tm.start(task.id, "session-1")
        assert started.status.value == "running"
        result = tm.complete(task.id, "All done", input_tokens=100)
        assert result["status"] == "done"
        # Check event was emitted
        events = store.poll_events()
        assert any(e.event_type == "step.done" for e in events)

    def test_start_full_queue_raises(self):
        tm, store = make_tm()
        for i in range(3):
            t = tm.add(f"Task {i}")
            tm.start(t.id)
        t4 = tm.add("Too many")
        with pytest.raises(RuntimeError, match="Queue full"):
            tm.start(t4.id)

    def test_fail_task(self):
        tm, store = make_tm()
        task = tm.add("Work")
        tm.start(task.id)
        result = tm.fail(task.id, "Broke")
        assert result["status"] == "failed"

    def test_cancel_task(self):
        tm, store = make_tm()
        task = tm.add("Work")
        result = tm.cancel(task.id)
        assert result["status"] == "cancelled"

    def test_show_task(self):
        tm, store = make_tm()
        task = tm.add("Work")
        d = tm.show(task.id)
        assert d["id"] == task.id
        assert d["description"] == "Work"

    def test_show_missing_raises(self):
        tm, store = make_tm()
        with pytest.raises(KeyError):
            tm.show("nope")

    def test_list_and_status(self):
        tm, store = make_tm()
        tm.add("A")
        tm.add("B")
        tasks = tm.list_tasks()
        assert len(tasks) == 2
        status = tm.status()
        assert status["total"] == 2
        assert status["counts"]["queued"] == 2

    def test_ready_and_active(self):
        tm, store = make_tm()
        t1 = tm.add("First")
        t2 = tm.add("Second", depends_on=[t1.id])
        ready = tm.ready()
        assert len(ready) == 1
        assert ready[0]["id"] == t1.id
        tm.start(t1.id)
        active = tm.active()
        assert len(active) == 1

    def test_set_session(self):
        tm, store = make_tm()
        task = tm.add("Work")
        result = tm.set_session(task.id, "new-session")
        assert result["session_key"] == "new-session"

    def test_health_check(self):
        tm, store = make_tm()
        task = tm.add("Stuck work")
        tm.start(task.id)
        # Can't easily simulate age, but ensure it runs without error
        result = tm.health_check()
        assert "stuck_failed" in result
        assert "active" in result

    def test_wait_and_respond(self):
        tm, store = make_tm()
        task = tm.add("Needs input")
        tm.start(task.id)
        wait_result = tm.wait(task.id, "confirm", "Proceed?", options=["Yes", "No"])
        assert wait_result["status"] == "waiting"

        t = store.get_task(task.id)
        assert t.status.value == "waiting"

        resp = tm.respond(task.id, "Yes")
        assert resp["status"] == "running"

    def test_respond_not_waiting_raises(self):
        tm, store = make_tm()
        task = tm.add("Work")
        with pytest.raises(ValueError, match="not waiting"):
            tm.respond(task.id, "whatever")
