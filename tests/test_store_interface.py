"""Tests for the StorageBackend interface using the MemoryBackend."""

from tests.memory_store import MemoryBackend


def make_store() -> MemoryBackend:
    return MemoryBackend()


class TestTaskCRUD:
    def test_add_and_get_task(self):
        store = make_store()
        task = store.add_task("t-1", "Do something")
        assert task.id == "t-1"
        assert task.description == "Do something"
        assert task.status.value == "queued"

        fetched = store.get_task("t-1")
        assert fetched is not None
        assert fetched.id == "t-1"

    def test_get_missing_task(self):
        store = make_store()
        assert store.get_task("nope") is None

    def test_list_tasks(self):
        store = make_store()
        store.add_task("t-1", "First")
        store.add_task("t-2", "Second")
        all_tasks = store.list_tasks()
        assert len(all_tasks) == 2

    def test_list_tasks_by_status(self):
        store = make_store()
        store.add_task("t-1", "First")
        store.add_task("t-2", "Second")
        store.start_task("t-1", "sess-1")
        assert len(store.list_tasks(status="running")) == 1
        assert len(store.list_tasks(status="queued")) == 1

    def test_update_task(self):
        store = make_store()
        store.add_task("t-1", "First")
        store.update_task("t-1", session_key="abc")
        task = store.get_task("t-1")
        assert task.session_key == "abc"

    def test_find_duplicate(self):
        store = make_store()
        store.add_task("t-1", "Do something")
        assert store.find_duplicate("Do something") == "t-1"
        assert store.find_duplicate("  do something  ") == "t-1"
        assert store.find_duplicate("Something else") is None

    def test_task_count_by_status(self):
        store = make_store()
        store.add_task("t-1", "A")
        store.add_task("t-2", "B")
        store.start_task("t-1", "s")
        counts = store.task_count_by_status()
        assert counts["running"] == 1
        assert counts["queued"] == 1


class TestTaskLifecycle:
    def test_start_task(self):
        store = make_store()
        store.add_task("t-1", "Work")
        store.start_task("t-1", "session-123")
        task = store.get_task("t-1")
        assert task.status.value == "running"
        assert task.session_key == "session-123"
        assert task.started_at is not None

    def test_complete_task(self):
        store = make_store()
        store.add_task("t-1", "Work")
        store.start_task("t-1")
        store.complete_task("t-1", "All done")
        task = store.get_task("t-1")
        assert task.status.value == "done"
        assert task.result == "All done"
        assert task.completed_at is not None

    def test_fail_task(self):
        store = make_store()
        store.add_task("t-1", "Work")
        store.start_task("t-1")
        store.fail_task("t-1", "Oops")
        task = store.get_task("t-1")
        assert task.status.value == "failed"
        assert task.result == "Oops"

    def test_cancel_task(self):
        store = make_store()
        store.add_task("t-1", "Work")
        store.cancel_task("t-1")
        task = store.get_task("t-1")
        assert task.status.value == "cancelled"

    def test_ready_tasks_no_deps(self):
        store = make_store()
        store.add_task("t-1", "No deps")
        ready = store.ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t-1"

    def test_ready_tasks_with_deps(self):
        store = make_store()
        store.add_task("t-1", "First")
        store.add_task("t-2", "Second", depends_on=["t-1"])
        # t-2 not ready because t-1 is not done
        ready = store.ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t-1"

        # Complete t-1, now t-2 should be ready
        store.complete_task("t-1", "done")
        ready = store.ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t-2"

    def test_active_tasks(self):
        store = make_store()
        store.add_task("t-1", "A")
        store.add_task("t-2", "B")
        store.start_task("t-1")
        active = store.active_tasks()
        assert len(active) == 1
        assert active[0].id == "t-1"

    def test_boost_priority(self):
        store = make_store()
        store.add_task("t-1", "Work", priority="normal")
        task = store.get_task("t-1")
        assert task.priority_value == 3
        store.boost_priority("t-1")
        task = store.get_task("t-1")
        assert task.priority_value == 2
        assert task.priority == "high"
        assert task.priority_boosted is True

    def test_update_task_cost(self):
        store = make_store()
        store.add_task("t-1", "Work")
        store.update_task_cost("t-1", input_tokens=100, output_tokens=50, cost_usd=0.01)
        task = store.get_task("t-1")
        assert task.input_tokens == 100
        assert task.output_tokens == 50
        assert task.cost_usd == 0.01
        # Test accumulation
        store.update_task_cost("t-1", input_tokens=200, output_tokens=100, cost_usd=0.02)
        task = store.get_task("t-1")
        assert task.input_tokens == 300
        assert task.output_tokens == 150

    def test_next_task_id_sequential(self):
        store = make_store()
        id1 = store.next_task_id()
        store.add_task(id1, "First")
        id2 = store.next_task_id()
        # IDs should be sequential
        assert id1 != id2
        seq1 = int(id1.split("-")[-1])
        seq2 = int(id2.split("-")[-1])
        assert seq2 == seq1 + 1
