"""Tests for the events module."""

from luna_os.events import ContractHelper, process_events, resolve_plan_for_task
from tests.memory_store import MemoryBackend


class TestContractHelper:
    def test_emit_lifecycle(self):
        store = MemoryBackend()
        ch = ContractHelper(store, task_id="t-1", step_id=1)
        ch.start("Starting work")
        ch.progress("50% done")
        ch.done("All done", succeeded=5, failed=0)

        events = store.poll_events(limit=10)
        assert len(events) == 3
        assert events[0].event_type == "contract.start"
        assert events[1].event_type == "contract.progress"
        assert events[2].event_type == "contract.done"
        assert events[2].payload["succeeded"] == 5

    def test_emit_fail(self):
        store = MemoryBackend()
        ch = ContractHelper(store, task_id="t-1", step_id=1)
        ch.fail("Broke")
        events = store.poll_events()
        assert len(events) == 1
        assert events[0].event_type == "contract.fail"
        assert events[0].payload["error"] == "Broke"

    def test_emit_partial(self):
        store = MemoryBackend()
        ch = ContractHelper(store, task_id="t-1", step_id=1)
        ch.partial("Partial result", succeeded=3, failed=2)
        events = store.poll_events()
        assert events[0].event_type == "contract.partial"


class TestResolvePlanForTask:
    def test_find_task_in_plan(self):
        store = MemoryBackend()
        store.create_plan(
            "plan-1",
            "chat-1",
            "Goal",
            [{"title": "Step A"}, {"title": "Step B"}],
        )
        store.update_plan_status("plan-1", "active")
        store.start_step("plan-1", 1, "t-1")

        plan_id, step_num = resolve_plan_for_task(store, "t-1")
        assert plan_id == "plan-1"
        assert step_num == 1

    def test_not_found(self):
        store = MemoryBackend()
        plan_id, step_num = resolve_plan_for_task(store, "t-nonexistent")
        assert plan_id is None
        assert step_num is None


class TestProcessEvents:
    def test_process_step_done(self):
        store = MemoryBackend()
        store.create_plan(
            "plan-1",
            "chat-1",
            "Goal",
            [{"title": "A"}, {"title": "B"}],
        )
        store.update_plan_status("plan-1", "active")
        store.start_step("plan-1", 1, "t-1")

        # Emit a step.done event
        store.emit_event("step.done", task_id="t-1", payload={"result": "ok"})

        done_calls = []

        def on_done(plan, step_num, result):
            done_calls.append((plan.id, step_num, result))

        processed = process_events(store, on_step_done=on_done)
        assert processed == 1
        assert len(done_calls) == 1
        assert done_calls[0] == ("plan-1", 1, "ok")

    def test_process_step_failed(self):
        store = MemoryBackend()
        store.create_plan("plan-1", "chat-1", "Goal", [{"title": "A"}])
        store.update_plan_status("plan-1", "active")
        store.start_step("plan-1", 1, "t-1")

        store.emit_event("step.failed", task_id="t-1", payload={"error": "broke"})

        fail_calls = []

        def on_fail(plan, step_num, error):
            fail_calls.append((plan.id, step_num, error))

        processed = process_events(store, on_step_fail=on_fail)
        assert processed == 1
        assert len(fail_calls) == 1
        assert fail_calls[0] == ("plan-1", 1, "broke")

    def test_ack_prevents_reprocessing(self):
        store = MemoryBackend()
        store.emit_event("test.ping", payload={"test": True})
        assert len(store.poll_events()) == 1
        process_events(store)
        assert len(store.poll_events()) == 0
