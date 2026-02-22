"""Tests for Planner.build_confirm_card."""

from luna_os.planner import Planner
from luna_os.types import Plan, PlanStatus, Step


def _make_plan(n_steps=3, goal="Test goal"):
    steps = [
        Step(
            plan_id="p1",
            step_num=i,
            title=f"Step {i}",
            depends_on=[i - 1] if i > 0 else [],
        )
        for i in range(n_steps)
    ]
    return Plan(id="p1", chat_id="oc_test", goal=goal, status=PlanStatus.DRAFT, steps=steps)


class TestBuildConfirmCard:
    def test_basic_structure(self):
        plan = _make_plan()
        card = Planner.build_confirm_card(plan)
        assert card["header"]["title"]["content"].startswith("📋 Plan 确认")
        assert card["header"]["template"] == "blue"
        # Must have action element with 3 buttons
        actions = [e for e in card["elements"] if e.get("tag") == "action"]
        assert len(actions) == 1
        buttons = actions[0]["actions"]
        assert len(buttons) == 3
        # Verify button action values
        assert buttons[0]["value"]["action"] == "plan_confirm"
        assert buttons[1]["value"]["action"] == "plan_modify"
        assert buttons[2]["value"]["action"] == "plan_cancel"

    def test_plan_id_and_chat_id_in_buttons(self):
        plan = _make_plan()
        card = Planner.build_confirm_card(plan)
        actions = [e for e in card["elements"] if e.get("tag") == "action"]
        for btn in actions[0]["actions"]:
            assert btn["value"]["chat_id"] == "oc_test"
            assert btn["value"]["plan_id"] == "p1"

    def test_steps_displayed(self):
        plan = _make_plan(n_steps=5)
        card = Planner.build_confirm_card(plan)
        # Find the div with step lines
        divs = [e for e in card["elements"] if e.get("tag") == "div"]
        step_text = "\n".join(d["text"]["content"] for d in divs if "text" in d)
        for i in range(5):
            assert f"Step {i}" in step_text

    def test_deps_shown(self):
        plan = _make_plan(n_steps=3)
        card = Planner.build_confirm_card(plan)
        divs = [e for e in card["elements"] if e.get("tag") == "div"]
        step_text = "\n".join(d["text"]["content"] for d in divs if "text" in d)
        assert "after" in step_text  # deps for step 1 and 2

    def test_empty_steps(self):
        plan = Plan(id="p2", chat_id="oc_test", goal="Empty", status=PlanStatus.DRAFT, steps=[])
        card = Planner.build_confirm_card(plan)
        divs = [e for e in card["elements"] if e.get("tag") == "div"]
        all_text = "\n".join(d["text"]["content"] for d in divs if "text" in d)
        assert "(no steps)" in all_text

    def test_long_goal_truncated_in_header(self):
        plan = _make_plan(goal="A" * 100)
        card = Planner.build_confirm_card(plan)
        header_text = card["header"]["title"]["content"]
        # Header should truncate to 50 chars
        assert len(header_text) < 70  # "📋 Plan 确认: " + 50 chars
