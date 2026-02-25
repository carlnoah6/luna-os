"""Check for plans with completed steps but next steps not started."""
from __future__ import annotations
import json
from datetime import datetime, timezone, timedelta
from ..models import Alert, Severity

NAME = "stuck_plans"
STUCK_MINUTES = 5


def check(db) -> list[Alert]:
    """Find active plans where a step is done but the next dependent step hasn't started."""
    alerts = []

    plans = db.execute(
        "SELECT id, chat_id, goal, status, updated_at FROM plans WHERE status = 'active'"
    )

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STUCK_MINUTES)

    for plan in plans:
        plan_id = plan["id"]

        steps = db.execute(
            """
            SELECT step_num, status, task_id, title, depends_on, completed_at
            FROM plan_steps
            WHERE plan_id = %s
            ORDER BY step_num
            """,
            (plan_id,)
        )

        step_list = list(steps)
        done_nums = {s["step_num"] for s in step_list if s["status"] == "done"}

        for s in step_list:
            if s["status"] != "pending":
                continue

            deps = s.get("depends_on") or []
            if isinstance(deps, str):
                try:
                    deps = json.loads(deps)
                except Exception:
                    deps = []

            if not deps:
                continue

            all_deps_done = all(d in done_nums for d in deps)
            if not all_deps_done:
                continue

            # All deps done — check how long ago
            dep_done_times = [
                st["completed_at"] for st in step_list
                if st["step_num"] in deps and st["completed_at"] is not None
            ]

            if dep_done_times and max(dep_done_times) < cutoff:
                mins = int((datetime.now(timezone.utc) - max(dep_done_times)).total_seconds() / 60)
                title = (s["title"] or "")[:50]
                goal = (plan["goal"] or "")[:40]
                alerts.append(Alert(
                    check_name=NAME,
                    key=f"stuck_plan:{plan_id}:step:{s['step_num']}",
                    message=f"Plan {plan_id} ({goal}) 步骤 {s['step_num']} 依赖已完成 {mins} 分钟但未启动: {title}",
                    severity=Severity.WARN,
                    context={"plan_id": plan_id, "chat_id": plan.get("chat_id"), "step_num": s["step_num"]},
                ))

    return alerts
