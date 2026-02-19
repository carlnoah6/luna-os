"""CLI entry point — preserves the existing task_manager.py and planner.py interface.

Usage:
  luna-os task add "description" [source_chat_id] [--no-chat] [--after tid-xxx] [--priority P]
  luna-os task show <id>
  luna-os task start <id> [session_key]
  luna-os task complete <id> ["result"] [--input-tokens N] [--output-tokens N] [--cost F]
  luna-os task fail <id> ["error"]
  luna-os task cancel <id>
  luna-os task list [status]
  luna-os task ready
  luna-os task active
  luna-os task status
  luna-os task set-session <id> <session_key>
  luna-os task cleanup [days]
  luna-os task health-check
  luna-os task wait <id> <type> <prompt> [--options ...]
  luna-os task respond <id> <response>

  luna-os plan init <chat_id> <goal> <steps_json>
  luna-os plan start <chat_id>
  luna-os plan show <chat_id|plan_id>
  luna-os plan step-done <chat_id> <step_num> "<result>"
  luna-os plan step-fail <chat_id> <step_num> "<error>"
  luna-os plan replan <chat_id> <new_steps_json> [--append]
  luna-os plan cancel <chat_id>
  luna-os plan advance <chat_id>
  luna-os plan pause <chat_id>
  luna-os plan resume <plan_id|chat_id|all>
  luna-os plan check-advances
  luna-os plan check-contracts
  luna-os plan list
  luna-os plan find-by-task <task_id>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from decimal import Decimal
from typing import Any

from luna_os.planner import Planner
from luna_os.store.postgres import PostgresBackend
from luna_os.task_manager import TaskManager


def _serialize(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, default=_serialize))


def _make_store() -> PostgresBackend:
    return PostgresBackend()


def _make_task_manager() -> TaskManager:
    store = _make_store()
    # Optional: create notification provider and agent runner from env vars
    notifications = None
    agent_runner = None
    try:
        import os

        if os.environ.get("LARK_APP_ID") and os.environ.get("LARK_APP_SECRET"):
            from luna_os.notifications.lark import LarkProvider

            notifications = LarkProvider()
    except Exception:
        pass
    try:
        from luna_os.agents.openclaw import OpenClawRunner

        agent_runner = OpenClawRunner()
    except Exception:
        pass
    return TaskManager(store, notifications=notifications, agent_runner=agent_runner)


def _make_planner() -> Planner:
    store = _make_store()
    notifications = None
    agent_runner = None
    try:
        import os

        if os.environ.get("LARK_APP_ID") and os.environ.get("LARK_APP_SECRET"):
            from luna_os.notifications.lark import LarkProvider

            notifications = LarkProvider()
    except Exception:
        pass
    try:
        from luna_os.agents.openclaw import OpenClawRunner

        agent_runner = OpenClawRunner()
    except Exception:
        pass
    return Planner(store, notifications=notifications, agent_runner=agent_runner)


# -- Task CLI ------------------------------------------------------------------


def task_cli(args: list[str]) -> None:
    """Handle 'task' subcommands."""
    if not args:
        print("Usage: luna-os task <command> [args...]")
        sys.exit(1)

    cmd = args[0]
    tm = _make_task_manager()

    try:
        if cmd == "add":
            depends_on = None
            create_chat = True
            priority = "normal"
            filtered = []
            i = 0
            rest = args[1:]
            while i < len(rest):
                if rest[i] == "--after" and i + 1 < len(rest):
                    depends_on = [x.strip() for x in rest[i + 1].split(",")]
                    i += 2
                elif rest[i] == "--no-chat":
                    create_chat = False
                    i += 1
                elif rest[i] == "--priority" and i + 1 < len(rest):
                    priority = rest[i + 1]
                    i += 2
                else:
                    filtered.append(rest[i])
                    i += 1
            desc = filtered[0] if filtered else ""
            source = filtered[1] if len(filtered) > 1 else None
            task = tm.add(
                desc,
                source_chat=source,
                depends_on=depends_on,
                priority=priority,
                create_chat=create_chat,
            )
            print_json(task.to_dict())

        elif cmd == "show":
            print_json(tm.show(args[1]))

        elif cmd == "start":
            task = tm.start(args[1], args[2] if len(args) > 2 else "")
            print_json(task.to_dict())

        elif cmd == "complete":
            result = args[2] if len(args) > 2 else ""
            inp_tok, out_tok, cost = 0, 0, 0.0
            i = 3
            rest = args
            while i < len(rest):
                if rest[i] == "--input-tokens" and i + 1 < len(rest):
                    inp_tok = int(rest[i + 1])
                    i += 2
                elif rest[i] == "--output-tokens" and i + 1 < len(rest):
                    out_tok = int(rest[i + 1])
                    i += 2
                elif rest[i] == "--cost" and i + 1 < len(rest):
                    cost = float(rest[i + 1])
                    i += 2
                else:
                    i += 1
            print_json(tm.complete(args[1], result, inp_tok, out_tok, cost))

        elif cmd == "fail":
            print_json(tm.fail(args[1], args[2] if len(args) > 2 else ""))

        elif cmd == "cancel":
            print_json(tm.cancel(args[1]))

        elif cmd == "list":
            f = args[1] if len(args) > 1 else None
            print_json(tm.list_tasks(f))

        elif cmd == "ready":
            print_json(tm.ready())

        elif cmd == "active":
            print_json(tm.active())

        elif cmd == "status":
            print_json(tm.status())

        elif cmd == "set-session":
            print_json(tm.set_session(args[1], args[2]))

        elif cmd == "cleanup":
            d = int(args[1]) if len(args) > 1 else 7
            print_json(tm.cleanup(d))

        elif cmd == "health-check":
            print_json(tm.health_check())

        elif cmd == "wait":
            opts = None
            for i, a in enumerate(args):
                if a == "--options" and i + 1 < len(args):
                    try:
                        opts = json.loads(args[i + 1])
                    except (json.JSONDecodeError, IndexError):
                        opts = args[i + 1 :]
                    break
            print_json(tm.wait(args[1], args[2], args[3], opts))

        elif cmd == "respond":
            print_json(tm.respond(args[1], " ".join(args[2:])))

        elif cmd == "cost":
            days = int(args[1]) if len(args) > 1 else 30
            print_json(tm.cost_report(days))

        else:
            print(f"Unknown task command: {cmd}", file=sys.stderr)
            sys.exit(1)

    except (ValueError, KeyError, RuntimeError) as e:
        print_json({"error": str(e)})
        sys.exit(1)


# -- Plan CLI ------------------------------------------------------------------


def plan_cli(args: list[str]) -> None:
    """Handle 'plan' subcommands."""
    if not args:
        print("Usage: luna-os plan <command> [args...]")
        sys.exit(1)

    cmd = args[0]
    planner = _make_planner()

    try:
        if cmd == "init":
            if len(args) < 4:
                print(
                    "Usage: luna-os plan init <chat_id> <goal> <steps_json>",
                    file=sys.stderr,
                )
                sys.exit(1)
            steps_raw = json.loads(args[3]) if isinstance(args[3], str) else args[3]
            print_json(planner.init(args[1], args[2], steps_raw))

        elif cmd == "start":
            print_json(planner.start(args[1]))

        elif cmd == "show":
            print(planner.show(args[1]))

        elif cmd == "step-done":
            print_json(planner.step_done(args[1], int(args[2]), args[3]))

        elif cmd == "step-fail":
            print_json(planner.step_fail(args[1], int(args[2]), args[3]))

        elif cmd == "replan":
            append = "--append" in args
            steps_raw = json.loads(args[2]) if isinstance(args[2], str) else args[2]
            print_json(planner.replan(args[1], steps_raw, append=append))

        elif cmd == "cancel":
            print_json(planner.cancel(args[1]))

        elif cmd == "advance":
            print_json(planner.advance(args[1]))

        elif cmd == "pause":
            print_json(planner.pause(args[1]))

        elif cmd == "resume":
            print_json(planner.resume(args[1]))

        elif cmd == "check-advances":
            print_json(planner.check_advances())

        elif cmd == "check-contracts":
            print_json(planner.check_contracts())

        elif cmd == "list":
            for item in planner.list_plans():
                status = item["status"]
                done = item["done"]
                total = item["total"]
                running = item.get("running_step")
                running_info = f" | running: Step {running}" if running else ""
                print(f"  {item['id']} [{status}] {item['goal']} ({done}/{total}){running_info}")

        elif cmd == "find-by-task":
            print_json(planner.find_by_task(args[1]))

        else:
            print(f"Unknown plan command: {cmd}", file=sys.stderr)
            sys.exit(1)

    except (ValueError, KeyError, RuntimeError) as e:
        print_json({"error": str(e)})
        sys.exit(1)


# -- Main entry point ---------------------------------------------------------


def main() -> None:
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    top = sys.argv[1]
    rest = sys.argv[2:]

    if top in ("--help", "-h", "help"):
        print(__doc__)
        sys.exit(0)
    elif top == "task":
        task_cli(rest)
    elif top == "plan":
        plan_cli(rest)
    else:
        print(f"Unknown command: {top}", file=sys.stderr)
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
