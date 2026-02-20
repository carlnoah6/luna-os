"""PostgreSQL (Neon) storage backend implementation."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg2
import psycopg2.extras

from luna_os.store.base import StorageBackend
from luna_os.types import (
    PRIORITY_MAP,
    PRIORITY_NAMES,
    Event,
    Plan,
    Step,
    Task,
    now_utc,
)

SGT = timezone(timedelta(hours=8))


def _validate_field_names(fields: dict[str, Any]) -> None:
    """Validate that field names are safe for SQL interpolation."""
    for key in fields:
        if not re.match(r"^[a-z_][a-z0-9_]*$", key):
            raise ValueError(f"Invalid field name: {key!r}")


def _row_to_dict(cursor: Any, row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row, strict=False))


def _rows_to_dicts(cursor: Any, rows: Any) -> list[dict[str, Any]]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r, strict=False)) for r in rows]


class PostgresBackend(StorageBackend):
    """PostgreSQL (Neon) implementation of the StorageBackend interface."""

    def __init__(self, db_url: str | None = None) -> None:
        if db_url is None:
            db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL environment variable or db_url parameter is required")
        self._db_url = db_url
        self._conn: Any = None

    def _get_conn(self) -> Any:
        if self._conn is not None:
            try:
                self._conn.isolation_level  # noqa: B018 — test connection
                return self._conn
            except Exception:
                self._conn = None
        self._conn = psycopg2.connect(self._db_url)
        self._conn.autocommit = True
        return self._conn

    def _execute(self, sql: str, params: Any = None, fetch: str | None = None) -> Any:
        for attempt in range(2):
            try:
                conn = self._get_conn()
                cur = conn.cursor()
                cur.execute(sql, params)
                if fetch == "one":
                    row = cur.fetchone()
                    return _row_to_dict(cur, row)
                elif fetch == "all":
                    rows = cur.fetchall()
                    return _rows_to_dicts(cur, rows)
                return cur
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                self._conn = None
                if attempt == 1:
                    raise

    # -- Helpers ---------------------------------------------------------------

    def _dict_to_task(self, d: dict[str, Any] | None) -> Task | None:
        if d is None:
            return None
        return Task.from_dict(d)

    def _dicts_to_tasks(self, rows: list[dict[str, Any]]) -> list[Task]:
        return [Task.from_dict(r) for r in rows]

    def _dict_to_step(self, d: dict[str, Any] | None) -> Step | None:
        if d is None:
            return None
        return Step.from_dict(d)

    def _dict_to_event(self, d: dict[str, Any] | None) -> Event | None:
        if d is None:
            return None
        if isinstance(d.get("payload"), str):
            d["payload"] = json.loads(d["payload"])
        return Event.from_dict(d)

    # -- Tasks -----------------------------------------------------------------

    def add_task(
        self,
        task_id: str,
        description: str,
        source_chat: str | None = None,
        priority: str = "normal",
        depends_on: list[str] | None = None,
    ) -> Task:
        now = now_utc()
        pv = PRIORITY_MAP.get(priority, 3)
        self._execute(
            """INSERT INTO tasks (id, description, status, source_chat, priority,
                   priority_value, depends_on, created_at, updated_at)
               VALUES (%s, %s, 'queued', %s, %s, %s, %s, %s, %s)""",
            (task_id, description, source_chat, priority, pv, depends_on or [], now, now),
        )
        return self.get_task(task_id)  # type: ignore[return-value]

    def get_task(self, task_id: str) -> Task | None:
        row = self._execute("SELECT * FROM tasks WHERE id = %s", (task_id,), fetch="one")
        return self._dict_to_task(row)

    def list_tasks(self, status: str | None = None) -> list[Task]:
        if status:
            rows = self._execute(
                "SELECT * FROM tasks WHERE status = %s ORDER BY priority_value, created_at",
                (status,),
                fetch="all",
            )
        else:
            rows = self._execute(
                "SELECT * FROM tasks ORDER BY priority_value, created_at",
                fetch="all",
            )
        return self._dicts_to_tasks(rows or [])

    def update_task(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        _validate_field_names(fields)
        fields["updated_at"] = now_utc()
        sets = ", ".join(f"{k} = %s" for k in fields)
        vals = list(fields.values()) + [task_id]
        self._execute(f"UPDATE tasks SET {sets} WHERE id = %s", vals)

    def start_task(self, task_id: str, session_key: str = "") -> None:
        now = now_utc()
        self._execute(
            """UPDATE tasks SET status='running', session_key=%s,
                   started_at=%s, updated_at=%s WHERE id=%s""",
            (session_key, now, now, task_id),
        )

    def complete_task(self, task_id: str, result: str = "") -> None:
        now = now_utc()
        self._execute(
            """UPDATE tasks SET status='done', result=%s,
                   completed_at=%s, updated_at=%s WHERE id=%s""",
            (result, now, now, task_id),
        )

    def fail_task(self, task_id: str, error: str = "") -> None:
        now = now_utc()
        self._execute(
            """UPDATE tasks SET status='failed', result=%s,
                   completed_at=%s, updated_at=%s WHERE id=%s""",
            (error, now, now, task_id),
        )

    def cancel_task(self, task_id: str) -> None:
        self._execute(
            "UPDATE tasks SET status='cancelled', updated_at=%s WHERE id=%s",
            (now_utc(), task_id),
        )

    def ready_tasks(self) -> list[Task]:
        all_queued = self._execute(
            "SELECT * FROM tasks WHERE status = 'queued' ORDER BY priority_value, created_at",
            fetch="all",
        )
        if not all_queued:
            return []
        dep_ids: set[str] = set()
        for t in all_queued:
            deps = t.get("depends_on") or []
            dep_ids.update(deps)
        done_ids: set[str] = set()
        if dep_ids:
            rows = self._execute(
                "SELECT id FROM tasks WHERE id = ANY(%s) AND status = 'done'",
                (list(dep_ids),),
                fetch="all",
            )
            done_ids = {r["id"] for r in (rows or [])}
        ready = [t for t in all_queued if all(d in done_ids for d in (t.get("depends_on") or []))]
        return self._dicts_to_tasks(ready)

    def active_tasks(self) -> list[Task]:
        rows = self._execute(
            "SELECT * FROM tasks WHERE status = 'running' ORDER BY started_at",
            fetch="all",
        )
        return self._dicts_to_tasks(rows or [])

    def cleanup_tasks(self, days: int = 7) -> int:
        cutoff = now_utc() - timedelta(days=days)
        cur = self._execute(
            """DELETE FROM tasks
               WHERE status IN ('done', 'cancelled', 'failed')
                 AND updated_at < %s""",
            (cutoff,),
        )
        return cur.rowcount if cur else 0

    def next_task_id(self) -> str:
        today = datetime.now(SGT)
        prefix = f"tid-{today.strftime('%m%d')}-"
        rows = self._execute(
            "SELECT id FROM tasks WHERE id LIKE %s",
            (prefix + "%",),
            fetch="all",
        )
        max_seq = 0
        for r in rows or []:
            try:
                seq = int(r["id"].split("-")[-1])
                if seq > max_seq:
                    max_seq = seq
            except (ValueError, IndexError):
                pass
        return f"{prefix}{max_seq + 1}"

    def update_task_cost(
        self,
        task_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0,
    ) -> None:
        self._execute(
            """UPDATE tasks SET input_tokens = input_tokens + %s,
                   output_tokens = output_tokens + %s,
                   cost_usd = cost_usd + %s,
                   updated_at = %s WHERE id = %s""",
            (input_tokens, output_tokens, cost_usd, now_utc(), task_id),
        )

    def find_duplicate(self, description: str) -> str | None:
        row = self._execute(
            """SELECT id FROM tasks
               WHERE status IN ('queued', 'running')
                 AND LOWER(TRIM(description)) = LOWER(TRIM(%s))
               LIMIT 1""",
            (description,),
            fetch="one",
        )
        return row["id"] if row else None

    def task_count_by_status(self) -> dict[str, int]:
        rows = self._execute(
            "SELECT status, COUNT(*) as cnt FROM tasks GROUP BY status",
            fetch="all",
        )
        return {r["status"]: r["cnt"] for r in (rows or [])}

    def boost_priority(self, task_id: str) -> None:
        task = self.get_task(task_id)
        if not task or task.status.value != "queued":
            return
        pv = task.priority_value
        if pv > 1:
            new_pv = pv - 1
            new_name = PRIORITY_NAMES.get(new_pv, "normal")
            self.update_task(
                task_id, priority=new_name, priority_value=new_pv, priority_boosted=True
            )

    def cost_summary(self, days: int = 30) -> list[dict[str, Any]]:
        return (
            self._execute(
                """SELECT
                     COUNT(*) as total_tasks,
                     SUM(input_tokens) as total_input,
                     SUM(output_tokens) as total_output,
                     SUM(cost_usd) as total_cost,
                     DATE(completed_at AT TIME ZONE 'Asia/Singapore') as day
                   FROM tasks
                   WHERE status = 'done'
                     AND completed_at > NOW() - (%s || ' days')::interval
                   GROUP BY day
                   ORDER BY day DESC""",
                (days,),
                fetch="all",
            )
            or []
        )

    # -- Plans -----------------------------------------------------------------

    def create_plan(
        self,
        plan_id: str,
        chat_id: str,
        goal: str,
        steps: list[dict[str, Any]],
        status: str = "draft",
    ) -> Plan:
        now = now_utc()
        self._execute(
            """INSERT INTO plans (id, chat_id, goal, status, created_at, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (plan_id, chat_id, goal, status, now, now),
        )
        for i, s in enumerate(steps, 1):
            raw_deps = s.get("depends_on") or []
            deps = [d + 1 for d in raw_deps]
            self._execute(
                """INSERT INTO plan_steps (plan_id, step_num, title, prompt, status, depends_on)
                   VALUES (%s, %s, %s, %s, 'pending', %s)""",
                (plan_id, i, s.get("title", ""), s.get("prompt", ""), deps),
            )
        return self.get_plan(plan_id)  # type: ignore[return-value]

    def get_plan(self, plan_id: str) -> Plan | None:
        plan_row = self._execute("SELECT * FROM plans WHERE id = %s", (plan_id,), fetch="one")
        if not plan_row:
            return None
        step_rows = (
            self._execute(
                "SELECT * FROM plan_steps WHERE plan_id = %s ORDER BY step_num",
                (plan_id,),
                fetch="all",
            )
            or []
        )
        plan_row["steps"] = step_rows
        return Plan.from_dict(plan_row)

    def get_plan_by_chat(self, chat_id: str, status_filter: str | None = None) -> Plan | None:
        if status_filter:
            plan_row = self._execute(
                """SELECT * FROM plans
                   WHERE chat_id = %s AND status = %s
                   ORDER BY created_at DESC LIMIT 1""",
                (chat_id, status_filter),
                fetch="one",
            )
            if not plan_row:
                plan_row = self._execute(
                    "SELECT * FROM plans WHERE id = %s AND status = %s",
                    (chat_id, status_filter),
                    fetch="one",
                )
            if not plan_row and len(chat_id) <= 8:
                plan_row = self._execute(
                    """SELECT * FROM plans
                       WHERE (chat_id LIKE %s OR id = %s) AND status = %s
                       ORDER BY created_at DESC LIMIT 1""",
                    ("%" + chat_id, chat_id, status_filter),
                    fetch="one",
                )
        else:
            plan_row = self._execute(
                """SELECT * FROM plans
                   WHERE chat_id = %s
                   ORDER BY created_at DESC LIMIT 1""",
                (chat_id,),
                fetch="one",
            )
            if not plan_row:
                plan_row = self._execute(
                    "SELECT * FROM plans WHERE id = %s ORDER BY created_at DESC LIMIT 1",
                    (chat_id,),
                    fetch="one",
                )
            if not plan_row and len(chat_id) <= 8:
                plan_row = self._execute(
                    """SELECT * FROM plans
                       WHERE chat_id LIKE %s OR id = %s
                       ORDER BY created_at DESC LIMIT 1""",
                    ("%" + chat_id, chat_id),
                    fetch="one",
                )
        if not plan_row:
            return None
        step_rows = (
            self._execute(
                "SELECT * FROM plan_steps WHERE plan_id = %s ORDER BY step_num",
                (plan_row["id"],),
                fetch="all",
            )
            or []
        )
        plan_row["steps"] = step_rows
        return Plan.from_dict(plan_row)

    def list_plans(self, status: str | None = None) -> list[Plan]:
        if status:
            rows = self._execute(
                "SELECT * FROM plans WHERE status = %s ORDER BY created_at DESC",
                (status,),
                fetch="all",
            )
        else:
            rows = self._execute("SELECT * FROM plans ORDER BY created_at DESC", fetch="all")
        return [Plan.from_dict(r) for r in (rows or [])]

    def update_plan_status(self, plan_id: str, status: str) -> None:
        self._execute(
            "UPDATE plans SET status = %s, updated_at = %s WHERE id = %s",
            (status, now_utc(), plan_id),
        )

    def cancel_plan(self, plan_id: str) -> None:
        now = now_utc()
        self._execute(
            "UPDATE plans SET status = 'cancelled', updated_at = %s WHERE id = %s",
            (now, plan_id),
        )
        self._execute(
            """UPDATE plan_steps SET status = 'cancelled'
               WHERE plan_id = %s AND status = 'pending'""",
            (plan_id,),
        )

    def next_plan_id(self) -> str:
        today = datetime.now(SGT)
        prefix = f"plan-{today.strftime('%m%d')}-"
        rows = self._execute(
            "SELECT id FROM plans WHERE id LIKE %s",
            (prefix + "%",),
            fetch="all",
        )
        max_seq = 0
        for r in rows or []:
            try:
                seq = int(r["id"].split("-")[-1])
                if seq > max_seq:
                    max_seq = seq
            except (ValueError, IndexError):
                pass
        return f"{prefix}{max_seq + 1}"

    # -- Steps -----------------------------------------------------------------

    def get_step(self, plan_id: str, step_num: int) -> Step | None:
        row = self._execute(
            "SELECT * FROM plan_steps WHERE plan_id = %s AND step_num = %s",
            (plan_id, step_num),
            fetch="one",
        )
        return self._dict_to_step(row)

    def update_step(self, plan_id: str, step_num: int, **fields: Any) -> None:
        if not fields:
            return
        _validate_field_names(fields)
        sets = ", ".join(f"{k} = %s" for k in fields)
        vals = list(fields.values()) + [plan_id, step_num]
        self._execute(
            f"UPDATE plan_steps SET {sets} WHERE plan_id = %s AND step_num = %s",
            vals,
        )

    def start_step(self, plan_id: str, step_num: int, task_id: str) -> None:
        self._execute(
            """UPDATE plan_steps SET status='running', task_id=%s, started_at=%s
               WHERE plan_id=%s AND step_num=%s""",
            (task_id, now_utc(), plan_id, step_num),
        )

    def complete_step(self, plan_id: str, step_num: int, result: str) -> None:
        self._execute(
            """UPDATE plan_steps SET status='done', result=%s, completed_at=%s
               WHERE plan_id=%s AND step_num=%s""",
            (result, now_utc(), plan_id, step_num),
        )

    def fail_step(self, plan_id: str, step_num: int, error: str) -> None:
        self._execute(
            """UPDATE plan_steps SET status='failed', result=%s, completed_at=%s
               WHERE plan_id=%s AND step_num=%s""",
            (error, now_utc(), plan_id, step_num),
        )

    def ready_steps(self, plan_id: str) -> list[Step]:
        all_rows = (
            self._execute(
                "SELECT * FROM plan_steps WHERE plan_id = %s ORDER BY step_num",
                (plan_id,),
                fetch="all",
            )
            or []
        )
        done_nums = {s["step_num"] for s in all_rows if s["status"] == "done"}
        ready = []
        for s in all_rows:
            if s["status"] != "pending":
                continue
            deps = s.get("depends_on") or []
            if all(d in done_nums for d in deps):
                ready.append(Step.from_dict(s))
        return ready

    def next_pending_step(self, plan_id: str) -> Step | None:
        steps = self.ready_steps(plan_id)
        return steps[0] if steps else None

    def delete_pending_steps(self, plan_id: str) -> None:
        self._execute(
            "DELETE FROM plan_steps WHERE plan_id = %s AND status = 'pending'",
            (plan_id,),
        )

    def insert_step(
        self,
        plan_id: str,
        step_num: int,
        title: str,
        prompt: str,
        depends_on: list[int] | None = None,
    ) -> None:
        self._execute(
            """INSERT INTO plan_steps (plan_id, step_num, title, prompt, status, depends_on)
               VALUES (%s, %s, %s, %s, 'pending', %s)""",
            (plan_id, step_num, title, prompt, depends_on or []),
        )

    # -- Events ----------------------------------------------------------------

    def emit_event(
        self,
        event_type: str,
        task_id: str | None = None,
        plan_id: str | None = None,
        step_num: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._execute(
            """INSERT INTO events (event_type, task_id, plan_id, step_num,
                   payload, processed, created_at)
               VALUES (%s, %s, %s, %s, %s, false, %s)""",
            (
                event_type,
                task_id,
                plan_id,
                step_num,
                json.dumps(payload) if payload else None,
                now_utc(),
            ),
        )

    def poll_events(self, limit: int = 10) -> list[Event]:
        rows = self._execute(
            """SELECT * FROM events
               WHERE processed = false
               ORDER BY created_at
               LIMIT %s""",
            (limit,),
            fetch="all",
        )
        events = []
        for r in rows or []:
            if isinstance(r.get("payload"), str):
                r["payload"] = json.loads(r["payload"])
            events.append(Event.from_dict(r))
        return events

    def ack_event(self, event_id: int) -> None:
        self._execute(
            "UPDATE events SET processed = true, processed_at = %s WHERE id = %s",
            (now_utc(), event_id),
        )

    def ack_events(self, event_ids: list[int]) -> None:
        if not event_ids:
            return
        self._execute(
            """UPDATE events SET processed = true, processed_at = %s
               WHERE id = ANY(%s)""",
            (now_utc(), event_ids),
        )
