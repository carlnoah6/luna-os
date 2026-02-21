"""Dashboard card builder for Luna OS.

Builds Lark interactive cards showing:
- Task manager status (running / queued tasks)
- Planner status (active plans, step progress)
- Session overview (active chat sessions)

Usage::

    from luna_os.dashboard import DashboardBuilder

    builder = DashboardBuilder(store)
    card = builder.build_global()          # full dashboard
    card = builder.build_plan(chat_id)     # plan-specific card
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any

from luna_os.store.base import StorageBackend

logger = logging.getLogger(__name__)

SGT = timezone(timedelta(hours=8))

PRIORITY_ICONS: dict[str, str] = {
    "P0": "🔴",
    "P1": "🟠",
    "P2": "🟡",
    "P3": "🟢",
    "normal": "🟢",
}

STATUS_EMOJI: dict[str, str] = {
    "done": "✅",
    "running": "🔄",
    "pending": "⏳",
    "failed": "❌",
    "cancelled": "🚫",
    "waiting": "⏸️",
}

PLAN_STATUS_STYLE: dict[str, tuple[str, str]] = {
    "active": ("blue", "🟢 进行中"),
    "completed": ("green", "✅ 已完成"),
    "cancelled": ("grey", "🚫 已取消"),
    "draft": ("yellow", "📝 草稿"),
    "paused": ("orange", "⏸️ 已暂停"),
}


# ── Formatting helpers ───────────────────────────────────────────


def _fmt_duration(minutes: float) -> str:
    if minutes < 1:
        return "<1min"
    if minutes < 60:
        return f"{minutes:.0f}min"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def _fmt_tokens(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.0f}K"
    return str(tokens)


def _col(text: str, weight: int = 1) -> dict[str, Any]:
    """Build a Lark card column element."""
    return {
        "tag": "column",
        "width": "weighted",
        "weight": weight,
        "vertical_align": "top",
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md", "content": text}}
        ],
    }


def _wrap_card(
    title: str,
    template: str,
    elements: list[dict[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    """Wrap elements in a Lark card with header, timestamp, and refresh button."""
    elements.append({
        "tag": "note",
        "elements": [
            {
                "tag": "plain_text",
                "content": f"🕐 最后更新: {now.strftime('%Y-%m-%d %H:%M:%S')} SGT",
            }
        ],
    })
    elements.append({
        "tag": "action",
        "actions": [{
            "tag": "button",
            "text": {"tag": "plain_text", "content": "🔄 刷新"},
            "type": "primary",
            "value": {"action": "refresh_dashboard"},
        }],
    })
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": template,
        },
        "elements": elements,
    }


# ── Dashboard builder ────────────────────────────────────────────


class DashboardBuilder:
    """Builds Lark interactive card dicts from store data.

    Args:
        store: A :class:`StorageBackend` instance for reading tasks and plans.
        max_concurrent: Max concurrent task slots shown in the header.
        session_data: Optional pre-loaded session overview list. Each entry
            should have keys like ``name``, ``planner``, ``tokens``,
            ``usage_pct``, ``relative_time``, ``last_activity``.
    """

    def __init__(
        self,
        store: StorageBackend,
        *,
        max_concurrent: int = 8,
        session_data: list[dict[str, Any]] | None = None,
        session_overview_file: str | None = None,
        session_overview_script: str | None = None,
    ) -> None:
        self.store = store
        self.max_concurrent = max_concurrent
        self._session_overview_file = session_overview_file or os.environ.get(
            "SESSION_OVERVIEW_FILE",
            os.path.expanduser("~/.openclaw/workspace/data/session-overview.json"),
        )
        self._session_overview_script = session_overview_script or os.environ.get(
            "SESSION_OVERVIEW_SCRIPT",
            os.path.expanduser("~/.openclaw/workspace/scripts/session-overview.py"),
        )
        self.session_data = session_data

    # ── Global dashboard ─────────────────────────────────────

    def _refresh_session_data(self) -> None:
        """Refresh session overview by running the script and loading the JSON."""
        # Try to refresh the data file
        if os.path.exists(self._session_overview_script):
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["python3", self._session_overview_script],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

        # Load from file
        if os.path.exists(self._session_overview_file):
            try:
                with open(self._session_overview_file) as f:
                    overview = json.load(f)
                self.session_data = overview.get("sessions", [])
            except Exception:
                self.session_data = []
        else:
            self.session_data = []

    def build_global(self) -> dict[str, Any]:
        """Build the global task dashboard card."""
        # Auto-load session data if not provided
        if self.session_data is None:
            self._refresh_session_data()

        now = datetime.now(SGT)
        tasks = self._list_tasks_as_dicts()

        running = [t for t in tasks if t["status"] == "running"]
        queued = [t for t in tasks if t["status"] == "queued"]

        elements: list[dict[str, Any]] = []

        # Stats header
        elements.append({
            "tag": "div",
            "fields": [
                {
                    "is_short": True,
                    "text": {
                        "tag": "lark_md",
                        "content": f"**🏃 运行中** {len(running)}/{self.max_concurrent}",
                    },
                },
                {
                    "is_short": True,
                    "text": {
                        "tag": "lark_md",
                        "content": f"**⏳ 等待中** {len(queued)}",
                    },
                },
            ],
        })
        elements.append({"tag": "hr"})

        # Running tasks
        elements.extend(self._task_section("🏃 运行中任务", running, now))
        elements.append({"tag": "hr"})

        # Queued tasks
        elements.extend(self._task_section("⏳ 等待中任务", queued, now, limit=5))
        elements.append({"tag": "hr"})

        # Session overview
        if self.session_data:
            elements.extend(self._session_section())
            elements.append({"tag": "hr"})

        return _wrap_card("🖥️ Luna 任务仪表盘", "blue", elements, now)

    # ── Plan-specific card ───────────────────────────────────

    def build_plan(self, chat_id: str) -> dict[str, Any]:
        """Build a plan-focused card for a specific group chat."""
        now = datetime.now(SGT)
        elements: list[dict[str, Any]] = []

        plans = self._plans_for_chat(chat_id)
        if not plans:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "📋 **当前群聊无规划任务**\n\n使用 planner 创建新的规划。",
                },
            })
            return _wrap_card("📋 规划仪表盘", "grey", elements, now)

        plan = plans[0]
        steps = plan.get("steps", [])
        total = len(steps)
        done_count = sum(1 for s in steps if s.get("status") == "done")
        running_count = sum(1 for s in steps if s.get("status") == "running")
        failed_count = sum(1 for s in steps if s.get("status") == "failed")

        color, status_text = PLAN_STATUS_STYLE.get(
            plan["status"], ("grey", plan["status"])
        )

        # Plan header
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**📋 {plan['goal']}**"},
        })
        elements.append({
            "tag": "div",
            "fields": [
                {
                    "is_short": True,
                    "text": {"tag": "lark_md", "content": f"**状态** {status_text}"},
                },
                {
                    "is_short": True,
                    "text": {"tag": "lark_md", "content": f"**进度** {done_count}/{total}"},
                },
            ],
        })

        if running_count > 0 or failed_count > 0:
            extra: list[dict[str, Any]] = []
            if running_count:
                extra.append({
                    "is_short": True,
                    "text": {"tag": "lark_md", "content": f"**🔄 运行中** {running_count}"},
                })
            if failed_count:
                extra.append({
                    "is_short": True,
                    "text": {"tag": "lark_md", "content": f"**❌ 失败** {failed_count}"},
                })
            elements.append({"tag": "div", "fields": extra})

        elements.append({"tag": "hr"})

        # Steps list
        for s in steps:
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": self._format_step(s, now)},
            })

        elements.append({"tag": "hr"})

        # Other plans
        other = [
            p for p in plans[1:]
            if p["status"] in ("active", "paused", "draft")
        ]
        if other:
            lines = ["**其他规划**"]
            for p in other[:3]:
                _, st = PLAN_STATUS_STYLE.get(p["status"], ("grey", p["status"]))
                lines.append(f"  {st} {p['goal'][:40]}")
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": "\n".join(lines)},
            })
            elements.append({"tag": "hr"})

        return _wrap_card(f"📋 {plan['goal'][:30]}", color, elements, now)

    # ── Internal helpers ─────────────────────────────────────

    def _list_tasks_as_dicts(self) -> list[dict[str, Any]]:
        raw = self.store.list_tasks()
        return [t.to_dict() if hasattr(t, "to_dict") else t for t in raw]

    def _plans_for_chat(self, chat_id: str) -> list[dict[str, Any]]:
        raw = self.store.list_plans()
        plans = [p.to_dict() if hasattr(p, "to_dict") else p for p in raw]
        chat_plans = [p for p in plans if p.get("chat_id") == chat_id]
        chat_plans.sort(
            key=lambda p: p.get("updated_at") or p.get("created_at") or "",
            reverse=True,
        )
        return chat_plans

    def _task_section(
        self,
        title: str,
        tasks: list[dict[str, Any]],
        now: datetime,
        *,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        if not tasks:
            return [{
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**{title}** — 无"},
            }]

        lines = [f"**{title}**"]
        show = tasks[:limit] if limit else tasks
        for t in show:
            icon = PRIORITY_ICONS.get(t.get("priority", "normal"), "🟢")
            elapsed = ""
            if t.get("started_at"):
                started = t["started_at"]
                if isinstance(started, str):
                    started = datetime.fromisoformat(started)
                if started.tzinfo is None:
                    started = started.replace(tzinfo=SGT)
                mins = (now - started.astimezone(SGT)).total_seconds() / 60
                elapsed = f" ⏱{_fmt_duration(mins)}"
            desc = t["description"][:50]
            deps = t.get("depends_on") or []
            dep_str = f" (等待 {','.join(deps)})" if deps else ""
            lines.append(f"{icon} `{t['id']}` {desc}{elapsed}{dep_str}")

        if limit and len(tasks) > limit:
            lines.append(f"... 还有 {len(tasks) - limit} 个")

        return [{
            "tag": "div",
            "text": {"tag": "lark_md", "content": "\n".join(lines)},
        }]

    def _format_step(self, step: dict[str, Any], now: datetime) -> str:
        num = step.get("step_num", 0)
        title = step.get("title", f"Step {num}")
        status = step.get("status", "pending")
        emoji = STATUS_EMOJI.get(status, "⬜")
        tid = step.get("task_id", "")

        line = f"{emoji} **S{num}.** {title}"
        if tid:
            line += f"  `{tid}`"

        if status == "running" and step.get("started_at"):
            started = step["started_at"]
            if isinstance(started, str):
                started = datetime.fromisoformat(started)
            if started.tzinfo is None:
                started = started.replace(tzinfo=SGT)
            mins = (now - started.astimezone(SGT)).total_seconds() / 60
            line += f"  ⏱{_fmt_duration(mins)}"

        if status == "done" and step.get("result"):
            line += f"\n    _{step['result'][:60]}_"

        deps = step.get("depends_on") or []
        if deps and status == "pending":
            line += f"  (等待 S{', S'.join(str(d) for d in deps)})"

        return line

    def _session_section(self) -> list[dict[str, Any]]:
        if not self.session_data:
            return [{
                "tag": "div",
                "text": {"tag": "lark_md", "content": "📊 **Session 概览** — 暂无数据"},
            }]

        elements: list[dict[str, Any]] = []

        # Header row
        elements.append({
            "tag": "column_set",
            "flex_mode": "none",
            "background_style": "grey",
            "columns": [
                _col("**Session**", 3),
                _col("**状态**", 2),
                _col("**Tokens**", 2),
                _col("**时长**", 1),
            ],
        })

        for s in self.session_data[:8]:
            name = s.get("name", "?")[:20]
            planner = s.get("planner")
            if planner and planner.get("goal"):
                name_cell = f"{name}\n📋 {planner['goal'][:30]}"
            else:
                name_cell = name

            if planner and planner.get("status_text"):
                status = planner["status_text"]
            else:
                status = s.get("last_activity") or "—"

            tok = s.get("tokens", 0)
            pct = s.get("usage_pct", 0)
            tokens_str = f"{_fmt_tokens(tok)} ({pct}%)" if tok else "0"
            age = s.get("relative_time", "?")

            elements.append({
                "tag": "column_set",
                "flex_mode": "none",
                "columns": [
                    _col(name_cell, 3),
                    _col(status, 2),
                    _col(tokens_str, 2),
                    _col(age, 1),
                ],
            })

        return elements
