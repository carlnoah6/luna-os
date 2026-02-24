"""Command handlers for the interceptor.

Each handler is an async function that takes (user_text, InterceptResult)
and returns a Feishu card response dict (or text dict).
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from collections.abc import Callable
from typing import Any

from luna_os.interceptor.types import InterceptResult

logger = logging.getLogger(__name__)

PREFIX = "[拦截器]"


def _make_card(title: str, content: str, template: str = "blue") -> dict[str, Any]:
    """Helper to create a Feishu card response."""
    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"content": f"{PREFIX} {title}", "tag": "plain_text"},
                "template": template,
            },
            "elements": [
                {"tag": "div", "text": {"content": content, "tag": "lark_md"}},
            ],
        },
    }


def _make_error_card(error_msg: str) -> dict[str, Any]:
    return _make_card("❌ 错误", f"```\n{error_msg}\n```", template="red")


def _run_cmd(*args: str, timeout: int = 10) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def handle_dashboard(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show the task/plan dashboard."""
    try:
        rc, stdout, stderr = _run_cmd("luna-os", "task", "status")
        if rc != 0:
            return _make_error_card(f"task status failed: {stderr}")

        status = json.loads(stdout)
        total = status.get("total", 0)
        counts = status.get("counts", {})

        content = f"""**任务统计**

- 总任务数：{total}
- 运行中：{counts.get('running', 0)}
- 排队中：{counts.get('queued', 0)}
- 已完成：{counts.get('done', 0)}
- 失败：{counts.get('failed', 0)}"""

        return _make_card("📊 仪表盘", content)
    except Exception as e:
        logger.exception("Dashboard handler failed")
        return _make_error_card(str(e))


async def handle_tasks(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """List current tasks."""
    try:
        rc, stdout, stderr = _run_cmd("luna-os", "task", "list", "--status", "running,queued")
        if rc != 0:
            return _make_error_card(f"task list failed: {stderr}")

        tasks = json.loads(stdout) if stdout.strip() else []
        if not tasks:
            return _make_card("📋 任务列表", "当前没有运行中或排队中的任务。", template="green")

        lines = []
        for t in tasks[:10]:
            tid = t.get("id", "?")
            desc = (t.get("description") or "")[:60]
            status = t.get("status", "?")
            emoji = {"running": "🔄", "queued": "⏳"}.get(status, "📌")
            lines.append(f"{emoji} `{tid}` {desc}")

        if len(tasks) > 10:
            lines.append(f"\n... 还有 {len(tasks) - 10} 个任务")

        return _make_card("📋 任务列表", "\n".join(lines))
    except Exception as e:
        logger.exception("Tasks handler failed")
        return _make_error_card(str(e))


async def handle_model(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show current model info."""
    try:
        rc, stdout, stderr = _run_cmd(
            "openclaw", "sessions", "--active", "5", "--json",
        )
        if rc != 0:
            return _make_error_card(f"sessions failed: {stderr}")

        sessions = json.loads(stdout) if stdout.strip() else []
        if sessions:
            model = sessions[0].get("model", "unknown")
        else:
            model = "unknown"

        content = f"""**当前模型**: `{model}`

切换方式：直接说"用 sonnet/opus/haiku"
或发送 `/model <provider/model>`"""

        return _make_card("🧠 模型", content)
    except Exception as e:
        logger.exception("Model handler failed")
        return _make_error_card(str(e))


async def handle_new(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Start a new session — forward to OpenClaw."""
    return _make_card(
        "🔄 新对话",
        "新对话请求已收到。\n\n⚠️ 此命令需要 OpenClaw 处理，消息已转发。",
        template="orange",
    )


async def handle_plan(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show current plan status."""
    try:
        rc, stdout, stderr = _run_cmd("luna-os", "plan", "list", "--status", "active")
        if rc != 0:
            return _make_error_card(f"plan list failed: {stderr}")

        plans = json.loads(stdout) if stdout.strip() else []
        if not plans:
            return _make_card("📐 计划状态", "当前没有活跃的计划。", template="green")

        lines = []
        for p in plans[:5]:
            pid = p.get("id", "?")
            goal = (p.get("goal") or "")[:60]
            total = p.get("total_steps", 0)
            done = p.get("done_steps", 0)
            lines.append(f"📐 `{pid}` {goal}")
            lines.append(f"   进度：{done}/{total} 步")

        return _make_card("📐 计划状态", "\n".join(lines))
    except Exception as e:
        logger.exception("Plan handler failed")
        return _make_error_card(str(e))


async def handle_usage(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show token usage / cost."""
    try:
        rc, stdout, stderr = _run_cmd(
            "openclaw", "sessions", "--active", "60", "--json",
        )
        if rc != 0:
            return _make_error_card(f"sessions failed: {stderr}")

        sessions = json.loads(stdout) if stdout.strip() else []
        total_in = 0
        total_out = 0
        for s in sessions:
            total_in += s.get("tokensIn", 0)
            total_out += s.get("tokensOut", 0)

        content = f"""**最近 1 小时用量**

- 活跃 session 数：{len(sessions)}
- 输入 tokens：{total_in:,}
- 输出 tokens：{total_out:,}
- 总计：{total_in + total_out:,}"""

        return _make_card("💰 用量统计", content)
    except Exception as e:
        logger.exception("Usage handler failed")
        return _make_error_card(str(e))


async def handle_help(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show available commands."""
    content = """**可用命令**

- `/dashboard` (仪表盘) — 任务面板
- `/tasks` (任务列表) — 查看任务
- `/model` (当前模型) — 查看/切换模型
- `/new` (新对话) — 开始新对话
- `/plan` (计划状态) — 查看计划进度
- `/usage` (花费) — Token 用量统计
- `/timeline` (时间线) — 活动记录
- `/status` (系统状态) — 系统信息
- `/commands` (命令列表) — 所有命令
- `/stop` (停止) — 停止当前任务
- `/compact` (压缩) — 压缩上下文
- `/help` (帮助) — 显示此帮助

💡 以上命令由拦截器直接处理，不消耗 LLM token"""

    return _make_card("❓ 帮助", content, template="green")


async def handle_timeline(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show activity timeline."""
    try:
        rc, stdout, stderr = _run_cmd(
            "luna-os", "task", "list", "--status", "done", "--limit", "5",
        )
        if rc != 0:
            return _make_error_card(f"task list failed: {stderr}")

        tasks = json.loads(stdout) if stdout.strip() else []
        if not tasks:
            return _make_card("📈 时间线", "暂无最近完成的任务。")

        lines = []
        for t in tasks[:5]:
            tid = t.get("id", "?")
            desc = (t.get("description") or "")[:50]
            completed = t.get("completed_at", "")
            if completed:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(completed.replace("Z", "+00:00"))
                    time_str = dt.strftime("%m-%d %H:%M")
                except (ValueError, AttributeError):
                    time_str = completed[:16]
            else:
                time_str = "?"
            lines.append(f"✅ `{tid}` {desc}")
            lines.append(f"   🕐 {time_str}")

        return _make_card("📈 时间线", "\n".join(lines))
    except Exception as e:
        logger.exception("Timeline handler failed")
        return _make_error_card(str(e))


async def handle_status(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show system status."""
    try:
        rc, stdout, stderr = _run_cmd(
            "openclaw", "sessions", "--active", "5", "--json",
        )
        sessions = json.loads(stdout) if stdout.strip() and rc == 0 else []
        model = sessions[0].get("model", "unknown") if sessions else "unknown"

        rc2, stdout2, _ = _run_cmd("luna-os", "task", "status")
        task_status = json.loads(stdout2) if stdout2.strip() and rc2 == 0 else {}
        counts = task_status.get("counts", {})

        content = f"""**系统状态**

- 模型：`{model}`
- 活跃 session：{len(sessions)}
- 运行中任务：{counts.get('running', 0)}
- 排队中任务：{counts.get('queued', 0)}
- 拦截器：✅ 运行中"""

        return _make_card("🔧 系统状态", content, template="green")
    except Exception as e:
        logger.exception("Status handler failed")
        return _make_error_card(str(e))


async def handle_commands(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """List all registered commands."""
    from luna_os.interceptor.registry import CommandRegistry

    registry = CommandRegistry()
    lines = ["**已注册命令**\n"]
    for cmd in registry.all_commands():
        primary = cmd.patterns[0] if cmd.patterns else cmd.id
        aliases = ", ".join(cmd.patterns[1:3]) if len(cmd.patterns) > 1 else ""
        alias_str = f" ({aliases})" if aliases else ""
        lines.append(f"- `{primary}`{alias_str}")

    lines.append(f"\n共 {len(registry.all_commands())} 个命令")
    return _make_card("📜 命令列表", "\n".join(lines))


async def handle_stop(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Stop — forward to OpenClaw."""
    return _make_card(
        "🛑 停止",
        "停止请求已收到。\n\n⚠️ 此命令需要 OpenClaw 处理，消息已转发。",
        template="red",
    )


async def handle_compact(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Compact — forward to OpenClaw."""
    return _make_card(
        "📦 压缩上下文",
        "压缩请求已收到。\n\n⚠️ 此命令需要 OpenClaw 处理，消息已转发。",
        template="orange",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HANDLER_REGISTRY: dict[str, Callable[[str, InterceptResult], Any]] = {
    "dashboard": handle_dashboard,
    "tasks": handle_tasks,
    "task_list": handle_tasks,
    "model": handle_model,
    "model_switch": handle_model,
    "new": handle_new,
    "new_session": handle_new,
    "plan": handle_plan,
    "plan_status": handle_plan,
    "usage": handle_usage,
    "token_usage": handle_usage,
    "cost": handle_usage,
    "help": handle_help,
    "timeline": handle_timeline,
    "status": handle_status,
    "commands": handle_commands,
    "stop": handle_stop,
    "compact": handle_compact,
}
