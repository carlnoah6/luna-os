"""Command handlers for the interceptor.

Each handler is an async function that takes (user_text, InterceptResult)
and returns a Feishu card response dict (or text dict).
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from typing import Any

from luna_os.interceptor.types import InterceptResult

logger = logging.getLogger(__name__)

PREFIX = "[拦截器]"


def _make_card(title: str, content: str, template: str = "blue") -> dict[str, Any]:
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
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def _get_sessions(active_minutes: int = 5) -> list[dict[str, Any]]:
    """Get active sessions from OpenClaw."""
    rc, stdout, stderr = _run_cmd(
        "openclaw", "sessions", "--active", str(active_minutes), "--json",
    )
    if rc != 0 or not stdout.strip():
        return []
    try:
        data = json.loads(stdout)
        # openclaw sessions --json returns {sessions: [...]}
        if isinstance(data, dict):
            return data.get("sessions", [])
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def handle_dashboard(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show the task/plan dashboard using the existing card builder."""
    try:
        import os
        card_builder = os.path.expanduser(
            "~/.openclaw/workspace/scripts/lark-card-builder.py"
        )
        rc, stdout, stderr = _run_cmd("python3", card_builder, timeout=15)
        if rc != 0:
            return _make_error_card(f"card builder failed: {stderr}")

        card = json.loads(stdout)
        # Prepend [拦截器] to the title
        header = card.get("header", {})
        title = header.get("title", {})
        if title.get("content"):
            title["content"] = f"{PREFIX} {title['content']}"

        return {"msg_type": "interactive", "card": card}
    except Exception as e:
        logger.exception("Dashboard handler failed")
        return _make_error_card(str(e))
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
        sessions = _get_sessions(10)
        model = "unknown"
        for s in sessions:
            if isinstance(s, dict) and s.get("model"):
                model = s["model"]
                break

        content = f"""**当前模型**: `{model}`

切换方式：直接说"用 sonnet/opus/haiku"
或发送 `/model <provider/model>`"""

        return _make_card("🧠 模型", content)
    except Exception as e:
        logger.exception("Model handler failed")
        return _make_error_card(str(e))


async def handle_new(user_text: str, result: InterceptResult) -> dict[str, Any]:
    return _make_card(
        "🔄 新对话",
        "新对话请求已收到，消息已转发给 OpenClaw 处理。",
        template="orange",
    )


async def handle_plan(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show current plan status."""
    try:
        rc, stdout, stderr = _run_cmd("luna-os", "plan", "list", "--status", "active")
        if rc != 0:
            return _make_error_card(f"plan list failed: {stderr}")

        if not stdout.strip():
            return _make_card("📐 计划状态", "当前没有活跃的计划。", template="green")

        # Parse text output
        lines = []
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if line and "[active]" in line:
                lines.append(f"🔄 {line}")
            elif line and "[paused]" in line:
                lines.append(f"⏸️ {line}")

        if not lines:
            return _make_card("📐 计划状态", "当前没有活跃的计划。", template="green")

        return _make_card("📐 计划状态", "\n".join(lines[:10]))
    except Exception as e:
        logger.exception("Plan handler failed")
        return _make_error_card(str(e))


async def handle_usage(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show token usage / cost."""
    try:
        sessions = _get_sessions(60)
        total_in = 0
        total_out = 0
        count = 0
        for s in sessions:
            if not isinstance(s, dict):
                continue
            total_in += s.get("inputTokens", 0) or 0
            total_out += s.get("outputTokens", 0) or 0
            count += 1

        def fmt_tokens(n: int) -> str:
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            if n >= 1_000:
                return f"{n / 1_000:.1f}k"
            return str(n)

        content = f"""**最近 1 小时用量**

- 活跃 Session：{count} 个
- 输入 Token：{fmt_tokens(total_in)}
- 输出 Token：{fmt_tokens(total_out)}
- 总计：{fmt_tokens(total_in + total_out)}"""

        return _make_card("💰 用量", content)
    except Exception as e:
        logger.exception("Usage handler failed")
        return _make_error_card(str(e))


async def handle_help(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show available commands."""
    content = """**拦截器命令**（不消耗 LLM token）

- `/dashboard` — 仪表盘（任务+计划+session）
- `/tasks` — 任务列表
- `/model` — 当前模型
- `/plan` — 计划状态
- `/usage` — Token 用量
- `/status` — 系统状态
- `/timeline` — 活动时间线
- `/commands` — 命令列表
- `/help` — 帮助

**转发命令**（拦截器通知 + OpenClaw 处理）
- `/new` — 新对话
- `/stop` — 停止任务
- `/compact` — 压缩上下文

中文别名也可以用，如"仪表盘"、"任务列表"等。"""

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
            time_str = completed[:16] if completed else "?"
            lines.append(f"✅ `{tid}` {desc}")
            lines.append(f"   🕐 {time_str}")

        return _make_card("📈 时间线", "\n".join(lines))
    except Exception as e:
        logger.exception("Timeline handler failed")
        return _make_error_card(str(e))


async def handle_status(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show system status."""
    try:
        sessions = _get_sessions(5)
        model = "unknown"
        context = "?"
        for s in sessions:
            if not isinstance(s, dict):
                continue
            if s.get("model"):
                model = s["model"]
            ctx = s.get("totalTokens", 0) or 0
            max_ctx = s.get("contextTokens", 128000) or 128000
            if ctx > 0:
                context = f"{ctx // 1000}k / {max_ctx // 1000}k ({ctx * 100 // max_ctx}%)"
                break

        # Task counts
        rc, stdout, _ = _run_cmd("luna-os", "task", "status")
        task_info = ""
        if rc == 0 and stdout.strip():
            status = json.loads(stdout)
            counts = status.get("counts", {})
            task_info = f"运行: {counts.get('running', 0)} | 排队: {counts.get('queued', 0)}"

        content = f"""**系统状态**

- 模型：`{model}`
- 上下文：{context}
- 任务：{task_info}
- Session 数：{len(sessions)} 个（5 分钟内活跃）"""

        return _make_card("📊 系统状态", content)
    except Exception as e:
        logger.exception("Status handler failed")
        return _make_error_card(str(e))


async def handle_commands(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """List all registered commands."""
    from luna_os.interceptor.registry import CommandRegistry

    registry = CommandRegistry()
    cmds = list(registry.all_commands())

    lines = [f"**已注册 {len(cmds)} 个命令**\n"]
    for cmd in cmds:
        patterns = ", ".join(cmd.patterns[:3])
        lines.append(f"- `{cmd.id}` — {patterns}")

    return _make_card("📜 命令列表", "\n".join(lines))


async def handle_stop(user_text: str, result: InterceptResult) -> dict[str, Any]:
    return _make_card(
        "🛑 停止",
        "停止请求已收到，消息已转发给 OpenClaw 处理。",
        template="red",
    )


async def handle_compact(user_text: str, result: InterceptResult) -> dict[str, Any]:
    return _make_card(
        "📦 压缩上下文",
        "压缩请求已收到，消息已转发给 OpenClaw 处理。",
        template="orange",
    )


async def handle_share(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Create a share link for the current conversation."""
    import subprocess
    import json
    
    session_id = result.session_id
    if not session_id:
        return _make_card(
            "❌ 分享失败",
            "无法获取当前 session ID",
            template="red",
        )
    
    # Call create_share_from_last_new.py
    script_path = "/home/ubuntu/.openclaw/workspace/projects/luna-share/create_share_from_last_new.py"
    try:
        proc = subprocess.run(
            ["python3", script_path, session_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if proc.returncode != 0:
            return _make_card(
                "❌ 分享失败",
                f"脚本执行失败：{proc.stderr[:200]}",
                template="red",
            )
        
        # Parse JSON output from last line
        lines = proc.stdout.strip().split('\n')
        for line in reversed(lines):
            if line.startswith('{'):
                data = json.loads(line)
                share_url = data.get('url', '')
                message_count = data.get('message_count', 0)
                
                return _make_card(
                    "✅ 分享链接已生成",
                    f"🔗 {share_url}\n\n📝 包含 {message_count} 条消息",
                    template="green",
                )
        
        return _make_card(
            "❌ 分享失败",
            "无法解析脚本输出",
            template="red",
        )
    
    except subprocess.TimeoutExpired:
        return _make_card(
            "❌ 分享失败",
            "脚本执行超时（30秒）",
            template="red",
        )
    except Exception as e:
        return _make_card(
            "❌ 分享失败",
            f"错误：{str(e)[:200]}",
            template="red",
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
    "share": handle_share,
}
