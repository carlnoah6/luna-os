"""Command handlers for the interceptor.

Each handler is an async function that takes (user_text, InterceptResult)
and returns a Feishu card response dict.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any, Callable

from luna_os.interceptor.types import InterceptResult

logger = logging.getLogger(__name__)


def _make_card(title: str, content: str, template: str = "blue") -> dict[str, Any]:
    """Helper to create a Feishu card response."""
    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"content": title, "tag": "plain_text"},
                "template": template,
            },
            "elements": [
                {"tag": "div", "text": {"content": content, "tag": "lark_md"}},
            ],
        },
    }


def _make_error_card(error_msg: str) -> dict[str, Any]:
    """Helper to create an error card."""
    return _make_card("❌ 错误", f"```\n{error_msg}\n```", template="red")


async def handle_dashboard(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show the task/plan dashboard."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "luna-os", "dashboard",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            error = stderr.decode() if stderr else "Unknown error"
            return _make_error_card(f"Dashboard command failed: {error}")
        
        output = stdout.decode().strip()
        if not output:
            output = "No active tasks or plans."
        
        return _make_card("📊 Dashboard", output, template="blue")
    
    except Exception as e:
        logger.exception("Dashboard handler failed")
        return _make_error_card(str(e))


async def handle_task_list(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """List current tasks."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "luna-os", "task", "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            error = stderr.decode() if stderr else "Unknown error"
            return _make_error_card(f"Task list command failed: {error}")
        
        output = stdout.decode().strip()
        if not output:
            output = "No tasks found."
        
        return _make_card("📋 Task List", output, template="blue")
    
    except Exception as e:
        logger.exception("Task list handler failed")
        return _make_error_card(str(e))


async def handle_model(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show or switch the current LLM model."""
    try:
        # Check if user wants to switch model
        text_lower = user_text.lower()
        model_keywords = {
            "sonnet": "claude-sonnet-4",
            "opus": "claude-opus-4",
            "haiku": "claude-haiku-3.5",
            "gpt-4": "gpt-4",
            "gpt-3.5": "gpt-3.5-turbo",
        }
        
        for keyword, model_name in model_keywords.items():
            if keyword in text_lower:
                # TODO: Implement model switching via config update
                return _make_card(
                    "🔄 Model Switch",
                    f"Model switching to `{model_name}` is not yet implemented.\n\n"
                    f"Please update your config manually.",
                    template="yellow",
                )
        
        # Show current model
        # TODO: Read from actual config
        return _make_card(
            "🤖 Current Model",
            "Current model: `claude-sonnet-4-6`\n\n"
            "To switch models, mention the model name (e.g., 'use opus', 'switch to haiku')",
            template="blue",
        )
    
    except Exception as e:
        logger.exception("Model handler failed")
        return _make_error_card(str(e))


async def handle_new(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Start a new conversation session."""
    try:
        return _make_card(
            "✨ New Session",
            "新对话已创建！\n\n请直接发送消息开始对话。",
            template="green",
        )
    except Exception as e:
        logger.exception("New session handler failed")
        return _make_error_card(str(e))


async def handle_help(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show available commands and help."""
    try:
        help_text = """**可用命令：**

📊 **仪表盘** - 查看任务和计划进度
   示例：`仪表盘` `dashboard` `看板`

📋 **任务列表** - 查看当前任务
   示例：`任务列表` `tasks` `待办`

🤖 **模型切换** - 切换或查看当前 LLM 模型
   示例：`/model` `切换模型` `用 opus`

✨ **新对话** - 开始新的对话会话
   示例：`/new` `新对话` `重新开始`

📈 **时间线** - 查看活动时间线
   示例：`时间线` `timeline` `最近做了什么`

💰 **费用统计** - 查看 token 使用情况
   示例：`cost` `花了多少钱` `token usage`

❓ **帮助** - 显示此帮助信息
   示例：`help` `帮助` `你能做什么`
"""
        return _make_card("❓ Help", help_text, template="blue")
    
    except Exception as e:
        logger.exception("Help handler failed")
        return _make_error_card(str(e))


async def handle_cost(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show token usage and cost statistics."""
    try:
        return _make_card(
            "💰 Cost Statistics",
            "Token 统计功能开发中...\n\n"
            "敬请期待！",
            template="yellow",
        )
    except Exception as e:
        logger.exception("Cost handler failed")
        return _make_error_card(str(e))


async def handle_timeline(user_text: str, result: InterceptResult) -> dict[str, Any]:
    """Show activity timeline."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "luna-os", "timeline",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            error = stderr.decode() if stderr else "Unknown error"
            return _make_error_card(f"Timeline command failed: {error}")
        
        output = stdout.decode().strip()
        if not output:
            output = "No recent activity."
        
        return _make_card("📈 Timeline", output, template="blue")
    
    except Exception as e:
        logger.exception("Timeline handler failed")
        return _make_error_card(str(e))


# Handler registry - maps handler names to functions
HANDLER_REGISTRY: dict[str, Callable[[str, InterceptResult], Any]] = {
    "dashboard": handle_dashboard,
    "task_list": handle_task_list,
    "model": handle_model,
    "new": handle_new,
    "help": handle_help,
    "cost": handle_cost,
    "timeline": handle_timeline,
}
