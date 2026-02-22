"""Default command definitions for the interceptor.

Each command has a canonical name, a handler key, and example phrases
that users might type.  The embedding engine uses these examples to
match incoming messages without an LLM call.
"""

from __future__ import annotations

from luna_os.interceptor.embedding import CommandDef

# fmt: off
DEFAULT_COMMANDS: list[CommandDef] = [
    CommandDef(
        name="dashboard",
        handler="dashboard",
        description="Show the task/plan dashboard",
        examples=(
            "仪表盘", "dashboard", "看板", "任务看板",
            "show dashboard", "打开仪表盘", "看一下进度",
            "项目进度", "plan status", "计划状态",
        ),
    ),
    CommandDef(
        name="task_list",
        handler="task_list",
        description="List current tasks",
        examples=(
            "任务列表", "task list", "tasks", "我的任务",
            "待办事项", "todo list", "show tasks",
            "有什么任务", "还有哪些任务",
        ),
    ),
    CommandDef(
        name="model_switch",
        handler="model",
        description="Switch or show current LLM model",
        examples=(
            "/model", "切换模型", "switch model", "用什么模型",
            "当前模型", "current model", "change model",
        ),
    ),
    CommandDef(
        name="new_session",
        handler="new",
        description="Start a new conversation session",
        examples=(
            "/new", "新对话", "new session", "new chat",
            "重新开始", "start over", "清空对话",
        ),
    ),
    CommandDef(
        name="help",
        handler="help",
        description="Show help / available commands",
        examples=(
            "帮助", "help", "你能做什么", "what can you do",
            "命令列表", "有哪些命令", "怎么用",
        ),
    ),
    CommandDef(
        name="cost",
        handler="cost",
        description="Show token/cost usage",
        examples=(
            "花了多少钱", "cost", "token usage", "用了多少 token",
            "费用", "消耗", "usage stats",
        ),
    ),
    CommandDef(
        name="timeline",
        handler="timeline",
        description="Show activity timeline",
        examples=(
            "时间线", "timeline", "活动记录", "最近做了什么",
            "activity log", "history",
        ),
    ),
]
# fmt: on
