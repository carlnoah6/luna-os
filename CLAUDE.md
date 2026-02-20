# CLAUDE.md - Instructions for luna-os refactoring

## Task
Refactor the task management and planner scripts in `reference/` into a clean Python package under `src/luna_os/`.

## Source Files (in reference/)
- `task_manager.py` (637 lines) — Task management CLI (add/start/complete/fail/cancel/list)
- `planner.py` (1117 lines) — Multi-step orchestration engine
- `planner_helpers.py` (324 lines) — Helper functions for planner
- `planner_spawn.py` (286 lines) — Step spawning logic
- `planner_templates.py` (76 lines) — Plan templates
- `task_store.py` (527 lines) — PostgreSQL data layer (Neon)
- `task-chat.py` (129 lines) — Lark temporary group chat management
- `contract_helper.py` (89 lines) — Event emission
- `dashboard_updater.py` — Dashboard update trigger
- `emit_event.py` — Event emission utility
- `plan_timeline.py` — Timeline graph generation
- `lark-card-builder.py` (378 lines) — Dashboard card builder
- `lark-task-dashboard.py` (176 lines) — Dashboard send/update

## Target Package Structure
```
src/luna_os/
  __init__.py              # Public API exports
  task_manager.py          # TaskManager class (add/start/complete/fail/cancel/list)
  planner.py               # Planner class (init/start/step_done/step_fail/replan/cancel/advance)
  store/
    __init__.py
    base.py                # Abstract StorageBackend interface
    postgres.py            # PostgreSQL (Neon) implementation
  notifications/
    __init__.py
    base.py                # Abstract NotificationProvider interface
    lark.py                # Lark/Feishu implementation (chat creation, messaging, cards, dashboard)
  agents/
    __init__.py
    base.py                # Abstract AgentRunner interface
    openclaw.py            # OpenClaw sessions_spawn implementation
  events.py                # Event emission and contract checking
  timeline.py              # Plan timeline/dependency graph generation
  types.py                 # Shared dataclasses (Task, Plan, Step, Event, etc.)
  cli.py                   # CLI entry point (preserves existing CLI interface)
```

## Critical Requirements

1. **THREE ABSTRACT LAYERS** — This is the core architectural change:
   - `StorageBackend` (store/base.py): abstract interface for task/plan/step/event CRUD
   - `NotificationProvider` (notifications/base.py): abstract interface for sending messages, creating chats, building cards
   - `AgentRunner` (agents/base.py): abstract interface for spawning agent sessions

2. **NO HARDCODED SECRETS** — Database connection string from `DATABASE_URL` env var. Lark credentials from `LARK_APP_ID`/`LARK_APP_SECRET` env vars. No hardcoded chat IDs, user IDs, etc.

3. **NO HARDCODED PATHS** — Remove all `/home/ubuntu/.openclaw/workspace/...` paths. Use env vars or config.

4. **English comments and docstrings** — All code in English.

5. **Keep all functionality** — Task lifecycle, plan orchestration, dependency-based step advancement, event-driven contract checking, timeline generation, dashboard cards — all must be preserved.

6. **Type hints** — Proper type hints throughout.

7. **Tests** — Write tests in `tests/` for: store interface, task lifecycle, plan creation with dependencies, ready_steps logic, event processing.

8. **pyproject.toml** — Already exists, update as needed. Add `lark-toolkit` as optional dependency (for the Lark notification provider).

9. **Delete reference/ when done** — Remove the reference directory after refactoring.

## Abstract Interface Examples

### StorageBackend
```python
class StorageBackend(ABC):
    @abstractmethod
    def add_task(self, task_id: str, description: str, ...) -> Task: ...
    @abstractmethod
    def get_task(self, task_id: str) -> Optional[Task]: ...
    @abstractmethod
    def start_task(self, task_id: str, session_key: str) -> Task: ...
    @abstractmethod
    def complete_task(self, task_id: str, result: str) -> Task: ...
    @abstractmethod
    def create_plan(self, plan_id: str, chat_id: str, goal: str, steps: list) -> Plan: ...
    @abstractmethod
    def ready_steps(self, plan_id: str) -> list[Step]: ...
    # etc.
```

### NotificationProvider
```python
class NotificationProvider(ABC):
    @abstractmethod
    def send_message(self, chat_id: str, text: str) -> None: ...
    @abstractmethod
    def create_chat(self, name: str, members: list[str]) -> str: ...
    @abstractmethod
    def send_dashboard(self, chat_id: str, card_data: dict) -> None: ...
    # etc.
```

### AgentRunner
```python
class AgentRunner(ABC):
    @abstractmethod
    def spawn(self, task_id: str, prompt: str, session_label: str) -> str: ...
    @abstractmethod
    def is_running(self, session_key: str) -> bool: ...
```

## Don't
- Don't add async unless the original code was async
- Don't change the CLI interface (task_manager.py add/start/complete/etc. should still work)
- Don't add heavy dependencies
- Don't over-abstract — keep it practical, the three layers above are enough

## After refactoring
- Run `ruff check src/` and fix any issues
- Run `pytest` and make sure tests pass
- Create a meaningful commit on the current branch
