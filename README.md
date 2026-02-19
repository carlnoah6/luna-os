# luna-os

Task orchestration & planner framework for AI agents.

## Components

- **TaskStore** — PostgreSQL-backed task registry (add / start / complete / fail / cancel / list)
- **Planner** — Multi-step plan orchestration with dependency tracking and automatic step progression
- **ContractHelper** — Event-driven contract system for step completion signaling
- **Dashboard** — Lark-integrated task dashboard with auto-refresh

## Install

```bash
pip install -e ".[dev]"
```

## Development

```bash
# lint
ruff check src/ tests/

# test
pytest
```

## License

MIT
