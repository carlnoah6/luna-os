"""Watchdog alert model and dedup logic."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Severity(Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass
class Alert:
    check_name: str
    key: str  # dedup key, e.g. "stale_task:tid-0225-10"
    message: str
    severity: Severity = Severity.WARN
    context: dict = field(default_factory=dict)


# Dedup state file
DEDUP_FILE = Path.home() / ".openclaw" / "share" / "watchdog_dedup.json"
COOLDOWN_SECONDS = 15 * 60  # 15 minutes


def _load_dedup() -> dict:
    if DEDUP_FILE.exists():
        try:
            return json.loads(DEDUP_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_dedup(state: dict):
    DEDUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEDUP_FILE.write_text(json.dumps(state))


def filter_dedup(alerts: list[Alert]) -> list[Alert]:
    """Remove alerts that were already fired within cooldown period."""
    state = _load_dedup()
    now = time.time()
    new_alerts = []
    for a in alerts:
        last_fired = state.get(a.key, 0)
        if now - last_fired > COOLDOWN_SECONDS:
            new_alerts.append(a)
            state[a.key] = now
    # Clean old entries (> 24h)
    state = {k: v for k, v in state.items() if now - v < 86400}
    _save_dedup(state)
    return new_alerts
