"""Session overview generator.

Reads OpenClaw session store + Lark chat name cache, generates a concise
overview of all active chat sessions (excludes subagent sessions).

Usage (CLI):
    luna-os session-overview [--stdout]
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta, timezone

logger = logging.getLogger(__name__)

SGT = timezone(timedelta(hours=8))

SESSIONS_FILE = os.path.expanduser("~/.openclaw/agents/main/sessions/sessions.json")
SESSIONS_DIR = os.path.expanduser("~/.openclaw/agents/main/sessions")
CHAT_CACHE = os.path.expanduser("~/.openclaw/workspace/data/lark-chats-cache.json")


def _load_chat_names() -> dict[str, str]:
    """Load chat_id -> name mapping from lark-lookup-chat cache."""
    mapping: dict[str, str] = {}
    if os.path.exists(CHAT_CACHE):
        try:
            with open(CHAT_CACHE) as f:
                data = json.load(f)
            for c in data.get("chats", []):
                mapping[c["chat_id"]] = c["name"]
        except Exception:
            pass
    return mapping


def _load_planners() -> dict[str, dict]:
    """Load chat_id -> planner info mapping from PostgreSQL."""
    mapping: dict[str, dict] = {}
    try:
        from luna_os.store.postgres import PostgresBackend

        store = PostgresBackend()
        all_plans_raw = store.list_plans() or []
        all_plans = [
            p.to_dict() if hasattr(p, "to_dict") else p for p in all_plans_raw
        ]
        now = datetime.now(UTC)

        def _still_relevant(p: dict) -> bool:
            status = p.get("status", "")
            if status == "cancelled":
                return False
            if status in ("completed", "failed"):
                created = p.get("created_at")
                if created and hasattr(created, "timestamp"):
                    age_h = (now - created).total_seconds() / 3600
                    if age_h > 24:
                        return False
            return True

        plans = sorted(
            [p for p in all_plans if _still_relevant(p)],
            key=lambda p: p.get("updated_at") or p.get("created_at") or now,
            reverse=True,
        )[:30]

        for p in plans:
            chat_id = p.get("chat_id")
            if not chat_id or chat_id in mapping:
                continue
            plan_id = p["id"]
            steps = p.get("steps", [])
            done = sum(
                1
                for s in steps
                if (s.get("status") if isinstance(s, dict) else getattr(s, "status", None))
                in ("done", "completed")
            )
            total = len(steps)
            status = p.get("status", "unknown")
            goal = p.get("goal", "")
            mapping[chat_id] = {
                "plan_id": plan_id,
                "status": status,
                "goal": goal[:60],
                "progress": f"{done}/{total}",
            }
    except Exception:
        logger.debug("Failed to load planners", exc_info=True)

    return mapping


def _relative_time(ts: float) -> str:
    """Convert timestamp to relative time string."""
    diff = time.time() - ts
    if diff < 60:
        return "just now"
    if diff < 3600:
        return f"{int(diff / 60)}m ago"
    if diff < 86400:
        return f"{int(diff / 3600)}h ago"
    return f"{int(diff / 86400)}d ago"


def generate_overview(*, output_file: str | None = None) -> dict:
    """Generate session overview.

    Returns dict with sessions list and counts.
    """
    # Load sessions.json
    if not os.path.exists(SESSIONS_FILE):
        return {"sessions": [], "total_count": 0, "chat_count": 0, "subagent_count": 0}

    with open(SESSIONS_FILE) as f:
        all_sessions = json.load(f)

    chat_names = _load_chat_names()
    planners = _load_planners()

    chat_sessions = []
    for key, meta in all_sessions.items():
        # Skip subagent sessions
        if "subagent" in key:
            continue

        # Extract chat_id from key
        chat_id = ""
        if key.startswith("feishu:group:") or key.startswith("feishu:dm:"):
            chat_id = key.split(":", 2)[2]

        # Session name
        name = chat_names.get(chat_id, chat_id or key)

        # Usage
        usage = meta.get("usage", {})
        input_tokens = usage.get("input", 0)
        output_tokens = usage.get("output", 0)
        total_tokens = input_tokens + output_tokens
        context_tokens = meta.get("contextTokens", 128000)
        usage_pct = round(total_tokens / context_tokens * 100, 1) if context_tokens else 0

        # Timing
        updated_at = meta.get("updatedAt", 0)
        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at).timestamp()
            except Exception:
                updated_at = 0
        relative = _relative_time(updated_at) if updated_at else "unknown"

        # Last activity
        last_msg = meta.get("lastMessage", {})
        last_activity = ""
        if isinstance(last_msg, dict):
            role = last_msg.get("role", "")
            text = last_msg.get("text", "")[:60]
            last_activity = f"[{role}] {text}" if text else ""

        # Planner info
        planner_info = planners.get(chat_id)

        chat_sessions.append({
            "key": key,
            "chat_id": chat_id,
            "name": name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "usage_pct": usage_pct,
            "updated_at": updated_at,
            "relative_time": relative,
            "last_activity": last_activity,
            "compactions": meta.get("compactionCount", 0),
            "planner": planner_info,
        })

    chat_sessions.sort(key=lambda s: s["name"])

    result = {
        "sessions": chat_sessions,
        "total_count": len(all_sessions),
        "chat_count": len(chat_sessions),
        "subagent_count": len(all_sessions) - len(chat_sessions),
        "generated_at": datetime.now(SGT).isoformat(),
    }

    # Save if output_file specified
    out = output_file or os.path.expanduser(
        "~/.openclaw/workspace/data/session-overview.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="[session-overview] %(message)s")

    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h", "help"):
        print(__doc__)
        sys.exit(0)

    result = generate_overview()
    if "--stdout" in sys.argv:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({
            "ok": True,
            "chat_sessions": len(result["sessions"]),
            "subagents": result["subagent_count"],
        }))
