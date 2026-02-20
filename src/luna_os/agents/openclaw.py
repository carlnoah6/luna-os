"""OpenClaw agent runner implementation."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from luna_os.agents.base import AgentRunner


class OpenClawRunner(AgentRunner):
    """Agent runner that spawns OpenClaw sessions."""

    def __init__(self, binary: str = "openclaw") -> None:
        self._binary = binary

    def spawn(self, task_id: str, prompt: str, session_label: str = "") -> str:
        """Spawn an OpenClaw agent session.

        Returns the session key used to identify this session.
        """
        session_id = session_label or f"task-{task_id}"
        cmd = [
            self._binary,
            "agent",
            "--session-id",
            session_id,
            "--timeout",
            "900",
            "--message",
            prompt,
        ]
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return session_id

    def is_running(self, session_key: str) -> bool:
        """Check if an OpenClaw agent session is still alive.

        Uses session file modification time as the primary signal:
        - If .jsonl was modified in the last 3 minutes, the session is alive.
        - If .jsonl.lock exists and was created recently, the session is alive.
        - If a circuit breaker .loop-* file exists, the session is dead.
        """
        sessions_dir = os.environ.get(
            "OPENCLAW_SESSIONS_DIR",
            str(Path.home() / ".openclaw" / "agents" / "main" / "sessions"),
        )
        session_dir = Path(sessions_dir)

        # Prevent path traversal
        safe_key = Path(session_key).name
        if safe_key != session_key or ".." in session_key:
            return False

        jsonl_path = session_dir / f"{safe_key}.jsonl"
        lock_path = session_dir / f"{safe_key}.jsonl.lock"

        # Circuit breaker check
        import glob as _glob

        if _glob.glob(str(session_dir / f"{safe_key}.jsonl.loop-*")):
            return False

        now = time.time()

        if jsonl_path.exists():
            mtime = os.path.getmtime(jsonl_path)
            if now - mtime < 180:
                return True

        if lock_path.exists():
            mtime = os.path.getmtime(lock_path)
            if now - mtime < 180:
                return True

        return False
