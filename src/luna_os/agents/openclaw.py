"""OpenClaw agent runner — spawns isolated sessions via the Gateway.

Uses ``openclaw gateway call agent`` to spawn subagent sessions directly
through the Gateway RPC. Sessions are visible in ``sessions_list``,
managed by the Gateway lifecycle, and use ~2.3x fewer tokens than
``openclaw agent`` (no full workspace context).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path

from luna_os.agents.base import AgentRunner

logger = logging.getLogger(__name__)

# Default path to streaming-bridge.py
_DEFAULT_BRIDGE = (
    Path.home() / ".openclaw" / "workspace" / "scripts" / "streaming-bridge.py"
)


class OpenClawRunner(AgentRunner):
    """Spawn subagent sessions via Gateway RPC (``openclaw gateway call agent``)."""

    def __init__(
        self,
        binary: str = "openclaw",
        bridge_script: str | Path | None = None,
    ) -> None:
        self._binary = binary
        self._bridge_script = Path(bridge_script) if bridge_script else _DEFAULT_BRIDGE

    def spawn(
        self,
        task_id: str,
        prompt: str,
        session_label: str = "",
        reply_chat_id: str = "",
    ) -> str:
        """Spawn a subagent session via Gateway RPC.

        Calls ``openclaw gateway call agent`` with ``--expect-final``
        in a background process. The Gateway manages the session
        lifecycle, and the session is visible in ``sessions_list``.

        Returns the session key.
        """
        child_id = session_label or f"task-{task_id}"
        session_key = f"agent:main:subagent:{child_id}"

        params = {
            "message": prompt,
            "sessionKey": session_key,
            "idempotencyKey": str(uuid.uuid4()),
            "deliver": False,
            "lane": "subagent",
            "timeout": 1800,
        }

        cmd = [
            self._binary, "gateway", "call", "agent",
            "--params", json.dumps(params),
            "--expect-final",
            "--json",
            "--timeout", "1810000",  # slightly over agent timeout
        ]

        # Start streaming bridge BEFORE launching agent
        if reply_chat_id:
            self._start_bridge(child_id, reply_chat_id, task_id)

        # Run in background (non-blocking)
        log_path = f"/tmp/agent-spawn-{child_id}.log"
        log_fh = open(log_path, "w")  # noqa: SIM115
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )

        logger.info(
            "Spawned subagent: key=%s pid=%d task=%s",
            session_key, proc.pid, task_id,
        )

        return session_key

    def _start_bridge(
        self, session_label: str, chat_id: str, task_id: str = "",
    ) -> None:
        """Launch streaming-bridge.py in the background."""
        if not self._bridge_script.exists():
            logger.warning(
                "Streaming bridge not found at %s, skipping",
                self._bridge_script,
            )
            return

        cmd = [
            sys.executable,
            str(self._bridge_script),
            session_label,
            chat_id,
            "--timeout", "1800",
        ]
        if task_id:
            cmd.extend(["--task-id", task_id])

        try:
            log_fh = open("/tmp/streaming-bridge.log", "a")  # noqa: SIM115
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=log_fh,
                start_new_session=True,
            )
            logger.info(
                "Started streaming bridge: pid=%d label=%s chat=%s",
                proc.pid, session_label, chat_id,
            )
        except Exception as exc:
            logger.warning("Failed to start streaming bridge: %s", exc)

    def is_running(self, session_key: str) -> bool:
        """Check if a spawned subagent is still running.

        Uses ``openclaw sessions --active 5 --json`` to check if the
        session was recently active. Falls back to session file check.
        """
        try:
            result = subprocess.run(
                [self._binary, "sessions", "--active", "5", "--json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return session_key in result.stdout
        except Exception:
            pass

        # Fallback: check session file
        return self._check_session_file(session_key)

    @staticmethod
    def _check_session_file(session_key: str) -> bool:
        """Fallback: check if a session file was recently modified."""
        import time

        sessions_dir = os.environ.get(
            "OPENCLAW_SESSIONS_DIR",
            str(Path.home() / ".openclaw" / "agents" / "main" / "sessions"),
        )
        # Session key format: agent:main:subagent:task-xxx
        # File might be named by the last segment or by UUID
        short_key = session_key.rsplit(":", 1)[-1] if ":" in session_key else session_key
        jsonl_path = Path(sessions_dir) / f"{short_key}.jsonl"
        if jsonl_path.exists():
            age = time.time() - jsonl_path.stat().st_mtime
            if age < 300:
                return True
        return False
