"""OpenClaw agent runner — spawns isolated sessions via the Gateway.

Uses ``openclaw agent`` to run isolated sessions directly. The Gateway
manages the session lifecycle, streaming, and delivery.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from luna_os.agents.base import AgentRunner

logger = logging.getLogger(__name__)

# Default path to streaming-bridge.py
_DEFAULT_BRIDGE = Path.home() / ".openclaw" / "workspace" / "scripts" / "streaming-bridge.py"


class OpenClawRunner(AgentRunner):
    """Spawn agent sessions via OpenClaw Gateway's cron one-shot mechanism."""

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
        """Spawn an isolated agent session via ``openclaw agent``.

        Runs the agent directly (no cron delay). The Gateway manages
        the session lifecycle. If *reply_chat_id* is provided, results
        are delivered to that Feishu chat and a streaming bridge is
        started for live progress.

        Returns the session id used.
        """
        name = session_label or f"task-{task_id}"

        cmd = [
            self._binary, "agent",
            "--message", prompt,
            "--session-id", name,
            "--timeout", "1800",
            "--json",
        ]

        # Deliver results to Feishu chat if available
        if reply_chat_id:
            cmd.extend([
                "--deliver",
                "--reply-channel", "feishu",
                "--reply-to", f"chat:{reply_chat_id}",
            ])

        # Start streaming bridge BEFORE launching agent
        if reply_chat_id:
            self._start_bridge(name, reply_chat_id, task_id)

        # Run agent in background (non-blocking)
        log_fh = open(  # noqa: SIM115
            f"/tmp/agent-spawn-{name}.log", "a"
        )
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )

        logger.info(
            "Spawned agent session: name=%s pid=%d task=%s",
            name, proc.pid, task_id,
        )

        return name

    def _start_bridge(
        self, session_label: str, chat_id: str, task_id: str = ""
    ) -> None:
        """Launch streaming-bridge.py in the background.

        The bridge watches for a new session file matching *session_label*
        and streams output to the Lark *chat_id* via CardKit.
        """
        if not self._bridge_script.exists():
            logger.warning(
                "Streaming bridge not found at %s, skipping", self._bridge_script
            )
            return

        cmd = [
            sys.executable,
            str(self._bridge_script),
            session_label,  # bridge's find_session_file will search for this
            chat_id,
            "--timeout", "1800",
        ]
        if task_id:
            cmd.extend(["--task-id", task_id])

        try:
            log_fh = open(  # noqa: SIM115
                "/tmp/streaming-bridge.log", "a"
            )
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=log_fh,
                start_new_session=True,  # detach from parent
            )
            logger.info(
                "Started streaming bridge: pid=%d label=%s chat=%s",
                proc.pid, session_label, chat_id,
            )
        except Exception as exc:
            logger.warning("Failed to start streaming bridge: %s", exc)

    def is_running(self, session_key: str) -> bool:
        """Check if a spawned agent session is still running.

        Checks if an ``openclaw agent`` process with the matching
        session-id is still alive, or if the session file was recently
        modified (within 5 minutes).
        """
        # Method 1: check for running process
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"--session-id {session_key}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except Exception:
            pass

        # Method 2: check session file recency
        return self._check_session_file(session_key)

    @staticmethod
    def _check_session_file(session_key: str) -> bool:
        """Fallback: check if a session file exists and was recently modified."""
        import time

        sessions_dir = os.environ.get(
            "OPENCLAW_SESSIONS_DIR",
            str(Path.home() / ".openclaw" / "agents" / "main" / "sessions"),
        )
        jsonl_path = Path(sessions_dir) / f"{session_key}.jsonl"
        if jsonl_path.exists():
            age = time.time() - jsonl_path.stat().st_mtime
            if age < 300:  # Modified within 5 minutes
                return True
        return False
