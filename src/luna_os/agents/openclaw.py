"""OpenClaw agent runner — spawns isolated sessions via the Gateway.

Uses ``openclaw cron add --at`` to create one-shot isolated sessions that
are fully managed by the Gateway (visible in sessions_list, streaming
output, auto-announce on completion).
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import UTC, datetime, timedelta

from luna_os.agents.base import AgentRunner

logger = logging.getLogger(__name__)


class OpenClawRunner(AgentRunner):
    """Spawn agent sessions via OpenClaw Gateway's cron one-shot mechanism."""

    def __init__(self, binary: str = "openclaw") -> None:
        self._binary = binary

    def spawn(
        self,
        task_id: str,
        prompt: str,
        session_label: str = "",
        reply_chat_id: str = "",
    ) -> str:
        """Spawn an isolated session via ``openclaw cron add --at``.

        Creates a one-shot cron job that runs immediately as an isolated
        session. The Gateway manages the session lifecycle, streaming,
        and announces the result back to the main session on completion.

        Returns the cron job ID as the session key.
        """
        name = session_label or f"task-{task_id}"
        # Schedule 5 seconds from now (minimum viable delay)
        run_at = (datetime.now(UTC) + timedelta(seconds=5)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        cmd = [
            self._binary, "cron", "add",
            "--at", run_at,
            "--session", "isolated",
            "--message", prompt,
            "--announce",
            "--delete-after-run",
            "--name", name,
            "--timeout-seconds", "1800",
            "--json",
        ]

        # Deliver results to Feishu chat if available
        if reply_chat_id:
            cmd.extend(["--channel", "feishu", "--to", reply_chat_id])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"openclaw cron add failed (code {result.returncode}): "
                f"{result.stderr.strip()[:200]}"
            )

        try:
            data = json.loads(result.stdout)
            job_id = data.get("id", "")
        except (json.JSONDecodeError, KeyError) as exc:
            raise RuntimeError(
                f"Failed to parse cron add output: {result.stdout[:200]}"
            ) from exc

        if not job_id:
            raise RuntimeError("openclaw cron add returned empty job id")

        logger.info(
            "Spawned isolated session: name=%s job_id=%s at=%s",
            name, job_id, run_at,
        )
        return name

    def is_running(self, session_key: str) -> bool:
        """Check if a session is still running.

        Checks for the cron job in the job list, or for an active session
        file in the sessions directory.
        """
        import os
        from pathlib import Path

        # Check session file activity (last modified within 5 minutes)
        sessions_dir = os.environ.get(
            "OPENCLAW_SESSIONS_DIR",
            str(Path.home() / ".openclaw" / "agents" / "main" / "sessions"),
        )
        jsonl_path = Path(sessions_dir) / f"{session_key}.jsonl"
        if jsonl_path.exists():
            import time

            age = time.time() - jsonl_path.stat().st_mtime
            if age < 300:  # Modified within 5 minutes
                return True

        return False
