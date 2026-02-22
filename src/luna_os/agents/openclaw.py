"""OpenClaw agent runner — spawns isolated sessions via the Gateway.

Uses ``openclaw cron add --at`` to create one-shot isolated sessions that
are fully managed by the Gateway (visible in sessions_list, streaming
output, auto-announce on completion).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
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
        """Spawn an isolated session via ``openclaw cron add --at``.

        Creates a one-shot cron job that runs immediately as an isolated
        session. The Gateway manages the session lifecycle, streaming,
        and announces the result back to the main session on completion.

        If *reply_chat_id* is provided and a streaming bridge script exists,
        a background bridge process is started to stream output to the Lark chat.

        Returns the cron job name as the session key.
        """
        name = session_label or f"task-{task_id}"
        # Schedule 15 seconds from now — must be enough for the cron add
        # command itself to complete + Gateway to register the job.
        # 5s was too tight: large prompts take 2-3s to serialize, and
        # if createdAtMs > schedule.at the job never fires.
        run_at = (datetime.now(UTC) + timedelta(seconds=15)).strftime(
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

        # Start streaming bridge if we have a chat to stream to
        if reply_chat_id:
            self._start_bridge(name, reply_chat_id, task_id)

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
        """Check if a spawned session is still running.

        Uses ``openclaw cron list --json`` to check if the cron job with
        the matching name still exists. Since jobs are created with
        ``--delete-after-run``, a job's presence means it's either
        pending or actively running.

        Falls back to session file check if cron list fails.
        """
        try:
            result = subprocess.run(
                [self._binary, "cron", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                jobs = data if isinstance(data, list) else data.get("jobs", [])
                for job in jobs:
                    if job.get("name") == session_key and job.get("enabled", False):
                        state = job.get("state", {})
                        # Job exists and hasn't errored out
                        if state.get("lastStatus") not in ("error",):
                            return True
                        # Even if last status was error, if it hasn't run yet
                        # (no lastRunAtMs), it's still pending
                        if not state.get("lastRunAtMs"):
                            return True
        except Exception as exc:
            logger.debug("cron list check failed, falling back: %s", exc)
            # Fall back to file check
            return self._check_session_file(session_key)

        return False

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
