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


class OpenClawRunner(AgentRunner):
    """Spawn subagent sessions via Gateway RPC (``openclaw gateway call agent``)."""

    def __init__(
        self,
        binary: str = "openclaw",
    ) -> None:
        self._binary = binary

    def spawn(
        self,
        task_id: str,
        prompt: str,
        session_label: str = "",
        reply_chat_id: str = "",
        timeout_minutes: int | None = None,
        model: str | None = None,
    ) -> str:
        """Spawn a subagent session via Gateway RPC.

        Calls ``openclaw gateway call agent`` with ``--expect-final``
        in a background process. The Gateway manages the session
        lifecycle, and the session is visible in ``sessions_list``.

        Returns the session key.
        """
        child_id = session_label or f"task-{task_id}"
        session_key = f"agent:main:subagent:{child_id}"

        timeout_sec = (timeout_minutes or 30) * 60

        # Set model override via sessions.patch BEFORE spawning
        if model:
            self._set_session_model(session_key, model)

        params: dict[str, object] = {
            "message": prompt,
            "sessionKey": session_key,
            "idempotencyKey": str(uuid.uuid4()),
            "deliver": False,
            "lane": "subagent",
            "timeout": timeout_sec,
        }

        cmd = [
            self._binary, "gateway", "call", "agent",
            "--params", json.dumps(params),
            "--expect-final",
            "--json",
            "--timeout", str((timeout_sec + 10) * 1000),  # convert to ms, add 10s buffer
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

        # Code-layer safety net: start a sentinel that watches the agent process
        # and auto-fails the task if the agent exits without calling emit_event.py.
        # More immediate than the watchdog cron (fires on process exit, not on schedule).
        if task_id:
            self._start_sentinel(proc.pid, task_id)

        return session_key

    def _start_sentinel(self, agent_pid: int, task_id: str) -> None:
        """Start a sentinel process that auto-fails the task if the agent exits without reporting.

        The sentinel waits for the agent process to exit, then checks if the task
        is still in 'running' state. If so, it calls emit_event.py step.failed —
        a code-layer safety net that fires immediately on process exit (not on cron schedule).
        """
        ws = os.environ.get("OPENCLAW_WORKSPACE", "/home/ubuntu/.openclaw/workspace")
        emit_script = f"{ws}/scripts/emit_event.py"

        # Inline Python sentinel: wait for pid, then check+fail if task still running
        sentinel_code = f"""
import subprocess, sys, time, os
pid = {agent_pid}
task_id = {task_id!r}
emit_script = {emit_script!r}

# Wait for the agent process to finish
try:
    os.waitpid(pid, 0)
except (ChildProcessError, ProcessLookupError):
    pass  # Already gone
except Exception:
    # waitpid may fail if pid is not a child of this process; poll instead
    while True:
        try:
            os.kill(pid, 0)
            time.sleep(2)
        except ProcessLookupError:
            break
        except Exception:
            break

# Give the task a moment to write its emit_event.py result
time.sleep(3)

# Check task status via emit_event.py noop: if task is still 'running', auto-fail
result = subprocess.run(
    ["python3", emit_script, "step.failed",
     "--task-id", task_id,
     "--result", "subagent process exited without calling emit_event.py (sentinel auto-fail)"],
    capture_output=True, text=True, timeout=30,
)
# emit_event.py returns non-zero if the task is already done/failed (idempotent guard)
# So we only log; the planner handles deduplication
if result.returncode == 0:
    import logging
    logging.getLogger("luna_os.sentinel").info(
        "Sentinel auto-failed task %s (agent pid=%s exited without reporting)", task_id, pid
    )
"""

        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", sentinel_code],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=open(f"/tmp/sentinel-{task_id}.log", "w"),  # noqa: SIM115
            )
            logger.info("Started sentinel: pid=%d for task=%s agent_pid=%d",
                        proc.pid, task_id, agent_pid)
        except Exception as exc:
            logger.warning("Failed to start sentinel for task %s: %s", task_id, exc)

    def _start_bridge(
        self, session_label: str, chat_id: str, task_id: str = "",
    ) -> None:
        """Launch streaming bridge in the background."""
        cmd = [
            sys.executable, "-m", "luna_os.cli",
            "streaming-bridge",
            session_label,
            chat_id,
        ]
        if task_id:
            cmd.extend(["--task-id", task_id])
        cmd.extend(["--timeout", "1800"])

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

    def _set_session_model(self, session_key: str, model: str) -> None:
        """Set model override for a session by directly modifying sessions.json.

        Parses model string (e.g., "api-proxy/kimi-k2.5") into provider/model
        and writes to the session store BEFORE spawning.
        """
        parts = model.split("/", 1)
        if len(parts) != 2:
            logger.warning("Invalid model format: %s (expected provider/model)", model)
            return

        provider, model_name = parts

        # Read sessions.json
        sessions_file = Path.home() / ".openclaw/agents/main/sessions/sessions.json"
        if not sessions_file.exists():
            logger.warning("Sessions file not found: %s", sessions_file)
            return

        try:
            with open(sessions_file) as f:
                sessions = json.load(f)

            # Create or update session entry
            if session_key not in sessions:
                sessions[session_key] = {
                    "sessionId": str(uuid.uuid4()),
                    "updatedAt": int(__import__("time").time() * 1000),
                }

            sessions[session_key]["providerOverride"] = provider
            sessions[session_key]["modelOverride"] = model_name

            # Write back atomically
            tmp_file = sessions_file.with_suffix(".tmp")
            with open(tmp_file, "w") as f:
                json.dump(sessions, f, indent=2)
            tmp_file.replace(sessions_file)

            logger.info(
                "Set model override: session=%s provider=%s model=%s",
                session_key, provider, model_name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to set model override: %s",
                exc,
            )

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

    def spawn_monitor(
        self,
        plan_id: str,
        goal: str,
        chat_id: str,
        running_steps: list[dict],
        timeout_minutes: int = 60,
    ) -> str:
        """Spawn a monitor subagent to watch running steps.
        
        Returns the monitor session key.
        """
        session_key = f"agent:main:subagent:monitor-{plan_id}"
        
        # Build running steps info with JSONL paths
        SESSIONS_DIR = "/home/ubuntu/.openclaw/agents/main/sessions"
        SESSIONS_JSON = f"{SESSIONS_DIR}/sessions.json"
        
        steps_info_lines = []
        for step in running_steps:
            # Get JSONL path for this session
            try:
                with open(SESSIONS_JSON) as f:
                    sessions_data = json.load(f)
                session_id = sessions_data.get(step["session_key"], {}).get("sessionId", "")
                jsonl_path = f"{SESSIONS_DIR}/{session_id}.jsonl" if session_id else "unknown"
            except Exception:
                jsonl_path = "unknown"
            
            steps_info_lines.append(f"""- Step {step['step_num']}: {step['title']}
  - Task ID: {step['task_id']}
  - Session key: {step['session_key']}
  - JSONL 文件路径: {jsonl_path}""")
        
        running_steps_info = "\n".join(steps_info_lines)
        
        # Build monitor prompt
        prompt = f"""你是 Plan Monitor（计划监控器）。

监控的 plan: {plan_id}
目标: {goal}
群聊: {chat_id}

当前运行中的 steps:
{running_steps_info}

你的工作：
1. 每 5 分钟检查一次所有 running steps 的状态
2. 检查方法：
   a. 读 session JSONL 文件最后 50 行：tail -50 {{jsonl_path}}
   b. 检查文件修改时间：stat -c %Y {{jsonl_path}}
   c. 查看 plan 状态：luna-os plan show {chat_id}
3. 用你的智能判断：
   - 任务在正常推进？（JSONL 有新内容，内容合理）
   - 卡住了？（5分钟没新内容，或反复重试同一操作）
   - 出错了？（有 error 但没恢复）
4. 如果发现异常，通过 webhook 通知群聊：
   curl -X POST http://localhost:3000/webhook/lark \\
     -H 'Content-Type: application/json' \\
     -d '{{"chat_id": "{chat_id}", "text": "⚠️ 监控告警: Step X 可能卡住了..."}}'
5. 如果所有 steps 都完成（done/failed），退出

开始第一次检查。之后每 5 分钟检查一次（用 sleep 300）。"""
        
        # Set Flash model for monitor (cheap)
        self._set_session_model(session_key, "api-proxy/gemini-3-flash-preview")
        
        # Spawn monitor (no reply_chat_id, monitor notifies directly via webhook)
        monitor_session_key = self.spawn(
            task_id=f"monitor-{plan_id}",
            prompt=prompt,
            session_label=f"monitor-{plan_id}",
            reply_chat_id="",
            timeout_minutes=timeout_minutes,
            model="api-proxy/gemini-3-flash-preview",
        )
        
        logger.info("Monitor spawned: plan=%s session=%s", plan_id, monitor_session_key)
        return monitor_session_key
