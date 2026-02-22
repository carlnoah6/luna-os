"""Tests for OpenClawRunner (subagent spawning via Gateway RPC)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from luna_os.agents.openclaw import OpenClawRunner


class TestOpenClawRunnerSpawn:
    """Test that spawn creates subagent sessions via gateway call agent."""

    def test_spawn_calls_gateway_rpc(self):
        runner = OpenClawRunner()

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=12345)
            result = runner.spawn("t1", "do something", "task-label")

        assert result == "agent:main:subagent:task-label"
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "openclaw"
        assert cmd[1] == "gateway"
        assert cmd[2] == "call"
        assert cmd[3] == "agent"
        assert "--params" in cmd
        assert "--expect-final" in cmd

        # Verify params JSON
        params_idx = cmd.index("--params") + 1
        params = json.loads(cmd[params_idx])
        assert params["sessionKey"] == "agent:main:subagent:task-label"
        assert params["message"] == "do something"
        assert params["deliver"] is False
        assert params["lane"] == "subagent"
        assert "idempotencyKey" in params

    def test_spawn_with_reply_chat_id(self):
        runner = OpenClawRunner()

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=12345)
            runner.spawn(
                "t2", "prompt", "label",
                reply_chat_id="oc_abc123",
            )

        # Bridge should be started (2 Popen calls: bridge + agent)
        assert mock_popen.call_count == 2

    def test_spawn_without_reply_chat_id(self):
        runner = OpenClawRunner()

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=12345)
            runner.spawn("t3", "prompt", "label")

        # Only agent, no bridge
        assert mock_popen.call_count == 1

    def test_spawn_default_session_label(self):
        runner = OpenClawRunner()

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=12345)
            result = runner.spawn("my-task-id", "prompt")

        assert result == "agent:main:subagent:task-my-task-id"


class TestOpenClawRunnerIsRunning:
    """Test is_running checks sessions list."""

    def test_session_active(self):
        runner = OpenClawRunner()
        key = "agent:main:subagent:task-abc"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(
                returncode=0,
                stdout=f'{{"sessions": ["{key}"]}}',
                stderr="",
            )
            assert runner.is_running(key) is True

    def test_session_not_found(self):
        runner = OpenClawRunner()
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(
                returncode=0,
                stdout='{"sessions": ["other-key"]}',
                stderr="",
            )
            assert runner.is_running("agent:main:subagent:task-abc") is False

    def test_sessions_cmd_fails_fallback_to_file(self, tmp_path):
        runner = OpenClawRunner()
        session_file = tmp_path / "task-abc.jsonl"
        session_file.write_text('{"type":"message"}\n')

        with patch("subprocess.run") as m, \
             patch.dict(
                 "os.environ",
                 {"OPENCLAW_SESSIONS_DIR": str(tmp_path)},
             ):
            m.side_effect = Exception("sessions failed")
            assert runner.is_running("agent:main:subagent:task-abc") is True

    def test_sessions_cmd_fails_no_file(self, tmp_path):
        runner = OpenClawRunner()

        with patch("subprocess.run") as m, \
             patch.dict(
                 "os.environ",
                 {"OPENCLAW_SESSIONS_DIR": str(tmp_path)},
             ):
            m.side_effect = Exception("sessions failed")
            assert runner.is_running("agent:main:subagent:task-abc") is False
