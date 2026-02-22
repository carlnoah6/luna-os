"""Tests for OpenClawRunner (isolated session spawning via openclaw agent)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from luna_os.agents.openclaw import OpenClawRunner


class TestOpenClawRunnerSpawn:
    """Test that spawn creates isolated sessions via openclaw agent."""

    def test_spawn_calls_agent(self):
        runner = OpenClawRunner()

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=12345)
            result = runner.spawn("t1", "do something", "task-label")

        assert result == "task-label"
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "openclaw"
        assert cmd[1] == "agent"
        assert "--session-id" in cmd
        assert "task-label" in cmd
        assert "--message" in cmd
        assert "do something" in cmd

    def test_spawn_with_reply_chat_id(self):
        runner = OpenClawRunner()

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=12345)
            runner.spawn(
                "t2", "prompt", "label",
                reply_chat_id="oc_abc123",
            )

        cmd = mock_popen.call_args[0][0]
        assert "--deliver" in cmd
        assert "--reply-channel" in cmd
        assert "feishu" in cmd
        assert "--reply-to" in cmd
        assert "chat:oc_abc123" in cmd

    def test_spawn_without_reply_chat_id(self):
        runner = OpenClawRunner()

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=12345)
            runner.spawn("t3", "prompt", "label")

        cmd = mock_popen.call_args[0][0]
        assert "--deliver" not in cmd
        assert "--reply-channel" not in cmd

    def test_spawn_default_session_label(self):
        runner = OpenClawRunner()

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=12345)
            result = runner.spawn("my-task-id", "prompt")

        assert result == "task-my-task-id"
        cmd = mock_popen.call_args[0][0]
        assert "task-my-task-id" in cmd


class TestOpenClawRunnerIsRunning:
    """Test is_running checks for running agent process."""

    def test_process_running(self):
        runner = OpenClawRunner()
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(
                returncode=0, stdout="12345\n", stderr="",
            )
            assert runner.is_running("task-abc") is True

    def test_process_not_running_no_file(self, tmp_path):
        runner = OpenClawRunner()
        with patch("subprocess.run") as m, \
             patch.dict(
                 "os.environ",
                 {"OPENCLAW_SESSIONS_DIR": str(tmp_path)},
             ):
            m.return_value = MagicMock(
                returncode=1, stdout="", stderr="",
            )
            assert runner.is_running("task-abc") is False

    def test_process_not_running_recent_file(self, tmp_path):
        runner = OpenClawRunner()
        session_file = tmp_path / "task-abc.jsonl"
        session_file.write_text('{"type":"message"}\n')

        with patch("subprocess.run") as m, \
             patch.dict(
                 "os.environ",
                 {"OPENCLAW_SESSIONS_DIR": str(tmp_path)},
             ):
            m.return_value = MagicMock(
                returncode=1, stdout="", stderr="",
            )
            assert runner.is_running("task-abc") is True

    def test_pgrep_fails_fallback_to_file(self, tmp_path):
        runner = OpenClawRunner()
        session_file = tmp_path / "task-abc.jsonl"
        session_file.write_text('{"type":"message"}\n')

        with patch("subprocess.run") as m, \
             patch.dict(
                 "os.environ",
                 {"OPENCLAW_SESSIONS_DIR": str(tmp_path)},
             ):
            m.side_effect = Exception("pgrep failed")
            assert runner.is_running("task-abc") is True

    def test_pgrep_fails_no_file(self, tmp_path):
        runner = OpenClawRunner()

        with patch("subprocess.run") as m, \
             patch.dict(
                 "os.environ",
                 {"OPENCLAW_SESSIONS_DIR": str(tmp_path)},
             ):
            m.side_effect = Exception("pgrep failed")
            assert runner.is_running("task-abc") is False
