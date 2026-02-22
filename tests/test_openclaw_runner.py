"""Tests for OpenClawRunner (isolated session spawning via cron)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from luna_os.agents.openclaw import OpenClawRunner


class TestOpenClawRunnerSpawn:
    """Test that spawn creates isolated sessions via openclaw cron add."""

    def test_spawn_calls_cron_add(self):
        runner = OpenClawRunner()
        fake_output = json.dumps({"id": "job-123", "name": "task-t1"})

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=fake_output,
                stderr="",
            )
            result = runner.spawn("t1", "do something", "task-label")

        assert result == "task-label"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "openclaw"
        assert cmd[1] == "cron"
        assert cmd[2] == "add"
        assert "--session" in cmd
        assert "isolated" in cmd
        assert "--announce" in cmd
        assert "--delete-after-run" in cmd
        assert "--message" in cmd
        assert "do something" in cmd

    def test_spawn_with_reply_chat_id(self):
        runner = OpenClawRunner()
        fake_output = json.dumps({"id": "job-456"})

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=fake_output,
                stderr="",
            )
            runner.spawn("t2", "prompt", "label", reply_chat_id="oc_abc123")

        cmd = mock_run.call_args[0][0]
        assert "--channel" in cmd
        assert "feishu" in cmd
        assert "--to" in cmd
        assert "oc_abc123" in cmd

    def test_spawn_without_reply_chat_id(self):
        runner = OpenClawRunner()
        fake_output = json.dumps({"id": "job-789"})

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=fake_output,
                stderr="",
            )
            runner.spawn("t3", "prompt", "label")

        cmd = mock_run.call_args[0][0]
        assert "--channel" not in cmd
        assert "--to" not in cmd

    def test_spawn_failure_raises(self):
        runner = OpenClawRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="Error: something went wrong",
            )
            with pytest.raises(RuntimeError, match="cron add failed"):
                runner.spawn("t4", "prompt")

    def test_spawn_bad_json_raises(self):
        runner = OpenClawRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="not json",
                stderr="",
            )
            with pytest.raises(RuntimeError, match="parse cron add output"):
                runner.spawn("t5", "prompt")

    def test_spawn_empty_job_id_raises(self):
        runner = OpenClawRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({"id": ""}),
                stderr="",
            )
            with pytest.raises(RuntimeError, match="empty job id"):
                runner.spawn("t6", "prompt")

    def test_spawn_default_session_label(self):
        runner = OpenClawRunner()
        fake_output = json.dumps({"id": "job-abc"})

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=fake_output,
                stderr="",
            )
            result = runner.spawn("my-task-id", "prompt")

        assert result == "task-my-task-id"
        cmd = mock_run.call_args[0][0]
        assert "task-my-task-id" in cmd


class TestOpenClawRunnerIsRunning:
    """Test is_running checks cron job list."""

    @staticmethod
    def _cron_output(jobs):
        return json.dumps({"jobs": jobs})

    def test_job_exists_enabled(self):
        runner = OpenClawRunner()
        out = self._cron_output([
            {"name": "task-abc", "enabled": True, "state": {"lastStatus": "ok"}},
        ])
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(returncode=0, stdout=out, stderr="")
            assert runner.is_running("task-abc") is True

    def test_job_not_found(self):
        runner = OpenClawRunner()
        out = self._cron_output([
            {"name": "other-task", "enabled": True, "state": {}},
        ])
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(returncode=0, stdout=out, stderr="")
            assert runner.is_running("task-abc") is False

    def test_job_disabled(self):
        runner = OpenClawRunner()
        out = self._cron_output([
            {"name": "task-abc", "enabled": False, "state": {"lastStatus": "ok"}},
        ])
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(returncode=0, stdout=out, stderr="")
            assert runner.is_running("task-abc") is False

    def test_job_errored_with_run(self):
        runner = OpenClawRunner()
        out = self._cron_output([
            {"name": "task-abc", "enabled": True, "state": {"lastStatus": "error", "lastRunAtMs": 123}},
        ])
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(returncode=0, stdout=out, stderr="")
            assert runner.is_running("task-abc") is False

    def test_job_pending_no_run_yet(self):
        runner = OpenClawRunner()
        out = self._cron_output([
            {"name": "task-abc", "enabled": True, "state": {}},
        ])
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(returncode=0, stdout=out, stderr="")
            assert runner.is_running("task-abc") is True

    def test_cron_list_fails_fallback_to_file(self, tmp_path):
        runner = OpenClawRunner()
        session_file = tmp_path / "task-abc.jsonl"
        session_file.write_text('{"type":"message"}\n')

        with patch("subprocess.run") as m, \
             patch.dict("os.environ", {"OPENCLAW_SESSIONS_DIR": str(tmp_path)}):
            m.side_effect = Exception("cron list failed")
            assert runner.is_running("task-abc") is True

    def test_cron_list_fails_no_file(self, tmp_path):
        runner = OpenClawRunner()

        with patch("subprocess.run") as m, \
             patch.dict("os.environ", {"OPENCLAW_SESSIONS_DIR": str(tmp_path)}):
            m.side_effect = Exception("cron list failed")
            assert runner.is_running("task-abc") is False

    def test_empty_job_list(self):
        runner = OpenClawRunner()
        out = self._cron_output([])
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(returncode=0, stdout=out, stderr="")
            assert runner.is_running("task-abc") is False

    def test_cron_list_returns_plain_list(self):
        """Handle case where cron list returns a plain array instead of {jobs:[]}."""
        runner = OpenClawRunner()
        out = json.dumps([
            {"name": "task-abc", "enabled": True, "state": {"lastStatus": "ok"}},
        ])
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(returncode=0, stdout=out, stderr="")
            assert runner.is_running("task-abc") is True
