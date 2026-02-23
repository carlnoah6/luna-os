"""Tests for the interceptor module."""

from __future__ import annotations

import asyncio
import json

import pytest

from luna_os.interceptor.matcher import CommandMatcher
from luna_os.interceptor.registry import CommandRegistry
from luna_os.interceptor.types import MatchResult


@pytest.fixture
def registry():
    return CommandRegistry()


@pytest.fixture
def matcher(registry):
    return CommandMatcher(registry)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestRegistry:
    def test_loads_commands(self, registry):
        cmds = list(registry.all_commands())
        assert len(cmds) >= 7

    def test_exact_lookup(self, registry):
        cmd = registry.lookup_pattern("仪表盘")
        assert cmd is not None
        assert cmd.id == "dashboard"

    def test_case_insensitive(self, registry):
        cmd = registry.lookup_pattern("Dashboard")
        assert cmd is not None
        assert cmd.id == "dashboard"

    def test_slash_prefix(self, registry):
        cmd = registry.lookup_pattern("/model sonnet")
        assert cmd is not None
        assert cmd.id == "model_switch"

    def test_no_match(self, registry):
        cmd = registry.lookup_pattern("今天天气怎么样")
        assert cmd is None


class TestMatcher:
    def test_exact_match(self, matcher):
        result = _run(matcher.match("仪表盘"))
        assert result.match == MatchResult.EXACT
        assert result.command_id == "dashboard"
        assert result.confidence == 1.0

    def test_passthrough(self, matcher):
        result = _run(matcher.match("帮我写一封邮件"))
        assert result.match == MatchResult.PASSTHROUGH
        assert result.command_id is None

    def test_empty_string(self, matcher):
        result = _run(matcher.match(""))
        assert result.match == MatchResult.PASSTHROUGH

    def test_slash_new(self, matcher):
        result = _run(matcher.match("/new"))
        assert result.match == MatchResult.EXACT
        assert result.command_id == "new_session"


class TestProxyExtractText:
    def test_feishu_v2_format(self):
        from luna_os.interceptor.proxy import InterceptorProxy

        payload = json.dumps({
            "event": {
                "message": {
                    "content": json.dumps({"text": "仪表盘"})
                }
            }
        }).encode()
        text = InterceptorProxy._extract_text(payload)
        assert text == "仪表盘"

    def test_invalid_json(self):
        from luna_os.interceptor.proxy import InterceptorProxy

        text = InterceptorProxy._extract_text(b"not json")
        assert text is None
