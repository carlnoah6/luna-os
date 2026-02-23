"""Tests for the embedding engine."""

from __future__ import annotations

import pytest

from luna_os.interceptor.commands import DEFAULT_COMMANDS
from luna_os.interceptor.embedding import EmbeddingEngine


@pytest.fixture(scope="module")
def engine():
    """Shared engine instance (model loading is expensive)."""
    eng = EmbeddingEngine(DEFAULT_COMMANDS, cache_dir=None)
    # Use /tmp for test cache
    import tempfile
    from pathlib import Path
    eng._cache_dir = Path(tempfile.mkdtemp(prefix="luna_emb_test_"))
    return eng


class TestEmbeddingEngine:
    def test_exact_match(self, engine: EmbeddingEngine):
        result = engine.match("仪表盘")
        assert result.is_match
        assert result.command.name == "dashboard"
        assert result.score > 0.8

    def test_fuzzy_match_dashboard(self, engine: EmbeddingEngine):
        result = engine.match("看一下项目进度怎么样了")
        assert result.is_match
        assert result.command.name == "dashboard"

    def test_task_list(self, engine: EmbeddingEngine):
        result = engine.match("我现在有哪些待办")
        assert result.is_match
        assert result.command.name == "task_list"

    def test_slash_command(self, engine: EmbeddingEngine):
        result = engine.match("/model")
        assert result.is_match
        assert result.command.name == "model_switch"

    def test_no_match_for_random(self, engine: EmbeddingEngine):
        result = engine.match("今天天气怎么样")
        # Weather query should NOT match any command
        assert result.score < 0.65

    def test_top_k(self, engine: EmbeddingEngine):
        results = engine.match_top_k("任务", k=3)
        assert len(results) == 3
        assert results[0].score >= results[1].score >= results[2].score

    def test_warmup(self, engine: EmbeddingEngine):
        ms = engine.warmup()
        assert ms >= 0

    def test_cache_roundtrip(self, engine: EmbeddingEngine):
        # First call populates cache, second should load from it
        engine._embeddings = None  # force reload
        engine._ensure_embeddings()
        engine._embeddings = None  # force reload again
        emb = engine._ensure_embeddings()
        assert emb.shape[0] == engine.num_examples
