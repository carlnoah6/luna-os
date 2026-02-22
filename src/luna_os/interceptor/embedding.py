"""Embedding engine for command matching.

Pre-computes embeddings for known commands/intents and matches incoming
messages via cosine similarity — no LLM call needed for exact or near-exact
command hits.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommandDef:
    """A registered command with its canonical name and example phrases."""
    name: str                       # e.g. "dashboard"
    handler: str                    # dotted path or key for dispatch
    examples: tuple[str, ...]       # phrases that should trigger this command
    description: str = ""


@dataclass
class MatchResult:
    """Result of matching an incoming message against known commands."""
    command: CommandDef | None
    score: float                    # cosine similarity (0-1)
    matched_example: str = ""
    elapsed_ms: float = 0.0

    @property
    def is_match(self) -> bool:
        return self.command is not None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path(os.environ.get(
    "LUNA_EMBEDDING_CACHE", "/tmp/luna-os/embedding-cache"
))
DEFAULT_MODEL = os.environ.get(
    "LUNA_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)
# Similarity threshold — above this we consider it a command match
DEFAULT_THRESHOLD = float(os.environ.get("LUNA_EMBEDDING_THRESHOLD", "0.65"))


def _fingerprint(commands: Sequence[CommandDef], model_name: str) -> str:
    """Deterministic hash of command definitions + model for cache invalidation."""
    blob = json.dumps(
        [(c.name, c.handler, c.examples) for c in commands],
        sort_keys=True,
    ) + "|" + model_name
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """Pre-computes and caches command embeddings for fast cosine matching."""

    def __init__(
        self,
        commands: Sequence[CommandDef],
        *,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self._commands = list(commands)
        self._model_name = model_name
        self._threshold = threshold
        self._cache_dir = cache_dir
        self._model = None  # lazy-loaded

        # Flat list: (index_into_commands, example_text)
        self._examples: list[tuple[int, str]] = []
        for i, cmd in enumerate(self._commands):
            for ex in cmd.examples:
                self._examples.append((i, ex))

        # Will be set by _ensure_embeddings
        self._embeddings: np.ndarray | None = None

    # -- lazy model loading --------------------------------------------------

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # -- cache management ----------------------------------------------------

    def _cache_path(self) -> Path:
        fp = _fingerprint(self._commands, self._model_name)
        return self._cache_dir / f"cmd_embeddings_{fp}.npz"

    def _load_cache(self) -> np.ndarray | None:
        p = self._cache_path()
        if p.exists():
            try:
                data = np.load(p)
                arr = data["embeddings"]
                if arr.shape[0] == len(self._examples):
                    logger.info("Loaded cached embeddings from %s", p)
                    return arr
                logger.warning("Cache shape mismatch, recomputing")
            except Exception:
                logger.warning("Corrupt cache file, recomputing")
        return None

    def _save_cache(self, embeddings: np.ndarray) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        p = self._cache_path()
        np.savez_compressed(p, embeddings=embeddings)
        logger.info("Saved embeddings cache to %s", p)

    # -- embedding computation -----------------------------------------------

    def _ensure_embeddings(self) -> np.ndarray:
        if self._embeddings is not None:
            return self._embeddings

        cached = self._load_cache()
        if cached is not None:
            self._embeddings = cached
            return cached

        model = self._get_model()
        texts = [ex for _, ex in self._examples]
        logger.info("Computing embeddings for %d example phrases", len(texts))
        t0 = time.monotonic()
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        elapsed = (time.monotonic() - t0) * 1000
        logger.info("Embeddings computed in %.1f ms", elapsed)

        self._embeddings = np.asarray(emb, dtype=np.float32)
        self._save_cache(self._embeddings)
        return self._embeddings

    # -- matching ------------------------------------------------------------

    def match(self, text: str) -> MatchResult:
        """Match *text* against pre-computed command embeddings.

        Returns the best MatchResult.  If score < threshold, command is None.
        """
        t0 = time.monotonic()
        emb_matrix = self._ensure_embeddings()
        model = self._get_model()

        query = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        query = np.asarray(query, dtype=np.float32)

        # cosine similarity (vectors already L2-normalised)
        scores = (emb_matrix @ query.T).squeeze()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        elapsed = (time.monotonic() - t0) * 1000

        cmd_idx, example_text = self._examples[best_idx]

        if best_score >= self._threshold:
            return MatchResult(
                command=self._commands[cmd_idx],
                score=best_score,
                matched_example=example_text,
                elapsed_ms=elapsed,
            )
        return MatchResult(command=None, score=best_score, elapsed_ms=elapsed)

    def match_top_k(self, text: str, k: int = 3) -> list[MatchResult]:
        """Return top-k matches (useful for debugging / fallback)."""
        t0 = time.monotonic()
        emb_matrix = self._ensure_embeddings()
        model = self._get_model()

        query = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        query = np.asarray(query, dtype=np.float32)
        scores = (emb_matrix @ query.T).squeeze()
        top_indices = np.argsort(scores)[::-1][:k]
        elapsed = (time.monotonic() - t0) * 1000

        results = []
        for idx in top_indices:
            cmd_idx, example_text = self._examples[int(idx)]
            score = float(scores[int(idx)])
            results.append(MatchResult(
                command=self._commands[cmd_idx] if score >= self._threshold else None,
                score=score,
                matched_example=example_text,
                elapsed_ms=elapsed,
            ))
        return results

    # -- info ----------------------------------------------------------------

    @property
    def num_commands(self) -> int:
        return len(self._commands)

    @property
    def num_examples(self) -> int:
        return len(self._examples)

    def warmup(self) -> float:
        """Pre-load model + compute embeddings. Returns elapsed ms."""
        t0 = time.monotonic()
        self._ensure_embeddings()
        return (time.monotonic() - t0) * 1000
