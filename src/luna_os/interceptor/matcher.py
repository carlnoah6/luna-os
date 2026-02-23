"""Command matcher — multi-stage classification pipeline.

Stage 1: Exact keyword / pattern match (free, instant)
Stage 2: Embedding cosine similarity (cheap, ~50ms)
Stage 3: Gemini Flash Lite zero-shot classification (fallback, ~200ms)

If all stages fail → passthrough to the LLM agent.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from luna_os.interceptor.types import InterceptResult, MatchResult

if TYPE_CHECKING:
    from luna_os.interceptor.registry import CommandRegistry

logger = logging.getLogger(__name__)

# Thresholds
EMBEDDING_THRESHOLD = float(os.environ.get("INTERCEPT_EMBED_THRESHOLD", "0.78"))
GEMINI_CONFIDENCE_THRESHOLD = float(os.environ.get("INTERCEPT_GEMINI_THRESHOLD", "0.70"))


class CommandMatcher:
    """Three-stage command matcher."""

    def __init__(
        self,
        registry: CommandRegistry,
        *,
        embedding_engine: EmbeddingEngine | None = None,
        gemini_classifier: GeminiClassifier | None = None,
    ) -> None:
        self.registry = registry
        self._embed = embedding_engine
        self._gemini = gemini_classifier

    async def match(self, text: str) -> InterceptResult:
        """Run the classification pipeline and return the result."""
        stripped = text.strip()
        if not stripped:
            return InterceptResult(match=MatchResult.PASSTHROUGH, raw_text=text)

        # Stage 1: exact keyword
        cmd = self.registry.lookup_pattern(stripped)
        if cmd:
            logger.debug("Exact match: %s -> %s", stripped, cmd.id)
            return InterceptResult(
                match=MatchResult.EXACT,
                command_id=cmd.id,
                confidence=1.0,
                handler=cmd.handler,
                raw_text=text,
            )

        # Stage 2: embedding similarity
        if self._embed:
            result = await self._embed.find_closest(stripped)
            if result and result.confidence >= EMBEDDING_THRESHOLD:
                logger.debug(
                    "Embedding match: %s -> %s (%.3f)",
                    stripped, result.command_id, result.confidence,
                )
                return InterceptResult(
                    match=MatchResult.EMBEDDING,
                    command_id=result.command_id,
                    confidence=result.confidence,
                    handler=result.handler,
                    raw_text=text,
                )

        # Stage 3: Gemini Flash Lite fallback
        if self._gemini:
            result = await self._gemini.classify(stripped)
            if result and result.confidence >= GEMINI_CONFIDENCE_THRESHOLD:
                logger.debug(
                    "Gemini match: %s -> %s (%.3f)",
                    stripped, result.command_id, result.confidence,
                )
                return InterceptResult(
                    match=MatchResult.FALLBACK_LLM,
                    command_id=result.command_id,
                    confidence=result.confidence,
                    handler=result.handler,
                    raw_text=text,
                )

        # No match — pass through to agent
        return InterceptResult(match=MatchResult.PASSTHROUGH, raw_text=text)


# ---------------------------------------------------------------------------
# Embedding engine interface (step 2 will implement the real one)
# ---------------------------------------------------------------------------

class EmbeddingMatch:
    """Result from embedding search."""

    __slots__ = ("command_id", "confidence", "handler")

    def __init__(self, command_id: str, confidence: float, handler: str) -> None:
        self.command_id = command_id
        self.confidence = confidence
        self.handler = handler


class EmbeddingEngine:
    """Abstract embedding engine — step 2 provides the real implementation."""

    async def build_index(self, registry: CommandRegistry) -> None:
        raise NotImplementedError

    async def find_closest(self, text: str) -> EmbeddingMatch | None:
        raise NotImplementedError


class GeminiClassifier:
    """Abstract Gemini Flash Lite classifier — step 2 provides the real implementation."""

    async def classify(self, text: str) -> InterceptResult | None:
        raise NotImplementedError
