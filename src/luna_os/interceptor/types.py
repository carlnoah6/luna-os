"""Data types for the interceptor module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MatchResult(Enum):
    """How a message was classified."""

    EXACT = "exact"  # keyword hit
    EMBEDDING = "embedding"  # embedding similarity above threshold
    FALLBACK_LLM = "fallback_llm"  # Gemini Flash Lite classified it
    PASSTHROUGH = "passthrough"  # not a command — forward to agent


@dataclass
class Command:
    """A registered command."""

    id: str
    patterns: list[str]
    handler: str
    description: str = ""
    examples: list[str] = field(default_factory=list)
    embedding_vectors: list[list[float]] | None = None


@dataclass
class InterceptResult:
    """Result of intercepting a message."""

    match: MatchResult
    command_id: str | None = None
    confidence: float = 0.0
    handler: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
