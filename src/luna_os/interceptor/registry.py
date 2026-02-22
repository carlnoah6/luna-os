"""Command registry — loads and indexes the commands.json file."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from luna_os.interceptor.types import Command

logger = logging.getLogger(__name__)

_DEFAULT_COMMANDS_PATH = Path(__file__).parent / "commands.json"


class CommandRegistry:
    """In-memory registry of interceptable commands."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else _DEFAULT_COMMANDS_PATH
        self._commands: dict[str, Command] = {}
        self._pattern_index: dict[str, str] = {}  # lowered pattern -> command id
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, command_id: str) -> Command | None:
        return self._commands.get(command_id)

    def lookup_pattern(self, text: str) -> Command | None:
        """Exact keyword match (case-insensitive, stripped)."""
        key = text.strip().lower()
        cmd_id = self._pattern_index.get(key)
        if cmd_id:
            return self._commands[cmd_id]
        # Substring match for short patterns (e.g. "/model" in "/model sonnet")
        for pattern, cid in self._pattern_index.items():
            if pattern.startswith("/") and key.startswith(pattern):
                return self._commands[cid]
        return None

    def all_commands(self) -> Iterator[Command]:
        yield from self._commands.values()

    def all_patterns_text(self) -> list[str]:
        """Return all patterns + examples as flat list (for embedding)."""
        texts: list[str] = []
        for cmd in self._commands.values():
            texts.extend(cmd.patterns)
            texts.extend(cmd.examples)
        return texts

    def pattern_to_command_id(self) -> dict[str, str]:
        """Map every pattern/example text to its command id."""
        mapping: dict[str, str] = {}
        for cmd in self._commands.values():
            for p in cmd.patterns:
                mapping[p] = cmd.id
            for e in cmd.examples:
                mapping[e] = cmd.id
        return mapping

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            logger.warning("Commands file not found: %s", self._path)
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        for entry in data.get("commands", []):
            cmd = Command(
                id=entry["id"],
                patterns=entry.get("patterns", []),
                handler=entry["handler"],
                description=entry.get("description", ""),
                examples=entry.get("examples", []),
            )
            self._commands[cmd.id] = cmd
            for pat in cmd.patterns:
                self._pattern_index[pat.lower()] = cmd.id
        logger.info("Loaded %d commands from %s", len(self._commands), self._path)
