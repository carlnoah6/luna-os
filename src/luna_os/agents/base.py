"""Abstract agent runner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AgentRunner(ABC):
    """Abstract interface for spawning and managing agent sessions."""

    @abstractmethod
    def spawn(
        self,
        task_id: str,
        prompt: str,
        session_label: str = "",
        reply_chat_id: str = "",
    ) -> str:
        """Spawn an agent session. Return a session key / identifier."""

    @abstractmethod
    def is_running(self, session_key: str) -> bool:
        """Check whether a session is still alive."""
