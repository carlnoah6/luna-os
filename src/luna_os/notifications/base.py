"""Abstract notification provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class NotificationProvider(ABC):
    """Abstract interface for sending messages, creating chats, and dashboards."""

    @abstractmethod
    def send_message(self, chat_id: str, text: str, msg_type: str = "text") -> dict[str, Any]:
        """Send a message to a chat. Return response metadata (e.g. message_id)."""

    @abstractmethod
    def create_chat(self, name: str, description: str, members: list[str]) -> str:
        """Create a group chat. Return the new chat_id."""

    @abstractmethod
    def dissolve_chat(self, chat_id: str) -> bool:
        """Dissolve (delete) a group chat. Return success flag."""

    @abstractmethod
    def send_card(self, chat_id: str, card_data: dict[str, Any]) -> dict[str, Any]:
        """Send an interactive card to a chat. Return response metadata."""

    @abstractmethod
    def update_card(self, message_id: str, card_data: dict[str, Any]) -> bool:
        """Update an existing card message. Return success flag."""

    @abstractmethod
    def upload_image(self, image_path: str) -> str:
        """Upload an image and return the image key."""

    @abstractmethod
    def send_image(self, chat_id: str, image_key: str) -> dict[str, Any]:
        """Send an image message to a chat. Return response metadata."""
