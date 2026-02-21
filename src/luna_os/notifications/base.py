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
    def update_chat(self, chat_id: str, **kwargs: Any) -> bool:
        """Update group chat properties (name, description, permissions, etc.).

        Supported kwargs vary by provider. Common keys:
        - name: New chat name.
        - description: New chat description.
        - edit_permission: Who can edit chat settings (e.g. "only_owner", "all_members").
        - moderation_permission: Who can send messages.

        Return success flag.
        """

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

    # ── Streaming cards ──────────────────────────────────────────

    @abstractmethod
    def create_streaming_card(self, chat_id: str) -> dict[str, Any]:
        """Create a streaming card in a chat.

        Returns:
            Dict with ``card_id`` and ``message_id`` keys.
        """

    @abstractmethod
    def update_streaming_card(
        self,
        card_id: str,
        content: str,
        *,
        seq: int = 0,
        status: str | None = None,
    ) -> bool:
        """Update the content (and optional status) of a streaming card.

        Args:
            card_id: The card ID returned by :meth:`create_streaming_card`.
            content: Markdown content for the main body.
            seq: Monotonically increasing sequence number.
            status: Optional status line shown below the main content.

        Returns:
            True on success.
        """

    @abstractmethod
    def close_streaming_card(
        self,
        card_id: str,
        content: str,
        *,
        seq: int = 0,
    ) -> bool:
        """Finalize a streaming card with final content and disable streaming.

        Args:
            card_id: The card ID.
            content: Final markdown content.
            seq: Sequence number (must be higher than last update).

        Returns:
            True on success.
        """

    # ── Optional hooks (default no-ops) ──────────────────────────

    def update_dashboard(self, trigger: str = "unknown") -> None:  # noqa: B027
        """Trigger dashboard refresh after state changes. Override to implement."""

    def update_group_title(self, chat_id: str) -> None:  # noqa: B027
        """Update group chat title to reflect plan progress. Override to implement."""
