"""Lark/Feishu notification provider implementation.

Handles chat creation, messaging, card sending, image upload, and dashboard updates.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from luna_os.notifications.base import NotificationProvider

BASE_URL = "https://open.larksuite.com/open-apis"


class LarkProvider(NotificationProvider):
    """Lark/Feishu notification provider using the Open API."""

    def __init__(
        self,
        app_id: str | None = None,
        app_secret: str | None = None,
    ) -> None:
        self._app_id = app_id or os.environ.get("LARK_APP_ID", "")
        self._app_secret = app_secret or os.environ.get("LARK_APP_SECRET", "")
        self._tenant_token: str | None = None
        self._token_expires_at: float = 0.0

    def _get_tenant_token(self) -> str:
        """Obtain (and cache) a tenant access token with expiry tracking."""
        if self._tenant_token and time.time() < self._token_expires_at:
            return self._tenant_token
        if not self._app_id or not self._app_secret:
            raise ValueError("LARK_APP_ID and LARK_APP_SECRET environment variables are required")
        body = json.dumps(
            {
                "app_id": self._app_id,
                "app_secret": self._app_secret,
            }
        ).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/auth/v3/tenant_access_token/internal",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        if data.get("code") != 0:
            raise RuntimeError(f"Failed to get tenant token: {data}")
        self._tenant_token = data["tenant_access_token"]
        # Lark tokens expire in ~2 hours; refresh 5 minutes early
        expire_secs = data.get("expire", 7200)
        self._token_expires_at = time.time() + expire_secs - 300
        return self._tenant_token  # type: ignore[return-value]

    def _api_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated API request to the Lark Open API."""
        token = self._get_tenant_token()
        url = f"{BASE_URL}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
            if result.get("code") != 0:
                raise RuntimeError(
                    f"Lark API error: code={result.get('code')} msg={result.get('msg')}"
                )
            return result.get("data", {})
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Lark API HTTP error: {e.code}") from e

    def send_message(self, chat_id: str, text: str, msg_type: str = "text") -> dict[str, Any]:
        """Send a message to a Lark chat."""
        if msg_type == "text":
            content = json.dumps({"text": text})
        elif msg_type == "interactive":
            content = text  # already JSON
        else:
            content = text
        data = self._api_request(
            "POST",
            "/im/v1/messages?receive_id_type=chat_id",
            body={
                "receive_id": chat_id,
                "msg_type": msg_type,
                "content": content,
            },
        )
        return {"message_id": data.get("message_id", "")}

    def create_chat(
        self,
        name: str,
        description: str,
        members: list[str],
        *,
        edit_permission: str = "only_owner",
    ) -> str:
        """Create a Lark group chat and return the chat_id.

        By default the chat is locked down so only the bot (owner) can
        edit settings.  Pass ``edit_permission="all_members"`` to relax.
        """
        data = self._api_request(
            "POST",
            "/im/v1/chats?set_bot_manager=true",
            body={
                "name": name,
                "description": description,
                "user_id_list": members,
                "chat_mode": "group",
                "chat_type": "private",
                "edit_permission": edit_permission,
            },
        )
        return data.get("chat_id", "")

    def update_chat(self, chat_id: str, **kwargs: Any) -> bool:
        """Update a Lark group chat's properties.

        Supported kwargs (passed directly to the Lark PUT /im/v1/chats API):
        - name, description, edit_permission, moderation_permission, etc.
        """
        if not kwargs:
            return True
        try:
            self._api_request("PUT", f"/im/v1/chats/{chat_id}", body=kwargs)
            return True
        except Exception:
            return False

    def dissolve_chat(self, chat_id: str) -> bool:
        """Dissolve (delete) a Lark group chat."""
        try:
            self._api_request("DELETE", f"/im/v1/chats/{chat_id}")
            return True
        except Exception:
            return False

    def send_card(self, chat_id: str, card_data: dict[str, Any]) -> dict[str, Any]:
        """Send an interactive card to a Lark chat."""
        card_json = json.dumps(card_data, ensure_ascii=False)
        return self.send_message(chat_id, card_json, msg_type="interactive")

    def update_card(self, message_id: str, card_data: dict[str, Any]) -> bool:
        """Update an existing interactive card."""
        card_json = json.dumps(card_data, ensure_ascii=False)
        try:
            token = self._get_tenant_token()
            body = json.dumps({"content": card_json}).encode()
            req = urllib.request.Request(
                f"{BASE_URL}/im/v1/messages/{message_id}",
                data=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="PATCH",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            return data.get("code") == 0
        except Exception:
            return False

    def upload_image(self, image_path: str) -> str:
        """Upload an image to Lark and return the image key."""
        import mimetypes

        token = self._get_tenant_token()
        boundary = "----LunaOSBoundary"
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"

        with open(image_path, "rb") as f:
            file_data = f.read()

        body_parts = []
        # image_type field
        body_parts.append(f"--{boundary}\r\n".encode())
        body_parts.append(b'Content-Disposition: form-data; name="image_type"\r\n\r\nmessage\r\n')
        # image field
        body_parts.append(f"--{boundary}\r\n".encode())
        fname = os.path.basename(image_path)
        body_parts.append(
            f'Content-Disposition: form-data; name="image"; filename="{fname}"\r\n'.encode()
        )
        body_parts.append(f"Content-Type: {mime_type}\r\n\r\n".encode())
        body_parts.append(file_data)
        body_parts.append(f"\r\n--{boundary}--\r\n".encode())

        body = b"".join(body_parts)
        req = urllib.request.Request(
            f"{BASE_URL}/im/v1/images",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        if data.get("code") != 0:
            raise RuntimeError(f"Image upload failed: {data}")
        return data["data"]["image_key"]

    def send_image(self, chat_id: str, image_key: str) -> dict[str, Any]:
        """Send an image message to a Lark chat."""
        content = json.dumps({"image_key": image_key})
        data = self._api_request(
            "POST",
            "/im/v1/messages?receive_id_type=chat_id",
            body={
                "receive_id": chat_id,
                "msg_type": "image",
                "content": content,
            },
        )
        return {"message_id": data.get("message_id", "")}
