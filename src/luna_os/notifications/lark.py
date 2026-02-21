"""Lark/Feishu notification provider implementation.

Handles chat creation, messaging, card sending, image upload, and dashboard updates.
"""

from __future__ import annotations

import contextlib
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
        restrict_messaging: bool = True,
    ) -> str:
        """Create a Lark group chat and return the chat_id.

        By default the chat is locked down:
        - ``edit_permission="only_owner"`` — members cannot change chat settings.
        - ``restrict_messaging=True`` — only the bot (moderator) can send
          messages; human members are read-only.

        Pass ``restrict_messaging=False`` to allow all members to post.
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
        chat_id = data.get("chat_id", "")
        if chat_id and restrict_messaging:
            self.set_moderation(chat_id, "moderator_list")
        return chat_id

    def set_moderation(self, chat_id: str, setting: str = "moderator_list") -> bool:
        """Set who can send messages in a chat.

        Args:
            chat_id: Target chat.
            setting: ``"all_members"`` or ``"moderator_list"`` (only
                     group owner / managers can post).
        """
        try:
            self._api_request(
                "PUT",
                f"/im/v1/chats/{chat_id}/moderation",
                body={"moderation_setting": setting},
            )
            return True
        except Exception:
            return False

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

    # ── Streaming cards (CardKit) ────────────────────────────────

    MAX_CARD_CONTENT: int = 3800

    def _truncate(self, text: str) -> str:
        if len(text) <= self.MAX_CARD_CONTENT:
            return text
        return text[: self.MAX_CARD_CONTENT - 20] + "\n\n... (truncated)"

    def create_streaming_card(self, chat_id: str) -> dict[str, Any]:
        """Create a CardKit streaming card and send it to *chat_id*."""
        card_json = {
            "schema": "2.0",
            "config": {"wide_screen_mode": True, "streaming_mode": True},
            "body": {
                "elements": [
                    {"tag": "markdown", "content": " ", "element_id": "content"},
                    {"tag": "markdown", "content": " ", "element_id": "status"},
                ]
            },
        }
        data = self._api_request(
            "POST",
            "/cardkit/v1/cards",
            body={"type": "card_json", "data": json.dumps(card_json)},
        )
        card_id = data.get("card_id", "")

        im_data = self._api_request(
            "POST",
            "/im/v1/messages?receive_id_type=chat_id",
            body={
                "receive_id": chat_id,
                "msg_type": "interactive",
                "content": json.dumps(
                    {"type": "card", "data": {"card_id": card_id}}
                ),
            },
        )
        return {
            "card_id": card_id,
            "message_id": im_data.get("message_id", ""),
        }

    def update_streaming_card(
        self,
        card_id: str,
        content: str,
        *,
        seq: int = 0,
        status: str | None = None,
    ) -> bool:
        """Update the content of a streaming card."""
        content = self._truncate(content)
        try:
            self._api_request(
                "PUT",
                f"/cardkit/v1/cards/{card_id}/elements/content/content",
                body={"content": content or " ", "sequence": seq},
            )
            if status is not None:
                self._api_request(
                    "PUT",
                    f"/cardkit/v1/cards/{card_id}/elements/status/content",
                    body={"content": status or " ", "sequence": seq + 1},
                )
            return True
        except (RuntimeError, urllib.error.URLError):
            return False

    def close_streaming_card(
        self,
        card_id: str,
        content: str,
        *,
        seq: int = 0,
    ) -> bool:
        """Finalize a streaming card and disable streaming mode."""
        content = self._truncate(content)
        try:
            self._api_request(
                "PUT",
                f"/cardkit/v1/cards/{card_id}/elements/content/content",
                body={"content": content or " ", "sequence": seq},
            )
            self._api_request(
                "PUT",
                f"/cardkit/v1/cards/{card_id}/elements/status/content",
                body={"content": " ", "sequence": seq + 1},
            )
            # Settings endpoint requires PATCH, not PUT
            token = self._get_tenant_token()
            url = f"{BASE_URL}/cardkit/v1/cards/{card_id}/settings"
            body = json.dumps({
                "settings": json.dumps({"streaming_mode": False}),
                "sequence": seq + 2,
            }).encode()
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                method="PATCH",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
            return result.get("code") == 0
        except (RuntimeError, urllib.error.URLError):
            return False

    # ── Optional hooks ───────────────────────────────────────────

    def update_dashboard(self, trigger: str = "unknown") -> None:
        """Trigger Lark dashboard card refresh (30s debounce)."""
        import subprocess
        from pathlib import Path

        state_file = Path(
            os.environ.get(
                "DASHBOARD_STATE_FILE",
                os.path.expanduser("~/.openclaw/workspace/data/dashboard-state.json"),
            )
        )
        # Rate limit: 30s debounce
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                last_ts = state.get("last_update_ts", 0)
                if time.time() - last_ts < 30:
                    return
            except Exception:
                pass

        script = os.environ.get(
            "DASHBOARD_SCRIPT",
            os.path.expanduser("~/.openclaw/workspace/scripts/lark-task-dashboard.py"),
        )
        if not os.path.exists(script):
            return
        with contextlib.suppress(Exception):
            subprocess.Popen(
                ["python3", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

    def update_group_title(self, chat_id: str) -> None:
        """Fire-and-forget group title update based on plan progress."""
        import subprocess
        from pathlib import Path

        config_path = Path(
            os.environ.get(
                "GROUP_TITLE_CONFIG",
                os.path.expanduser("~/.openclaw/workspace/data/group-title-config.json"),
            )
        )
        if not config_path.exists():
            return
        try:
            with open(config_path) as f:
                config = json.load(f)
            if not config.get("enabled", True):
                return
            group_config = config.get("groups", {}).get(chat_id, {})
            if not group_config.get("enabled", config.get("default_enabled", False)):
                return
        except Exception:
            return

        script = os.environ.get(
            "GROUP_TITLE_SCRIPT",
            os.path.expanduser("~/.openclaw/workspace/scripts/update-group-title.py"),
        )
        if not os.path.exists(script):
            return
        with contextlib.suppress(Exception):
            subprocess.Popen(
                ["python3", script, "--chat-id", chat_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
