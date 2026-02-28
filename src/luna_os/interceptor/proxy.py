"""HTTP proxy — intercepts Feishu webhook requests.

Runs as an aiohttp server. Incoming POST /webhook/event is inspected:
  - If the message matches a command → handle locally, return response
  - Otherwise → forward to the upstream agent endpoint

Architecture:
  Feishu → [Nginx] → InterceptorProxy(:8280) → upstream(:3000)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any
import base64
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from aiohttp import ClientSession, web

from luna_os.interceptor.matcher import CommandMatcher
from luna_os.interceptor.types import MatchResult

logger = logging.getLogger(__name__)

# Summary cache: {chat_id: {summary, message_count, cached_at (epoch float)}}
_summary_cache: dict[str, dict[str, Any]] = {}
_SUMMARY_TTL = 3600  # seconds

DEFAULT_LISTEN_PORT = int(os.environ.get("INTERCEPT_PORT", "8280"))
DEFAULT_UPSTREAM = os.environ.get("INTERCEPT_UPSTREAM", "http://127.0.0.1:3000")
FEISHU_ENCRYPT_KEY = os.environ.get("FEISHU_ENCRYPT_KEY", "")


class InterceptorProxy:
    """aiohttp-based reverse proxy with command interception."""

    # Commands that should be forwarded to upstream after sending the card
    # (they need OpenClaw to actually process them)
    FORWARD_AFTER_CARD = {"new", "stop", "compact"}

    def __init__(
        self,
        matcher: CommandMatcher,
        *,
        listen_port: int = DEFAULT_LISTEN_PORT,
        upstream: str = DEFAULT_UPSTREAM,
        handler_registry: dict[str, Any] | None = None,
    ) -> None:
        self.matcher = matcher
        self.listen_port = listen_port
        self.upstream = upstream.rstrip("/")
        self._handlers: dict[str, Any] = handler_registry or {}
        self._session: ClientSession | None = None
        self._seen_events: set[str] = set()
        self._app = web.Application()
        self._app.router.add_post("/webhook/event", self._handle_event)
        self._app.router.add_post("/feishu/events", self._handle_event)
        self._app.router.add_get("/nav/chats", self._handle_nav_chats)
        self._app.router.add_get("/nav/chat/{chat_id}/summary", self._handle_nav_summary)
        self._app.router.add_post("/nav/cache/clear", self._handle_nav_cache_clear)
        self._app.router.add_route("*", "/{path:.*}", self._proxy_passthrough)
        self._app.on_startup.append(self._on_startup)
        self._app.on_cleanup.append(self._on_cleanup)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _on_startup(self, app: web.Application) -> None:
        self._session = ClientSession()
        logger.info(
            "Interceptor proxy started on :%d → %s", self.listen_port, self.upstream
        )

    async def _on_cleanup(self, app: web.Application) -> None:
        if self._session:
            await self._session.close()

    def run(self) -> None:
        """Blocking entry point."""
        web.run_app(self._app, port=self.listen_port, print=logger.info)

    async def start_async(self) -> web.AppRunner:
        """Non-blocking start for testing / embedding."""
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.listen_port)
        await site.start()
        return runner

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    async def _handle_event(self, request: web.Request) -> web.Response:
        """Main webhook handler — intercept or forward."""
        t0 = time.monotonic()
        body = await request.read()

        # Debug: log request body
        logger.debug("Request body: %s", body.decode()[:500])
        
        # Decrypt for event parsing
        decrypted_body = self._decrypt_feishu(body)

        # Deduplicate events (Feishu retries if no response within 3s)
        event_id = self._extract_event_id(decrypted_body)
        if event_id and event_id in self._seen_events:
            logger.info("Duplicate event %s, skipping", event_id)
            return web.json_response({})
        if event_id:
            self._seen_events.add(event_id)
            # Keep set bounded (last 200 events)
            if len(self._seen_events) > 200:
                self._seen_events = set(list(self._seen_events)[-100:])

        # Decrypt for parsing (forward original encrypted body to upstream)
        decrypted_body = self._decrypt_feishu(body)

        # Parse the Feishu event to extract user text and chat_id
        user_text = self._extract_text(decrypted_body)
        chat_id = self._extract_chat_id(decrypted_body)
        logger.debug("Extracted text: %r, chat_id: %r", user_text, chat_id)
        
        if not user_text:
            logger.debug("No text extracted, forwarding to upstream")
            return await self._forward(request, body)

        # Run matcher
        result = await self.matcher.match(user_text)
        
        # Attach chat_id to result for handlers that need it
        if chat_id:
            result.extra['chat_id'] = chat_id

        if result.match == MatchResult.PASSTHROUGH:
            logger.debug("Passthrough: %.40s", user_text)
            return await self._forward(request, body)

        # Command matched — handle locally
        logger.info(
            "Intercepted [%s] cmd=%s conf=%.2f text=%.40s (%.1fms)",
            result.match.value,
            result.command_id,
            result.confidence,
            user_text,
            (time.monotonic() - t0) * 1000,
        )

        handler_fn = self._handlers.get(result.handler or "")
        if not handler_fn:
            logger.warning("No handler for %s, forwarding", result.command_id)
            return await self._forward(request, body)

        try:
            response_data = await handler_fn(user_text, result)
        except Exception:
            logger.exception("Handler %s failed", result.handler)
            return await self._forward(request, body)

        # Send the response back to the chat via Feishu API
        chat_id = self._extract_chat_id(decrypted_body)
        if chat_id and response_data:
            try:
                await self._send_to_chat(chat_id, response_data)
            except Exception:
                logger.exception("Failed to send response to chat %s", chat_id)

        # Some commands need OpenClaw to process them too
        if result.command_id in self.FORWARD_AFTER_CARD:
            logger.info("Forwarding %s to upstream after card", result.command_id)
            return await self._forward(request, body)

        # Return empty 200 to Feishu webhook (must respond within 3s)
        return web.json_response({})

    async def _proxy_passthrough(self, request: web.Request) -> web.Response:
        """Forward any non-event request to upstream."""
        body = await request.read()
        return await self._forward(request, body)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _forward(self, request: web.Request, body: bytes) -> web.Response:
        """Forward request to upstream and relay the response."""
        import aiohttp

        assert self._session is not None
        # Use request.path to preserve the original URL path (named routes
        # don't populate match_info['path']).
        path = request.path.lstrip("/")
        url = f"{self.upstream}/{path}"
        if request.query_string:
            url += f"?{request.query_string}"

        try:
            timeout = aiohttp.ClientTimeout(total=2.5)  # Feishu expects <3s
            async with self._session.request(
                method=request.method,
                url=url,
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                data=body,
                timeout=timeout,
            ) as resp:
                resp_body = await resp.read()
                return web.Response(
                    status=resp.status,
                    body=resp_body,
                    headers={
                        k: v for k, v in resp.headers.items()
                        if k.lower() != "transfer-encoding"
                    },
                )
        except Exception:
            logger.exception("Failed to forward to %s", url)
            return web.json_response({"error": "upstream_unavailable"}, status=502)

    async def _handle_nav_chats(self, request: web.Request) -> web.Response:
        """GET /nav/chats — return permanent Luna group chats with share links and cached summaries."""
        try:
            loop = asyncio.get_event_loop()
            chats = await loop.run_in_executor(None, _fetch_nav_chats)
            now = time.time()
            missing = []
            for chat in chats:
                chat_id = chat.get("chat_id", "")
                cached = _summary_cache.get(chat_id)
                if cached and (now - cached.get("cached_at", 0)) <= _SUMMARY_TTL:
                    chat["summary"] = cached.get("summary")
                    chat["entity"] = cached.get("entity")
                else:
                    chat["summary"] = None
                    chat["entity"] = None
                    missing.append(chat_id)

            # Kick off background computation for uncached chats
            async def _bg_compute(cid: str) -> None:
                try:
                    result = await loop.run_in_executor(None, _fetch_chat_summary, cid)
                    result["cached_at"] = time.time()
                    _summary_cache[cid] = result
                except Exception:
                    logger.exception("Background summary compute failed for %s", cid)

            for cid in missing:
                asyncio.create_task(_bg_compute(cid))

            return web.json_response(chats, headers={"Access-Control-Allow-Origin": "*"})
        except Exception:
            logger.exception("Failed to fetch nav chats")
            return web.json_response({"error": "fetch_failed"}, status=500)

    async def _handle_nav_summary(self, request: web.Request) -> web.Response:
        """GET /nav/chat/{chat_id}/summary — AI summary of recent Carl messages."""
        import datetime
        chat_id = request.match_info["chat_id"]
        now = time.time()
        cached = _summary_cache.get(chat_id)

        def _to_response(data: dict, stale: bool = False) -> web.Response:
            payload = dict(data)
            payload["cached_at"] = datetime.datetime.utcfromtimestamp(
                data["cached_at"]
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            if stale:
                payload["stale"] = True
            return web.json_response(payload, headers={"Access-Control-Allow-Origin": "*"})

        async def _bg_refresh():
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, _fetch_chat_summary, chat_id)
                result["cached_at"] = time.time()
                _summary_cache[chat_id] = result
            except Exception:
                logger.exception("Background summary refresh failed for %s", chat_id)

        if cached:
            age = now - cached["cached_at"]
            if age <= _SUMMARY_TTL:
                # Fresh cache hit
                return _to_response(cached)
            else:
                # Stale: return old data immediately, refresh in background
                asyncio.create_task(_bg_refresh())
                return _to_response(cached, stale=True)

        # No cache: compute synchronously
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _fetch_chat_summary, chat_id)
            result["cached_at"] = now
            _summary_cache[chat_id] = result
            return _to_response(result)
        except Exception:
            logger.exception("Failed to fetch summary for %s", chat_id)
            return web.json_response({"error": "summary_failed"}, status=500)

    async def _handle_nav_cache_clear(self, request: web.Request) -> web.Response:
        """POST /nav/cache/clear — clear all summary cache entries."""
        _summary_cache.clear()
        return web.json_response({"cleared": True}, headers={"Access-Control-Allow-Origin": "*"})


    @staticmethod
    def _decrypt_feishu(body: bytes) -> bytes:
        """Decrypt Feishu encrypted event payload. Returns decrypted body or original."""
        encrypt_key = FEISHU_ENCRYPT_KEY
        if not encrypt_key:
            return body
        try:
            data = json.loads(body)
            encrypt_str = data.get("encrypt")
            if not encrypt_str:
                return body
            # Feishu AES-256-CBC: key = SHA256(encrypt_key), IV = first 16 bytes of ciphertext
            key = hashlib.sha256(encrypt_key.encode()).digest()
            ciphertext = base64.b64decode(encrypt_str)
            iv = ciphertext[:16]
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext[16:]) + decryptor.finalize()
            # Remove PKCS7 padding
            pad_len = plaintext[-1]
            if 0 < pad_len <= 16:
                plaintext = plaintext[:-pad_len]
            return plaintext
        except Exception as e:
            logger.debug("Feishu decrypt failed: %s", e)
            return body

    @staticmethod
    def _extract_event_id(body: bytes) -> str | None:
        """Extract event_id from a Feishu event payload for deduplication."""
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return data.get("header", {}).get("event_id")

    @staticmethod
    def _extract_text(body: bytes) -> str | None:
        """Extract user message text from a Feishu event payload."""
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        # Feishu v2 event format
        event = data.get("event", {})
        message = event.get("message", {})
        content_str = message.get("content", "")
        if content_str:
            try:
                content = json.loads(content_str)
                return content.get("text", "").strip()
            except (json.JSONDecodeError, AttributeError):
                pass

        # Fallback: check for plain text field
        return event.get("text", data.get("text"))

    @staticmethod
    def _extract_chat_id(body: bytes) -> str | None:
        """Extract chat_id from a Feishu event payload."""
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        event = data.get("event", {})
        message = event.get("message", {})
        return message.get("chat_id") or event.get("chat_id")

    async def _send_to_chat(self, chat_id: str, response_data: dict[str, Any]) -> None:
        """Send handler response to the chat via Feishu API (async, non-blocking)."""
        from luna_os.notifications.lark import LarkProvider

        loop = asyncio.get_event_loop()

        def _send() -> None:
            lark = LarkProvider()
            logger.info("Sending response to chat %s, msg_type=%s", chat_id, response_data.get("msg_type"))
            if response_data.get("msg_type") == "interactive":
                result = lark.send_card(chat_id, response_data["card"])
                logger.info("Card sent: %s", result)
            else:
                text = response_data.get("text") or response_data.get("msg", "")
                if text:
                    result = lark.send_message(chat_id, text)
                    logger.info("Text sent: %s", result)
                else:
                    logger.warning("No text/card to send for response: %s", response_data)

        await loop.run_in_executor(None, _send)


# ---------------------------------------------------------------------------
# Nav chats helper (used by GET /nav/chats)
# ---------------------------------------------------------------------------

TEMP_PREFIXES = ("🤖", "🔧", "Task tid-", "Step ")
TEMP_DESC_KEYWORDS = ("Luna OS 子任务",)
EXCLUDE_NAME_KEYWORDS = ("Test", "测试")
EXCLUDE_NAMES_EXACT = {"QJunyi, Carl"}


def _is_permanent(chat: dict[str, Any]) -> bool:
    name = chat.get("name", "")
    desc = chat.get("description", "") or ""
    if any(name.startswith(p) for p in TEMP_PREFIXES):
        return False
    if any(k in desc for k in TEMP_DESC_KEYWORDS):
        return False
    if any(k in name for k in EXCLUDE_NAME_KEYWORDS):
        return False
    if name in EXCLUDE_NAMES_EXACT:
        return False
    if chat.get("chat_status") != "normal":
        return False
    return True


def _fetch_nav_chats() -> list[dict[str, Any]]:
    """Fetch permanent Luna group chats with share links from Feishu API."""
    import urllib.parse
    import urllib.request

    from luna_os.notifications.lark import LarkProvider

    lark = LarkProvider()
    token = lark._get_tenant_token()

    # Fetch all chats
    chats: list[dict[str, Any]] = []
    page_token = ""
    while True:
        url = "https://open.larksuite.com/open-apis/im/v1/chats?page_size=100"
        if page_token:
            url += f"&page_token={urllib.parse.quote(page_token)}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        d = json.loads(urllib.request.urlopen(req).read())
        chats.extend(d.get("data", {}).get("items", []))
        if not d.get("data", {}).get("has_more"):
            break
        page_token = d["data"]["page_token"]

    permanent = [c for c in chats if _is_permanent(c)]

    result = []
    for chat in permanent:
        chat_id = chat["chat_id"]
        share_link = ""
        try:
            url = f"https://open.larksuite.com/open-apis/im/v1/chats/{chat_id}/link"
            req = urllib.request.Request(
                url,
                data=json.dumps({"validity_period": "permanently"}).encode(),
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                method="POST",
            )
            resp = json.loads(urllib.request.urlopen(req).read())
            share_link = resp.get("data", {}).get("share_link", "")
        except Exception as e:
            logger.warning("share_link failed for %s: %s", chat_id, e)

        result.append({
            "chat_id": chat_id,
            "name": chat.get("name", ""),
            "description": chat.get("description", ""),
            "avatar": chat.get("avatar", ""),
            "share_link": share_link,
            "last_active": _fetch_last_active(chat_id, token),
        })

    return result


def _fetch_last_active(chat_id: str, token: str) -> str | None:
    """Return ISO timestamp of the most recent message in a chat."""
    import urllib.request
    try:
        url = f"https://open.larksuite.com/open-apis/im/v1/messages?container_id_type=chat&container_id={chat_id}&page_size=1&sort_type=ByCreateTimeDesc"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        d = json.loads(urllib.request.urlopen(req).read())
        items = d.get("data", {}).get("items", [])
        if items:
            ts_ms = int(items[0].get("create_time", 0))
            if ts_ms:
                from datetime import datetime, timezone
                return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
    except Exception as e:
        logger.warning("last_active failed for %s: %s", chat_id, e)
    return None


# Carl's sender ID — messages from this user represent his intent
CARL_SENDER_ID = "ou_35f664e694dd100adf97b867e68e1d3a"


def _fetch_chat_summary(chat_id: str) -> dict[str, Any]:
    """Fetch recent Carl messages from a chat and summarize with a cheap model."""
    import urllib.request
    from luna_os.notifications.lark import LarkProvider

    lark = LarkProvider()
    token = lark._get_tenant_token()

    # Fetch last 50 messages
    url = (
        f"https://open.larksuite.com/open-apis/im/v1/messages"
        f"?container_id_type=chat&container_id={chat_id}&page_size=50&sort_type=ByCreateTimeDesc"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    d = json.loads(urllib.request.urlopen(req).read())
    items = d.get("data", {}).get("items", [])

    # Filter to Carl's messages and extract text
    carl_msgs: list[str] = []
    for item in items:
        sender = item.get("sender", {})
        if sender.get("id") != CARL_SENDER_ID:
            continue
        try:
            body = json.loads(item.get("body", {}).get("content", "{}"))
            text = body.get("text", "").strip()
            if text:
                carl_msgs.append(text)
        except Exception:
            pass

    if not carl_msgs:
        return {"summary": None, "message_count": 0}

    # Summarize with cheap model via api-proxy
    msgs_text = "\n".join(f"- {m}" for m in carl_msgs[:20])
    prompt = (
        "以下是用户在一个飞书群里最近发的消息（最新在前）。\n"
        "请返回一个 JSON 对象，包含两个字段：\n"
        "1. summary: 一句话（不超过30字）概括他目前在这个群里关注或推进的事情\n"
        "2. entity: 从摘要中提取最核心的实体词（1-4个字，例如：播客、代码审查、融资），用于作为群聊标题关键字\n"
        "只输出 JSON，不要任何其他内容。\n\n" + msgs_text
    )

    try:
        import os
        api_key = os.environ.get("API_PROXY_KEY", "sk-luna-2026-openclaw")
        body = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }).encode()
        req2 = urllib.request.Request(
            "http://localhost:8180/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req2, timeout=15).read())
        raw = resp["choices"][0]["message"]["content"].strip()
        parsed = json.loads(raw)
        summary = parsed.get("summary", "").strip() or None
        entity = parsed.get("entity", "").strip() or None
    except Exception as e:
        logger.warning("summary LLM call failed: %s", e)
        summary = None
        entity = None

    return {"summary": summary, "entity": entity, "message_count": len(carl_msgs)}
