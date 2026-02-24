"""HTTP proxy — intercepts Feishu webhook requests.

Runs as an aiohttp server. Incoming POST /webhook/event is inspected:
  - If the message matches a command → handle locally, return response
  - Otherwise → forward to the upstream agent endpoint

Architecture:
  Feishu → [Nginx] → InterceptorProxy(:8280) → upstream(:3000)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from aiohttp import ClientSession, web

from luna_os.interceptor.matcher import CommandMatcher
from luna_os.interceptor.types import MatchResult

logger = logging.getLogger(__name__)

DEFAULT_LISTEN_PORT = int(os.environ.get("INTERCEPT_PORT", "8280"))
DEFAULT_UPSTREAM = os.environ.get("INTERCEPT_UPSTREAM", "http://127.0.0.1:3000")


class InterceptorProxy:
    """aiohttp-based reverse proxy with command interception."""

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
        self._app = web.Application()
        self._app.router.add_post("/webhook/event", self._handle_event)
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

        # Parse the Feishu event to extract user text
        user_text = self._extract_text(body)
        logger.debug("Extracted text: %r", user_text)
        
        if not user_text:
            logger.debug("No text extracted, forwarding to upstream")
            return await self._forward(request, body)

        # Run matcher
        result = await self.matcher.match(user_text)

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
        if handler_fn:
            try:
                response_data = await handler_fn(user_text, result)
            except Exception:
                logger.exception("Handler %s failed", result.handler)
                return await self._forward(request, body)
        else:
            response_data = {
                "msg": f"Command '{result.command_id}' matched but no handler registered yet."
            }

        return web.json_response(response_data)

    async def _proxy_passthrough(self, request: web.Request) -> web.Response:
        """Forward any non-event request to upstream."""
        body = await request.read()
        return await self._forward(request, body)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _forward(self, request: web.Request, body: bytes) -> web.Response:
        """Forward request to upstream and relay the response."""
        assert self._session is not None
        url = f"{self.upstream}/{request.match_info.get('path', '')}"
        if request.query_string:
            url += f"?{request.query_string}"

        try:
            async with self._session.request(
                method=request.method,
                url=url,
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                data=body,
                timeout=None,
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
