"""CLI entry point for the interceptor.

Usage:
  luna-os intercept serve [--port 8280] [--upstream http://127.0.0.1:3000]
  luna-os intercept test "some user message"
  luna-os intercept list-commands
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys


def intercept_cli(args: list[str]) -> None:
    """Handle `luna-os intercept ...` subcommands."""
    if not args:
        print(__doc__)
        sys.exit(1)

    sub = args[0]
    rest = args[1:]

    if sub == "serve":
        _serve(rest)
    elif sub == "test":
        _test_match(rest)
    elif sub == "list-commands":
        _list_commands()
    elif sub in ("--help", "-h", "help"):
        print(__doc__)
    else:
        print(f"Unknown intercept subcommand: {sub}", file=sys.stderr)
        sys.exit(1)


def _serve(args: list[str]) -> None:
    """Start the interceptor proxy server."""
    import argparse

    parser = argparse.ArgumentParser(prog="luna-os intercept serve")
    parser.add_argument("--port", type=int, default=8280)
    parser.add_argument("--upstream", default="http://127.0.0.1:3000")
    parser.add_argument("--log-level", default="INFO")
    opts = parser.parse_args(args)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [interceptor] %(levelname)s %(message)s",
    )

    from luna_os.interceptor.handlers import HANDLER_REGISTRY
    from luna_os.interceptor.matcher import CommandMatcher
    from luna_os.interceptor.proxy import InterceptorProxy
    from luna_os.interceptor.registry import CommandRegistry

    registry = CommandRegistry()
    matcher = CommandMatcher(registry)
    proxy = InterceptorProxy(
        matcher,
        listen_port=opts.port,
        upstream=opts.upstream,
        handler_registry=HANDLER_REGISTRY,
    )
    proxy.run()


def _test_match(args: list[str]) -> None:
    """Test a message against the matcher (keyword-only, no embedding)."""
    if not args:
        print("Usage: luna-os intercept test \"message text\"", file=sys.stderr)
        sys.exit(1)

    text = " ".join(args)

    from luna_os.interceptor.matcher import CommandMatcher
    from luna_os.interceptor.registry import CommandRegistry

    registry = CommandRegistry()
    matcher = CommandMatcher(registry)
    result = asyncio.run(matcher.match(text))

    print(json.dumps({
        "text": text,
        "match": result.match.value,
        "command_id": result.command_id,
        "confidence": result.confidence,
        "handler": result.handler,
    }, ensure_ascii=False, indent=2))


def _list_commands() -> None:
    """Print all registered commands."""
    from luna_os.interceptor.registry import CommandRegistry

    registry = CommandRegistry()
    for cmd in registry.all_commands():
        print(f"  {cmd.id:20s}  patterns={cmd.patterns}")
