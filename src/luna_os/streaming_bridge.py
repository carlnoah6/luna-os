"""Streaming bridge — monitor an OpenClaw session and stream output to Lark via CardKit.

Usage (CLI):
    luna-os streaming-bridge <session_id> <chat_id> [--timeout 900] [--task-id TID]

Creates a streaming card in the Lark chat, monitors the session transcript,
and updates the card in real-time as the agent produces output.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

UPDATE_INTERVAL = 0.8
MAX_CONTENT_LEN = 3800
PLACEHOLDERS = ("...", "..", ".", "…")


# ---------------------------------------------------------------------------
# Path helpers (replaces openclaw_config dependency)
# ---------------------------------------------------------------------------

def _get_sessions_dir() -> Path:
    env = os.environ.get("OPENCLAW_SESSIONS_DIR")
    if env:
        return Path(env)
    home = os.environ.get("OPENCLAW_HOME", os.path.expanduser("~/.openclaw"))
    return Path(home) / "agents" / "main" / "sessions"


# ---------------------------------------------------------------------------
# Session file resolution
# ---------------------------------------------------------------------------

def find_session_file(
    session_id: str,
    sessions_dir: Path | None = None,
    max_wait: int = 60,
) -> Path | None:
    """Find the .jsonl transcript file for a session ID.

    Searches by: (1) direct filename match, (2) ``openclaw sessions`` list
    to resolve session key -> UUID, (3) file content search.
    Waits up to *max_wait* seconds for the file to appear.
    """
    sdir = sessions_dir or _get_sessions_dir()
    deadline = time.time() + max_wait

    while True:
        # Method 1: Direct filename match
        for f in sdir.glob("*.jsonl"):
            if session_id in f.name:
                return f

        # Method 2: Resolve via sessions list (key -> sessionId -> file)
        try:
            res = subprocess.run(
                ["openclaw", "sessions", "--active", "10", "--json"],
                capture_output=True, text=True, timeout=8,
            )
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                sessions = data if isinstance(data, list) else data.get("sessions", [])
                for s in sessions:
                    key = s.get("key", "")
                    sid = s.get("sessionId", s.get("id", ""))
                    if session_id in key and sid:
                        candidate = sdir / f"{sid}.jsonl"
                        if candidate.exists():
                            logger.info("Resolved via sessions list: %s -> %s", session_id, sid)
                            return candidate
        except Exception:
            pass

        # Method 3: Content search (first 10 lines, recent files only)
        now = time.time()
        for f in sorted(sdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                if now - f.stat().st_mtime > 120:
                    break
                with open(f) as fh:
                    for _i, line in enumerate(fh):
                        if _i >= 10:
                            break
                        if session_id in line:
                            logger.info(
                                "Found in file content: %s -> %s",
                                session_id, f.name,
                            )
                            return f
            except Exception:
                continue

        if time.time() >= deadline:
            logger.warning("Session file not found after %ds, closing", max_wait)
            return None

        time.sleep(3)


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def extract_content(
    jsonl_path: str | Path,
    last_offset: int = 0,
) -> tuple[list[str], str | None, int, list[dict]]:
    """Read new lines from the session transcript.

    Returns ``(texts, tool_status, new_offset, usage_entries)``.
    """
    texts: list[str] = []
    tool_status: str | None = None
    usage_entries: list[dict] = []
    new_offset = last_offset

    try:
        with open(jsonl_path, "rb") as f:
            f.seek(last_offset)
            raw = f.read()
            new_offset = f.tell()

        for line in raw.decode("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            msg = entry.get("message", entry)
            role = msg.get("role", "")

            if role == "assistant":
                # Collect usage data
                usage = msg.get("usage")
                if isinstance(usage, dict) and (usage.get("input") or usage.get("output")):
                    cost_obj = usage.get("cost", {})
                    usage_entries.append({
                        "input": usage.get("input", 0),
                        "output": usage.get("output", 0),
                        "cost": cost_obj.get("total", 0) if isinstance(cost_obj, dict) else 0,
                    })

                c = msg.get("content", "")
                if isinstance(c, str) and c.strip():
                    t = c.strip()
                    if t not in PLACEHOLDERS:
                        texts.append(t)
                elif isinstance(c, list):
                    for block in c:
                        if not isinstance(block, dict):
                            continue
                        bt = block.get("type", "")
                        if bt == "text" and block.get("text", "").strip():
                            t = block["text"].strip()
                            if t not in PLACEHOLDERS:
                                texts.append(t)
                        elif bt == "thinking":
                            thought = block.get("thinking", "")
                            if isinstance(thought, str) and thought.strip():
                                preview = thought.strip()[:80].replace("\n", " ")
                                tool_status = f"\U0001f4ad {preview}..."
                        elif bt in ("tool_use", "toolCall"):
                            name = block.get("name") or block.get("toolName") or "tool"
                            inp = block.get("input", {})
                            detail = ""
                            if isinstance(inp, dict):
                                for key in ("command", "url", "query", "file_path", "path"):
                                    if key in inp:
                                        detail = str(inp[key])[:60]
                                        break
                            tool_status = (
                                f"\U0001f527 {name}: `{detail}`"
                                if detail
                                else f"\U0001f527 {name}"
                            )

            elif role == "toolResult":
                c = msg.get("content", "")
                rt = ""
                if isinstance(c, str):
                    rt = c.strip()
                elif isinstance(c, list):
                    for block in c:
                        if isinstance(block, dict) and block.get("type") == "text":
                            rt = block.get("text", "").strip()
                            break
                if rt:
                    if len(rt) > 80:
                        rt = rt[:80] + "..."
                    tool_status = f"\u2705 {rt}"

    except Exception:
        logger.debug("Error reading %s", jsonl_path, exc_info=True)

    return texts, tool_status, new_offset, usage_entries


# ---------------------------------------------------------------------------
# Gateway idle check
# ---------------------------------------------------------------------------

def _is_session_idle(session_id: str) -> bool:
    """Check if a session is no longer active via openclaw sessions."""
    try:
        res = subprocess.run(
            ["openclaw", "sessions", "--active", "2", "--json"],
            capture_output=True, text=True, timeout=8,
        )
        if res.returncode != 0:
            return False
        data = json.loads(res.stdout)
        sessions = data if isinstance(data, list) else data.get("sessions", [])
        for s in sessions:
            key = s.get("key", "")
            if session_id in key:
                return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main bridge loop
# ---------------------------------------------------------------------------

def run_bridge(
    session_id: str,
    chat_id: str,
    timeout: int = 900,
    *,
    task_id: str = "",
) -> None:
    """Run the streaming bridge for a session."""
    try:
        from lark_toolkit import LarkClient
        from lark_toolkit.cards import StreamingCard
    except ImportError:
        logger.error("lark-toolkit not installed, cannot run streaming bridge")
        return

    client = LarkClient()
    sessions_dir = _get_sessions_dir()

    logger.info("Starting bridge: session=%s chat=%s timeout=%d", session_id, chat_id, timeout)

    # Create streaming card
    card = StreamingCard(client, chat_id)
    card.create()

    session_file: Path | None = None
    last_offset = 0
    accumulated: list[str] = []
    all_usage: list[dict] = []
    idle_count = 0
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Find session file if not yet found
        if session_file is None:
            session_file = find_session_file(session_id, sessions_dir, max_wait=0)
            if session_file is None:
                idle_count += 1
                if idle_count > 60:  # 60 * 0.8s = 48s no file
                    logger.warning("Session file not found after 48s, closing")
                    break
                time.sleep(UPDATE_INTERVAL)
                continue

        # Read new content
        if session_file.exists():
            texts, tool_status, new_offset, usage = extract_content(
                session_file, last_offset,
            )
            all_usage.extend(usage)

            if texts or tool_status:
                last_offset = new_offset
                logger.info("Read %d texts, offset->%d", len(texts), new_offset)

                if texts:
                    accumulated.extend(texts)
                    full_text = "\n\n".join(accumulated)
                    if len(full_text) > MAX_CONTENT_LEN:
                        full_text = full_text[-MAX_CONTENT_LEN:]
                    ok = card.update(full_text)
                    logger.info("Card update: ok=%s len=%d", ok, len(full_text))

                if tool_status:
                    card.update_status(tool_status)
                    logger.info("Status: %s", tool_status[:60])
            else:
                idle_count += 1
                if idle_count >= 12 and idle_count % 12 == 0 and _is_session_idle(session_id):
                    # Final read
                    if session_file.exists():
                        texts2, _ts, new_offset2, final_usage = extract_content(
                            session_file, last_offset,
                        )
                        all_usage.extend(final_usage)
                        if texts2:
                            accumulated.extend(texts2)
                            last_offset = new_offset2
                    logger.info("Session %s confirmed idle by gateway", session_id)
                    break
        else:
            idle_count += 1

        time.sleep(UPDATE_INTERVAL)

    # Close card with final content
    final = "\n\n".join(accumulated) if accumulated else "\u2705 \u5b8c\u6210"
    card.close(final)
    logger.info("Bridge finished")


def main() -> None:
    """CLI entry point for streaming-bridge."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[streaming-bridge] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Streaming bridge")
    parser.add_argument("session_id")
    parser.add_argument("chat_id")
    parser.add_argument("--task-id", default="")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    run_bridge(args.session_id, args.chat_id, args.timeout, task_id=args.task_id)
