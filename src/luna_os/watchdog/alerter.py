"""Send watchdog alerts via Feishu message to the monitoring group."""
from __future__ import annotations
import logging
import subprocess
from .models import Alert, Severity

logger = logging.getLogger(__name__)

# Fixed monitoring channel — Carl's main group
MONITOR_CHAT_ID = "oc_630995d9b870d2ff6ab3fa34a4e7315a"

SEVERITY_EMOJI = {
    Severity.INFO: "ℹ️",
    Severity.WARN: "⚠️",
    Severity.CRITICAL: "🚨",
}


def send_alerts(alerts: list[Alert]):
    """Send alerts directly to the monitoring Feishu group (not via wake event)."""
    if not alerts:
        return

    lines = ["🐕 Watchdog 检测到异常：", ""]
    for a in alerts:
        emoji = SEVERITY_EMOJI.get(a.severity, "⚠️")
        lines.append(f"{emoji} [{a.check_name}] {a.message}")

    text = "\n".join(lines)
    logger.info(f"Sending {len(alerts)} alert(s) to {MONITOR_CHAT_ID}")

    try:
        # Send directly to Feishu group via openclaw message
        proc = subprocess.run(
            ["openclaw", "message", "send",
             "--channel", "feishu",
             "--target", f"chat:{MONITOR_CHAT_ID}",
             "-m", text],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            logger.info("Alert sent to monitoring group")
        else:
            logger.error(f"Alert send failed (rc={proc.returncode}): {proc.stderr}")
            # Fallback to wake event
            _send_wake_event(text)
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")
        _send_wake_event(text)


def _send_wake_event(text: str):
    """Fallback: send as wake event."""
    try:
        subprocess.run(
            ["openclaw", "system", "event", "--text", text, "--mode", "now"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        pass
