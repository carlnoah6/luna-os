"""Send watchdog alerts via openclaw system event."""
from __future__ import annotations
import logging
import subprocess
from .models import Alert, Severity

logger = logging.getLogger(__name__)

SEVERITY_EMOJI = {
    Severity.INFO: "ℹ️",
    Severity.WARN: "⚠️",
    Severity.CRITICAL: "🚨",
}


def send_alerts(alerts: list[Alert]):
    """Send alerts as a single wake event to Luna."""
    if not alerts:
        return

    lines = ["🐕 Watchdog 检测到异常：", ""]
    for a in alerts:
        emoji = SEVERITY_EMOJI.get(a.severity, "⚠️")
        lines.append(f"{emoji} [{a.check_name}] {a.message}")

    text = "\n".join(lines)
    logger.info(f"Sending {len(alerts)} alert(s)")

    try:
        subprocess.run(
            ["openclaw", "system", "event", "--text", text, "--mode", "now"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")
