"""Watchdog runner — discovers and executes all checks, sends alerts."""
from __future__ import annotations
import importlib
import logging
import os
import pkgutil
import sys
from pathlib import Path

from . import checks as checks_pkg
from .models import Alert, filter_dedup
from .alerter import send_alerts

logger = logging.getLogger(__name__)


def _get_db():
    """Get a database connection that returns dicts."""
    import psycopg2
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    conn = psycopg2.connect(url)
    conn.autocommit = True  # Each query is independent
    return conn


class _DictCursorDB:
    """Thin wrapper that provides db.execute() returning list of dicts."""
    def __init__(self, conn):
        self.conn = conn

    def execute(self, query, params=None):
        import psycopg2.extras
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall()


def discover_checks():
    """Auto-discover all check modules in the checks/ package."""
    check_modules = []
    checks_path = Path(checks_pkg.__file__).parent

    for importer, modname, ispkg in pkgutil.iter_modules([str(checks_path)]):
        if modname.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f".checks.{modname}", package="luna_os.watchdog")
            if hasattr(mod, "check"):
                check_modules.append(mod)
                logger.debug(f"Loaded check: {modname}")
        except Exception as e:
            logger.warning(f"Failed to load check {modname}: {e}")

    return check_modules


def run(dry_run: bool = False, verbose: bool = False) -> list[Alert]:
    """Run all checks, dedup, and send alerts. Returns alerts found."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [watchdog] %(message)s")

    modules = discover_checks()
    logger.info(f"Discovered {len(modules)} check(s): {[m.NAME for m in modules if hasattr(m, 'NAME')]}")

    conn = _get_db()
    db = _DictCursorDB(conn)

    all_alerts = []
    for mod in modules:
        name = getattr(mod, "NAME", mod.__name__)
        try:
            alerts = mod.check(db)
            if alerts:
                logger.info(f"  {name}: {len(alerts)} alert(s)")
            else:
                logger.debug(f"  {name}: OK")
            all_alerts.extend(alerts)
        except Exception as e:
            logger.error(f"  {name}: ERROR — {e}")

    conn.close()

    # Dedup
    new_alerts = filter_dedup(all_alerts)
    logger.info(f"Total: {len(all_alerts)} alert(s), {len(new_alerts)} new (after dedup)")

    if new_alerts and not dry_run:
        send_alerts(new_alerts)
    elif new_alerts and dry_run:
        for a in new_alerts:
            print(f"  [{a.severity.value}] {a.message}")

    return new_alerts


def status():
    """Show current watchdog status — recent alerts and check health."""
    from .models import _load_dedup
    import time

    state = _load_dedup()
    now = time.time()

    print(f"Watchdog status — {len(state)} tracked alert(s)")
    print()

    if not state:
        print("  No recent alerts. All clear. ✅")
        return

    for key, ts in sorted(state.items(), key=lambda x: x[1], reverse=True):
        age_min = int((now - ts) / 60)
        cooldown_left = max(0, 15 - age_min)
        status = f"(cooldown {cooldown_left}m)" if cooldown_left > 0 else "(can re-fire)"
        print(f"  {key}: last fired {age_min}m ago {status}")
