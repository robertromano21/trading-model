"""POST an alert to the trading-model dashboard. Never raises to the caller.

Usage:
    from alerts_emit import emit
    emit({"type": "fill", "ticker": "IEF", "text": "Bought 18 IEF", "meta": {"side": "buy", "qty": 18, "price": 96.2}})

The dashboard must be running. Default URL is http://127.0.0.1:3003/api/alerts
(override with TM_ALERTS_URL or PRICES_PORT). Failures are logged and return False.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def _url() -> str:
    explicit = os.environ.get("TM_ALERTS_URL") or os.environ.get("ALERTS_URL")
    if explicit:
        return explicit.rstrip("/")
    port = os.environ.get("PRICES_PORT") or "3003"
    return f"http://127.0.0.1:{port}/api/alerts"


def emit(payload: dict | None = None, **fields) -> bool:
    """Send one alert. Returns True on success, False on any failure. Never raises."""
    try:
        body = dict(payload or {})
        body.update(fields)
        if not body.get("type"):
            return False
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            _url(),
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=float(os.environ.get("TM_ALERTS_TIMEOUT", "2"))) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        try:
            print(f"[alerts_emit] skipped: {exc}", file=sys.stderr)
        except Exception:
            pass
        return False
    except Exception as exc:  # noqa: BLE001 — must never fail the caller
        try:
            print(f"[alerts_emit] skipped: {exc}", file=sys.stderr)
        except Exception:
            pass
        return False


if __name__ == "__main__":
    ok = emit(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"type": "error", "text": "alerts_emit self-test"})
    sys.exit(0 if ok else 1)
