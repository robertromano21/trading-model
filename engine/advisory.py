"""Optional hook for per-ticker advisory verdicts (e.g. from the AI analyst layer). The risk engine stays final.

Sources (both optional; the newest verdict per ticker wins):
  1. JSON file  Massive/Research/advisory_verdicts.json  (override with ADVISORY_VERDICTS_PATH):
       {"generatedAt": "...", "verdicts": {"AAPL": {"flag": "GREEN", "conviction": 0.7, "entry_low": 350,
        "entry_high": 358, "stop": 340, "overextension": 0.4, "asof": "2026-11-25", "source": "ai"}}}
     ("verdicts" may also be a list of objects with a "ticker" field.)
  2. SQLite table advisory_verdicts in Massive/paper_trading.sqlite3:
       ticker TEXT, asof TEXT (YYYY-MM-DD), flag TEXT (GREEN|YELLOW|RED), conviction REAL (0-1),
       entry_low REAL, entry_high REAL, stop REAL, overextension REAL, source TEXT, created_at TEXT

Rules (mode 'apply'): verdicts can only REDUCE risk, never add it.
  RED    -> no new entries or increases in that ticker (existing positions and exits are untouched)
  YELLOW -> new entries / increases scaled by ``yellow_scale`` (default 0.5)
  GREEN  -> no change (cannot enlarge a position beyond the risk engine's target)
  entry band -> a new entry whose reference price is outside [entry_low, entry_high] is deferred
  stop / overextension / conviction -> recorded only
Missing, unreadable, malformed or stale (older than ``max_age_days``) verdicts mean no change.
Mode 'log' (default) records what would have changed without changing anything; 'off' ignores verdicts.
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from fetch_and_model import DATA_DIR

DEFAULT_JSON = DATA_DIR / "Research" / "advisory_verdicts.json"
FLAGS = {"GREEN", "YELLOW", "RED"}
TABLE_SQL = """CREATE TABLE IF NOT EXISTS advisory_verdicts (
  ticker TEXT NOT NULL, asof TEXT NOT NULL, flag TEXT NOT NULL, conviction REAL, entry_low REAL, entry_high REAL,
  stop REAL, overextension REAL, source TEXT, created_at TEXT, PRIMARY KEY (ticker, asof, source))"""


@dataclass
class Verdict:
    ticker: str
    flag: str
    asof: str | None = None
    conviction: float | None = None
    entry_low: float | None = None
    entry_high: float | None = None
    stop: float | None = None
    overextension: float | None = None
    source: str | None = None


def _num(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _parse(ticker, item):
    flag = str(item.get("flag", "")).upper()
    if flag not in FLAGS:
        return None
    return Verdict(ticker.upper(), flag, item.get("asof"), _num(item.get("conviction")), _num(item.get("entry_low")),
                   _num(item.get("entry_high")), _num(item.get("stop")), _num(item.get("overextension")),
                   item.get("source"))


def load_verdicts(asof=None, max_age_days=3, json_path=None, connection=None):
    """Return {ticker: Verdict}. Never raises: any problem yields an empty or partial dict plus warnings."""
    verdicts, warnings = {}, []
    asof = date.fromisoformat(str(asof)[:10]) if asof else date.today()

    def fresh(v):
        if not v.asof:
            return True
        try:
            age = (asof - date.fromisoformat(str(v.asof)[:10])).days
        except ValueError:
            return False
        return 0 <= age <= max_age_days

    def keep(v):
        if v and fresh(v):
            old = verdicts.get(v.ticker)
            if old is None or str(v.asof or "") >= str(old.asof or ""):
                verdicts[v.ticker] = v

    path = Path(json_path or os.environ.get("ADVISORY_VERDICTS_PATH") or DEFAULT_JSON)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            items = data.get("verdicts", data) if isinstance(data, dict) else data
            pairs = items.items() if isinstance(items, dict) else ((i.get("ticker", ""), i) for i in items)
            for ticker, item in pairs:
                if isinstance(item, dict) and ticker:
                    keep(_parse(ticker, {"asof": data.get("asof") if isinstance(data, dict) else None, **item}))
        except (OSError, ValueError, AttributeError, TypeError) as error:
            warnings.append(f"advisory JSON ignored: {error}")
    if connection is not None:
        try:
            if connection.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name='advisory_verdicts'").fetchone():
                for row in connection.execute("SELECT ticker, asof, flag, conviction, entry_low, entry_high, stop, "
                                              "overextension, source FROM advisory_verdicts"):
                    keep(_parse(row[0], dict(zip(("ticker", "asof", "flag", "conviction", "entry_low", "entry_high",
                                                  "stop", "overextension", "source"), row))))
        except sqlite3.Error as error:
            warnings.append(f"advisory table ignored: {error}")
    return verdicts, warnings


def adjust_entry(verdict, weight_or_size, ref_price, yellow_scale=0.5):
    """Return (adjusted amount, reason or None) for a NEW entry or increase. Never increases the amount."""
    if verdict is None or weight_or_size <= 0:
        return weight_or_size, None
    if verdict.flag == "RED":
        return 0.0, "advisory_red"
    if verdict.entry_low is not None and ref_price is not None and ref_price < verdict.entry_low:
        return 0.0, "advisory_below_band"
    if verdict.entry_high is not None and ref_price is not None and ref_price > verdict.entry_high:
        return 0.0, "advisory_above_band"
    if verdict.flag == "YELLOW":
        return weight_or_size * min(max(yellow_scale, 0.0), 1.0), "advisory_yellow"
    return weight_or_size, None


# ---------------------------------------------------------------- scorecard gate (engine/analyst_scoring.py)
def gated_adjust_entry(verdict, weight_or_size, ref_price, yellow_scale=0.5, horizons=(20, 60)):
    """adjust_entry plus the analyst-scorecard gate. Returns (amount_if_applied, note, gate_open, gate_reason).

    Callers may apply the reduced amount ONLY when gate_open is True: the verdict's flag type has beaten a
    date-matched random pick at one of `horizons` (n_eff >= 30, Holm-adjusted permutation p < 0.05) in a scorecard
    no older than 7 days. Missing, stale or unreadable scorecards and unproven flags mean log-only (no effect)."""
    allowed, note = adjust_entry(verdict, weight_or_size, ref_price, yellow_scale)
    if note is None:
        return allowed, None, False, None
    try:
        from analyst_scoring import gate_open
        ok, why = gate_open(verdict.flag, horizons)
    except Exception as error:   # the gate fails closed
        ok, why = False, f"analyst scorecard gate unavailable: {error}"
    return allowed, note, bool(ok), why
