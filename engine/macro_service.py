"""Dashboard-facing CLI for the macro calendar, watcher status, surprises and event study.

  python macro_service.py status
  python macro_service.py refresh          # rebuild calendar (official + fallback), no watcher restart
  python macro_service.py context          # [MACRO_EVENTS] block
  python macro_service.py consensus --id cpi --period 2026-10 --value 0.2
  python macro_service.py study            # run event study + overnight study
  python macro_service.py overnight-features
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically
from macro_calendar import (
    CALENDAR_PATH, build_calendar, context_block, load_calendar, upcoming, event_catalog,
)
from macro_surprise import latest_surprises, save_consensus_entry, load_consensus
from macro_reaction import status as reaction_status

ET = ZoneInfo("America/New_York")
MACRO_DIR = DATA_DIR / "Macro"
STATUS_PATH = MACRO_DIR / "status_calendar.json"
RESEARCH_STATUS = DATA_DIR / "Research" / "macro_status.json"
WATCHER_STATUS = MACRO_DIR / "watcher_status.json"
FEATURES_PATH = MACRO_DIR / "overnight_features.json"


def _read(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def overnight_features(asof=None, panel=None):
    """Pre-US-open features from overnight overseas moves (ETF lagged + cash index if cached)."""
    from market_data import load_panel
    from macro_sources import OVERSEAS_ETFS, OVERSEAS_INDICES, OVERSEAS_DIR

    asof = asof or datetime.now(ET)
    tickers = list(OVERSEAS_ETFS) + ["SPY", "QQQ", "EFA", "EEM"]
    if panel is None:
        panel = load_panel(tickers=tickers)
    close = panel.close.copy()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index).normalize())
    last = close.iloc[-1]
    prev = close.iloc[-2] if len(close) >= 2 else None
    features = {"asof": str(close.index[-1].date()), "generatedAt": datetime.now(ET).isoformat(timespec="seconds")}
    etfs = {}
    for t in OVERSEAS_ETFS:
        if t in close.columns and prev is not None and prev.get(t):
            etfs[t] = None if not prev[t] else float(last[t] / prev[t] - 1)
    features["laggedEtfReturn"] = etfs
    asia = [etfs.get(t) for t in ("EWJ", "FXI", "MCHI", "EWY") if etfs.get(t) is not None]
    europe = [etfs.get(t) for t in ("EWG", "EWU", "EZU") if etfs.get(t) is not None]
    features["asiaEtfAvg"] = sum(asia) / len(asia) if asia else None
    features["europeEtfAvg"] = sum(europe) / len(europe) if europe else None
    features["notes"] = [
        "Lagged ETF returns include the previous US session and are not this morning's cash close.",
        "Same-morning cash-index features appear below when a cached Yahoo (or other) file is present.",
    ]
    cash = {}
    if OVERSEAS_DIR.is_dir():
        for name, meta in OVERSEAS_INDICES.items():
            path = OVERSEAS_DIR / f"{meta['etf']}_{name.replace(' ', '_').lower()}.csv"
            if not path.is_file():
                continue
            try:
                bars = pd.read_csv(path, parse_dates=[0], index_col=0)
                s = bars["close"].dropna()
                if len(s) >= 2:
                    cash[name] = {"last": float(s.iloc[-1]), "chg": float(s.iloc[-1] / s.iloc[-2] - 1),
                                  "region": meta["region"], "etf": meta["etf"]}
            except Exception:
                continue
    features["cashIndex"] = cash
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    write_atomically(FEATURES_PATH, json.dumps(features, indent=2, default=str).encode("utf-8"))
    return features


def status_payload():
    cal = load_calendar()
    coming = upcoming(days=7, include_past_hours=2)
    watcher = _read(WATCHER_STATUS, {}) or {}
    surprises = latest_surprises(12)
    try:
        from macro_sources import gb12_available
        gb12 = gb12_available()
    except Exception:
        gb12 = False
    study = _read(MACRO_DIR / "event_study.json", {}) or {}
    overnight = _read(MACRO_DIR / "overnight_study.json", {}) or {}
    payload = {
        "generatedAt": datetime.now(ET).isoformat(timespec="seconds"),
        "timezone": "America/New_York",
        "calendar": {
            "builtAt": cal.get("builtAt"),
            "eventCount": cal.get("eventCount") or len(cal.get("events") or []),
            "start": cal.get("start"),
            "end": cal.get("end"),
            "sources": cal.get("sources") or [],
            "notes": cal.get("notes") or [],
            "path": str(CALENDAR_PATH),
        },
        "upcoming": coming[:20],
        "latestSurprises": surprises,
        "watcher": watcher,
        "gb12MacroLoaders": gb12,
        "eventStudy": {k: study.get(k) for k in ("generatedAt", "nTests", "nProvenHolm", "honesty") if k in study},
        "overnightStudy": {k: overnight.get(k) for k in ("generatedAt", "nTests", "nProvenOosHolm", "honesty") if k in overnight},
        "reaction": reaction_status(),
        "consensusEntries": len((load_consensus().get("entries") or [])),
        "catalogCount": len(event_catalog()),
    }
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    write_atomically(STATUS_PATH, json.dumps(payload, indent=2, default=str).encode("utf-8"))
    try:
        write_atomically(RESEARCH_STATUS, json.dumps(payload, indent=2, default=str).encode("utf-8"))
    except Exception:
        pass
    return payload


def tool_payload(days=7, surprises=12):
    """Compact JSON for the MACRO_CALENDAR analyst tool."""
    cal = load_calendar()
    coming = upcoming(days=days, include_past_hours=2)
    return {
        "ok": True,
        "timezone": "America/New_York",
        "builtAt": cal.get("builtAt"),
        "upcoming": [
            {k: r.get(k) for k in ("key", "id", "name", "agency", "country", "datetimeEt", "timeEt",
                                   "importance", "entry", "consensus", "actual", "surpriseZ", "notes")
             if r.get(k) is not None}
            for r in coming
        ],
        "latestSurprises": latest_surprises(surprises),
        "watcher": _read(WATCHER_STATUS, {}) or {},
        "context": context_block(upcoming_days=days),
        "eventStudy": (_read(MACRO_DIR / "event_study.json", {}) or {}).get("honesty"),
        "provenRules": ((_read(MACRO_DIR / "event_study.json", {}) or {}).get("nProvenHolm") or 0),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Macro calendar service (times in ET).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("refresh")
    r.add_argument("--no-fetch", action="store_true")
    r.add_argument("--start")
    r.add_argument("--end")
    sub.add_parser("context")
    c = sub.add_parser("consensus")
    c.add_argument("--id", required=True)
    c.add_argument("--period", required=True)
    c.add_argument("--value", required=True, type=float)
    c.add_argument("--source", default="manual")
    c.add_argument("--field", default="headline")
    s = sub.add_parser("study")
    s.add_argument("--no-fetch", action="store_true")
    sub.add_parser("overnight-features")
    t = sub.add_parser("tool")
    t.add_argument("--days", type=int, default=7)
    args = parser.parse_args(argv)

    if args.cmd == "refresh":
        doc = build_calendar(start=args.start, end=args.end, fetch=not args.no_fetch)
        payload = status_payload()
        print(json.dumps({"ok": True, "eventCount": doc["eventCount"], "builtAt": doc["builtAt"],
                          "sources": doc["sources"], "notes": doc["notes"]}, indent=2))
        return 0
    if args.cmd == "context":
        print(context_block())
        return 0
    if args.cmd == "consensus":
        entry = save_consensus_entry(args.id, args.period, args.value, source=args.source, field=args.field)
        print(json.dumps({"ok": True, "entry": entry}, indent=2))
        return 0
    if args.cmd == "study":
        from macro_event_study import run_all
        doc = run_all(fetch_indices=not args.no_fetch)
        ev, ov = doc["eventStudy"], doc["overnight"]
        print(json.dumps({
            "ok": True,
            "eventStudy": {k: ev.get(k) for k in ("generatedAt", "nTests", "nProvenHolm")},
            "overnight": {k: ov.get(k) for k in ("generatedAt", "nTests", "nProvenOosHolm")},
        }, indent=2))
        return 0
    if args.cmd == "overnight-features":
        print(json.dumps(overnight_features(), indent=2, default=str))
        return 0
    if args.cmd == "tool":
        print(json.dumps(tool_payload(days=args.days), indent=2, default=str))
        return 0
    print(json.dumps(status_payload(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
