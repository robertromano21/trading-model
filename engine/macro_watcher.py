"""Release watcher: poll official sources in a tight loop around the scheduled ET time.

Started by the dashboard (`python macro_watcher.py watch`) or run standalone. From about
30 seconds before a release, the official source is polled until the headline is parsed
or the window expires. Rate limits are respected; the polite user-agent is always sent.
Live trading is never triggered from here — the watcher only writes local cache, scores
the surprise, emits a `macro_release` alert, and records chat/memory entries.

  python macro_watcher.py watch
  python macro_watcher.py poll-once
  python macro_watcher.py replay --key cpi:2026-11-13   # past-release / mock latency
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fetch_and_model import DATA_DIR, write_atomically
from macro_calendar import (
    CALENDAR_PATH, due_window, event_by_id, iso_et, load_calendar, save_calendar, upcoming, build_calendar,
)

ET = ZoneInfo("America/New_York")
MACRO_DIR = DATA_DIR / "Macro"
STATUS_PATH = MACRO_DIR / "watcher_status.json"
RELEASES_PATH = MACRO_DIR / "releases.jsonl"
DEFAULT_BEFORE_SEC = 30
DEFAULT_AFTER_SEC = 15 * 60
IDLE_SLEEP = 15.0
TIGHT_SLEEP = 2.0
# Unregistered BLS API is 25 queries/day; during a window we prefer the news-release HTML.
BLS_PAGES = {
    "cpi": "https://www.bls.gov/news.release/cpi.nr0.htm",
    "core_cpi": "https://www.bls.gov/news.release/cpi.nr0.htm",
    "ppi": "https://www.bls.gov/news.release/ppi.nr0.htm",
    "nfp": "https://www.bls.gov/news.release/empsit.nr0.htm",
    "unrate": "https://www.bls.gov/news.release/empsit.nr0.htm",
    "jolts": "https://www.bls.gov/news.release/jolts.nr0.htm",
    "eci": "https://www.bls.gov/news.release/eci.nr0.htm",
}
CLAIMS_URLS = (
    "https://www.dol.gov/ui/data.pdf",
    "https://oui.doleta.gov/unemploy/wkclaims/report.asp",
    "https://www.dol.gov/newsroom/releases",
)
FED_RSS = "https://www.federalreserve.gov/feeds/press_monetary.xml"


def now_et():
    return datetime.now(ET)


def write_status(payload):
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updatedAt"] = iso_et(now_et())
    write_atomically(STATUS_PATH, json.dumps(payload, indent=2, default=str).encode("utf-8"))
    return payload


def append_release(record):
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    with RELEASES_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# ---- parsers (headline numbers). All take text and return {field: value} or None.

def _pct(text):
    if text is None:
        return None
    s = str(text).replace(",", "").replace("+", "").replace("percent", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_cpi_html(text):
    """BLS CPI news-release page: all-items and core monthly / 12-month changes."""
    out = {}
    # Typical: "The all items index rose 0.2 percent in October"
    m = re.search(r"all items index (?:rose|increased|fell|decreased|was unchanged(?: at)?)(?: by)?\s*(\d+\.\d+)?\s*percent",
                  text, re.IGNORECASE)
    if m:
        out["mom"] = 0.0 if "unchanged" in m.group(0).lower() and not m.group(1) else _pct(m.group(1))
    m = re.search(r"all items[^\.]{0,80}?(\d+\.\d+)\s*percent (?:over the last 12 months|increase over the last 12)",
                  text, re.IGNORECASE)
    if m:
        out["yoy"] = _pct(m.group(1))
    m = re.search(r"(?:index for )?all items less food and energy[^\.]{0,80}?(?:rose|increased|fell|decreased|unchanged)(?: by)?\s*(\d+\.\d+)?\s*percent",
                  text, re.IGNORECASE)
    if m:
        out["core_mom"] = 0.0 if "unchanged" in m.group(0).lower() and not m.group(1) else _pct(m.group(1))
    m = re.search(r"all items less food and energy[^\.]{0,80}?(\d+\.\d+)\s*percent (?:over the last 12 months)",
                  text, re.IGNORECASE)
    if m:
        out["core_yoy"] = _pct(m.group(1))
    return out or None


def parse_empsit_html(text):
    """BLS Employment Situation: payroll change (thousands) and unemployment rate."""
    out = {}
    m = re.search(r"total nonfarm(?: payroll)?(?: employment)?[^\.]{0,100}?(?:increased|rose|decreased|fell|was unchanged)(?: by)?\s*(\d{1,3}(?:,\d{3})*)?(?:\s*thousand)?",
                  text, re.IGNORECASE)
    if m:
        raw = (m.group(1) or "0").replace(",", "")
        signed = -1 if re.search(r"decreased|fell", m.group(0), re.IGNORECASE) else 1
        if "unchanged" in m.group(0).lower() and not m.group(1):
            out["nfp_change"] = 0.0
        else:
            out["nfp_change"] = signed * float(raw)
    m = re.search(r"unemployment rate[^\.]{0,60}?(\d+\.\d+)\s*percent", text, re.IGNORECASE)
    if m:
        out["unrate"] = _pct(m.group(1))
    return out or None


def parse_ppi_html(text):
    out = {}
    m = re.search(r"final demand[^\.]{0,80}?(?:increased|rose|decreased|fell|unchanged)(?: by)?\s*(\d+\.\d+)?\s*percent",
                  text, re.IGNORECASE)
    if m:
        out["mom"] = 0.0 if "unchanged" in m.group(0).lower() and not m.group(1) else _pct(m.group(1))
    return out or None


def parse_claims_text(text):
    # "Initial claims were 224,000" or similar.
    m = re.search(r"initial(?: unemployment)? claims[^\.]{0,80}?(\d{2,3},\d{3}|\d{5,7})", text, re.IGNORECASE)
    if not m:
        m = re.search(r"(\d{2,3},\d{3})(?:\s+in\s+)?initial", text, re.IGNORECASE)
    if m:
        return {"value": float(m.group(1).replace(",", ""))}
    return None


def parse_fomc_statement(text):
    """Extract the target range from an FOMC statement / RSS body."""
    # "target range for the federal funds rate at 3-3/4 to 4 percent" or "4.50 to 4.75 percent"
    frac = {"1/8": 0.125, "1/4": 0.25, "3/8": 0.375, "1/2": 0.5, "5/8": 0.625, "3/4": 0.75, "7/8": 0.875}
    def _rate(raw):
        raw = raw.strip()
        m = re.match(r"(\d+)\s*-\s*(\d+)/(\d+)", raw)
        if m:
            return float(m.group(1)) + (int(m.group(2)) / int(m.group(3)))
        m = re.match(r"(\d+)\s+(\d+)/(\d+)", raw)
        if m:
            return float(m.group(1)) + (int(m.group(2)) / int(m.group(3)))
        try:
            return float(raw)
        except ValueError:
            return None
    m = re.search(
        r"target range for the federal funds rate(?: at| of| to)?\s*([0-9./\-\s]+)\s*(?:to|-)\s*([0-9./\-\s]+)\s*percent",
        text, re.IGNORECASE)
    if not m:
        m = re.search(r"federal funds rate[^\.]{0,80}?(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*percent",
                      text, re.IGNORECASE)
    if m:
        lo, hi = _rate(m.group(1)), _rate(m.group(2))
        if lo is not None and hi is not None:
            return {"lower": lo, "upper": hi, "mid": (lo + hi) / 2.0}
    m = re.search(r"federal funds rate[^\.]{0,60}?(\d+(?:\.\d+)?)\s*percent", text, re.IGNORECASE)
    if m:
        mid = _rate(m.group(1))
        return {"mid": mid, "lower": mid, "upper": mid} if mid is not None else None
    return None


def parse_bls_api_latest(doc, series_id):
    series = (((doc.get("Results") or {}).get("series")) or [])
    for item in series:
        if item.get("seriesID") != series_id:
            continue
        data = item.get("data") or []
        if not data:
            return None
        latest = data[0]
        return {"value": _pct(latest.get("value")), "period": latest.get("year") + latest.get("period"),
                "periodName": latest.get("periodName")}
    return None


PARSERS = {
    "cpi": ("html", parse_cpi_html, "mom"),
    "core_cpi": ("html", parse_cpi_html, "core_mom"),
    "ppi": ("html", parse_ppi_html, "mom"),
    "nfp": ("html", parse_empsit_html, "nfp_change"),
    "unrate": ("html", parse_empsit_html, "unrate"),
    "claims": ("text", parse_claims_text, "value"),
    "fomc_decision": ("text", parse_fomc_statement, "mid"),
    "fomc_minutes": ("text", (lambda t: {"released": True} if "minutes of the federal open market committee" in t.lower() else None), "released"),
}


def parser_for(event_id):
    return PARSERS.get(event_id)


def fetch_and_parse(row, opener=None, clock=None):
    """Hit the official source for one calendar row and try to parse a headline. Returns a record or None."""
    from macro_sources import auction_results, http_text, parse_fed_rss, FED_MONETARY_RSS, FetchError

    started = clock() if clock else time.perf_counter()
    event_id = row["id"]
    source_url = None
    body = None
    parsed = None
    error = None
    try:
        if event_id in BLS_PAGES:
            source_url = BLS_PAGES[event_id]
            body, _, _ = http_text(source_url, opener=opener, accept="text/html")
            kind, fn, _field = parser_for(event_id)
            parsed = fn(body)
        elif event_id == "claims":
            for url in CLAIMS_URLS:
                source_url = url
                try:
                    body, _, _ = http_text(url, opener=opener)
                except Exception:
                    continue
                parsed = parse_claims_text(body or "")
                if parsed:
                    break
        elif event_id in {"fomc_decision", "fomc_minutes"}:
            source_url = FED_MONETARY_RSS
            body, _, _ = http_text(source_url, opener=opener, accept="application/rss+xml")
            items = parse_fed_rss(body)
            kind = event_id
            for item in items:
                if item.get("kind") != kind:
                    continue
                parsed = parse_fomc_statement((item.get("title") or "") + "\n" + (item.get("text") or ""))
                if event_id == "fomc_minutes":
                    parsed = parsed or {"released": True, "title": item.get("title")}
                if parsed:
                    parsed["link"] = item.get("link")
                    break
        elif event_id == "treasury_auction":
            source_url = "fiscaldata:auctions_query"
            results = auction_results(opener=opener, start=row.get("date"), end=row.get("date"))
            term = (row.get("securityTerm") or "").lower()
            for item in results:
                t = (item.get("security_term") or "").lower()
                if term and term not in t and t not in term:
                    continue
                parsed = item
                break
            if parsed is None and results:
                parsed = results[0]
        else:
            # Generic: no dedicated parser — a manual entry or later FRED vintage will fill this in.
            error = "no_live_parser"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    elapsed_ms = int(round(1000 * ((clock() if clock else time.perf_counter()) - started)))
    if not parsed:
        return None if not error else {
            "ok": False, "eventId": event_id, "key": row.get("key"), "url": source_url,
            "error": error, "parseMs": elapsed_ms,
        }
    return {
        "ok": True, "eventId": event_id, "key": row.get("key"), "url": source_url,
        "parsed": parsed, "parseMs": elapsed_ms, "fetchedAt": iso_et(now_et()),
    }


def _headline(event_id, parsed):
    if not parsed:
        return None
    if event_id in PARSERS:
        field = PARSERS[event_id][2]
        if field in parsed:
            return parsed[field]
    for key in ("value", "mom", "nfp_change", "unrate", "mid", "high_yield", "yoy"):
        if key in parsed and parsed[key] is not None:
            return parsed[key]
    return None


def _score(row, parsed, opener=None):
    from macro_surprise import (
        history_of, record_surprise, score_auction, score_fomc, score_print,
    )
    event_id = row["id"]
    period = row.get("period") or row.get("date")
    if event_id == "fomc_decision":
        mid = (parsed or {}).get("mid")
        prior = row.get("priorMid")
        hist = history_of("fomc_decision")
        return record_surprise(score_fomc(mid, prior, row.get("date"), opener=opener,
                                          history_surprises=hist),
                               calendar_key=row.get("key"))
    if event_id == "treasury_auction":
        from macro_sources import load_curve_yield, tenor_proxy_series
        term = (parsed or {}).get("security_term") or row.get("securityTerm")
        typ = (parsed or {}).get("security_type") or row.get("securityType")
        series = tenor_proxy_series(term or "", typ or "")
        prior_yield = None
        if series:
            # Prior session close: use the auction date itself as as-of (GB-12 lag is applied by the loader when used).
            prior_yield, _, _ = load_curve_yield(series, row.get("date"))
        hist_doc = {}
        try:
            from macro_surprise import HISTORY_PATH, _read
            raw = _read(HISTORY_PATH, default={})
            # Auction history is stored under treasury_auction as generic surprises; also look at a structured cache.
        except Exception:
            raw = {}
        return record_surprise(score_auction(parsed or {}, prior_yield=prior_yield,
                                             when_issued=(parsed or {}).get("when_issued"),
                                             history=hist_doc),
                               calendar_key=row.get("key"))
    actual = _headline(event_id, parsed)
    hist_actuals = []
    return record_surprise(score_print(event_id, period, actual,
                                       history_actuals=hist_actuals,
                                       history_surprises=history_of(event_id)),
                           calendar_key=row.get("key"))


def _big_surprise(score):
    z = score.get("surpriseZ") or score.get("tailZ")
    if z is None:
        # FOMC: |surprise vs implied| of 12.5 bp or more is "big"
        bp = score.get("surpriseVsImpliedBp") or score.get("changeBp")
        return bp is not None and abs(bp) >= 12.5
    return abs(z) >= 1.5


def emit_macro_alert(row, score, parsed=None):
    from alerts_emit import emit
    z = score.get("surpriseZ") if score else None
    if z is None and score:
        z = score.get("tailZ")
    big = bool(score and _big_surprise(score))
    actual = None if not score else score.get("actual", score.get("actualMid", score.get("highYield")))
    cons = None if not score else score.get("consensus", score.get("impliedRate", score.get("benchmark")))
    parts = [row.get("name") or row["id"], "is out"]
    if actual is not None:
        parts.append(f"at {actual}")
    if cons is not None:
        parts.append(f"versus {cons}")
    if z is not None:
        parts.append(f"(z {z:+.2f})")
    text = " ".join(str(p) for p in parts) + "."
    if big:
        text = "Large surprise. " + text
    href = "/paper2.html#macro"
    payload = {
        "type": "macro_release",
        "text": text,
        "ticker": None,
        "severity": "critical" if big else "info",
        "href": href,
        "meta": {
            "eventId": row.get("id"),
            "key": row.get("key"),
            "datetimeEt": row.get("datetimeEt"),
            "surpriseZ": z,
            "actual": actual,
            "consensus": cons,
            "big": big,
        },
    }
    if big:
        payload["speech"] = text
    emit(payload)
    return payload, big


def post_chat_and_memory(row, score, alert_text):
    notes = []
    try:
        from decision_feed import CHAT_DB, _insert, connect_chat
        chat = connect_chat(CHAT_DB)
        with chat:
            _insert(chat, f"macro:{row.get('key')}", kind="macro", role="trader",
                    decision_date=row.get("date"), text=alert_text,
                    rationale="Economic release watcher (times in ET). Not a trade.",
                    data=score or row)
        notes.append("chat")
    except Exception as error:  # noqa: BLE001
        notes.append(f"chat skipped: {error}")
    try:
        # memory_ingest lives next to the AI layer, not under engine/.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server" / "ai"))
        from memory_ingest import ingest
        ingest("EVENT", alert_text, ticker=None, ts=row.get("datetimeEt"),
               meta={"kind": "macro_release", "eventId": row.get("id"), "key": row.get("key"),
                     "dedupeKey": f"macro:{row.get('key')}", "score": score})
        notes.append("memory")
    except Exception as error:  # noqa: BLE001
        notes.append(f"memory skipped: {error}")
    return notes


def handle_print(row, fetch_result, opener=None):
    """Score, alert, persist one successful (or auction) parse."""
    parsed = (fetch_result or {}).get("parsed")
    score = None
    try:
        score = _score(row, parsed, opener=opener)
    except Exception as error:  # noqa: BLE001
        score = {"eventId": row["id"], "error": str(error)}
    payload, big = emit_macro_alert(row, score if isinstance(score, dict) else {}, parsed)
    post_chat_and_memory(row, score, payload["text"])
    record = {
        "key": row.get("key"), "eventId": row["id"], "datetimeEt": row.get("datetimeEt"),
        "fetchedAt": fetch_result.get("fetchedAt"), "parseMs": fetch_result.get("parseMs"),
        "url": fetch_result.get("url"), "parsed": parsed, "score": score,
        "bigSurprise": big, "scheduledEt": row.get("datetimeEt"),
    }
    # Latency from the official time to a parsed number (seconds). Negative means we
    # parsed a number already on the page before the official timestamp (clock skew / early leak / replay).
    try:
        official = datetime.fromisoformat(row["datetimeEt"])
        fetched = datetime.fromisoformat(fetch_result["fetchedAt"])
        record["latencySec"] = (fetched - official).total_seconds()
        record["latencyMsFromRequest"] = fetch_result.get("parseMs")
    except Exception:
        pass
    append_release(record)
    return record


def poll_once(asof=None, opener=None, before_sec=DEFAULT_BEFORE_SEC, after_sec=DEFAULT_AFTER_SEC,
              clock=None):
    """Poll every due release once. Returns a list of records (prints and misses)."""
    asof = asof or now_et()
    due = due_window(asof=asof, before_sec=before_sec, after_sec=after_sec)
    results = []
    for row in due:
        fetched = fetch_and_parse(row, opener=opener, clock=clock)
        if fetched and fetched.get("ok"):
            results.append(handle_print(row, fetched, opener=opener))
        else:
            results.append({"key": row.get("key"), "eventId": row["id"], "ok": False,
                            "error": (fetched or {}).get("error") or "not_yet"})
    return results


def watch(poll_idle=IDLE_SLEEP, poll_tight=TIGHT_SLEEP, opener=None, once=False,
          before_sec=DEFAULT_BEFORE_SEC, after_sec=DEFAULT_AFTER_SEC, stop_after=None,
          clock=None, sleeper=None):
    """Long-running loop. `stop_after` and `sleeper` are for tests."""
    sleeper = sleeper or time.sleep
    started = now_et()
    polls = 0
    prints = 0
    write_status({"state": "running", "startedAt": iso_et(started), "polls": 0, "prints": 0,
                  "next": [r["key"] for r in upcoming(days=2)[:8]]})
    deadline = time.monotonic() + stop_after if stop_after else None
    try:
        while True:
            asof = now_et()
            due = due_window(asof=asof, before_sec=before_sec, after_sec=after_sec)
            coming = upcoming(asof=asof, days=2, include_past_hours=0)
            if due:
                batch = poll_once(asof=asof, opener=opener, before_sec=before_sec,
                                  after_sec=after_sec, clock=clock)
                polls += 1
                prints += sum(1 for r in batch if r.get("score") or r.get("parsed"))
                write_status({"state": "tight_loop", "polls": polls, "prints": prints,
                              "due": [r.get("key") for r in due],
                              "lastBatch": [{"key": r.get("key"), "ok": bool(r.get("parsed") or r.get("score")),
                                             "latencySec": r.get("latencySec"),
                                             "parseMs": r.get("parseMs") or r.get("latencyMsFromRequest")}
                                            for r in batch]})
                sleeper(poll_tight)
            else:
                polls += 1
                nxt = coming[0] if coming else None
                write_status({"state": "idle", "polls": polls, "prints": prints,
                              "next": None if not nxt else {"key": nxt.get("key"),
                                                            "name": nxt.get("name"),
                                                            "datetimeEt": nxt.get("datetimeEt")}})
                sleeper(poll_idle)
            if once:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
    except KeyboardInterrupt:
        write_status({"state": "stopped", "polls": polls, "prints": prints, "reason": "interrupt"})
        return {"polls": polls, "prints": prints, "state": "stopped"}
    write_status({"state": "exited", "polls": polls, "prints": prints})
    return {"polls": polls, "prints": prints, "state": "exited"}


def replay(key, body, event_id=None, official_et=None, clock_offset_sec=0.0):
    """Replay a past (or mock) release body through the parser and measure request-to-parse time.

    `body` is the HTML/RSS/text that would have been fetched. No network is used.
    """
    cal = load_calendar()
    row = None
    for item in cal.get("events") or []:
        if item.get("key") == key or (event_id and item.get("id") == event_id and key in {item.get("key"), item.get("date")}):
            row = item
            break
    if row is None:
        event = event_by_id(event_id or (key.split(":")[0] if key else "cpi"))
        official = official_et or iso_et(now_et())
        row = {"id": event["id"], "name": event["name"], "key": key or f"{event['id']}:replay",
               "datetimeEt": official, "date": official[:10], "period": official[:7]}
    if official_et:
        row = dict(row, datetimeEt=official_et)

    # A fake opener returns the provided body instantly; we still time parse + bookkeeping.
    def opener(url, method="GET", data=None, headers=None):
        if clock_offset_sec:
            time.sleep(clock_offset_sec)
        if isinstance(body, bytes):
            return body, 200, {"Content-Type": "text/html"}
        return body.encode("utf-8"), 200, {"Content-Type": "text/html"}

    # http_get with opener returns whatever the opener returns; our http_get already
    # unpacks (raw, status, headers) if the opener is used... check: in macro_sources
    # opener is expected to return bytes only! Look at http_get:
    #   if opener is not None:
    #       return opener(url, method=method, data=data, headers=hdrs)
    # And http_text does: raw, status, headers = http_get(...)
    # That's inconsistent if opener returns only bytes. I'll support both in a thin
    # wrapper used here by going through fetch_and_parse's opener via a bytes-only
    # shim AND a 3-tuple shim. We'll wrap to always return the 3-tuple and also
    # patch fetch to accept it. Easier: define opener to return bytes, and patch
    # http_get usage... fetch_and_parse uses http_text which unpacks 3 values.
    # I need to fix macro_sources.http_get to normalize opener return values.
    marks = []
    def ticking():
        marks.append(time.perf_counter())
        return marks[-1]

    fetched = fetch_and_parse(row, opener=opener, clock=ticking)
    # Measure parse/request time from the clock marks or parseMs.
    latency_ms = (fetched or {}).get("parseMs")
    return {"row": row, "fetch": fetched, "latencyMs": latency_ms,
            "ok": bool(fetched and fetched.get("ok"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Economic release watcher (times in ET).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--once", action="store_true")
    w.add_argument("--tight", type=float, default=TIGHT_SLEEP)
    w.add_argument("--idle", type=float, default=IDLE_SLEEP)
    sub.add_parser("poll-once")
    sub.add_parser("status")
    r = sub.add_parser("replay")
    r.add_argument("--key", required=True)
    r.add_argument("--file", help="Path to a saved HTML/RSS body.")
    r.add_argument("--event")
    args = parser.parse_args(argv)
    if args.cmd == "status":
        if not STATUS_PATH.is_file():
            print(json.dumps({"state": "never_started"}))
            return 0
        print(STATUS_PATH.read_text(encoding="utf-8"))
        return 0
    if args.cmd == "poll-once":
        # Make sure there is a calendar; do not block on a failed official fetch.
        if not CALENDAR_PATH.is_file():
            build_calendar(fetch=False)
        print(json.dumps(poll_once(), indent=2, default=str))
        return 0
    if args.cmd == "replay":
        body = Path(args.file).read_text(encoding="utf-8") if args.file else ""
        print(json.dumps(replay(args.key, body, event_id=args.event), indent=2, default=str))
        return 0
    if not CALENDAR_PATH.is_file():
        build_calendar(fetch=True)
    result = watch(poll_idle=args.idle, poll_tight=args.tight, once=args.once)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
