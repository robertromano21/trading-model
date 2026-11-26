"""Share-dilution flags, SEC offering watcher, and buy-block helpers.

Point-in-time rules use *filing* dates, never period-end dates: a 10-K filed in February
does not change the January book. Split jumps (known split records, or exact 2x/3x/4x/10x
style ratios) are ignored so a split is not mistaken for issuance.

This module never places orders. SEC requests send a User-Agent (SEC_USER_AGENT) and are
cached and rate-limited to well under 10 requests/second.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

DEFAULTS = {
    "red_threshold": 0.10,
    "warn_threshold": 0.05,
    "offering_days": 30,
    "dilution_exit": False,
    "dilution_filter": True,
    "min_interval_sec": 0.12,  # ~8 req/s, under the SEC 10 req/s fair-access cap
    "cache_seconds": 6 * 3600,
    # Size-relative offering flag (2026-11-26). Offering size vs shares outstanding (or market cap):
    # <5% LOW (warning only, no buy block), 5-15% MEDIUM (block new buys, YELLOW), >15% HIGH (block, RED).
    # Unknown size keeps the old conservative behaviour (block). Set offering_size_relative=false to restore
    # the plain binary flag.
    "offering_size_relative": True,
    "offering_small_pct": 0.05,
    "offering_large_pct": 0.15,
}

# Equity registration / prospectus forms that indicate a possible share issuance.
# S-8 (employee plans) is intentionally omitted — too noisy and not a "new offering" in the same sense.
OFFERING_FORMS = frozenset({
    "S-1", "S-1/A", "S-3", "S-3/A", "S-3ASR",
    "F-1", "F-1/A", "F-3", "F-3/A", "F-3ASR",
    "424B", "424B1", "424B2", "424B3", "424B4", "424B5",
})
# 8-K is only an offering flag when these items (or ATM language) are present.
EIGHT_K_DILUTION_ITEMS = frozenset({"3.02"})
ATM_RE = re.compile(
    r"at[-\s]?the[-\s]?market|\batm (offering|program|facility|issuance)\b|"
    r"equity distribution agreement|sales agreement|controlled equity offering",
    re.I,
)
SPLIT_RATIOS = (
    [n for n in (2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100)]
    + [1 / n for n in (2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100)]
    + [1.5, 2 / 3, 3 / 2, 5 / 4, 4 / 3]
)
# Tight: a 100% issuance that happens to land at 2.000 should still look like a split, but 2.10x should not.
SPLIT_RATIO_TOLERANCE = 0.005
FLAGS_NAME = "dilution_flags.json"
HISTORY_DIRNAME = "dilution"


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    raw = os.environ.get("PRICES_DATA_DIR")
    return Path(raw) if raw else _root() / "Massive"


def _research_dir() -> Path:
    return _data_dir() / "Research"


def flags_path() -> Path:
    return _research_dir() / FLAGS_NAME


def history_dir() -> Path:
    return _research_dir() / HISTORY_DIRNAME


def config_from_settings(settings=None, **overrides):
    cfg = dict(DEFAULTS)
    src = dict(settings or {})
    src.update(overrides)
    mapping = {
        "red_threshold": ("dilution_red_threshold", "red_threshold"),
        "warn_threshold": ("dilution_warn_threshold", "warn_threshold"),
        "offering_days": ("dilution_offering_days", "offering_days"),
        "dilution_exit": ("dilution_exit",),
        "dilution_filter": ("dilution_filter",),
        "min_interval_sec": ("dilution_min_interval_sec", "min_interval_sec"),
        "cache_seconds": ("dilution_cache_seconds", "cache_seconds"),
        "offering_size_relative": ("dilution_offering_size_relative", "offering_size_relative"),
        "offering_small_pct": ("dilution_offering_small_pct", "offering_small_pct"),
        "offering_large_pct": ("dilution_offering_large_pct", "offering_large_pct"),
    }
    for dest, keys in mapping.items():
        for key in keys:
            if key in src and src[key] is not None:
                value = src[key]
                if dest in {"red_threshold", "warn_threshold", "min_interval_sec", "offering_small_pct", "offering_large_pct"}:
                    cfg[dest] = float(value)
                elif dest in {"offering_days", "cache_seconds"}:
                    cfg[dest] = int(value)
                elif dest in {"dilution_exit", "dilution_filter", "offering_size_relative"}:
                    if isinstance(value, str):
                        cfg[dest] = value.strip().lower() in {"1", "true", "yes", "on"}
                    else:
                        cfg[dest] = bool(value)
                break
    if cfg["warn_threshold"] > cfg["red_threshold"]:
        cfg["warn_threshold"] = cfg["red_threshold"]
    return cfg


def as_date(value):
    """Parse a date-like value to a datetime.date, or None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip().replace("/", "-")[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _today():
    return datetime.now().date()


# --------------------------------------------------------------------------- splits / share growth
def split_factor(splits, start, end):
    """Cumulative to/from factor for splits with start < execution_date <= end."""
    start, end = as_date(start), as_date(end)
    if not start or not end or not splits:
        return 1.0
    factor = 1.0
    for item in splits:
        if isinstance(item, dict):
            day = as_date(item.get("date") or item.get("execution_date") or item.get("executionDate"))
            frm = item.get("from") if item.get("from") is not None else item.get("split_from")
            to = item.get("to") if item.get("to") is not None else item.get("split_to")
        else:
            day = as_date(item[0]) if len(item) >= 1 else None
            frm, to = (item[1], item[2]) if len(item) >= 3 else (None, None)
        if day is None or frm in (None, 0) or to in (None, 0):
            continue
        if start < day <= end:
            try:
                factor *= float(to) / float(frm)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
    return factor


def looks_like_split(ratio, tolerance=SPLIT_RATIO_TOLERANCE):
    """Return the canonical split ratio if `ratio` is essentially 2x, 3x, 4x, 10x, 0.5x, ..."""
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        return None
    if not ratio or ratio <= 0:
        return None
    for canonical in SPLIT_RATIOS:
        if abs(ratio - canonical) <= max(tolerance, abs(canonical) * tolerance):
            return canonical
    return None


def yoy_share_change(current_val, prior_val, splits=None, current_end=None, prior_end=None):
    """Split-adjusted year-over-year change: (current / prior) / split_factor - 1.

    If after the known-split adjustment the residual ratio is still exactly a common
    split ratio, treat it as an unrecorded split (change = 0) rather than issuance.
    """
    try:
        current_val = float(current_val)
        prior_val = float(prior_val)
    except (TypeError, ValueError):
        return None
    if prior_val <= 0 or current_val <= 0:
        return None
    raw = current_val / prior_val
    factor = split_factor(splits, prior_end, current_end)
    if factor and factor > 0:
        adjusted = raw / factor
    else:
        adjusted = raw
    residual = looks_like_split(adjusted)
    if residual:
        return 0.0
    return adjusted - 1.0


def point_in_time(rows, asof):
    """Keep observations whose *filed* date is on or before asof (period end is not enough)."""
    asof = as_date(asof) or _today()
    out = []
    for row in rows or []:
        filed = as_date(row.get("filed") or row.get("filedAt") or row.get("filed_date"))
        if filed is None or filed > asof:
            continue
        try:
            val = float(row.get("val") if row.get("val") is not None else row.get("value"))
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        end = as_date(row.get("end") or row.get("periodEnd") or row.get("period_end"))
        out.append({**row, "val": val, "filed": filed.isoformat(), "end": end.isoformat() if end else row.get("end")})
    out.sort(key=lambda r: (r["filed"], r.get("end") or ""))
    return out


def _closest_prior(rows, latest_end, lo_days=300, hi_days=430):
    """Pick the observation whose period end is ~1 year before latest_end."""
    latest = as_date(latest_end)
    if not latest:
        return None
    best, best_dist = None, None
    for row in rows:
        end = as_date(row.get("end"))
        if not end or end >= latest:
            continue
        dist = (latest - end).days
        if lo_days <= dist <= hi_days and (best_dist is None or abs(dist - 365) < abs(best_dist - 365)):
            best, best_dist = row, dist
    return best


def share_growth_asof(series, splits=None, asof=None, alt_series=None):
    """Point-in-time YoY share change.

    Prefer `series` (typically diluted weighted-average). If a split falls in the
    comparison window, prefer `alt_series` (typically basic shares outstanding, an
    instant) when it can form a YoY pair — weighted-average shares are only partly
    adjusted in the split quarter. If `series` is missing, fall back to `alt_series`.
    """
    asof = as_date(asof) or _today()
    primary = point_in_time(series, asof)
    fallback = point_in_time(alt_series, asof)

    def pair(rows, source):
        if len(rows) < 2:
            return None
        latest = rows[-1]
        prior = _closest_prior(rows, latest.get("end"))
        if not prior:
            return None
        factor = split_factor(splits, prior.get("end"), latest.get("end"))
        change = yoy_share_change(latest["val"], prior["val"], splits, latest.get("end"), prior.get("end"))
        if change is None:
            return None
        return {
            "source": source,
            "shareChangeYoY": change,
            "shareChangeRawYoY": (latest["val"] / prior["val"] - 1.0) if prior["val"] else None,
            "splitFactor": factor,
            "current": {"end": latest.get("end"), "filed": latest.get("filed"), "val": latest["val"]},
            "prior": {"end": prior.get("end"), "filed": prior.get("filed"), "val": prior["val"]},
            "splitInWindow": abs(factor - 1.0) > 1e-9,
        }

    main = pair(primary, "diluted")
    alt = pair(fallback, "basic")
    if main and main["splitInWindow"] and alt:
        chosen = alt
    elif main:
        chosen = main
    else:
        chosen = alt
    return chosen


def classify_growth(change, config=None):
    cfg = config_from_settings(config)
    if change is None:
        return None
    if change >= cfg["red_threshold"]:
        return "DILUTION"
    if change >= cfg["warn_threshold"]:
        return "DILUTION_WARNING"
    return None


# --------------------------------------------------------------------------- offerings
def _norm_form(form):
    return re.sub(r"\s+", "", str(form or "").upper())


def _item_tokens(items):
    text = str(items or "")
    return {part.strip() for part in re.split(r"[;,]", text) if part.strip()}


def parse_submissions(payload):
    """Normalize data.sec.gov/submissions/CIK##########.json (or a test fixture) into filing dicts."""
    if not payload:
        return []
    recent = (payload.get("filings") or {}).get("recent") or {}
    # Already a list of filing dicts?
    if isinstance(payload, list):
        rows = payload
    elif isinstance(recent, list):
        rows = recent
    elif isinstance(recent, dict) and "form" in recent:
        n = len(recent.get("form") or [])
        rows = []
        keys = set(recent)
        for i in range(n):
            rows.append({k: (recent[k][i] if i < len(recent[k] or []) else None) for k in keys})
    else:
        rows = payload.get("filings") or payload.get("recent") or []
        if isinstance(rows, dict):
            rows = []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        form = _norm_form(row.get("form") or row.get("formType"))
        filed = as_date(row.get("filingDate") or row.get("filed") or row.get("acceptanceDateTime"))
        if not form or not filed:
            continue
        items = row.get("items") or row.get("item") or ""
        desc = row.get("primaryDocDescription") or row.get("description") or ""
        out.append({
            "form": form,
            "filingDate": filed.isoformat(),
            "reportDate": (as_date(row.get("reportDate")) or None) and as_date(row.get("reportDate")).isoformat(),
            "accession": row.get("accessionNumber") or row.get("accession") or "",
            "items": items,
            "description": desc,
            "primaryDocument": row.get("primaryDocument") or "",
        })
    out.sort(key=lambda r: r["filingDate"], reverse=True)
    return out


DEBT_RE = re.compile(
    r"\b(senior notes?|debt securities|notes due|medium[-\s]term notes?|commercial paper|"
    r"debentures?|bonds? payable|depositary shares representing (bonds|notes))\b",
    re.I,
)
EQUITY_RE = re.compile(
    r"\b(common stock|ordinary shares|class [a-z] common|preferred stock|depositary shares|"
    r"at[-\s]?the[-\s]?market|\batm\b|equity distribution|warrants? to purchase|forward sale|forward prosupp)\b",
    re.I,
)


def is_offering_filing(filing, text=None):
    """True if this filing should raise DILUTION_OFFERING.

    S-1/F-1 and 8-K item 3.02 always count (new equity registration or unregistered sales).
    S-3/F-3 (non-ASR) and 424B1/3/4 count unless the text is clearly a debt issuance.
    424B2 and automatic shelves (S-3ASR/F-3ASR) only count with equity / ATM language —
    mega-caps file these constantly for senior notes, which is not share dilution.
    424B5 counts unless it reads as debt (forward-sale prospectus supplements are equity).
    """
    form = _norm_form(filing.get("form") if isinstance(filing, dict) else filing)
    items = filing.get("items", "") if isinstance(filing, dict) else ""
    desc = filing.get("description", "") if isinstance(filing, dict) else ""
    blob = " ".join(x for x in (desc, text or "") if x)
    if form in {"8-K", "8-K/A"}:
        tokens = _item_tokens(items)
        if tokens & EIGHT_K_DILUTION_ITEMS:
            return True
        if re.search(r"unregistered sales of equity|3\.02", f"{items} {blob}", re.I):
            return True
        if ATM_RE.search(blob) or ATM_RE.search(str(items)) or EQUITY_RE.search(blob):
            return True
        return False
    if form in {"S-1", "S-1/A", "F-1", "F-1/A"}:
        return True
    if text and (ATM_RE.search(text) or EQUITY_RE.search(text)) and not DEBT_RE.search(text):
        return True
    if form in {"S-3ASR", "F-3ASR", "424B2", "424B5"}:
        return bool(ATM_RE.search(blob) or EQUITY_RE.search(blob)) and not DEBT_RE.search(blob)
    if form in {"S-3", "S-3/A", "F-3", "F-3/A", "424B", "424B1", "424B3", "424B4"}:
        if DEBT_RE.search(blob):
            return False
        return True
    if ATM_RE.search(blob):
        return True
    return False


def offerings_asof(filings, asof=None, window_days=None, document_texts=None, config=None):
    cfg = config_from_settings(config, offering_days=window_days if window_days is not None else None)
    asof = as_date(asof) or _today()
    start = asof - timedelta(days=int(cfg["offering_days"]))
    texts = document_texts or {}
    hits = []
    for filing in parse_submissions(filings) if isinstance(filings, dict) else list(filings or []):
        if isinstance(filings, dict) and "filings" in filings and not isinstance(filing, dict):
            continue
        row = filing if isinstance(filing, dict) and "form" in filing and "filingDate" in filing else None
        if row is None:
            continue
        filed = as_date(row.get("filingDate"))
        if not filed or filed > asof or filed < start:
            continue
        key = row.get("accession") or row.get("primaryDocument") or ""
        text = texts.get(key) or texts.get(row.get("primaryDocument") or "")
        if is_offering_filing(row, text):
            hits.append({**row, "atm": bool(text and ATM_RE.search(text)) or bool(ATM_RE.search(row.get("description") or ""))})
    return hits


# --------------------------------------------------------------------------- offering size (size-relative flag)
SHARES_RE = re.compile(
    r"(?<![\d.,$])(\d{1,3}(?:,\d{3}){1,4}|\d{5,12})\s+(?:shares|ordinary shares|american depositary shares|ADSs)"
    r"(?:\s+of\s+(?:our\s+)?(?:class\s+[a-z]\s+)?(?:common\s+stock|ordinary\s+shares))?",
    re.I,
)
DOLLARS_RE = re.compile(
    r"(?:aggregate\s+(?:gross\s+)?(?:offering\s+price|sales\s+price|gross\s+proceeds)\s+of\s+(?:up\s+to\s+)?|"
    r"having\s+an\s+aggregate\s+offering\s+price\s+of\s+(?:up\s+to\s+)?|up\s+to\s+)"
    r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(million|billion)?",
    re.I,
)
UNDERWRITTEN_RE = re.compile(r"\bunderwrit(?:ten|ers?|ing)\b|\bbook[-\s]running\s+manager", re.I)
SIZE_TIERS = ("SMALL", "MEDIUM", "LARGE")


def parse_offering_size(text):
    """Best-effort cover-page parse: shares offered and/or dollar amount. Returns dict (values may be None)."""
    out = {"shares": None, "dollars": None}
    if not text:
        return out
    head = text[:40000]
    m = SHARES_RE.search(head)
    if m:
        try:
            out["shares"] = int(m.group(1).replace(",", ""))
        except ValueError:
            pass
    d = DOLLARS_RE.search(head)
    if d:
        try:
            val = float(d.group(1).replace(",", ""))
            unit = (d.group(2) or "").lower()
            val *= 1e9 if unit == "billion" else 1e6 if unit == "million" else 1.0
            if val >= 1e5:  # ignore per-share prices
                out["dollars"] = val
        except ValueError:
            pass
    return out


def offering_type(filing, text=None):
    form = _norm_form(filing.get("form"))
    blob = " ".join(x for x in (filing.get("description") or "", text or "") if x)
    if form in {"8-K", "8-K/A"}:
        return "ATM" if ATM_RE.search(blob) else "PRIVATE_PLACEMENT"
    if ATM_RE.search(blob) or filing.get("atm"):
        return "ATM"
    if form in {"S-1", "S-1/A", "F-1", "F-1/A"}:
        return "IPO_OR_S1"
    if form in {"424B1", "424B4"} or (form in {"424B", "424B2", "424B3", "424B5"} and UNDERWRITTEN_RE.search(blob)):
        return "UNDERWRITTEN"
    if form in {"S-3", "S-3/A", "F-3", "F-3/A", "S-3ASR", "F-3ASR"}:
        return "SHELF"
    return "OTHER"


def size_tier(size_pct, config=None):
    cfg = config_from_settings(config)
    if size_pct is None:
        return "UNKNOWN"
    if size_pct < cfg["offering_small_pct"]:
        return "SMALL"
    if size_pct <= cfg["offering_large_pct"]:
        return "MEDIUM"
    return "LARGE"


def annotate_offering(hit, text=None, shares_outstanding=None, price=None, config=None):
    """Add sizePct / sizeTier / offeringType to one offering hit. Pure, no I/O."""
    size = parse_offering_size(text) if text else {"shares": None, "dollars": None}
    pct = None
    if shares_outstanding and shares_outstanding > 0:
        if size["shares"]:
            pct = size["shares"] / float(shares_outstanding)
        elif size["dollars"] and price and price > 0:
            pct = size["dollars"] / (float(shares_outstanding) * float(price))
    if pct is not None and not (0 < pct < 20):   # implausible parse -> unknown
        pct = None
    kind = offering_type(hit, text)
    tier = size_tier(pct, config)
    # A shelf registration with no takedown only creates capacity; size it if known, else LOW-information UNKNOWN.
    return {**hit, "offeringType": kind, "sizeShares": size["shares"], "sizeDollars": size["dollars"],
            "sizePct": pct, "sizeTier": tier}


def offering_severity(offers, config=None):
    """HIGH (>15%), MEDIUM (5-15%), LOW (all <5%), UNKNOWN (size not parseable), or None (no offerings)."""
    if not offers:
        return None
    tiers = [o.get("sizeTier") or "UNKNOWN" for o in offers]
    if "LARGE" in tiers:
        return "HIGH"
    if "MEDIUM" in tiers:
        return "MEDIUM"
    if "UNKNOWN" in tiers:
        return "UNKNOWN"
    return "LOW"


def latest_shares_outstanding(basic_series=None, share_series=None, asof=None):
    for series in (basic_series, share_series):
        rows = point_in_time(series, asof)
        if rows:
            return rows[-1]["val"]
    return None


# --------------------------------------------------------------------------- evaluate / buy block
def evaluate(ticker, asof=None, share_series=None, basic_series=None, splits=None,
             filings=None, document_texts=None, config=None, price=None):
    """Pure evaluation: no I/O. Returns flags, the YoY number, and offering hits.

    `offering` / DILUTION_OFFERING stay binary for compatibility; `offeringSeverity` adds the size-relative
    tier (LOW / MEDIUM / HIGH / UNKNOWN) and `blocksBuy` honours it when offering_size_relative is on."""
    cfg = config_from_settings(config)
    asof = as_date(asof) or _today()
    growth = share_growth_asof(share_series, splits, asof, alt_series=basic_series)
    flags = []
    change = growth["shareChangeYoY"] if growth else None
    kind = classify_growth(change, cfg)
    if kind:
        flags.append(kind)
    offers = offerings_asof(filings, asof, config=cfg, document_texts=document_texts) if filings is not None else []
    severity = None
    if offers:
        flags.append("DILUTION_OFFERING")
        shares_out = latest_shares_outstanding(basic_series, share_series, asof)
        texts = document_texts or {}
        offers = [annotate_offering(o, texts.get(o.get("accession") or "") or texts.get(o.get("primaryDocument") or ""),
                                    shares_out, price, cfg) for o in offers]
        severity = offering_severity(offers, cfg)
    size_relative = bool(cfg.get("offering_size_relative"))
    offer_blocks = "DILUTION_OFFERING" in flags and not (size_relative and severity == "LOW")
    largest = max((o for o in offers if o.get("sizePct") is not None), key=lambda o: o["sizePct"], default=None)
    return {
        "ticker": (ticker or "").upper(),
        "asof": asof.isoformat(),
        "flags": flags,
        "shareChangeYoY": change,
        "growth": growth,
        "offerings": offers,
        "red": "DILUTION" in flags,
        "warning": "DILUTION_WARNING" in flags,
        "offering": "DILUTION_OFFERING" in flags,
        "offeringSeverity": severity,
        "offeringSizePct": largest["sizePct"] if largest else None,
        "offeringType": (largest or (offers[0] if offers else {})).get("offeringType") if offers else None,
        "blocksBuy": "DILUTION" in flags or offer_blocks,
        "config": {k: cfg[k] for k in ("red_threshold", "warn_threshold", "offering_days", "dilution_exit", "dilution_filter",
                                       "offering_size_relative", "offering_small_pct", "offering_large_pct")},
    }


def blocks_new_buy(result, has_position=False, config=None):
    """Whether a new buy (entry or add) should be blocked.

    Existing holdings are not forced out unless dilution_exit is true; the caller
    should still log a warning. Returns (blocked, reason).
    """
    cfg = config_from_settings(config or (result or {}).get("config"))
    if not cfg["dilution_filter"]:
        return False, None
    flags = (result or {}).get("flags") or []
    active = [f for f in flags if f in {"DILUTION", "DILUTION_OFFERING"}]
    if (cfg.get("offering_size_relative") and "DILUTION_OFFERING" in active
            and (result or {}).get("offeringSeverity") == "LOW"):
        active = [f for f in active if f != "DILUTION_OFFERING"]
        if not active:
            return False, "dilution_offering_small"
    # 5-10% warning does not block buys on its own.
    if not active:
        return False, None
    if has_position and not cfg["dilution_exit"]:
        return False, "dilution_warning"
    reason = "dilution_offering" if "DILUTION_OFFERING" in active and "DILUTION" not in active else "dilution"
    if has_position and cfg["dilution_exit"]:
        return True, "dilution_exit"
    if has_position:
        return False, "dilution_warning"
    return True, reason


def apply_to_weights(weights, flag_lookup, config=None):
    """Return a copy of the target-weight frame with new buys blocked point-in-time.

    `flag_lookup(date, ticker) -> result-dict or flags-list`. Uses only information
    that lookup returns (the lookup itself must be point-in-time). Sells are allowed.
    Existing holdings are capped at the prior target (no add) unless dilution_exit
    forces a flat position.
    """
    import pandas as pd

    cfg = config_from_settings(config)
    if not cfg["dilution_filter"] or weights is None or getattr(weights, "empty", False):
        return weights
    out = weights.copy().astype(float)
    held = {col: 0.0 for col in out.columns}
    for day in out.index:
        asof = day if not hasattr(day, "date") else day
        for ticker in out.columns:
            target = float(out.at[day, ticker]) if ticker in out.columns else 0.0
            if target != target:  # NaN
                target = 0.0
            result = flag_lookup(asof, ticker) if flag_lookup else None
            if isinstance(result, (list, tuple, set)):
                result = {"flags": list(result)}
            blocked, reason = blocks_new_buy(result or {}, has_position=held[ticker] > 1e-12, config=cfg)
            if reason == "dilution_exit":
                target = 0.0
            elif blocked and held[ticker] <= 1e-12:
                target = 0.0
            elif reason == "dilution_warning" and target > held[ticker]:
                target = held[ticker]  # allow reduce/exit, block adds
            out.at[day, ticker] = target
            held[ticker] = max(target, 0.0)
    return out


def risk_entry(result):
    """A RISK_KIND=DILUTION dict suitable for an analyst verdict."""
    if not result:
        return None
    change = result.get("shareChangeYoY")
    offers = result.get("offerings") or []
    flags = result.get("flags") or []
    if not flags:
        return None
    pct = f"{change * 100:.1f}%" if isinstance(change, (int, float)) and change is not None else "n/a"
    parts = []
    if "DILUTION" in flags:
        parts.append(f"split-adjusted diluted (or basic) share count is up {pct} year over year")
    elif "DILUTION_WARNING" in flags:
        parts.append(f"share count is up {pct} year over year (warning, below the 10% red threshold)")
    offer_sev = result.get("offeringSeverity")
    if "DILUTION_OFFERING" in flags and offers:
        latest = offers[0]
        size = result.get("offeringSizePct")
        size_txt = f", about {size * 100:.1f}% of shares outstanding" if isinstance(size, (int, float)) else ""
        kind = latest.get("offeringType")
        parts.append(f"recent {latest.get('form')} filed {latest.get('filingDate')}"
                     + (" (ATM language)" if latest.get("atm") else "")
                     + (f" [{kind}]" if kind else "") + size_txt
                     + (f", offering severity {offer_sev}" if offer_sev else ""))
    if "DILUTION" in flags or offer_sev == "HIGH":
        severity = "HIGH"
    elif "DILUTION_OFFERING" in flags and offer_sev == "LOW" and (result.get("config") or {}).get("offering_size_relative", True):
        severity = "LOW"
    elif "DILUTION_OFFERING" in flags:
        severity = "HIGH" if offer_sev is None else "MEDIUM"
    else:
        severity = "MEDIUM"
    return {
        "kind": "DILUTION",
        "severity": severity,
        "detail": "; ".join(parts) + ".",
        "shareChangeYoY": change,
        "flags": flags,
    }


def apply_to_verdict(verdict, result):
    """Force RED / at least WEAK health / a DILUTION risk when the 10% flag is on.

    Offering-only flags force at least YELLOW and the same risk entry. Never upgrades
    a more cautious action into a buy. Returns (verdict, clamps).
    """
    v = dict(verdict or {})
    clamps = list(v.get("clamps") or [])
    if not result:
        return v, clamps
    flags = result.get("flags") or []
    health = v.get("fundamentalHealth") or "UNKNOWN"
    rank = ["UNKNOWN", "STRONG", "ADEQUATE", "WEAK", "DISTRESSED"]

    def at_least(current, floor):
        if current not in rank:
            return floor
        return floor if rank.index(current) < rank.index(floor) else current

    risk = risk_entry(result)
    risks = list(v.get("risks") or [])
    if risk and not any((r or {}).get("kind") == "DILUTION" for r in risks if isinstance(r, dict)):
        risks.append({k: risk[k] for k in ("kind", "severity", "detail")})
        v["risks"] = risks
        clamps.append("dilution risk entry added")

    offer_sev = result.get("offeringSeverity")
    size_rel = (result.get("config") or {}).get("offering_size_relative", True)
    if "DILUTION" not in flags and "DILUTION_OFFERING" in flags and size_rel and offer_sev == "HIGH":
        if v.get("flag") != "RED":
            clamps.append(f"flag {v.get('flag')} -> RED (large offering, {offer_sev})")
            v["flag"] = "RED"
        if v.get("action") in {"BUY", "ADD"}:
            clamps.append(f"action {v.get('action')} -> WATCH (large offering)")
            v["action"] = "WATCH"
        entry = dict(v.get("entry") or {})
        if entry.get("bandLow") is not None or entry.get("bandHigh") is not None or entry.get("orderType") not in {None, "NONE"}:
            v["entry"] = {**entry, "orderType": "NONE", "bandLow": None, "bandHigh": None, "limitPrice": None}
            clamps.append("entry band removed (large offering)")
    elif "DILUTION" not in flags and "DILUTION_OFFERING" in flags and size_rel and offer_sev == "LOW":
        clamps.append("small offering (<5% of shares): risk noted, no clamp")
    elif "DILUTION" in flags:
        if v.get("flag") != "RED":
            clamps.append(f"flag {v.get('flag')} -> RED (dilution {result.get('shareChangeYoY')})")
            v["flag"] = "RED"
        if v.get("fundamentalHealth") not in {"WEAK", "DISTRESSED"}:
            clamps.append(f"fundamentalHealth {health} -> WEAK (dilution)")
            v["fundamentalHealth"] = at_least(health, "WEAK")
        if v.get("action") in {"BUY", "ADD"}:
            clamps.append(f"action {v.get('action')} -> WATCH (dilution red flag)")
            v["action"] = "WATCH"
        entry = dict(v.get("entry") or {})
        if entry.get("bandLow") is not None or entry.get("bandHigh") is not None or entry.get("orderType") not in {None, "NONE"}:
            v["entry"] = {**entry, "orderType": "NONE", "bandLow": None, "bandHigh": None, "limitPrice": None}
            clamps.append("entry band removed (dilution red flag)")
    elif "DILUTION_OFFERING" in flags:
        if v.get("flag") == "GREEN" or not v.get("flag"):
            clamps.append(f"flag {v.get('flag')} -> YELLOW (dilution offering)")
            v["flag"] = "YELLOW"
        if v.get("fundamentalHealth") in {None, "STRONG"}:
            v["fundamentalHealth"] = at_least(health, "ADEQUATE")
        if v.get("action") in {"BUY", "ADD"}:
            clamps.append(f"action {v.get('action')} -> WATCH (dilution offering)")
            v["action"] = "WATCH"
    elif "DILUTION_WARNING" in flags:
        # Warning only: do not force RED. Health stays, but the risk entry is already attached.
        if v.get("flag") == "GREEN":
            clamps.append("flag GREEN stays GREEN with a DILUTION warning (5-10%)")
    if clamps:
        v["clamps"] = list(dict.fromkeys([*(v.get("clamps") or []), *clamps]))
    return v, clamps


# --------------------------------------------------------------------------- I/O: cache, SEC, flags, alerts
class RateLimiter:
    def __init__(self, interval=DEFAULTS["min_interval_sec"]):
        self.interval = max(float(interval), 0.11)
        self.next = 0.0

    def wait(self):
        delay = self.next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next = time.monotonic() + self.interval


_LIMITER = RateLimiter()


def sec_user_agent():
    from local_env import env_value
    agent = env_value("SEC_USER_AGENT")
    if agent:
        return agent
    return "Trading-Model-main research (personal use; set SEC_USER_AGENT to 'Name email@domain')"


def _cache_file(kind, key):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(key))
    path = history_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{kind}-{safe}.json"


def _read_cache(path, max_age):
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    fetched = payload.get("fetchedAt")
    try:
        ts = datetime.fromisoformat(fetched.replace("Z", "+00:00")) if fetched else None
    except (TypeError, ValueError):
        ts = None
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if (datetime.now(timezone.utc) - ts).total_seconds() > max_age:
        return None
    return payload.get("value")


def _write_cache(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "fetchedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "value": value,
    }, default=str), encoding="utf-8")
    tmp.replace(path)


def sec_get_json(url, cache_key=None, kind="sec", config=None, fetch_impl=None, limiter=None):
    """GET JSON from SEC with User-Agent, cache, and rate limit. No secrets are logged."""
    cfg = config_from_settings(config)
    if cache_key:
        cached = _read_cache(_cache_file(kind, cache_key), cfg["cache_seconds"])
        if cached is not None:
            return cached, True
    limiter = limiter or _LIMITER
    fetch_impl = fetch_impl or _default_fetch
    last_error = None
    for attempt in range(1, 4):
        limiter.wait()
        try:
            data = fetch_impl(url, headers={"User-Agent": sec_user_agent(), "Accept": "application/json"})
            if cache_key:
                _write_cache(_cache_file(kind, cache_key), data)
            return data, False
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt < 3:
                time.sleep(1.5 * attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.0 * attempt)
                continue
            raise
    raise last_error  # pragma: no cover


def _default_fetch(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _default_fetch_text(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(600_000).decode("utf-8", errors="replace")


def _strip_html(raw):
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw or "")
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    return re.sub(r"\s+", " ", text)


def fetch_offering_text(cik, filing, fetch_text=None, config=None, limiter=None):
    """Cover-page text of one offering filing (cached, rate-limited, SEC User-Agent). Returns '' on failure."""
    acc = re.sub(r"\D", "", str(filing.get("accession") or ""))
    doc = filing.get("primaryDocument") or ""
    if not acc or not doc:
        return ""
    cfg = config_from_settings(config)
    cache = _cache_file("sec-doc", f"{acc}-{doc}")
    cached = _read_cache(cache, 30 * 86400)
    if cached is not None:
        return cached
    url = f"https://www.sec.gov/Archives/edgar/data/{int(pad_cik(cik))}/{acc}/{doc}"
    (limiter or _LIMITER).wait()
    try:
        raw = (fetch_text or _default_fetch_text)(url, headers={"User-Agent": sec_user_agent(), "Accept": "text/html"})
    except Exception:  # noqa: BLE001
        return ""
    text = _strip_html(raw)[:60000]
    _write_cache(cache, text)
    return text


def latest_close(ticker, asof=None):
    """Last close on or before asof from the local price database (for dollar-size offerings)."""
    try:
        import sqlite3
        from fetch_and_model import DATABASE
        con = sqlite3.connect(Path(DATABASE).resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
        day = (as_date(asof) or _today()).isoformat()
        row = con.execute("SELECT close FROM daily_prices WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT 1",
                          (ticker.upper(), day)).fetchone()
        con.close()
        return float(row[0]) if row and row[0] else None
    except Exception:  # noqa: BLE001
        return None


def pad_cik(cik):
    return str(int(re.sub(r"\D", "", str(cik)) or 0)).zfill(10)


def load_ticker_cik_map(fetch_impl=None, config=None, limiter=None):
    data, _ = sec_get_json(
        "https://www.sec.gov/files/company_tickers.json",
        cache_key="company_tickers", kind="sec-tickers",
        config=config, fetch_impl=fetch_impl, limiter=limiter,
    )
    out = {}
    for row in (data or {}).values() if isinstance(data, dict) else (data or []):
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper()
        cik = row.get("cik_str") or row.get("cik")
        if ticker and cik is not None:
            out[ticker] = str(cik)
            out[ticker.replace(".", "-")] = str(cik)
            out[ticker.replace("-", ".")] = str(cik)
    return out


def fetch_submissions(cik, fetch_impl=None, config=None, limiter=None):
    padded = pad_cik(cik)
    data, cached = sec_get_json(
        f"https://data.sec.gov/submissions/CIK{padded}.json",
        cache_key=f"submissions-{padded}", kind="sec-submissions",
        config=config, fetch_impl=fetch_impl, limiter=limiter,
    )
    return data, cached


def fetch_company_facts(cik, fetch_impl=None, config=None, limiter=None):
    padded = pad_cik(cik)
    data, cached = sec_get_json(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{padded}.json",
        cache_key=f"facts-{padded}", kind="sec-facts",
        config=config, fetch_impl=fetch_impl, limiter=limiter,
    )
    return data, cached


def _concept_facts(facts, namespace, names, kind="instant"):
    """Extract 10-K/10-Q facts for the first concept that has a usable series."""
    block = (facts or {}).get("facts") or {}
    ns = block.get(namespace) or {}
    day = 86400 * 1000  # unused; period length is computed from dates
    for name in names:
        units = (ns.get(name) or {}).get("units") or {}
        rows = units.get("shares") or units.get("pure") or units.get("USD") or []
        if not rows:
            # some filers use a single unit key
            for values in units.values():
                if values:
                    rows = values
                    break
        by_end = {}
        for fact in rows or []:
            form = str(fact.get("form") or "")
            if not re.match(r"^10-[QK]", form):
                continue
            end = as_date(fact.get("end"))
            filed = as_date(fact.get("filed"))
            if not end or not filed:
                continue
            try:
                val = float(fact.get("val"))
            except (TypeError, ValueError):
                continue
            if kind == "flow":
                start = as_date(fact.get("start"))
                if not start:
                    continue
                length = (end - start).days
                period = "Q" if 80 <= length <= 100 else ("FY" if 350 <= length <= 380 else None)
                if not period:
                    continue
                key = period + end.isoformat()
                prev = by_end.get(key)
                if prev is None or filed.isoformat() > prev["filed"]:
                    by_end[key] = {"period": period, "end": end.isoformat(), "val": val, "filed": filed.isoformat(), "concept": name}
            else:
                key = end.isoformat()
                prev = by_end.get(key)
                if prev is None or filed.isoformat() > prev["filed"]:
                    by_end[key] = {"end": end.isoformat(), "val": val, "filed": filed.isoformat(), "concept": name}
        if by_end:
            return sorted(by_end.values(), key=lambda r: (r["end"], r["filed"]))
    return []


def share_series_from_facts(facts):
    diluted = _concept_facts(facts, "us-gaap", [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstandingBasicAndDiluted",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ], kind="flow")
    # Weighted-average "flow" series: keep quarter rows (and FY when no quarter lands on that end).
    quarterly = [r for r in diluted if r.get("period") == "Q"]
    for fy in (r for r in diluted if r.get("period") == "FY"):
        if not any(abs((as_date(q["end"]) - as_date(fy["end"])).days) < 10 for q in quarterly if as_date(q["end"]) and as_date(fy["end"])):
            quarterly.append(fy)
    quarterly.sort(key=lambda r: r["end"])
    basic = _concept_facts(facts, "dei", ["EntityCommonStockSharesOutstanding"], kind="instant")
    if not basic:
        basic = _concept_facts(facts, "us-gaap", [
            "CommonStockSharesOutstanding",
            "EntityCommonStockSharesOutstanding",
        ], kind="instant")
    return quarterly, basic


def load_splits(tickers):
    """Best-effort split records from the local corporate_splits table. Returns {TICKER: [..]}."""
    out = {t.upper(): [] for t in tickers}
    try:
        from fetch_and_model import DATABASE
        from price_history import open_history
        from corporate_actions import ensure_schema
        with open_history(DATABASE) as connection:
            ensure_schema(connection)
            placeholders = ",".join("?" * len(out))
            if not out:
                return out
            rows = connection.execute(
                f"SELECT ticker, execution_date, split_from, split_to FROM corporate_splits "
                f"WHERE ticker IN ({placeholders})",
                list(out),
            ).fetchall()
            for ticker, execution_date, split_from, split_to in rows:
                out.setdefault(ticker.upper(), []).append({
                    "date": execution_date, "from": split_from, "to": split_to,
                })
    except Exception:  # noqa: BLE001 — splits are optional; missing table must not break a scan
        pass
    return out


def load_flags(path=None):
    path = Path(path) if path else flags_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def save_flags(payload, path=None):
    path = Path(path) if path else flags_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def flag_lookup_from_payload(payload):
    """Build a (date, ticker) -> result lookup from a flags file or in-memory history.

    History entries must include `asof` and flags computed with filing dates <= asof.
    If only a current snapshot is present, the snapshot is used on or after its asof
    (not before — that would leak future filings into a backtest).
    """
    tickers = (payload or {}).get("tickers") or {}
    histories = (payload or {}).get("history") or {}

    def lookup(day, ticker):
        key = str(ticker).upper()
        asof = as_date(day)
        hist = histories.get(key) or []
        chosen = None
        for row in hist:
            row_asof = as_date(row.get("asof"))
            if row_asof and asof and row_asof <= asof:
                if chosen is None or as_date(chosen.get("asof")) < row_asof:
                    chosen = row
        if chosen:
            return chosen
        snap = tickers.get(key)
        if snap and asof and as_date(snap.get("asof")) and as_date(snap.get("asof")) <= asof:
            return snap
        return {"flags": []}

    return lookup


def _emit_alert(payload):
    try:
        from alerts_emit import emit
        return emit(payload)
    except Exception:  # noqa: BLE001
        return False


def watchlist_tickers(extra=None):
    """Watchlist + paper holdings + day-trade symbols + env targets. Never raises."""
    names = []
    if extra:
        names.extend(extra)
    env = os.environ.get("TRADING_AI_TARGETS") or ""
    names.extend(t.strip().upper() for t in re.split(r"[\s,]+", env) if t.strip())
    names.extend(["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO", "AMD"])
    try:
        from universe import CORE_ETFS
        names.extend(CORE_ETFS)
    except Exception:
        pass
    for path in (
        _research_dir() / "paper_status.json",
        _research_dir() / "paper_status_dryrun.json",
        _research_dir() / "daytrade_status.json",
        _research_dir() / "daytrade_status_dryrun.json",
    ):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for pos in data.get("positions") or []:
            if isinstance(pos, dict) and pos.get("ticker"):
                names.append(str(pos["ticker"]).upper())
            elif isinstance(pos, dict) and pos.get("symbol"):
                names.append(str(pos["symbol"]).upper())
        for t in data.get("targets") or []:
            if isinstance(t, dict) and t.get("ticker"):
                names.append(str(t["ticker"]).upper())
        for row in data.get("trades") or data.get("recentDecisions") or []:
            if isinstance(row, dict) and (row.get("ticker") or row.get("symbol")):
                names.append(str(row.get("ticker") or row.get("symbol")).upper())
    # live sqlite ledgers (empty if the account has not been created yet)
    try:
        import sqlite3
        from fetch_and_model import DATA_DIR
        db = DATA_DIR / "paper_trading.sqlite3"
        if db.is_file():
            with sqlite3.connect(db) as con:
                for (t,) in con.execute("SELECT DISTINCT ticker FROM positions"):
                    names.append(str(t).upper())
                try:
                    for (t,) in con.execute("SELECT DISTINCT ticker FROM signal_state WHERE target_weight > 0"):
                        names.append(str(t).upper())
                except sqlite3.DatabaseError:
                    pass
                try:
                    for (t,) in con.execute("SELECT DISTINCT symbol FROM dt_trades"):
                        names.append(str(t).upper())
                except sqlite3.DatabaseError:
                    pass
    except Exception:
        pass
    # unique, stable order, skip obvious blanks
    seen, out = set(), []
    for t in names:
        t = re.sub(r"[^A-Z0-9.-]", "", str(t).upper())
        if t and t not in seen and len(t) <= 12:
            seen.add(t)
            out.append(t)
    return out


def _write_history(ticker, result, diluted, basic):
    folder = history_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{ticker}.json"
    payload = {
        "ticker": ticker,
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest": result,
        "dilutedShares": diluted,
        "basicShares": basic,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def load_history(ticker):
    path = history_dir() / f"{str(ticker).upper()}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def history_query(ticker, asof=None, config=None):
    """Analyst-facing dilution history for one ticker (from the last scan, no network)."""
    ticker = str(ticker).upper()
    stored = load_history(ticker)
    flags = load_flags() or {}
    snap = (flags.get("tickers") or {}).get(ticker)
    if not stored and not snap:
        return {"ticker": ticker, "available": False, "message": "No dilution history yet. Run: python dilution.py scan"}
    diluted = (stored or {}).get("dilutedShares") or []
    basic = (stored or {}).get("basicShares") or []
    splits = ((stored or {}).get("latest") or {}).get("growth")  # splits not always stored; recompute if we can
    # Reconstruct a point-in-time view if asof is requested.
    day = as_date(asof)
    if day and (diluted or basic):
        result = evaluate(
            ticker, asof=day, share_series=diluted, basic_series=basic,
            splits=load_splits([ticker]).get(ticker),
            filings={"filings": {"recent": []}},  # offerings: use stored latest offerings filtered by asof
            config=config,
        )
        stored_offers = ((stored or {}).get("latest") or snap or {}).get("offerings") or []
        result["offerings"] = [o for o in stored_offers if as_date(o.get("filingDate")) and as_date(o.get("filingDate")) <= day
                               and (day - as_date(o.get("filingDate"))).days <= config_from_settings(config)["offering_days"]]
        if result["offerings"] and "DILUTION_OFFERING" not in result["flags"]:
            result["flags"] = [*(result["flags"] or []), "DILUTION_OFFERING"]
            result["offering"] = True
            sev = offering_severity(result["offerings"])
            result["offeringSeverity"] = sev
            sized = [o for o in result["offerings"] if o.get("sizePct") is not None]
            result["offeringSizePct"] = max((o["sizePct"] for o in sized), default=None)
            small_ok = config_from_settings(config).get("offering_size_relative") and sev == "LOW"
            result["blocksBuy"] = bool(result["blocksBuy"]) or not small_ok
    else:
        result = (stored or {}).get("latest") or snap
    return {
        "ticker": ticker,
        "available": True,
        "asof": (as_date(asof) or as_date((result or {}).get("asof")) or _today()).isoformat(),
        "latest": result,
        "dilutedShares": diluted,
        "basicShares": basic,
        "snapshot": snap,
    }


def scan(tickers=None, asof=None, config=None, fetch_impl=None, emit=True, limiter=None, fetch_text=None):
    """Refresh flags for the watchlist / holdings. Network errors are recorded, never raised to the daily job."""
    cfg = config_from_settings(config)
    asof = as_date(asof) or _today()
    tickers = watchlist_tickers(tickers)
    limiter = limiter or RateLimiter(cfg["min_interval_sec"])
    previous = (load_flags() or {}).get("tickers") or {}
    cik_map, cik_error = {}, None
    try:
        cik_map = load_ticker_cik_map(fetch_impl=fetch_impl, config=cfg, limiter=limiter)
    except Exception as exc:  # noqa: BLE001
        cik_error = str(exc)
    splits = load_splits(tickers)
    results, errors, alerts = {}, [], []
    for ticker in tickers:
        cik = cik_map.get(ticker) or cik_map.get(ticker.replace(".", "-"))
        if not cik:
            # ETFs / funds typically have no corporate 10-K share count — not an error.
            results[ticker] = {
                "ticker": ticker, "asof": asof.isoformat(), "flags": [], "shareChangeYoY": None,
                "skipped": "no_cik", "offerings": [], "blocksBuy": False,
            }
            continue
        try:
            facts, _ = fetch_company_facts(cik, fetch_impl=fetch_impl, config=cfg, limiter=limiter)
            diluted, basic = share_series_from_facts(facts)
            submissions, _ = fetch_submissions(cik, fetch_impl=fetch_impl, config=cfg, limiter=limiter)
            result = evaluate(
                ticker, asof=asof, share_series=diluted, basic_series=basic,
                splits=splits.get(ticker), filings=submissions, config=cfg,
            )
            # Size-relative offering flag: read the cover page of recent offering filings (at most 3).
            # Tests that inject fetch_impl without fetch_text stay offline.
            if result.get("offerings") and cfg.get("offering_size_relative") and (fetch_text or fetch_impl is None):
                texts = {}
                for hit in result["offerings"][:3]:
                    text = fetch_offering_text(cik, hit, fetch_text=fetch_text, config=cfg, limiter=limiter)
                    if text:
                        texts[hit.get("accession") or hit.get("primaryDocument") or ""] = text
                if texts:
                    result = evaluate(
                        ticker, asof=asof, share_series=diluted, basic_series=basic,
                        splits=splits.get(ticker), filings=submissions, config=cfg,
                        document_texts=texts, price=latest_close(ticker, asof),
                    )
            result["cik"] = pad_cik(cik)
            results[ticker] = result
            _write_history(ticker, result, diluted, basic)
            prev_flags = set((previous.get(ticker) or {}).get("flags") or [])
            new_flags = [f for f in result["flags"] if f in {"DILUTION", "DILUTION_OFFERING"} and f not in prev_flags]
            if emit and new_flags:
                change = result.get("shareChangeYoY")
                pct = f"{change * 100:.1f}%" if isinstance(change, (int, float)) and change is not None else "n/a"
                offers = result.get("offerings") or []
                offer_bit = ""
                if offers:
                    offer_bit = f" Latest offering-related filing: {offers[0].get('form')} on {offers[0].get('filingDate')}."
                sz = result.get("offeringSizePct")
                size_txt = ((f"Offering size: {result.get('offeringSeverity')} ("
                             + ("n/a" if sz is None else f"{sz * 100:.1f}%") + " of shares outstanding). ")
                            if result.get("offeringSeverity") else "")
                text = (f"{ticker} is flagged for dilution ({', '.join(new_flags)}). "
                        f"Split-adjusted share-count change is {pct} year over year.{offer_bit} "
                        + size_txt
                        + ("New buys are blocked while this is active." if result.get("blocksBuy")
                           else "Small offering (under 5% of shares): noted, new buys are not blocked."))
                ok = _emit_alert({
                    "type": "dilution",
                    "ticker": ticker,
                    "severity": "warning",
                    "title": f"{ticker} dilution flag",
                    "text": text,
                    "speech": f"{ticker} is flagged for dilution. Split-adjusted share count is up {pct} year over year. I'm blocking new buys.",
                    "href": f"/ticker.html?symbol={ticker}",
                    "meta": {"flags": result["flags"], "shareChangeYoY": change, "offerings": offers[:3]},
                })
                alerts.append({"ticker": ticker, "flags": new_flags, "emitted": bool(ok)})
        except Exception as exc:  # noqa: BLE001 — one ticker must not abort the scan
            errors.append({"ticker": ticker, "error": str(exc)})
            results[ticker] = {
                "ticker": ticker, "asof": asof.isoformat(), "flags": [], "error": str(exc),
                "shareChangeYoY": None, "offerings": [], "blocksBuy": False,
            }
    flagged = {t: r for t, r in results.items() if r.get("flags")}
    payload = {
        "schema": "dilution_flags/v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "asof": asof.isoformat(),
        "config": {k: cfg[k] for k in ("red_threshold", "warn_threshold", "offering_days", "dilution_exit", "dilution_filter")},
        "tickers": results,
        "flagged": sorted(flagged),
        "errors": errors,
        "alerts": alerts,
        "cikMapError": cik_error,
        "scanned": len(tickers),
    }
    path = save_flags(payload)
    payload["path"] = str(path)
    return payload


def is_buy_blocked(ticker, asof=None, has_position=False, config=None, payload=None):
    """Lookup helper for the paper trader / day trader. Missing data = not blocked."""
    payload = payload if payload is not None else load_flags()
    if not payload:
        return False, None, None
    lookup = flag_lookup_from_payload(payload)
    result = lookup(asof or payload.get("asof") or _today(), ticker)
    blocked, reason = blocks_new_buy(result, has_position=has_position, config=config or payload.get("config"))
    return blocked, reason, result


# --------------------------------------------------------------------------- backtest comparison
def compare_tsmom(panel=None, start=None, end=None, train_min=252, test_len=63, config=None, payload=None):
    """Walk-forward-style comparison of plain TSMOM with vs without the dilution buy block.

    Uses the default TSMOM parameters (not a parameter grid) and stitched expanding-window
    OOS periods so the only difference is the filter. The lookup is point-in-time. When
    the universe is CORE_ETFS, almost every name has no CIK / no share-count flag, so the
    delta is expected to be zero or very small.
    """
    import pandas as pd
    from costs import CostModel
    from portfolio_sim import simulate
    from strategies import build

    cfg = config_from_settings(config)
    if panel is None:
        from market_data import load_panel
        from universe import CORE_ETFS
        panel = load_panel(tickers=list(CORE_ETFS))
    strategy = build({"strategy": "tsmom", "params": {}})
    weights = strategy.target_weights(panel)
    stops = strategy.stops(panel)
    payload = payload if payload is not None else (load_flags() or {})
    lookup = flag_lookup_from_payload(payload)
    filtered = apply_to_weights(weights, lookup, cfg)

    dates = list(panel.dates)
    if start is not None:
        dates = [d for d in dates if d >= pd.Timestamp(start)]
    if end is not None:
        dates = [d for d in dates if d <= pd.Timestamp(end)]
    n = len(dates)
    if n < train_min + max(test_len, 20):
        # Fall back to a single full-period (post warm-up) compare.
        folds = [{"oosStart": str(dates[min(train_min, n - 1)].date()) if dates else None,
                  "oosEnd": str(dates[-1].date()) if dates else None}]
        oos_start = dates[min(train_min, n - 1)] if dates else None
        base = simulate(panel, weights, stops, CostModel.for_broker("schwab"), 10_000.0, "next_open",
                        start=oos_start, end=end, whole_shares=True, min_order_value=100.0)
        filt = simulate(panel, filtered, stops, CostModel.for_broker("schwab"), 10_000.0, "next_open",
                        start=oos_start, end=end, whole_shares=True, min_order_value=100.0)
        return _compare_payload(base, filt, folds, payload, "full_period_after_warmup")

    folds, oos_index = [], None
    for i, start_i in enumerate(range(train_min, n, test_len)):
        end_i = min(start_i + test_len, n)
        fold_start, fold_end = dates[start_i], dates[end_i - 1]
        folds.append({"fold": i + 1, "oosStart": str(fold_start.date()), "oosEnd": str(fold_end.date())})
        if oos_index is None:
            oos_index = start_i
    oos_start = dates[oos_index]
    base = simulate(panel, weights, stops, CostModel.for_broker("schwab"), 10_000.0, "next_open",
                    start=oos_start, end=dates[-1], whole_shares=True, min_order_value=100.0)
    filt = simulate(panel, filtered, stops, CostModel.for_broker("schwab"), 10_000.0, "next_open",
                    start=oos_start, end=dates[-1], whole_shares=True, min_order_value=100.0)
    return _compare_payload(base, filt, folds, payload, "expanding_window_oos")


def _perf(result):
    import numpy as np
    r = result.returns.dropna()
    if r.empty:
        return {"cagr": None, "sharpe": None, "maxDrawdown": None, "totalReturn": None, "sessions": 0}
    ppy = 252
    mu, sig = float(r.mean()), float(r.std(ddof=0))
    equity = result.equity.dropna()
    roll_max = equity.cummax()
    dd = (equity / roll_max - 1.0).min() if len(equity) else 0.0
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 and equity.iloc[0] else None
    years = len(r) / ppy
    cagr = (1.0 + total) ** (1 / years) - 1 if total is not None and years > 0 and (1.0 + total) > 0 else None
    return {
        "cagr": cagr,
        "sharpe": (mu / sig * (ppy ** 0.5)) if sig else None,
        "maxDrawdown": float(dd) if dd == dd else None,
        "totalReturn": total,
        "sessions": int(len(r)),
        "finalEquity": float(equity.iloc[-1]) if len(equity) else None,
    }


def _compare_payload(base, filt, folds, payload, mode):
    bp, fp = _perf(base), _perf(filt)
    def delta(key):
        a, b = bp.get(key), fp.get(key)
        if a is None or b is None:
            return None
        return b - a
    flagged = (payload or {}).get("flagged") or [
        t for t, r in ((payload or {}).get("tickers") or {}).items() if r.get("flags")
    ]
    blocked_days = 0
    try:
        # Count sessions where any name's filtered weight is below the unfiltered weight while positive.
        diff = (base.holdings.fillna(0) - filt.holdings.fillna(0)).abs()
        blocked_days = int((diff.sum(axis=1) > 1e-9).sum()) if hasattr(diff, "sum") else 0
    except Exception:
        blocked_days = 0
    return {
        "mode": mode,
        "folds": folds,
        "withoutFilter": bp,
        "withFilter": fp,
        "delta": {k: delta(k) for k in ("cagr", "sharpe", "maxDrawdown", "totalReturn", "finalEquity")},
        "flaggedInSnapshot": flagged,
        "sessionsWithHoldingDiff": blocked_days,
        "note": ("The plain TSMOM baseline trades CORE_ETFS, which usually have no corporate 10-K "
                 "share-count series, so the dilution filter is expected to change little or nothing. "
                 "A non-zero delta would come from names that had a DILUTION or DILUTION_OFFERING "
                 "flag on a filing date inside the OOS window."),
    }


# --------------------------------------------------------------------------- CLI
def _print(payload):
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dilution flags, SEC offering watcher, buy-block helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_scan = sub.add_parser("scan", help="Refresh flags for the watchlist and paper holdings (non-fatal).")
    p_scan.add_argument("--tickers", default="", help="Extra comma-separated tickers")
    p_scan.add_argument("--asof", default="", help="YYYY-MM-DD (default: today)")
    p_scan.add_argument("--no-alerts", action="store_true")
    p_hist = sub.add_parser("history", help="Print stored dilution history for one ticker (no network).")
    p_hist.add_argument("ticker")
    p_hist.add_argument("--asof", default="")
    p_flag = sub.add_parser("flag", help="Look up whether a ticker is currently blocked.")
    p_flag.add_argument("ticker")
    p_flag.add_argument("--asof", default="")
    p_flag.add_argument("--has-position", action="store_true")
    p_cmp = sub.add_parser("compare", help="TSMOM walk-forward with vs without the dilution filter.")
    p_cmp.add_argument("--train-min", type=int, default=252)
    p_cmp.add_argument("--test-len", type=int, default=63)
    args = parser.parse_args(argv)
    if args.cmd == "scan":
        extra = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        payload = scan(tickers=extra or None, asof=args.asof or None, emit=not args.no_alerts)
        # Always exit 0 so a network hiccup never fails the daily paper run.
        _print({k: payload[k] for k in payload if k != "tickers"} | {
            "flaggedDetail": {t: payload["tickers"][t] for t in payload.get("flagged") or []},
        })
        return 0
    if args.cmd == "history":
        _print(history_query(args.ticker, asof=args.asof or None))
        return 0
    if args.cmd == "flag":
        blocked, reason, result = is_buy_blocked(args.ticker, asof=args.asof or None, has_position=args.has_position)
        _print({"ticker": args.ticker.upper(), "blocked": blocked, "reason": reason, "result": result})
        return 0
    if args.cmd == "compare":
        _print(compare_tsmom(train_min=args.train_min, test_len=args.test_len))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
