"""Config-gated paper / day-trader reactions to macro releases and overnight signals.

Default is off. In `log` mode every suggested adjustment is recorded but the book is unchanged.
In `apply` mode only rules that have beaten chance out of sample (Holm-proven in the event study)
are allowed to change targets. Everything else is marked experimental and logged only, unless
`macro_allow_experimental` is on (still never places a live broker order).

Examples (all experimental until the event study says otherwise):
  * Pre-release: cut gross exposure ahead of a high-impact CPI / NFP / FOMC print.
  * Large hawkish CPI surprise: cut gross or hedge with puts / SH (options module if present).
  * Jobs surprise: tilt duration (IEF / TLT).
  * Adverse regime + proven downside rule: puts, bear put spreads, or inverse ETFs.

PDT limits and the risk engine stay final. Spreads widen at the release, so day-trading
rules should sit out the first minutes rather than chase the print.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from fetch_and_model import DATA_DIR
from macro_calendar import upcoming, load_calendar, event_by_id

ET = ZoneInfo("America/New_York")
MACRO_DIR = DATA_DIR / "Macro"
STUDY_PATH = MACRO_DIR / "event_study.json"
OVERNIGHT_PATH = MACRO_DIR / "overnight_study.json"
STATE_PATH = MACRO_DIR / "reaction_state.json"

# High-impact names that can trigger pre-release risk reduction (still experimental until proven).
PRE_RELEASE_HIGH_IMPACT = {"cpi", "core_cpi", "nfp", "unrate", "core_pce", "pce", "fomc_decision", "gdp"}
DEFAULT_PRE_RELEASE_SCALE = 0.70
DEFAULT_HAWKISH_GROSS_SCALE = 0.75
DEFAULT_DURATION_TILT = 0.05
HEDGE_ETF = "SH"
DURATION_ETF = {"long": "TLT", "intermediate": "IEF"}


def _load(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def proven_set(study_path=STUDY_PATH, overnight_path=OVERNIGHT_PATH):
    try:
        from macro_event_study import proven_rules
        return proven_rules(_load(study_path, {}), _load(overnight_path, {}))
    except Exception:
        ev = _load(study_path, {}) or {}
        ov = _load(overnight_path, {}) or {}
        return [{"kind": "macro_event", **t} for t in ev.get("proven") or []] + [
            {"kind": "overnight", **t} for t in ov.get("proven") or []
        ]


def _proven_for(event_id, asset=None, horizon=None, rules=None):
    rules = rules if rules is not None else proven_set()
    hits = []
    for r in rules:
        if r.get("kind") == "macro_event" and r.get("eventId") == event_id:
            if asset and r.get("asset") != asset:
                continue
            if horizon and r.get("horizon") != horizon:
                continue
            hits.append(r)
        if r.get("kind") == "overnight" and event_id == "overnight":
            hits.append(r)
    return hits


def next_high_impact(asof, hours=24, path=None):
    rows = upcoming(asof=asof, days=2, include_past_hours=0, path=path) if path else upcoming(asof=asof, days=2, include_past_hours=0)
    out = []
    asof = asof if getattr(asof, "tzinfo", None) else datetime.now(ET)
    horizon = asof + timedelta(hours=hours)
    for row in rows:
        if row.get("id") not in PRE_RELEASE_HIGH_IMPACT:
            continue
        if row.get("importance") != "high":
            continue
        when = datetime.fromisoformat(row["datetimeEt"])
        if when.tzinfo is None:
            when = when.replace(tzinfo=ET)
        if asof <= when <= horizon:
            out.append(row)
    return out


def latest_print(event_ids, hours=12, path=None):
    doc = load_calendar(path) if path else load_calendar()
    asof = datetime.now(ET)
    found = []
    for row in reversed(doc.get("events") or []):
        if row.get("id") not in event_ids:
            continue
        if row.get("status") != "printed" and row.get("actual") is None:
            continue
        when = datetime.fromisoformat(row.get("printedAtEt") or row.get("datetimeEt"))
        if when.tzinfo is None:
            when = when.replace(tzinfo=ET)
        if asof - timedelta(hours=hours) <= when <= asof:
            found.append(row)
    return found


def try_options_hedge(kind, score, settings):
    """Ask the GB-13 options module to size a put / put-spread hedge. Fail soft to SH."""
    notes = []
    try:
        # Possible names the options worker might land under; we do not invent an order.
        for name in ("options_paper", "options_engine", "paper_options"):
            try:
                mod = __import__(name)
            except ImportError:
                continue
            fn = getattr(mod, "macro_hedge_suggestion", None) or getattr(mod, "hedge_for_macro", None)
            if callable(fn):
                suggestion = fn(kind=kind, score=score, settings=settings)
                return suggestion, f"{name}"
        notes.append("Options module not available yet; suggesting the inverse ETF SH as a paper hedge.")
    except Exception as error:  # noqa: BLE001
        notes.append(f"options hedge skipped: {error}")
    return {"instrument": "etf", "ticker": HEDGE_ETF, "role": "inverse_index_hedge",
            "notes": notes}, "inverse_etf_fallback"


def planned_actions(asof, settings, equity=None):
    """Return a list of suggested actions and whether each one is proven."""
    mode = str(settings.get("macro_reaction") or "off").lower()
    allow_exp = bool(settings.get("macro_allow_experimental"))
    actions = []
    rules = proven_set()
    # 1. Pre-release risk reduction.
    upcoming_hi = next_high_impact(asof, hours=int(settings.get("macro_prerelease_hours") or 24))
    if upcoming_hi:
        proven = any(_proven_for(r["id"], horizon="open_gap", rules=rules) or
                     _proven_for(r["id"], horizon="session_oc", rules=rules) for r in upcoming_hi)
        scale = float(settings.get("macro_prerelease_scale") or DEFAULT_PRE_RELEASE_SCALE)
        actions.append({
            "kind": "pre_release_cut",
            "scaleGross": scale,
            "events": [r.get("key") for r in upcoming_hi],
            "names": [r.get("name") for r in upcoming_hi],
            "proven": proven,
            "experimental": not proven,
            "reason": "High-impact release due within the look-ahead window. Cutting gross is a defensive default; it stays experimental until the event study shows a pre-release cut beats chance out of sample.",
        })
    # 2. Fresh hawkish CPI / core PCE surprise.
    cpi = latest_print({"cpi", "core_cpi", "core_pce", "pce"}, hours=12)
    for row in cpi:
        z = row.get("surpriseZ")
        if z is None or z < 1.5:
            continue
        proven = bool(_proven_for(row["id"], asset="SPY", rules=rules))
        hedge, hedge_src = try_options_hedge("hawkish_cpi", {"z": z, "row": row}, settings)
        actions.append({
            "kind": "hawkish_inflation_cut",
            "scaleGross": float(settings.get("macro_hawkish_scale") or DEFAULT_HAWKISH_GROSS_SCALE),
            "eventKey": row.get("key"),
            "surpriseZ": z,
            "hedge": hedge,
            "hedgeSource": hedge_src,
            "proven": proven,
            "experimental": not proven,
            "reason": f"Large hawkish inflation surprise (z={z:+.2f}). Cut gross and/or hedge. Spreads at the print are wide; this is not a first-tick trade.",
        })
    # 3. Jobs surprise → duration tilt.
    jobs = latest_print({"nfp", "unrate", "claims"}, hours=12)
    for row in jobs:
        z = row.get("surpriseZ")
        if z is None or abs(z) < 1.25:
            continue
        # Hawkish labor (hot NFP / low claims / low unemployment) → shorter duration.
        tilt = -DEFAULT_DURATION_TILT if z > 0 else DEFAULT_DURATION_TILT
        proven = bool(_proven_for(row["id"], asset="TLT", rules=rules) or
                      _proven_for(row["id"], asset="IEF", rules=rules))
        actions.append({
            "kind": "jobs_duration_tilt",
            "tltDelta": tilt,
            "iefDelta": tilt * 0.5,
            "eventKey": row.get("key"),
            "surpriseZ": z,
            "proven": proven,
            "experimental": not proven,
            "reason": f"Labor surprise z={z:+.2f}: tilt duration ({'shorter' if tilt < 0 else 'longer'}).",
        })
    # 4. FOMC hawkish surprise.
    fomc = latest_print({"fomc_decision"}, hours=18)
    for row in fomc:
        z = row.get("surpriseZ")
        surprise_bp = row.get("surprise")
        big = (z is not None and abs(z) >= 1.25) or (isinstance(surprise_bp, (int, float)) and abs(surprise_bp) >= 12.5)
        if not big:
            continue
        proven = bool(_proven_for("fomc_decision", rules=rules))
        actions.append({
            "kind": "fomc_reaction",
            "scaleGross": DEFAULT_HAWKISH_GROSS_SCALE if (z or 0) > 0 else 1.0,
            "eventKey": row.get("key"),
            "surpriseZ": z,
            "proven": proven,
            "experimental": not proven,
            "reason": "FOMC surprise vs the implied path. Reaction stays paper-only and experimental until proven.",
        })
    for act in actions:
        act["apply"] = (mode == "apply") and (act["proven"] or allow_exp)
        act["logOnly"] = not act["apply"]
    return actions, rules, mode


def adjust_paper_targets(target, day, settings, equity=None, asof=None):
    """Scale / tilt a paper-trader target weight series. Returns (new_target, actions)."""
    mode = str(settings.get("macro_reaction") or "off").lower()
    if mode in {"off", "", "none"}:
        return target, []
    if asof is None:
        # Session decision is made at the close of `day` for next-open execution: look ahead
        # from the close (16:00 ET) through the next morning's releases.
        asof = datetime(day.year, day.month, day.day, 16, 0, tzinfo=ET) if hasattr(day, "year") else datetime.now(ET)
    actions, _rules, _mode = planned_actions(asof, settings, equity=equity)
    if target is None:
        target = pd.Series(dtype=float)
    new = target.astype(float).copy()
    applied = []
    for act in actions:
        if not act.get("apply"):
            applied.append(act)
            continue
        if act["kind"] in {"pre_release_cut", "hawkish_inflation_cut", "fomc_reaction"} and act.get("scaleGross") is not None:
            new = new * float(act["scaleGross"])
        if act["kind"] == "jobs_duration_tilt":
            for ticker, delta in ((DURATION_ETF["long"], act.get("tltDelta") or 0),
                                  (DURATION_ETF["intermediate"], act.get("iefDelta") or 0)):
                new[ticker] = float(new.get(ticker, 0.0) + delta)
        if act.get("hedge") and act["kind"] == "hawkish_inflation_cut":
            hedge = act["hedge"]
            if hedge.get("instrument") == "etf" and hedge.get("ticker"):
                extra = max(0.0, 1.0 - float(act.get("scaleGross") or 1.0)) * 0.5
                new[hedge["ticker"]] = float(new.get(hedge["ticker"], 0.0) + extra)
        applied.append(act)
    # Risk engine still caps later; do not lever up here.
    if float(new.abs().sum() or 0) > 1.05 and float(target.abs().sum() or 0) <= 1.05:
        new = new * (1.0 / float(new.abs().sum()))
    return new, applied


def adjust_daytrade_candidates(candidates, session_date, settings, asof=None):
    """Filter or mark ORB candidates around a release. PDT / breaker stay in the day trader."""
    mode = str(settings.get("macro_reaction") or settings.get("macro_daytrade") or "off").lower()
    if mode in {"off", "", "none"}:
        return candidates, []
    if asof is None:
        # Morning of the session, 9:25 ET — just before the opening range.
        d = pd.Timestamp(session_date).to_pydatetime()
        asof = datetime(d.year, d.month, d.day, 9, 25, tzinfo=ET)
    actions, _rules, _mode = planned_actions(asof, {**settings, "macro_reaction": mode}, equity=None)
    notes = []
    if candidates is None or (hasattr(candidates, "empty") and candidates.empty):
        return candidates, actions
    out = candidates.copy()
    sit_out = any(a["kind"] in {"pre_release_cut", "hawkish_inflation_cut", "fomc_reaction"}
                  and (a.get("apply") or a.get("proven"))
                  for a in actions)
    # Even experimental high-impact prints: sit out the open rather than trade the first print
    # when the user has enabled apply/log — we only skip in apply mode if proven or allow_exp,
    # but we always *note* that spreads are wide.
    wide = any(a["kind"] in {"pre_release_cut", "hawkish_inflation_cut", "fomc_reaction"} for a in actions)
    if wide:
        notes.append("Spreads typically widen around the release; ORB entries that fire at 9:30–9:45 are trading a noisy print.")
    if sit_out and any(a.get("apply") for a in actions):
        if "skip_reason" not in out.columns:
            out["skip_reason"] = None
        # Don't delete rows (PDT accounting needs the trail); mark a skip reason the day trader
        # already understands, or a dedicated one it can log.
        out["macro_skip"] = True
        notes.append("Day-trade candidates marked macro_skip due to a proven (or explicitly allowed) release rule.")
    return out, actions + ([{"kind": "note", "reason": n, "proven": False, "experimental": True, "apply": False} for n in notes])


def status():
    ev = _load(STUDY_PATH, {}) or {}
    ov = _load(OVERNIGHT_PATH, {}) or {}
    return {
        "provenEventRules": len(ev.get("proven") or []),
        "provenOvernightRules": len(ov.get("proven") or []),
        "eventStudyAt": ev.get("generatedAt"),
        "overnightAt": ov.get("generatedAt"),
        "defaultMode": "off",
    }
