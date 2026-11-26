"""Geopolitical and black swan monitor: detect, confirm, escalate, alert.

Three layers (see also black_swan_feeds.py and black_swan_playbook.py):
  1. Detection: GDELT, GPR, public RSS, OFAC, central-bank press, classified by
     the AI analyst (Retort forced function) when a key is set, otherwise a
     deterministic keyword + GDELT-spike fallback.
  2. Market confirmation: a composite stress/relief score from VIX, the VIX
     term structure, gold, JPY/CHF, Treasuries, oil, credit spreads, overseas
     moves and the SPY gap. A flag becomes ACTIONABLE only when news severity
     and market confirmation agree.
  3. Playbooks: a NORMAL / WATCH / SHOCK / RECOVERY state machine with
     hysteresis. Paper-only; see black_swan_playbook.py.

Polling: `python black_swan.py poll`. Honors TRADING_BLACKSWAN=off.
Never places broker orders or previews.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically

ET = ZoneInfo("America/New_York")
OUT_DIR = DATA_DIR / "Research" / "black_swan"
STATE_PATH = OUT_DIR / "state.json"
EVENTS_PATH = OUT_DIR / "events.jsonl"
HEADLINES_PATH = OUT_DIR / "headlines.jsonl"
THRESHOLDS_PATH = OUT_DIR / "thresholds.json"
CHAT_DB = DATA_DIR / "chat_log.sqlite3"

EVENT_TYPES = (
    "war", "sanctions", "terror", "pandemic", "cyber", "disaster", "coup",
    "sovereign_default", "financial_contagion", "trade_war", "peace_deal",
    "ceasefire", "stimulus", "other",
)
DIRECTIONS = ("negative", "positive", "mixed", "unclear")
STATES = ("NORMAL", "WATCH", "SHOCK", "RECOVERY")

# Keyword maps: (pattern, event_type, direction, severity). First reasonably
# specific match wins per headline; severity is the max across matches.
# Patterns are searched in title + summary (case-insensitive).
KEYWORD_RULES = (
    # Positive / relief first so "ceasefire" is not swallowed by "war".
    (r"\bcease[- ]?fire\b|\btruce\b|\bhumanitarian pause\b", "ceasefire", "positive", 4),
    (r"\bpeace deal\b|\bpeace agreement\b|\barmistice\b|\bde-escalat", "peace_deal", "positive", 4),
    (r"\bhostages? (freed|released)\b|\bprisoner exchange\b", "peace_deal", "positive", 3),
    (r"\bemergency (rate )?cut\b|\bunscheduled cut\b|\bstimulus (bill|package|check)", "stimulus", "positive", 4),
    (r"\bbailout\b|\brescue package\b|\bbank holiday lifted\b|\bfed swap lines?\b", "stimulus", "positive", 4),
    (r"\bsanctions (lifted|eased|relief)\b|\bembargo lifted\b", "sanctions", "positive", 3),
    # Negative shocks.
    (r"\binvasion\b|\binvades?\b|\bdeclares war\b|\bstate of war\b|\bground offensive\b", "war", "negative", 5),
    (r"\bnuclear (strike|weapon|launch|attack|detonat)", "war", "negative", 5),
    (r"\bmissile(s)? (strike|attack|barrage)\b|\bair strikes?\b|\bbombard", "war", "negative", 4),
    (r"\bwar (breaks out|erupts|begins)\b|\barmed conflict\b|\bescalat(es|ion) to war\b", "war", "negative", 5),
    (r"\b(troops|forces) cross(ed)? the border\b|\bannex(?:ed|ation)\b", "war", "negative", 4),
    (r"\bsanctions?\b|\bembargo\b|\bexport control", "sanctions", "negative", 3),
    (r"\bofac\b|\bsdn list\b|\bdesignat(ed|es|ion) .*sanction", "sanctions", "negative", 3),
    (r"\bterror(ist)? (attack|bombing|shooting|massacre)\b|\bsuicide bomb", "terror", "negative", 5),
    (r"\bhostage(s)? (taken|crisis)\b|\bmass shooting\b|\bcar bomb\b", "terror", "negative", 4),
    (r"\b(who|who has)?\s*(declares|has declared) (a )?(pheic|phec|public health emergency)", "pandemic", "negative", 5),
    (r"\bglobal pandemic (declared|begins|has begun)\b|\bnovel (corona)?virus outbreak\b", "pandemic", "negative", 5),
    (r"\bpublic health emergency of international concern\b|\bhuman[- ]to[- ]human", "pandemic", "negative", 4),
    (r"\b(new|fresh|emerging) outbreak\b|\bcity[- ]wide lockdown\b", "pandemic", "negative", 3),
    (r"\bcyber[ -]?(attack|war|warfare|offensive)\b|\bransomware\b|\bcritical infrastructure (hack|breach)", "cyber", "negative", 4),
    (r"\bdata breach\b|\bhack(ed|ers)\b", "cyber", "negative", 2),
    (r"\bearthquake\b|\btsunami\b|\bhurricane\b|\btyphoon\b|\bcyclone\b|\bwildfire\b|\bfukushima\b", "disaster", "negative", 3),
    (r"\bcoup (d['’]etat|attempt)?\b|\bmilitary junta\b|\boverthrow(s|n)? the government\b", "coup", "negative", 4),
    (r"\bsovereign default\b|\bdebt moratorium\b|\bmissed (a )?coupon\b|\bimf bailout\b", "sovereign_default", "negative", 5),
    (r"\bbank (run|failure|collapse|holiday)\b|\breceivership\b|\bsvb\b|\bcredit suisse\b|\bcontagion\b",
     "financial_contagion", "negative", 5),
    (r"\bsystemic (risk|crisis)\b|\bfunding freeze\b|\bliquidity crisis\b|\brepo shock\b",
     "financial_contagion", "negative", 4),
    (r"\breciprocal tariffs?\b|\btrade war\b|\bsection 301\b|\bimport tariff", "trade_war", "negative", 4),
    (r"\btariffs?\b|\blevies on (chinese|eu|european|mexican|canadian) (goods|imports)", "trade_war", "negative", 3),
    (r"\bshooting down\b|\bfighter jet\b|\bnaval clash\b|\bskirmish\b", "war", "negative", 3),
)

REGION_RULES = (
    (r"\bukraine\b|\bkyiv\b|\bcrimea\b", "Ukraine"),
    (r"\brussia\b|\bmoscow\b|\bkremlin\b", "Russia"),
    (r"\bisrael\b|\bgaza\b|\bhamas\b|\bhezbollah\b|\blebanon\b|\bwest bank\b", "Israel-Palestine"),
    (r"\biran\b|\btehran\b|\birgc\b", "Iran"),
    (r"\btaiwan\b|\btaipei\b|\bstrait\b", "Taiwan"),
    (r"\bchina\b|\bbeijing\b|\bprc\b", "China"),
    (r"\bnorth korea\b|\bdprk\b|\bpyongyang\b", "North Korea"),
    (r"\bunited states\b|\b\bU\.?S\.?\b|\bwashington\b|\bwhite house\b", "United States"),
    (r"\beuropean union\b|\beurope\b|\bbrussels\b|\bnato\b", "Europe"),
    (r"\bjapan\b|\btokyo\b|\byen\b", "Japan"),
    (r"\bturkey\b|\bankara\b|\bsyria\b|\bdamascus\b", "Middle East"),
    (r"\bsaudi\b|\bgulf\b|\bhouthi\b|\byemen\b|\bred sea\b", "Gulf / Red Sea"),
    (r"\bvenezuela\b|\bargentina\b|\bbrazil\b", "Latin America"),
    (r"\bpakistan\b|\bindia\b|\bkashmir\b", "South Asia"),
)

ASSET_RULES = (
    (r"\boil\b|\bbrent\b|\bwti\b|\bcrude\b|\benergy\b", "USO", "energy"),
    (r"\bgold\b|\bbullion\b|\bsafe haven\b", "GLD", "commodities"),
    (r"\byen\b|\bboj\b|\bcarry trade\b", "FXY", "fx"),
    (r"\bdollar\b|\bdxy\b|\bfed\b|\brates?\b|\btreasury\b", "UUP", "fx"),
    (r"\bbank(s|ing)?\b|\bcredit spread\b|\bhigh yield\b", "XLF", "financials"),
    (r"\bsemiconductor|\bchip\b|\btaiwan\b", "SMH", "technology"),
    (r"\bairline|\btravel|\btouris", "XLY", "consumer"),
    (r"\bdefense\b|\bweapons\b|\barms\b", "ITA", "industrials"),
)

# Default thresholds (overwritten by event-study calibration when present).
DEFAULT_THRESHOLDS = {
    "watchNews": 2,
    "shockNews": 4,
    "watchStress": 0.35,
    "shockStress": 0.55,
    "reliefScore": 0.35,
    "hysteresis": 0.12,
    "shockMinSessions": 1,
    "shockMaxSessions": 10,
    "recoverySessions": 5,
    "gprZWatch": 2.0,
    "gprZShock": 3.0,
    "gdeltZWatch": 2.0,
    "gdeltZShock": 3.0,
    "vixLevelWatch": 20.0,
    "vixLevelShock": 28.0,
    "vixChgShock": 0.20,
    "vixPtsShock": 4.0,
    "spyGapShock": -0.015,
    "calibrated": False,
    "calibratedAt": None,
}

CLASSIFY_SCHEMA = {
    "name": "classifyBlackSwanHeadline",
    "description": "Classify a news headline as a geopolitical or financial black-swan event.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "event_type": {"type": "string", "enum": list(EVENT_TYPES)},
            "direction": {"type": "string", "enum": list(DIRECTIONS)},
            "severity": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "affected_regions": {"type": "array", "items": {"type": "string"}},
            "affected_assets": {"type": "array", "items": {"type": "string"}},
            "affected_sectors": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": ["event_type", "direction", "severity", "affected_regions",
                     "affected_assets", "affected_sectors", "rationale"],
    },
}


def data_dir():
    """Resolve the data root at call time so tests can set PRICES_DATA_DIR
    after fetch_and_model (which freezes DATA_DIR at import) is already loaded."""
    env = os.environ.get("PRICES_DATA_DIR")
    return Path(env) if env else DATA_DIR


def out_dir(root=None):
    path = Path(root) if root else data_dir() / "Research" / "black_swan"
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_et():
    return datetime.now(ET)


def iso_et(dt=None):
    dt = dt or now_et()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    return dt.isoformat(timespec="seconds")


def disabled(env=None):
    env = env or os.environ
    return str(env.get("TRADING_BLACKSWAN") or "").strip().lower() in {"off", "0", "false", "no"}


def load_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps(payload, indent=2, default=str).encode("utf-8"))


def load_thresholds(path=None):
    doc = load_json(path or (out_dir() / "thresholds.json"), {}) or {}
    out = dict(DEFAULT_THRESHOLDS)
    out.update({k: v for k, v in doc.items() if k in out or k in {"calibrated", "calibratedAt", "notes", "falsePositiveRate"}})
    return out


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path, last=200):
    path = Path(path)
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[-last:]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


# ---------------------------------------------------------------------------
# Classification (keyword fallback + optional Retort)
# ---------------------------------------------------------------------------
def _keyword_matches(text):
    matches = []
    for pat, etype, direction, sev in KEYWORD_RULES:
        if re.search(pat, text or "", re.IGNORECASE):
            matches.append((etype, direction, sev, pat))
    return matches


def classify_headline(headline, method="keyword"):
    title = (headline.get("title") or "") if isinstance(headline, dict) else str(headline)
    summary = (headline.get("summary") or "") if isinstance(headline, dict) else ""
    text = f"{title} {summary}"
    title_matches = _keyword_matches(title)
    matches = title_matches or [(e, d, min(sev, 3), p) for e, d, sev, p in _keyword_matches(text)]
    if not matches:
        return {
            "event_type": "other",
            "direction": "unclear",
            "severity": 1,
            "affected_regions": _regions(text),
            "affected_assets": [a for a, _ in _assets(text)],
            "affected_sectors": list({s for _, s in _assets(text)}),
            "rationale": "No black-swan keyword matched.",
            "method": method,
            "relevant": False,
        }
    # Highest severity wins; mixed if both directions at that severity.
    top = max(m[2] for m in matches)
    top_m = [m for m in matches if m[2] == top]
    types = list(dict.fromkeys(m[0] for m in top_m))
    dirs = list(dict.fromkeys(m[1] for m in top_m))
    direction = dirs[0] if len(dirs) == 1 else "mixed"
    return {
        "event_type": types[0],
        "direction": direction,
        "severity": int(top),
        "affected_regions": _regions(text),
        "affected_assets": [a for a, _ in _assets(text)],
        "affected_sectors": list({s for _, s in _assets(text)}),
        "rationale": f"Keyword match: {', '.join(sorted({m[0] for m in top_m}))} (severity {top}).",
        "method": method,
        "relevant": top >= 2,
        "allTypes": types,
    }


def _regions(text):
    found = []
    for pat, name in REGION_RULES:
        if re.search(pat, text, re.IGNORECASE) and name not in found:
            found.append(name)
    return found


def _assets(text):
    found = []
    for pat, asset, sector in ASSET_RULES:
        if re.search(pat, text, re.IGNORECASE) and (asset, sector) not in found:
            found.append((asset, sector))
    return found



def headline_is_recent(headline, hours=72, now=None):
    """Keep headlines from the last hours hours. WHO's RSS is a deep archive; drop stale or undated WHO items."""
    from datetime import datetime, timezone, timedelta
    now = now or datetime.now(timezone.utc)
    pub = None
    raw = (headline or {}).get("published") or (headline or {}).get("seendate")
    if raw:
        raw = str(raw).strip()
        try:
            pub = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            try:
                pub = datetime.strptime(raw[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pub = None
        if pub is not None and pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
    if pub is None:
        return (headline or {}).get("feed") not in {"who_news"}
    return pub >= now - timedelta(hours=hours)


def classify_headlines(headlines, gpr=None, gdelt_spike=None, ai_fn=None):
    """Classify a batch. Uses ai_fn(headline)->dict when provided; otherwise keywords."""
    headlines = [h for h in headlines if headline_is_recent(h)]
    classified = []
    for h in headlines:
        if ai_fn is not None:
            try:
                rec = dict(ai_fn(h) or {})
                rec.setdefault("method", "retort")
                rec.setdefault("event_type", "other")
                rec.setdefault("direction", "unclear")
                rec.setdefault("severity", 1)
                rec["relevant"] = int(rec.get("severity") or 1) >= 2 and rec.get("event_type") != "other"
            except Exception:  # noqa: BLE001 — never fail the poll
                rec = classify_headline(h)
        else:
            rec = classify_headline(h)
        item = dict(h)
        item["classification"] = rec
        classified.append(item)

    relevant = [c for c in classified if (c.get("classification") or {}).get("relevant")]
    # News score: max severity among relevant headlines, plus GDELT/GPR lifts.
    max_sev = max((int(c["classification"]["severity"]) for c in relevant), default=0)
    direction = _majority_direction(relevant)
    event_type = _majority_type(relevant)
    lift_notes = []
    if gdelt_spike and gdelt_spike.get("shockSpike"):
        max_sev = max(max_sev, 4)
        lift_notes.append(f"GDELT volume z={gdelt_spike.get('z'):.2f}" if gdelt_spike.get("z") is not None else "GDELT shock spike")
    elif gdelt_spike and gdelt_spike.get("spike"):
        max_sev = max(max_sev, 3)
        lift_notes.append(f"GDELT volume z={gdelt_spike.get('z'):.2f}" if gdelt_spike.get("z") is not None else "GDELT spike")
    if gpr and gpr.get("extreme"):
        max_sev = max(max_sev, 4)
        lift_notes.append(f"GPR z={gpr.get('z'):.2f}")
    elif gpr and gpr.get("elevated"):
        max_sev = max(max_sev, 3)
        lift_notes.append(f"GPR z={gpr.get('z'):.2f}")
    max_sev = int(min(5, max_sev))
    return {
        "items": classified,
        "relevant": relevant,
        "newsSeverity": max_sev,
        "direction": direction,
        "eventType": event_type,
        "nRelevant": len(relevant),
        "liftNotes": lift_notes,
        "regions": list(dict.fromkeys(r for c in relevant for r in (c["classification"].get("affected_regions") or []))),
        "assets": list(dict.fromkeys(a for c in relevant for a in (c["classification"].get("affected_assets") or []))),
        "sectors": list(dict.fromkeys(s for c in relevant for s in (c["classification"].get("affected_sectors") or []))),
        "method": "retort" if ai_fn is not None else "keyword",
        "schema": CLASSIFY_SCHEMA["name"],
    }


def _majority_direction(relevant):
    if not relevant:
        return "unclear"
    counts = {}
    for c in relevant:
        d = c["classification"].get("direction") or "unclear"
        w = int(c["classification"].get("severity") or 1)
        counts[d] = counts.get(d, 0) + w
    # Ignore unclear in the vote when something else exists.
    voted = {k: v for k, v in counts.items() if k != "unclear"} or counts
    top = max(voted.values())
    leaders = [k for k, v in voted.items() if v == top]
    if len(leaders) == 1:
        return leaders[0]
    if "positive" in leaders and "negative" in leaders:
        return "mixed"
    return leaders[0]


def _majority_type(relevant):
    if not relevant:
        return "other"
    counts = {}
    for c in relevant:
        t = c["classification"].get("event_type") or "other"
        w = int(c["classification"].get("severity") or 1)
        counts[t] = counts.get(t, 0) + w
    return max(counts, key=counts.get)


def try_ai_classifier():
    """Return a classify function if a Retort helper can be spawned; else None.

    The AI key is not set yet. When it is, `server/ai/blackSwanClassify.js` can
    be called; we never print or log secrets. Failure always falls back to keywords.
    """
    if str(os.environ.get("TRADING_BLACKSWAN_AI") or "on").lower() in {"off", "0", "false"}:
        return None
    # Only attempt if the helper exists and a key-presence flag is on. We do not
    # read the key value here (Node already knows how).
    helper = Path(__file__).resolve().parents[1] / "server" / "ai" / "blackSwanClassify.js"
    if not helper.is_file():
        return None
    if not any(os.environ.get(k) for k in ("OPENAI_API_KEY", "XAI_API_KEY", "GROQ_API_KEY", "TRADING_AI_HAS_KEY")):
        return None

    def _call(headline):
        import subprocess
        payload = json.dumps({"title": headline.get("title"), "summary": headline.get("summary"),
                              "schema": CLASSIFY_SCHEMA}, default=str).encode("utf-8")
        proc = subprocess.run(
            ["node", str(helper)], input=payload, capture_output=True, timeout=45, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError("classifier helper failed")
        out = json.loads(proc.stdout.decode("utf-8") or "{}")
        if not out.get("ok"):
            raise RuntimeError(out.get("error") or "classifier failed")
        return out.get("classification") or out.get("result")
    return _call


# ---------------------------------------------------------------------------
# Market confirmation
# ---------------------------------------------------------------------------
CONFIRM_TICKERS = ("SPY", "GLD", "TLT", "IEF", "USO", "HYG", "LQD", "FXY", "EWJ", "EWG", "EFA")


def _pct_change(series, lag=1):
    if series is None or len(series) <= lag:
        return None
    a, b = series.iloc[-1 - lag], series.iloc[-1]
    if pd.isna(a) or pd.isna(b) or a == 0:
        return None
    return float(b / a - 1.0)


def _z(series, window=60):
    if series is None or len(series) < 15:
        return None
    s = series.dropna()
    if len(s) < 15:
        return None
    w = s.iloc[-window:] if len(s) >= 20 else s
    sd = float(w.std())
    if sd < 1e-12:
        return 0.0
    return float((w.iloc[-1] - w.mean()) / sd)


def _last(series):
    if series is None or series.empty:
        return None
    v = series.dropna()
    return float(v.iloc[-1]) if len(v) else None


def load_confirm_inputs(asof=None, panel=None, vix=None, macro=None, fx=None):
    """Gather causal market series as of `asof` (inclusive). Missing pieces are omitted."""
    asof = pd.Timestamp(asof) if asof is not None else None
    out = {"asof": str(asof.date()) if asof is not None else None, "sources": {}}

    if vix is None:
        try:
            from vix_data import load as load_vix, features as vix_features
            vix = vix_features(load_vix())
            out["sources"]["vix"] = "vix_daily.csv"
        except Exception as exc:  # noqa: BLE001
            vix = None
            out["sources"]["vix"] = f"unavailable: {exc}"
    if vix is not None and asof is not None:
        vix = vix.loc[:asof]
    out["vix"] = vix

    if panel is None:
        try:
            from market_data import load_panel
            start = None
            if asof is not None:
                start = (asof - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
            end = asof.strftime("%Y-%m-%d") if asof is not None else None
            panel = load_panel(tickers=list(CONFIRM_TICKERS), start=start, end=end)
            out["sources"]["panel"] = "prices.sqlite3"
        except Exception as exc:  # noqa: BLE001
            panel = None
            out["sources"]["panel"] = f"unavailable: {exc}"
    if panel is not None and asof is not None:
        try:
            panel = panel.upto(asof)
        except Exception:
            pass
    out["panel"] = panel

    if macro is None:
        path = DATA_DIR / "Macro" / "macro_daily.csv"
        if path.is_file():
            try:
                macro = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
                out["sources"]["macro"] = "macro_daily.csv"
            except Exception as exc:  # noqa: BLE001
                out["sources"]["macro"] = f"unavailable: {exc}"
                macro = None
    if macro is not None and asof is not None:
        macro = macro.loc[:asof]
    out["macro"] = macro

    if fx is None:
        fx = {}
        fx_dir = DATA_DIR / "Macro" / "fx"
        for pair in ("USDJPY", "USDCHF"):
            p = fx_dir / f"{pair}.csv"
            if p.is_file():
                try:
                    frame = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
                    col = "close" if "close" in frame.columns else frame.columns[0]
                    fx[pair] = frame[col].astype(float)
                    out["sources"][pair] = str(p.name)
                except Exception:
                    continue
    if fx and asof is not None:
        fx = {k: v.loc[:asof] for k, v in fx.items()}
    out["fx"] = fx or {}
    return out


def confirmation_score(inputs=None, thresholds=None, asof=None, **kwargs):
    """Composite stress / relief score in [0, 1]. Causal: uses only data <= asof."""
    th = thresholds or load_thresholds()
    inputs = inputs or load_confirm_inputs(asof=asof, **kwargs)
    parts = []
    notes = []

    vix = inputs.get("vix")
    vix_level = vix_chg = vix_ratio = vix_z = vix_pct = peak20 = None
    if vix is not None and len(vix):
        row = vix.iloc[-1]
        vix_level = _num(row.get("vix") if hasattr(row, "get") else row["vix"] if "vix" in vix.columns else None)
        vix_ratio = _num(row.get("ratio") if hasattr(row, "get") else row["ratio"] if "ratio" in vix.columns else None)
        vix_z = _num(row.get("z") if hasattr(row, "get") else row["z"] if "z" in vix.columns else None)
        vix_pct = _num(row.get("pct") if hasattr(row, "get") else row["pct"] if "pct" in vix.columns else None)
        peak20 = _num(row.get("peak20") if hasattr(row, "get") else row["peak20"] if "peak20" in vix.columns else None)
        if "vix" in vix.columns and len(vix) >= 2:
            prev = float(vix["vix"].iloc[-2]) if pd.notna(vix["vix"].iloc[-2]) else None
            if prev and vix_level is not None:
                vix_chg = vix_level / prev - 1.0
        # Stress from level, change, backwardation.
        if vix_level is not None:
            parts.append(("vix_level", _clamp((vix_level - 15.0) / 25.0)))
        if vix_chg is not None:
            parts.append(("vix_chg", _clamp(vix_chg / max(th["vixChgShock"], 0.05))))
        if vix_ratio is not None:
            # <1 contango (calm), >1 backwardation (stress)
            parts.append(("vix_term", _clamp((vix_ratio - 0.85) / 0.30)))
        if vix_z is not None:
            parts.append(("vix_z", _clamp(vix_z / 2.5)))

    panel = inputs.get("panel")
    spy_gap = gold_ret = tlt_ret = oil_ret = hyg_lqd = overseas = None
    close = getattr(panel, "close", None) if panel is not None else None
    open_ = getattr(panel, "open", None) if panel is not None else None
    if close is not None:
        if "SPY" in close.columns:
            if open_ is not None and "SPY" in open_.columns and len(close) >= 2:
                prev_c = close["SPY"].iloc[-2]
                o = open_["SPY"].iloc[-1]
                if pd.notna(prev_c) and pd.notna(o) and prev_c:
                    spy_gap = float(o / prev_c - 1.0)
            spy_ret = _pct_change(close["SPY"])
            if spy_gap is None:
                spy_gap = spy_ret
            if spy_gap is not None:
                # Negative gap is stress.
                parts.append(("spy_gap", _clamp(-spy_gap / max(abs(th["spyGapShock"]), 0.005))))
        if "GLD" in close.columns:
            gold_ret = _pct_change(close["GLD"])
            if gold_ret is not None:
                parts.append(("gold", _clamp(gold_ret / 0.015)))  # gold up = haven bid = stress
        if "TLT" in close.columns:
            tlt_ret = _pct_change(close["TLT"])
            if tlt_ret is not None:
                parts.append(("treasuries", _clamp(tlt_ret / 0.015)))
        elif "IEF" in close.columns:
            tlt_ret = _pct_change(close["IEF"])
            if tlt_ret is not None:
                parts.append(("treasuries", _clamp(tlt_ret / 0.010)))
        if "USO" in close.columns:
            oil_ret = _pct_change(close["USO"])
            if oil_ret is not None:
                # Large absolute oil move confirms an event; sign is informational.
                parts.append(("oil_abs", _clamp(abs(oil_ret) / 0.03)))
        if "HYG" in close.columns and "LQD" in close.columns:
            hyg = _pct_change(close["HYG"])
            lqd = _pct_change(close["LQD"])
            if hyg is not None and lqd is not None:
                hyg_lqd = hyg - lqd
                parts.append(("credit_etf", _clamp(-hyg_lqd / 0.01)))
        # Overnight overseas: last change in EWJ / EFA (US-listed, so this is
        # the prior session unless we have a same-morning index cache).
        for t in ("EWJ", "EWG", "EFA"):
            if close is not None and t in close.columns:
                overseas = _pct_change(close[t])
                if overseas is not None:
                    parts.append((f"overseas_{t}", _clamp(-overseas / 0.015)))
                    break

    macro = inputs.get("macro")
    hy_oas_chg = None
    if macro is not None and "hy_oas" in getattr(macro, "columns", []):
        s = macro["hy_oas"].dropna()
        if len(s) >= 6:
            hy_oas_chg = float(s.iloc[-1] - s.iloc[-6])  # ~1 week, in OAS percent
            parts.append(("hy_oas", _clamp(hy_oas_chg / 0.40)))

    fx = inputs.get("fx") or {}
    jpy_ret = chf_ret = None
    if "USDJPY" in fx:
        jpy_ret = _pct_change(fx["USDJPY"])
        if jpy_ret is not None:
            # USDJPY down = yen stronger = stress.
            parts.append(("yen", _clamp(-jpy_ret / 0.012)))
    if "USDCHF" in fx:
        chf_ret = _pct_change(fx["USDCHF"])
        if chf_ret is not None:
            parts.append(("chf", _clamp(-chf_ret / 0.010)))

    stress = float(np.mean([s for _, s in parts])) if parts else 0.0
    relief = float(np.mean([max(0.0, 1.0 - s) for _, s in parts])) if parts else 0.0
    # VIX rollover from a spike: peak20 exists and VIX has dropped 10%+ off that peak
    # while still above a calm level — used for SVIX / re-risk.
    vix_rollover = False
    if vix_level is not None and peak20 and peak20 >= th["vixLevelShock"] and vix_level <= peak20 * 0.90:
        vix_rollover = True
        notes.append("VIX rolling over from a spike")

    if vix_level and vix_level >= th["vixLevelShock"]:
        notes.append(f"VIX {vix_level:.1f}")
    if vix_chg and vix_chg >= th["vixChgShock"]:
        notes.append(f"VIX {vix_chg:.0%} day")
    if spy_gap is not None and spy_gap <= th["spyGapShock"]:
        notes.append(f"SPY gap {spy_gap:.2%}")
    if hy_oas_chg and hy_oas_chg >= 0.25:
        notes.append(f"HY OAS {hy_oas_chg:+.2f}pp / week")

    return {
        "stress": round(stress, 4),
        "relief": round(relief, 4),
        "nParts": len(parts),
        "parts": {k: round(v, 4) for k, v in parts},
        "vix": vix_level, "vixChg": vix_chg, "vixRatio": vix_ratio, "vixZ": vix_z,
        "vixPercentile": vix_pct, "vixPeak20": peak20, "vixRollover": vix_rollover,
        "spyGap": spy_gap, "goldRet": gold_ret, "tltRet": tlt_ret, "oilRet": oil_ret,
        "hygVsLqd": hyg_lqd, "hyOasChg5": hy_oas_chg, "jpyRet": jpy_ret, "chfRet": chf_ret,
        "overseasRet": overseas, "notes": notes,
        "asof": inputs.get("asof"),
        "sources": inputs.get("sources") or {},
        "missing": [k for k, v in (("vix", vix_level), ("spy", spy_gap), ("gold", gold_ret),
                                   ("credit", hy_oas_chg or hyg_lqd), ("yen", jpy_ret)) if v is None],
    }


def _clamp(x, lo=0.0, hi=1.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(lo, min(hi, x))


def _num(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def judge(news, confirm, thresholds=None):
    """Combine news + confirmation into a flag: CLEAR / WATCH / ACTIONABLE."""
    th = thresholds or load_thresholds()
    sev = int(news.get("newsSeverity") or 0)
    direction = news.get("direction") or "unclear"
    stress, relief = float(confirm.get("stress") or 0), float(confirm.get("relief") or 0)
    vix_roll = bool(confirm.get("vixRollover"))

    actionable_neg = (
        sev >= th["shockNews"] and stress >= th["shockStress"] and direction in {"negative", "mixed"}
    )
    actionable_pos = (
        (sev >= th["shockNews"] and direction == "positive" and (relief >= th["reliefScore"] or vix_roll or stress < th["watchStress"]))
        or (vix_roll and direction == "positive" and sev >= th["watchNews"])
    )
    watch = (sev >= th["watchNews"] or stress >= th["watchStress"] or (news.get("gpr") or {}).get("elevated"))
    if actionable_neg:
        flag, agreed = "ACTIONABLE", "negative"
    elif actionable_pos:
        flag, agreed = "ACTIONABLE", "positive"
    elif watch:
        flag, agreed = "WATCH", direction if sev or stress else "unclear"
    else:
        flag, agreed = "CLEAR", "unclear"
    return {
        "flag": flag,
        "agreedDirection": agreed,
        "newsSeverity": sev,
        "stress": stress,
        "relief": relief,
        "reason": _judge_reason(flag, agreed, sev, stress, relief, confirm, news),
    }


def _judge_reason(flag, agreed, sev, stress, relief, confirm, news):
    bits = [f"news severity {sev}/5 ({news.get('eventType') or 'n/a'}, {news.get('direction') or 'unclear'})",
            f"market stress {stress:.2f}", f"relief {relief:.2f}"]
    if confirm.get("notes"):
        bits.append("; ".join(confirm["notes"][:3]))
    if news.get("liftNotes"):
        bits.append("; ".join(news["liftNotes"]))
    return f"{flag} {agreed}: " + " — ".join(bits)


# ---------------------------------------------------------------------------
# State machine (NORMAL / WATCH / SHOCK / RECOVERY) with hysteresis
# ---------------------------------------------------------------------------
def default_state():
    return {
        "state": "NORMAL",
        "direction": "unclear",
        "since": None,
        "sessionsInState": 0,
        "shockPeakStress": None,
        "lastFlag": "CLEAR",
        "lastEventType": "other",
        "updatedAt": None,
        "headline": "No elevated geopolitical or black-swan flag.",
        "policy": None,
    }


def load_state(path=None):
    return load_json(path or (out_dir() / "state.json"), default_state()) or default_state()


def step_state(prev, judgment, confirm, news, thresholds=None, asof=None):
    th = thresholds or load_thresholds()
    prev = dict(prev or default_state())
    state = prev.get("state") or "NORMAL"
    direction = prev.get("direction") or "unclear"
    sessions = int(prev.get("sessionsInState") or 0) + 1
    flag = judgment.get("flag") or "CLEAR"
    agreed = judgment.get("agreedDirection") or "unclear"
    stress = float(confirm.get("stress") or 0)
    hyst = float(th["hysteresis"])
    next_state, next_dir = state, direction

    if state == "NORMAL":
        if flag == "ACTIONABLE" and agreed == "negative":
            next_state, next_dir, sessions = "SHOCK", "negative", 1
        elif flag == "ACTIONABLE" and agreed == "positive":
            next_state, next_dir, sessions = "RECOVERY", "positive", 1
        elif flag == "WATCH":
            next_state, next_dir, sessions = "WATCH", agreed, 1
    elif state == "WATCH":
        if flag == "ACTIONABLE" and agreed == "negative":
            next_state, next_dir, sessions = "SHOCK", "negative", 1
        elif flag == "ACTIONABLE" and agreed == "positive":
            next_state, next_dir, sessions = "RECOVERY", "positive", 1
        elif flag == "CLEAR" and stress < (th["watchStress"] - hyst) and int(news.get("newsSeverity") or 0) < th["watchNews"]:
            next_state, next_dir, sessions = "NORMAL", "unclear", 1
        else:
            next_dir = agreed if agreed != "unclear" else direction
    elif state == "SHOCK":
        cooled = stress < (th["shockStress"] - hyst)
        expired = sessions >= int(th["shockMaxSessions"])
        rollover = bool(confirm.get("vixRollover"))
        if (cooled and sessions >= int(th["shockMinSessions"])) or expired or rollover:
            next_state, next_dir, sessions = "RECOVERY", "negative" if not (flag == "ACTIONABLE" and agreed == "positive") else "positive", 1
        else:
            next_dir = "negative"
            prev["shockPeakStress"] = max(float(prev.get("shockPeakStress") or 0), stress)
    elif state == "RECOVERY":
        if flag == "ACTIONABLE" and agreed == "negative":
            next_state, next_dir, sessions = "SHOCK", "negative", 1
        elif sessions >= int(th["recoverySessions"]) and stress < (th["watchStress"] - hyst) and flag == "CLEAR":
            next_state, next_dir, sessions = "NORMAL", "unclear", 1
        else:
            if agreed == "positive":
                next_dir = "positive"
    else:
        next_state, next_dir, sessions = "NORMAL", "unclear", 1

    from black_swan_playbook import policy_for
    policy = policy_for(next_state, next_dir, confirm, th)
    asof_s = str(asof) if asof is not None else iso_et()
    headline = _state_headline(next_state, next_dir, news, judgment)
    new = {
        "state": next_state,
        "direction": next_dir,
        "since": prev.get("since") if next_state == state else asof_s,
        "sessionsInState": sessions,
        "shockPeakStress": prev.get("shockPeakStress") if next_state in {"SHOCK", "RECOVERY"} else None,
        "lastFlag": flag,
        "lastEventType": news.get("eventType") or prev.get("lastEventType") or "other",
        "updatedAt": iso_et(),
        "asof": asof_s,
        "headline": headline,
        "judgment": judgment,
        "news": {k: news.get(k) for k in ("newsSeverity", "direction", "eventType", "nRelevant", "liftNotes", "regions", "method")},
        "confirm": {k: confirm.get(k) for k in
                    ("stress", "relief", "vix", "vixChg", "vixRatio", "vixRollover", "spyGap", "goldRet",
                     "tltRet", "oilRet", "hyOasChg5", "jpyRet", "notes", "missing", "asof")},
        "policy": policy,
        "transitioned": next_state != state or (next_dir != direction and next_state in {"SHOCK", "RECOVERY"}),
        "prevState": state,
    }
    if new["since"] is None:
        new["since"] = asof_s
    return new


def _state_headline(state, direction, news, judgment):
    etype = (news.get("eventType") or "event").replace("_", " ")
    sev = news.get("newsSeverity")
    if state == "SHOCK":
        return f"Confirmed negative shock ({etype}, severity {sev}/5). Playbook: cut gross, freeze new longs, allow vol/short/bond sleeves."
    if state == "RECOVERY" and direction == "positive":
        return f"Confirmed relief / positive shock ({etype}). Playbook: re-risk; SVIX allowed if VIX is rolling over."
    if state == "RECOVERY":
        return f"Leaving a shock ({etype}). Re-risking with hysteresis; SVIX allowed on a VIX rollover."
    if state == "WATCH":
        return f"Watch: {etype} in the news (severity {sev}/5) or market stress elevated, but not yet confirmed as ACTIONABLE."
    return judgment.get("reason") or "No elevated geopolitical or black-swan flag."


# ---------------------------------------------------------------------------
# Alerts, chat, poll
# ---------------------------------------------------------------------------
def emit_black_swan_alert(state, top_headlines=None):
    try:
        from alerts_emit import emit
    except Exception:
        return False
    sev_map = {"NORMAL": "info", "WATCH": "warning", "SHOCK": "critical", "RECOVERY": "warning"}
    titles = []
    for h in (top_headlines or [])[:3]:
        titles.append((h.get("title") or "")[:160])
    text = state.get("headline") or "Black swan state update."
    if titles:
        text = text + " Headlines: " + " | ".join(titles)
    return emit({
        "type": "black_swan",
        "severity": sev_map.get(state.get("state"), "warning"),
        "title": f"Black swan: {state.get('state')}",
        "text": text[:800],
        "href": "/paper2.html#black-swan",
        "meta": {
            "state": state.get("state"),
            "direction": state.get("direction"),
            "flag": state.get("lastFlag"),
            "eventType": state.get("lastEventType"),
            "stress": (state.get("confirm") or {}).get("stress"),
            "newsSeverity": (state.get("news") or {}).get("newsSeverity"),
        },
    })


CHAT_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, role TEXT NOT NULL, kind TEXT NOT NULL,
  account TEXT, ticker TEXT, side TEXT, qty REAL, price REAL, decision_date TEXT, text TEXT NOT NULL,
  rationale TEXT, verdict TEXT, data TEXT, source_key TEXT UNIQUE, parent_id INTEGER);
CREATE INDEX IF NOT EXISTS chat_by_ticker ON chat_messages (ticker, id);
CREATE INDEX IF NOT EXISTS chat_by_kind ON chat_messages (kind, id);
"""


def log_chat(state, path=None):
    path = Path(path) if path else data_dir() / "chat_log.sqlite3"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(path))
        db.executescript(CHAT_SCHEMA)
        key = f"black_swan:{state.get('state')}:{state.get('since')}:{state.get('updatedAt')}"
        payload = json.dumps({k: state.get(k) for k in ("state", "direction", "lastFlag", "lastEventType",
                                                        "sessionsInState", "policy", "judgment", "confirm", "news")}, default=str)
        db.execute(
            "INSERT OR IGNORE INTO chat_messages (at, role, kind, text, rationale, verdict, data, source_key) "
            "VALUES (?, 'trader', 'black_swan', ?, ?, ?, ?, ?)",
            (state.get("updatedAt") or iso_et(), state.get("headline") or "Black swan update.",
             (state.get("judgment") or {}).get("reason"), state.get("state"), payload, key),
        )
        db.commit()
        db.close()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[black_swan] chat log skipped: {exc}", file=sys.stderr)
        return False


def poll(opener=None, root=None, asof=None, use_cache=True, emit=True, env=None):
    """Run one detection + confirmation + state-machine step. Honors TRADING_BLACKSWAN=off."""
    if disabled(env):
        return {"ok": True, "skipped": True, "reason": "TRADING_BLACKSWAN=off"}
    root = out_dir(root)
    from black_swan_feeds import collect_headlines
    bundle = collect_headlines(opener=opener, root=root, use_cache=use_cache)
    ai_fn = None
    try:
        ai_fn = try_ai_classifier()
    except Exception:
        ai_fn = None
    news = classify_headlines(bundle.get("headlines") or [], gpr=bundle.get("gpr"),
                              gdelt_spike=bundle.get("gdeltSpike"), ai_fn=ai_fn)
    news["gpr"] = bundle.get("gpr") or {}
    confirm = confirmation_score(asof=asof)
    th = load_thresholds()
    judgment = judge(news, confirm, th)
    prev = load_state(root / "state.json")
    state = step_state(prev, judgment, confirm, news, th, asof=asof or iso_et())
    dollar = None
    try:
        from black_swan_dollar import attach_to_poll
        dollar = attach_to_poll(state, asof=asof, emit=emit, root=root)
    except Exception as exc:  # noqa: BLE001 — never fail the geo poll
        dollar = {"ok": False, "error": str(exc), "tradingChange": False}
        state["dollarShock"] = dollar
    save_json(root / "state.json", state)
    top = sorted(news.get("relevant") or [], key=lambda h: int((h.get("classification") or {}).get("severity") or 0), reverse=True)[:8]
    event_row = {
        "at": iso_et(),
        "asof": state.get("asof"),
        "state": state["state"],
        "prevState": state.get("prevState"),
        "transitioned": state.get("transitioned"),
        "flag": judgment["flag"],
        "direction": state["direction"],
        "eventType": news.get("eventType"),
        "newsSeverity": news.get("newsSeverity"),
        "stress": confirm.get("stress"),
        "relief": confirm.get("relief"),
        "headline": state["headline"],
        "headlines": [{"title": h.get("title"), "feed": h.get("feed"), "classification": h.get("classification")} for h in top],
        "feeds": {k: _feed_ok(v) for k, v in (bundle.get("status") or {}).items()},
        "classifyMethod": news.get("method"),
    }
    if state.get("transitioned") or judgment["flag"] == "ACTIONABLE":
        append_jsonl(root / "events.jsonl", event_row)
        if emit:
            emit_black_swan_alert(state, top)
            log_chat(state)
    # Always keep a short rolling snapshot of headlines.
    append_jsonl(root / "headlines.jsonl", {"at": iso_et(), "n": news.get("nRelevant"),
                                            "titles": [h.get("title") for h in top[:5]]})
    save_json(root / "feeds_status.json", bundle.get("status") or {})
    return {
        "ok": True,
        "state": state["state"],
        "direction": state["direction"],
        "flag": judgment["flag"],
        "transitioned": state.get("transitioned"),
        "headline": state["headline"],
        "newsSeverity": news.get("newsSeverity"),
        "stress": confirm.get("stress"),
        "relief": confirm.get("relief"),
        "nHeadlines": (bundle.get("status") or {}).get("nHeadlines"),
        "nRelevant": news.get("nRelevant"),
        "classifyMethod": news.get("method"),
        "feeds": bundle.get("status"),
        "policy": state.get("policy"),
        "path": str(root / "state.json"),
        "dollarShock": dollar,
    }


def _feed_ok(value):
    if isinstance(value, dict):
        return {k: value.get(k) for k in ("ok", "error", "n", "nArticles", "source") if k in value} or value
    if isinstance(value, list):
        return [{"id": x.get("id"), "ok": x.get("ok"), "n": x.get("n"), "error": x.get("error")} for x in value if isinstance(x, dict)]
    return value


def status(root=None):
    root = out_dir(root)
    state = load_state(root / "state.json")
    feeds = load_json(root / "feeds_status.json", {}) or {}
    study = load_json(root / "study.json", {}) or {}
    dollar_study = load_json(root / "dollar_shock.json", {}) or {}
    events = read_jsonl(root / "events.jsonl", last=20)
    return {
        "state": state,
        "feeds": feeds,
        "recentEvents": events[-10:],
        "thresholds": load_thresholds(root / "thresholds.json"),
        "study": {k: study.get(k) for k in ("generatedAt", "nEvents", "honesty", "coverage", "calibratedThresholds") if k in study},
        "dollarShock": state.get("dollarShock"),
        "dollarStudy": {k: dollar_study.get(k) for k in ("generatedAt", "events", "verdict", "honesty", "holdoutStart", "latest") if k in dollar_study} or None,
        "disabled": disabled(),
        "paths": {"state": str(root / "state.json"), "events": str(root / "events.jsonl"), "study": str(root / "study.json"),
                  "dollarStudy": str(root / "dollar_shock.json")},
    }


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["poll", "status", "study", "classify", "dollar"], nargs="?", default="status")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-emit", action="store_true")
    parser.add_argument("--asof")
    parser.add_argument("--text", help="Classify a single headline (classify command).")
    args = parser.parse_args(argv)
    if args.command == "classify":
        rec = classify_headline({"title": args.text or ""})
        print(json.dumps(rec, indent=2))
        return 0
    if args.command == "study":
        from black_swan_study import run_study
        print(json.dumps(run_study(), indent=2, default=str))
        return 0
    if args.command == "dollar":
        from black_swan_dollar import main as dollar_main
        return dollar_main(["study"])
    if args.command == "poll":
        if disabled():
            print(json.dumps({"ok": True, "skipped": True, "reason": "TRADING_BLACKSWAN=off"}))
            return 0
        result = poll(use_cache=not args.no_cache, emit=not args.no_emit, asof=args.asof)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 2
    print(json.dumps(status(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
