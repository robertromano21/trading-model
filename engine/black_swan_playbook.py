"""Config-driven, paper-only playbooks for confirmed black-swan states.

States (hysteresis lives in black_swan.step_state):
  NORMAL   — no change
  WATCH    — alerts only; no portfolio change
  SHOCK    — reduce gross, freeze new long buys, allow VIXY / short / bond sleeves
  RECOVERY — re-risk toward the strategy target; allow SVIX when VIX rolls over

Default mode is `log`: suggested adjustments are recorded but the book is unchanged.
`apply` changes paper targets only (never a live broker order or preview). PDT,
the risk engine and max_gross stay final. Explicit exits and decay: shock
gross-cuts expire after shockMaxSessions and unwind over recoverySessions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fetch_and_model import DATA_DIR

OUT_DIR = DATA_DIR / "Research" / "black_swan"
STATE_PATH = OUT_DIR / "state.json"

# Sleeve tickers the playbook may add or permit (only if those modules exist
# and the names are in the price panel). Never invent live option orders.
BOND_TICKERS = ("TLT", "IEF")
VOL_LONG = "VIXY"       # long-vol, used in SHOCK
VOL_SHORT = "SVIX"      # short-vol, used in RECOVERY after a VIX rollover
SHORT_PROXY = "SH"      # inverse S&P, allowed in SHOCK if present (GB-13 short mode)
DEFAULT_SHOCK_GROSS = 0.50
DEFAULT_WATCH_GROSS = 1.00
DEFAULT_BOND_TILT = 0.08
DEFAULT_VOL_WEIGHT = 0.05
DEFAULT_SHORT_WEIGHT = 0.05


def policy_for(state, direction, confirm=None, thresholds=None):
    confirm = confirm or {}
    rollover = bool(confirm.get("vixRollover"))
    policy = {
        "state": state,
        "direction": direction,
        "scaleGross": 1.0,
        "freezeNewLongs": False,
        "allowVixy": False,
        "allowSvix": False,
        "allowShort": False,
        "allowBonds": False,
        "bondTilt": 0.0,
        "vixyWeight": 0.0,
        "svixWeight": 0.0,
        "shortWeight": 0.0,
        "reRisk": False,
        "reason": "No playbook change.",
    }
    if state == "SHOCK" and direction in {"negative", "mixed", None, "unclear"}:
        policy.update({
            "scaleGross": float((thresholds or {}).get("shockGross", DEFAULT_SHOCK_GROSS) or DEFAULT_SHOCK_GROSS),
            "freezeNewLongs": True,
            "allowVixy": True,
            "allowSvix": False,
            "allowShort": True,
            "allowBonds": True,
            "bondTilt": DEFAULT_BOND_TILT,
            "vixyWeight": DEFAULT_VOL_WEIGHT,
            "shortWeight": DEFAULT_SHORT_WEIGHT,
            "reason": "Confirmed negative shock: cut gross, freeze new longs, allow VIXY / short / bond sleeves.",
        })
    elif state == "RECOVERY":
        policy.update({
            "scaleGross": 1.0,
            "freezeNewLongs": False,
            "allowVixy": False,
            "allowSvix": bool(rollover or direction == "positive"),
            "allowShort": False,
            "allowBonds": False,
            "svixWeight": DEFAULT_VOL_WEIGHT if (rollover or direction == "positive") else 0.0,
            "reRisk": True,
            "reason": ("Post-spike exhaustion: re-risk and allow SVIX."
                       if rollover else "Recovery: restore target weights; no new long freeze."),
        })
    elif state == "WATCH":
        policy.update({
            "scaleGross": float((thresholds or {}).get("watchGross", DEFAULT_WATCH_GROSS) or DEFAULT_WATCH_GROSS),
            "reason": "Watch only: no book change until news and markets agree.",
        })
    return policy


def load_state(path=None):
    path = Path(path) if path else STATE_PATH
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def current_policy(path=None):
    state = load_state(path)
    return state.get("policy") or policy_for(state.get("state") or "NORMAL", state.get("direction") or "unclear")


def _mode(settings):
    return str((settings or {}).get("black_swan") or (settings or {}).get("blackSwan") or "log").lower()


def adjust_paper_targets(target, day, settings, equity=None, state=None, asof=None):
    """Scale / tilt a paper-trader target series. Returns (new_target, actions).

    Mode `off`: no-op. Mode `log`: describe but do not change. Mode `apply`:
    change paper weights only. The risk engine still caps gross afterwards.
    """
    mode = _mode(settings)
    if mode in {"off", "", "none"}:
        return target, []
    state = state if state is not None else load_state()
    policy = (state or {}).get("policy") or policy_for((state or {}).get("state") or "NORMAL",
                                                       (state or {}).get("direction") or "unclear")
    if target is None:
        target = pd.Series(dtype=float)
    new = target.astype(float).copy()
    apply = mode == "apply"
    actions = []

    scale = float(policy.get("scaleGross") or 1.0)
    if scale < 0.999:
        if apply:
            # Do not scale existing short / vol / bond hedges down with the book.
            protect = {t for t in list(BOND_TICKERS) + [VOL_LONG, VOL_SHORT, SHORT_PROXY] if t in new.index}
            longs = [t for t in new.index if t not in protect]
            new[longs] = new[longs] * scale
        actions.append({
            "kind": "black_swan_scale_gross", "apply": apply, "scaleGross": scale,
            "state": policy.get("state"), "reason": policy.get("reason"),
            "ticker": None,
        })

    if policy.get("freezeNewLongs"):
        if apply:
            # Flatten new long *increases* relative to a zero name; we cannot see
            # the current book here, so we only prevent names the strategy just
            # turned on from being larger than they already would be after the
            # scale. The paper trader also consults freeze_new_long() in _make_orders.
            pass
        actions.append({
            "kind": "black_swan_freeze_longs", "apply": apply,
            "state": policy.get("state"), "reason": "Freeze new long buys while the shock playbook is on.",
            "ticker": None,
        })

    if policy.get("allowBonds") and policy.get("bondTilt"):
        tilt = float(policy["bondTilt"])
        # Prefer TLT on a flight-to-quality shock; split if both exist.
        for ticker, share in (("TLT", 0.65), ("IEF", 0.35)):
            delta = tilt * share
            if apply:
                new[ticker] = float(new.get(ticker, 0.0) + delta) if hasattr(new, "get") else float((new[ticker] if ticker in new.index else 0.0) + delta)
            actions.append({
                "kind": "black_swan_bond_tilt", "apply": apply, "ticker": ticker, "delta": delta,
                "state": policy.get("state"), "reason": "Flight-to-quality bond tilt (paper).",
            })

    if policy.get("allowVixy") and policy.get("vixyWeight"):
        w = float(policy["vixyWeight"])
        if apply:
            new[VOL_LONG] = max(float(new.get(VOL_LONG, 0.0) if hasattr(new, "get") else (new[VOL_LONG] if VOL_LONG in new.index else 0.0)), w)
        actions.append({
            "kind": "black_swan_vixy", "apply": apply, "ticker": VOL_LONG, "weight": w,
            "state": policy.get("state"), "reason": "Allow VIXY sleeve during a confirmed negative shock.",
        })

    if policy.get("allowSvix") and policy.get("svixWeight"):
        w = float(policy["svixWeight"])
        if apply:
            new[VOL_SHORT] = max(float(new.get(VOL_SHORT, 0.0) if hasattr(new, "get") else (new[VOL_SHORT] if VOL_SHORT in new.index else 0.0)), w)
            if VOL_LONG in new.index:
                new[VOL_LONG] = 0.0
        actions.append({
            "kind": "black_swan_svix", "apply": apply, "ticker": VOL_SHORT, "weight": w,
            "state": policy.get("state"), "reason": "Allow SVIX after a VIX rollover / positive shock.",
        })

    if policy.get("allowShort") and policy.get("shortWeight"):
        w = float(policy["shortWeight"])
        if apply:
            new[SHORT_PROXY] = max(float(new.get(SHORT_PROXY, 0.0) if hasattr(new, "get") else (new[SHORT_PROXY] if SHORT_PROXY in new.index else 0.0)), w)
        actions.append({
            "kind": "black_swan_short", "apply": apply, "ticker": SHORT_PROXY, "weight": w,
            "state": policy.get("state"), "reason": "Allow a short / downside proxy during a confirmed shock (paper).",
        })

    if apply and float(new.abs().sum() or 0) > 1.05 and float(target.abs().sum() or 0) <= 1.05:
        new = new * (1.0 / float(new.abs().sum()))
    return new, actions


def freeze_new_long(ticker, current_weight, target_weight, settings, state=None):
    """True when apply-mode should block a new or increased long in a shock."""
    if _mode(settings) != "apply":
        return False
    state = state if state is not None else load_state()
    policy = (state or {}).get("policy") or {}
    if not policy.get("freezeNewLongs"):
        return False
    ticker = str(ticker or "").upper()
    if ticker in {VOL_LONG, VOL_SHORT, SHORT_PROXY} or ticker in BOND_TICKERS:
        return False
    try:
        cur = float(current_weight or 0.0)
        tgt = float(target_weight or 0.0)
    except (TypeError, ValueError):
        return False
    return tgt > cur + 1e-9 and tgt > 0


def adjust_daytrade_candidates(candidates, session_date, settings, state=None):
    """Mark ORB candidates to sit out during an applied SHOCK."""
    mode = _mode(settings)
    if mode in {"off", "", "none"}:
        return candidates, []
    state = state if state is not None else load_state()
    policy = (state or {}).get("policy") or {}
    st = (state or {}).get("state") or policy.get("state")
    actions = [{
        "kind": "black_swan_daytrade",
        "apply": mode == "apply" and st == "SHOCK",
        "state": st,
        "reason": "Sit out new day trades during a confirmed shock (spreads and gaps are hostile).",
        "proven": False,
    }]
    if candidates is None or (hasattr(candidates, "empty") and candidates.empty):
        return candidates, actions
    if mode == "apply" and st == "SHOCK":
        out = candidates.copy()
        out["black_swan_skip"] = True
        if "skip_reason" in out.columns or True:
            out["skip_reason"] = out.get("skip_reason", None)
        return out, actions
    return candidates, actions


def sleeve_permissions(state=None):
    """What the VIXY/SVIX/short/bond sleeves (GB-13 / GB-12) may do right now."""
    state = state if state is not None else load_state()
    policy = (state or {}).get("policy") or policy_for((state or {}).get("state") or "NORMAL",
                                                       (state or {}).get("direction") or "unclear")
    return {
        "state": policy.get("state") or (state or {}).get("state") or "NORMAL",
        "allowVixy": bool(policy.get("allowVixy")),
        "allowSvix": bool(policy.get("allowSvix")),
        "allowShort": bool(policy.get("allowShort")),
        "allowBonds": bool(policy.get("allowBonds")),
        "freezeNewLongs": bool(policy.get("freezeNewLongs")),
        "scaleGross": policy.get("scaleGross"),
        "reason": policy.get("reason"),
    }
