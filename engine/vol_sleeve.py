"""Volatility sleeve (primary rule, Robert t567/t568): long VIXY when volatility is low and calm; long SVIX right
after a volatility spike once it starts to fade. 'Low' is defined on the VIX index (level, 252-day percentile,
60-day z-score) and the VIX / VIX3M term-structure ratio, never on VIXY's decaying share price.

ENTRIES
  VIXY  calm:  VIX percentile <= vixy_pct AND VIX <= vixy_level AND VIX/VIX3M <= vixy_ratio (steep contango)
  SVIX  post-spike: a spike in the last spike_lookback sessions (percentile >= spike_pct or VIX/VIX3M >= 1, i.e.
        backwardation) AND contango is back (ratio <= svix_entry_ratio) AND VIX is >= svix_drop below its 20-day
        peak. No SVIX entry within event_buffer sessions before a major macro event (FOMC; add CPI/NFP dates in
        Massive/Options/macro_events.json).
EXITS (every exit reason is recorded and sent to the decision chat by options_paper.py)
  VIXY  take_profit_spike: VIX z >= vixy_tp_z or percentile >= vixy_tp_pct
        backwardation_rollback: the term structure flipped to backwardation (ratio > 1) after entry and has rolled
        back below vixy_rollback_ratio (the spike is fading: take what is left)
        max_hold: held vixy_max_hold sessions;  decay_stop: close <= entry * (1 - vixy_decay_stop)
  SVIX  normalized: percentile <= svix_norm_pct AND ratio <= svix_norm_ratio (vol is back to normal: take profit)
        macro_event: exit at the close before a major macro event
        hard_stop: low <= entry * (1 - svix_hard_stop) (gap-aware: fills at the open if it opens below)
        trailing_stop: close <= peak close since entry * (1 - svix_trail)
        regime_flip: ratio >= svix_flip_ratio (backwardation again) or z >= svix_flip_z
        max_hold: held svix_max_hold sessions
Decisions are made at the close and filled at the next open (hard stops intraday at the stop or the open).
Costs: Schwab model (no commission; SEC + TAF on sells; spread and impact from dollar volume and volatility).

  python vol_sleeve.py walkforward     # writes Massive/Research/vol_sleeve.json (+ equity / trades CSV)
  python vol_sleeve.py signal          # today's decision for the paper runner (no orders anywhere)
"""

import itertools
import json
import math
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from costs import CostModel
from fetch_and_model import DATA_DIR, write_atomically

OUT_DIR = DATA_DIR / "Research"
EVENTS_PATH = DATA_DIR / "Options" / "macro_events.json"
TRADING_DAYS = 252
# FOMC decision dates (Federal Reserve published calendars). Extend via Massive/Options/macro_events.json.
FOMC = ["2024-11-07", "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17",
        "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16",
        "2026-10-28", "2026-12-09"]

DEFAULTS = {
    "vixy_enabled": True, "svix_enabled": True,
    "vixy_weight": 0.05, "svix_weight": 0.10,
    "vixy_pct": 0.20, "vixy_level": 16.0, "vixy_ratio": 0.90,
    "vixy_tp_z": 2.0, "vixy_tp_pct": 0.90, "vixy_rollback_ratio": 0.98, "vixy_max_hold": 40, "vixy_decay_stop": 0.15,
    "spike_lookback": 15, "spike_pct": 0.90, "svix_entry_ratio": 0.97, "svix_drop": 0.15,
    "svix_norm_pct": 0.35, "svix_norm_ratio": 0.90, "svix_hard_stop": 0.10, "svix_trail": 0.08,
    "svix_flip_ratio": 1.0, "svix_flip_z": 2.0, "svix_max_hold": 60, "event_buffer": 1,
    "cooldown": 3, "exits": "full",            # full | time_only (ablation: max-hold exits only)
}
GRID = {"vixy_pct": [0.20, 0.30], "svix_drop": [0.10, 0.20], "svix_trail": [0.08, 0.12], "vixy_max_hold": [20, 40]}


def macro_events():
    days = set(FOMC)
    if EVENTS_PATH.exists():
        try:
            extra = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
            days |= {str(d)[:10] for d in (extra.get("dates", extra) if isinstance(extra, dict) else extra)}
        except Exception:  # noqa: BLE001
            pass
    return sorted(days)


def _sessions_until_event(dates, events):
    ev = pd.DatetimeIndex(pd.to_datetime(events))
    out = pd.Series(999, index=dates)
    for i, d in enumerate(dates):
        nxt = ev[ev > d]
        if len(nxt):
            out.iloc[i] = int(((dates > d) & (dates <= nxt[0])).sum())
    return out


class VolSleeve:
    """State machine shared by the backtest and the paper runner. ``step`` sees one session's VIX features and
    ETF prices (no future rows) and returns the orders to fill at the next open plus exit reasons."""

    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"vol_sleeve: unknown parameters {sorted(unknown)}")
        self.p = {**DEFAULTS, **params}
        self.pos = {}           # ticker -> {"shares", "entry", "entryDate", "held", "peak", "sawBackwardation"}
        self.cooldown = {"VIXY": 0, "SVIX": 0}
        self.spike_age = 999

    def state(self):
        return {"params": self.p, "pos": self.pos, "cooldown": self.cooldown, "spikeAge": self.spike_age}

    @classmethod
    def from_state(cls, s):
        obj = cls(**s.get("params", {}))
        obj.pos, obj.cooldown, obj.spike_age = s.get("pos", {}), s.get("cooldown", obj.cooldown), s.get("spikeAge", 999)
        return obj

    def step(self, asof, f, close, sessions_to_event=999):
        """f: dict of VIX features for asof (vix, ratio, pct, z, peak20). close: {"VIXY": px, "SVIX": px}.
        Returns list of {"action": "buy"|"sell", "ticker", "weight", "reason", "detail"} for the next open."""
        p, out = self.p, []
        vix, ratio, pct, z, peak = (f.get(k) for k in ("vix", "ratio", "pct", "z", "peak20"))
        if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in (vix, ratio, pct)):
            return out
        z = 0.0 if z is None or (isinstance(z, float) and math.isnan(z)) else z
        spike_now = pct >= p["spike_pct"] or ratio >= 1.0
        self.spike_age = 0 if spike_now else self.spike_age + 1
        for t in self.cooldown:
            self.cooldown[t] = max(0, self.cooldown[t] - 1)
        detail = {"vix": round(vix, 2), "ratio": round(ratio, 3), "pct": round(pct, 3), "z": round(z, 2)}
        # ---------------- exits
        for t, pos in list(self.pos.items()):
            pos["held"] += 1
            px = close.get(t)
            if px is None:
                continue
            pos["peak"] = max(pos["peak"], px)
            if ratio > 1.0:
                pos["sawBackwardation"] = True
            reason = None
            if t == "VIXY":
                if p["exits"] == "full":
                    if z >= p["vixy_tp_z"] or pct >= p["vixy_tp_pct"]:
                        reason = "take_profit_spike"
                    elif pos.get("sawBackwardation") and ratio < p["vixy_rollback_ratio"]:
                        reason = "backwardation_rollback"
                    elif px <= pos["entry"] * (1 - p["vixy_decay_stop"]):
                        reason = "decay_stop"
                if not reason and pos["held"] >= p["vixy_max_hold"]:
                    reason = "max_hold"
            else:
                if p["exits"] == "full":
                    if pct <= p["svix_norm_pct"] and ratio <= p["svix_norm_ratio"]:
                        reason = "normalized"
                    elif ratio >= p["svix_flip_ratio"] or z >= p["svix_flip_z"]:
                        reason = "regime_flip"
                    elif sessions_to_event <= p["event_buffer"]:
                        reason = "macro_event"
                    elif px <= pos["peak"] * (1 - p["svix_trail"]):
                        reason = "trailing_stop"
                if not reason and pos["held"] >= p["svix_max_hold"]:
                    reason = "max_hold"
            if reason:
                out.append({"action": "sell", "ticker": t, "weight": 0.0, "reason": reason, "detail": detail})
        exiting = {o["ticker"] for o in out}
        # ---------------- entries
        if p["vixy_enabled"] and "VIXY" not in self.pos and "VIXY" not in exiting and not self.cooldown["VIXY"]:
            if pct <= p["vixy_pct"] and vix <= p["vixy_level"] and ratio <= p["vixy_ratio"]:
                out.append({"action": "buy", "ticker": "VIXY", "weight": p["vixy_weight"], "reason": "calm_low_vol",
                            "detail": detail})
        if p["svix_enabled"] and "SVIX" not in self.pos and "SVIX" not in exiting and not self.cooldown["SVIX"]:
            fading = peak and vix <= peak * (1 - p["svix_drop"])
            if (self.spike_age <= p["spike_lookback"] and not spike_now and ratio <= p["svix_entry_ratio"] and fading
                    and sessions_to_event > p["event_buffer"]):
                out.append({"action": "buy", "ticker": "SVIX", "weight": p["svix_weight"], "reason": "post_spike_fade",
                            "detail": {**detail, "peak20": round(peak, 2)}})
        return out

    def filled(self, order, price, shares, asof):
        t = order["ticker"]
        if order["action"] == "buy":
            self.pos[t] = {"shares": shares, "entry": price, "entryDate": str(asof)[:10], "held": 0, "peak": price,
                           "sawBackwardation": False}
        else:
            self.pos.pop(t, None)
            self.cooldown[t] = self.p["cooldown"]

    def hard_stop(self, t):
        pos = self.pos.get(t)
        if t == "SVIX" and pos and self.p["exits"] == "full":
            return pos["entry"] * (1 - self.p["svix_hard_stop"])
        return None


def simulate(panel, vixf, params=None, start=None, end=None, equity0=10_000.0, events=None, state_out=None):
    """Daily sleeve returns on a $10k account (unused cash earns 0), whole shares, Schwab costs, next-open fills.
    state_out (dict, optional) receives the final sleeve state, cash, shares and the orders pending for the next open."""
    sl = VolSleeve(**(params or {}))
    cost = CostModel.for_broker("schwab")
    dates = panel.dates
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    f = vixf.reindex(panel.dates).ffill()
    to_event = _sessions_until_event(panel.dates, events if events is not None else macro_events())
    dv = (panel.close * panel.volume).rolling(20, min_periods=5).mean()
    vol20 = panel.total_return().rolling(20, min_periods=10).std()
    cash, shares = equity0, {}
    eq_prev = equity0
    rets, eqs, trades, pending = [], [], [], []
    open_trades = {}
    for d in dates:
        o, h, l, c = panel.open.loc[d], panel.high.loc[d], panel.low.loc[d], panel.close.loc[d]
        # fills at the open for yesterday's decisions
        for order in pending:
            t = order["ticker"]
            px = float(o.get(t, np.nan))
            if not math.isfinite(px):
                continue
            if order["action"] == "buy":
                n = math.floor(order["weight"] * eq_prev / px)
                if n < 1:
                    continue
                fee = n * px * cost.cost_bps(n * px, dv.at[d, t] if t in dv else None, vol20.at[d, t] if t in vol20 else None, "buy", px) / 1e4
                cash -= n * px + fee
                shares[t] = n
                sl.filled(order, px, n, d)
                open_trades[t] = {"ticker": t, "entryDate": str(d.date()), "entry": px, "shares": n, "entryReason": order["reason"],
                                  "fees": fee}
            else:
                n = shares.pop(t, 0)
                if not n:
                    continue
                fee = n * px * cost.cost_bps(n * px, dv.at[d, t], vol20.at[d, t], "sell", px) / 1e4
                cash += n * px - fee
                sl.filled(order, px, n, d)
                tr = open_trades.pop(t)
                tr.update({"exitDate": str(d.date()), "exit": px, "exitReason": order["reason"], "fees": tr["fees"] + fee,
                           "pnl": n * (px - tr["entry"]) - tr["fees"] - fee})
                trades.append(tr)
        pending = []
        # intraday hard stop (gap-aware)
        for t in list(shares):
            stop = sl.hard_stop(t)
            if stop and math.isfinite(float(l.get(t, np.nan))) and l[t] <= stop:
                px = min(float(o[t]), stop)
                n = shares.pop(t)
                fee = n * px * cost.cost_bps(n * px, dv.at[d, t], vol20.at[d, t], "sell", px) / 1e4
                cash += n * px - fee
                sl.filled({"action": "sell", "ticker": t}, px, n, d)
                tr = open_trades.pop(t)
                tr.update({"exitDate": str(d.date()), "exit": px, "exitReason": "hard_stop" + ("_gap" if o[t] < stop else ""),
                           "fees": tr["fees"] + fee, "pnl": n * (px - tr["entry"]) - tr["fees"] - fee})
                trades.append(tr)
        eq = cash + sum(n * float(c[t]) for t, n in shares.items())
        rets.append(eq / eq_prev - 1)
        eqs.append(eq)
        eq_prev = eq
        row = f.loc[d].to_dict() if d in f.index else {}
        pending = sl.step(d, row, {t: float(c[t]) for t in ("VIXY", "SVIX") if t in c.index and math.isfinite(c[t])},
                          int(to_event.get(d, 999)))
    for t, tr in open_trades.items():
        tr.update({"exitDate": None, "exitReason": "open", "pnl": shares.get(t, 0) * (float(panel.close[t].loc[dates[-1]]) - tr["entry"]) - tr["fees"]})
        trades.append(tr)
    if state_out is not None:
        state_out.update({"sleeve": sl.state(), "cash": cash, "shares": dict(shares), "pending": pending,
                          "hardStops": {t: sl.hard_stop(t) for t in shares}})
    return pd.Series(rets, index=dates), pd.Series(eqs, index=dates), trades


def _sharpe(r):
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std() * math.sqrt(TRADING_DAYS)) if len(r) > 5 and r.std() > 0 else 0.0


def perf(r, eq=None):
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod() if eq is None else eq
    years = len(r) / TRADING_DAYS
    total = float((1 + r).prod() - 1)
    return {"totalReturn": total, "cagr": (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else None,
            "vol": float(r.std() * math.sqrt(TRADING_DAYS)), "sharpe": _sharpe(r),
            "maxDrawdown": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()), "days": int(len(r))}


def exit_table(trades):
    out = {}
    for t in trades:
        k = f"{t['ticker']}:{t['exitReason']}"
        o = out.setdefault(k, {"n": 0, "pnl": 0.0, "wins": 0})
        o["n"] += 1
        o["pnl"] += t.get("pnl") or 0.0
        o["wins"] += 1 if (t.get("pnl") or 0) > 0 else 0
    return {k: {**v, "pnl": round(v["pnl"], 2)} for k, v in sorted(out.items())}


def walk_forward(panel, vixf, grid=None, train_min=252, test_len=63, warmup=0):
    """Each combo simulated causally over the whole history; per fold the best training Sharpe is applied to the
    next test window; stitched OOS returns (positions inherited from the chosen combo: an approximation, noted)."""
    grid = grid or GRID
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    start = panel.dates[warmup]
    sims = [simulate(panel, vixf, c, start=start) for c in combos]
    dates = sims[0][0].index
    stitched, folds = pd.Series(np.nan, index=dates), []
    for s in range(train_min, len(dates), test_len):
        e = min(s + test_len, len(dates))
        scores = [_sharpe(x[0].iloc[:s]) for x in sims]
        best = int(np.argmax(scores))
        stitched.iloc[s:e] = sims[best][0].iloc[s:e].to_numpy()
        folds.append({"testStart": str(dates[s].date()), "testEnd": str(dates[e - 1].date()), "params": combos[best],
                      "trainSharpe": scores[best], "testSharpe": _sharpe(sims[best][0].iloc[s:e])})
    oos = stitched.dropna()
    oos_trades = [t for x, c in zip(sims, combos) for t in x[2] if t.get("exitDate") and t["exitDate"] >= str(oos.index[0].date())
                  and any(f["params"] == c and f["testStart"] <= t["entryDate"] <= f["testEnd"] for f in folds)] if len(oos) else []
    return {"combos": len(combos), "folds": folds, "oosStart": str(oos.index[0].date()) if len(oos) else None,
            "outOfSample": perf(oos) if len(oos) else None, "oosExitReasons": exit_table(oos_trades),
            "oosTrades": len(oos_trades)}, oos, sims, combos


def _block_p(diff, block=10, n=5000, seed=3):
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 20:
        return None
    rng = np.random.default_rng(seed)
    T = len(d)
    obs = d.mean()
    c = d - obs
    hits = 0
    for _ in range(n):
        starts = rng.integers(0, T, size=math.ceil(T / block))
        idx = np.concatenate([(np.arange(s, s + block) % T) for s in starts])[:T]
        hits += c[idx].mean() >= obs
    return float((1 + hits) / (1 + n))


def research(write=True):
    from quant_tools import fast_panel
    import vix_data
    panel = fast_panel(["VIXY", "SVIX", "SPY"])
    vixf = vix_data.features(vix_data.load())
    wf, oos, sims, combos = walk_forward(panel, vixf)
    start = panel.dates[0]
    variants = {
        "defaults_full_exits": DEFAULTS,
        "defaults_time_only_exits": {**DEFAULTS, "exits": "time_only"},
        "vixy_only": {**DEFAULTS, "svix_enabled": False},
        "svix_only": {**DEFAULTS, "vixy_enabled": False},
    }
    ablation = {}
    for name, prm in variants.items():
        r, eq, tr = simulate(panel, vixf, {k: v for k, v in prm.items() if k in DEFAULTS}, start=start)
        r_oos = r.loc[oos.index[0]:] if len(oos) else r
        ablation[name] = {"fullPeriod": perf(r), "oosWindow": perf(r_oos), "trades": len([t for t in tr if t.get("exitDate")]),
                          "exitReasons": exit_table(tr)}
    # buy-and-hold references at the same weights
    ret = panel.total_return()
    bh = {}
    for t, w in (("VIXY", DEFAULTS["vixy_weight"]), ("SVIX", DEFAULTS["svix_weight"])):
        r = (w * ret[t]).fillna(0.0)
        bh[f"hold_{t}_{int(w * 100)}pct"] = perf(r.loc[oos.index[0]:] if len(oos) else r)
    p_full_vs_time = None
    if len(oos):
        a = simulate(panel, vixf, DEFAULTS, start=start)[0].loc[oos.index[0]:]
        b = simulate(panel, vixf, {**DEFAULTS, "exits": "time_only"}, start=start)[0].loc[oos.index[0]:]
        p_full_vs_time = _block_p(a - b)
    p_oos_gt0 = _block_p(oos) if len(oos) else None
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"), "rule": "VIXY when VIX calm/low (level+percentile+contango); "
        "SVIX after a spike fades; explicit exits", "data": {"sessions": int(len(panel.dates)), "first": str(panel.dates[0].date()),
                                                             "last": str(panel.dates[-1].date()), "vixSource": "FRED VIXCLS / VXVCLS"},
        "walkForward": wf, "pOosMeanReturnGt0": p_oos_gt0, "ablation": ablation, "buyHoldReferences": bh,
        "pFullExitsBeatTimeOnly": p_full_vs_time,
        "honesty": ("Two years of ETF history contain about one major VIX spike (April 2025) and a few smaller ones, so "
                    "the sleeve has only a handful of independent trades. Treat these numbers as a smoke test of the "
                    "rule and its exits, not evidence of an edge. Sleeve returns are on the whole $10k account "
                    "(5% VIXY / 10% SVIX positions), idle cash earns 0."),
        "defaults": DEFAULTS, "grid": GRID,
    }
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_atomically(OUT_DIR / "vol_sleeve.json", json.dumps(summary, indent=1, default=str).encode())
        oos.rename("vol_sleeve_oos").to_csv(OUT_DIR / "vol_sleeve_oos_returns.csv")
    return summary


def signal():
    """Latest VIX regime features (the paper runner, options_paper.py, replays the sleeve to get today's orders)."""
    import vix_data
    return vix_data.latest()


if __name__ == "__main__":
    from structured_log import log_run
    cmd = sys.argv[1] if len(sys.argv) > 1 else "walkforward"
    with log_run("vol_sleeve"):
        if cmd == "walkforward":
            s = research()
            print(json.dumps({k: s[k] for k in ("walkForward", "pOosMeanReturnGt0", "pFullExitsBeatTimeOnly", "buyHoldReferences",
                                                "honesty")}, indent=1, default=str)[:6000])
            print(json.dumps({k: {"oos": v["oosWindow"], "exits": v["exitReasons"]} for k, v in s["ablation"].items()}, indent=1, default=str))
        else:
            print(json.dumps(signal(), indent=1))
