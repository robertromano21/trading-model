"""Walk-forward tests of the option / downside overlays on top of the core TSMOM book ($10k, whole shares, Schwab
costs). Option prices are MODELED (Black-Scholes on a VIX-scaled vol forecast with skew and spread models) because
there are no historical option quotes locally; every result says so. Inverse-ETF overlays use real SH/PSQ/SDS
prices. Paper short stock (config-gated) uses real SPY prices plus a modeled borrow fee and Reg-T margin.

For each overlay a small parameter grid is simulated causally over the whole history; per fold (expanding train,
63-session test) the combination with the best training Sharpe of the TOTAL account is applied to the next test
window; the stitched out-of-sample returns are compared with the base book over the same days (block-bootstrap p
of the daily return difference, Holm across overlays). Two sizing modes:
  account     integer contracts in the $10k account (what Robert can actually trade)
  fractional  fractional contracts: reference only, NOT tradable at $10k; shows the economics without lot size

Swarm interface (GB-11): ``apply_overlay(panel, base_result, signals, params, start, fractional=False, vix=None)``
returns {"returns", "pnl", "trades", "events", "skips", "pricing"}; ``OVERLAY_GRID`` holds the design space.

  python options_research.py            # writes Massive/Research/options_overlays.json
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
from options_book import ModelPricer, OptionsBook, OptionsConfig, RiskRefusal
from options_pricing import Leg, leg_from
from strategies.options_overlay import OptionsOverlay

OUT_PATH = DATA_DIR / "Research" / "options_overlays.json"
ETF_LIKE = {"SPY", "QQQ", "IWM", "DIA", "SH", "PSQ", "SDS", "TLT", "IEF", "GLD", "XLF", "XLK", "XLE", "SMH", "HYG"}
OVERLAY_GRID = {
    "protective_put": {"trigger": ["hmm", "top"], "structure": ["put", "put_spread"], "otm_pct": [0.03, 0.07]},
    "bearish_puts": {"trigger": ["top"], "top_k": [2, 3], "structure": ["put", "put_spread"]},
    "inverse_etf": {"trigger": ["hmm", "top"], "inverse": ["SH", "SDS"]},
    "short_stock": {"trigger": ["hmm", "top"]},
    "covered_call": {"call_otm_pct": [0.03, 0.06]},
    "csp_entry": {"csp_otm_pct": [0.03, 0.06]},
}
TRADING_DAYS = 252


def vol_surface(panel, vix=None, tickers=("SPY", "QQQ", "IWM")):
    """Causal ATM vol per date and underlying. SPY = VIX / 100 when VIX is available (real 30-day implied vol of the
    S&P 500); others = their blended EWMA/realized vol * clip(VIX / (100 * SPY blended vol), 0.8, 2.0) (a VIX-implied
    premium); without VIX, blended vol * 1.12."""
    ret = panel.total_return()
    out = {}
    for t in set(tickers) | {"SPY"}:
        if t not in ret.columns:
            continue
        r = ret[t]
        ewma = np.sqrt((r ** 2).ewm(alpha=0.06, min_periods=20).mean() * TRADING_DAYS)
        real = r.rolling(60, min_periods=20).std() * math.sqrt(TRADING_DAYS)
        out[t] = (0.5 * (ewma + real)).clip(lower=0.08)
    base = pd.DataFrame(out)
    if vix is not None:
        v = (vix.reindex(panel.dates).ffill() / 100.0)
        ratio = (v / base["SPY"]).clip(0.8, 2.0)
        surf = base.mul(ratio, axis=0)
        surf["SPY"] = v.fillna(base["SPY"] * 1.12)
        surf.attrs["source"] = "VIX-scaled (FRED VIXCLS)"
    else:
        surf = base * 1.12
        surf.attrs["source"] = "blended realized vol * 1.12"
    return surf.ffill()


def _next_dividends(panel, d, tickers, days=4):
    out = {}
    dv = panel.dividends
    for t in tickers:
        if t not in dv.columns:
            continue
        window = dv[t].loc[d:].iloc[1:days + 1]
        hit = window[window > 0]
        if len(hit):
            out[t] = (str(hit.index[0].date()), float(hit.iloc[0]))
    return out


def apply_overlay(panel, base, signals, params, start, fractional=False, vix=None, surface=None):
    """Run one overlay over [start, end] on top of a base SimResult (equity book). Causal; decisions at the close,
    fills at the next open. Returns daily total-account returns and the overlay P&L series."""
    mode = params.get("mode")
    ov = OptionsOverlay(enabled=True, fractional=fractional, **params)
    cfg = OptionsConfig(enabled=True, initial_equity=0.0, short_stock_enabled=(mode == "short_stock"),
                        max_position_loss=max(0.03, params.get("risk_budget", 0.02) * 1.05),
                        max_premium_at_risk=0.10, max_abs_delta=1.5)
    book = OptionsBook(cfg)
    book.cash = 0.0
    surface = surface if surface is not None else vol_surface(panel, vix)
    pricer = ModelPricer(lambda d, u: float(surface[u].loc[pd.Timestamp(d)]) if u in surface.columns and pd.Timestamp(d) in surface.index
                         else float(surface["SPY"].loc[pd.Timestamp(d)]), is_etf_fn=lambda u: u in ETF_LIKE)
    cost = CostModel.for_broker("schwab")
    dv = (panel.close * panel.volume).rolling(20, min_periods=5).mean()
    vol20 = panel.total_return().rolling(20, min_periods=10).std()
    dates = panel.dates[panel.dates >= pd.Timestamp(start)]
    beq = base.equity.reindex(dates).ffill()
    hold_w = base.holdings.reindex(dates).fillna(0.0) if base.holdings is not None else pd.DataFrame(index=dates)
    gross = base.gross_exposure.reindex(dates).fillna(0.0)
    sma100 = panel.close.rolling(100, min_periods=100).mean()
    sh_cash, sh_pos = 0.0, {}       # inverse ETF sleeve: ticker -> {"shares", "stop", "entry"}
    pending, trades, skips = [], [], []
    pnl_prev, rets, pnls = 0.0, [], []
    total_prev = float(beq.iloc[0])
    leg_quote = lambda u, l, d, s: pricer.quote(u, leg_from(l), d, s)
    for i, d in enumerate(dates):
        o, c, lo = panel.open.loc[d], panel.close.loc[d], panel.low.loc[d]
        # ---------------- fills at the open
        for it in pending:
            a = it["action"]
            try:
                if a == "open":
                    u = it["underlying"]
                    spot = float(o.get(u, np.nan))
                    if not math.isfinite(spot):
                        continue
                    book.open(it["strategy"], u, it["legs"], pricer, d, spot, spots={**{k: float(v) for k, v in o.items() if math.isfinite(v)}},
                              tag=it.get("tag", mode), reason=it.get("reason", ""), covered_shares=it.get("covered_shares", 0))
                elif a == "close":
                    p = next((x for x in book.positions if x["id"] == it["id"]), None)
                    if p:
                        book.close(it["id"], pricer, d, float(o[p["underlying"]]), it.get("reason", "close"))
                elif a == "buy":
                    t = it["ticker"]; px = float(o[t]); n = it["shares"]
                    fee = n * px * cost.cost_bps(n * px, dv.at[d, t], vol20.at[d, t], "buy", px) / 1e4
                    sh_cash -= n * px + fee
                    sh_pos[t] = {"shares": n, "stop": px * (1 - params.get("stop_pct", 0.08)), "entry": px, "date": str(d.date()), "fee": fee}
                elif a == "sell" and it["ticker"] in sh_pos:
                    t = it["ticker"]; px = float(o[t]); pos = sh_pos.pop(t)
                    fee = pos["shares"] * px * cost.cost_bps(pos["shares"] * px, dv.at[d, t], vol20.at[d, t], "sell", px) / 1e4
                    sh_cash += pos["shares"] * px - fee
                    trades.append({"ticker": t, "entry": pos["entry"], "exit": px, "entryDate": pos["date"], "exitDate": str(d.date()),
                                   "pnl": pos["shares"] * (px - pos["entry"]) - pos["fee"] - fee, "exitReason": it.get("reason")})
                elif a == "short":
                    book.account_equity = total_prev
                    book.short(it["ticker"], it["shares"], float(o[it["ticker"]]), d, {it["ticker"]: float(o[it["ticker"]])}, pricer, it.get("reason", ""))
                elif a == "cover":
                    book.cover(it["ticker"], float(o[it["ticker"]]), d, it.get("reason", "cover"))
            except RiskRefusal as error:
                skips.append({"date": str(d.date()), "intent": a, "why": str(error)})
            except KeyError:
                continue
        pending = []
        # ---------------- inverse ETF stops (gap-aware)
        for t in list(sh_pos):
            pos = sh_pos[t]
            if float(lo[t]) <= pos["stop"]:
                px = min(float(o[t]), pos["stop"]); sh_pos.pop(t)
                fee = pos["shares"] * px * cost.cost_bps(pos["shares"] * px, dv.at[d, t], vol20.at[d, t], "sell", px) / 1e4
                sh_cash += pos["shares"] * px - fee
                trades.append({"ticker": t, "entry": pos["entry"], "exit": px, "entryDate": pos["date"], "exitDate": str(d.date()),
                               "pnl": pos["shares"] * (px - pos["entry"]) - pos["fee"] - fee, "exitReason": "stop" + ("_gap" if o[t] < pos["stop"] else "")})
        # ---------------- close: expirations, assignment, short-stock housekeeping
        spots = {k: float(v) for k, v in c.items() if math.isfinite(v)}
        opens = {k: float(v) for k, v in o.items() if math.isfinite(v)}
        divs = {t: float(panel.dividends.at[d, t]) for t in [s["ticker"] for s in book.shorts] if t in panel.dividends.columns}
        book.daily(pricer, d, spots, opens, divs, _next_dividends(panel, d, {p["underlying"] for p in book.positions}))
        # ---------------- account value
        sleeve = book.equity(pricer, d, spots) + sh_cash + sum(p["shares"] * spots[t] for t, p in sh_pos.items())
        total = float(beq.loc[d]) + sleeve
        rets.append(total / total_prev - 1 if total_prev else 0.0)
        pnls.append(sleeve)
        total_prev = total
        # ---------------- decisions for tomorrow's open
        hw = hold_w.loc[d] if len(hold_w.columns) else pd.Series(dtype=float)
        holdings = {t: float(w) * float(beq.loc[d]) / spots[t] for t, w in hw.items() if w > 0 and t in spots}
        idle = max(0.0, (1 - float(gross.loc[d])) * float(beq.loc[d]))
        book.account_equity, book.cash_budget, book.external_delta = total, idle, float(gross.loc[d]) * float(beq.loc[d])
        cands = sorted([t for t in ETF_LIKE | set(hw.index) if t in spots and t in sma100.columns and spots[t] > (sma100.at[d, t] or np.inf)
                        and t not in ("SH", "PSQ", "SDS")], key=lambda t: spots[t])
        sig_row = signals.loc[d].to_dict() if d in signals.index else {}
        ctx = {"spots": spots, "holdings": holdings, "equity": total, "idle_cash": idle, "book_delta": book.external_delta,
               "signal": sig_row, "positions": book.positions, "candidates": cands,
               "share_positions": {**{t: p["shares"] for t, p in sh_pos.items()}, **{s["ticker"]: -s["shares"] for s in book.shorts}},
               "quote": lambda u, l, _d=d: leg_quote(u, l, _d, spots[u]),
               "value_of": lambda p, _d=d: book.position_value(p, pricer, _d, spots[p["underlying"]], side="exit")[0]}
        for it in ov.intents(d, ctx):
            if it["action"] == "skip":
                skips.append({"date": str(d.date()), "intent": "skip", "why": it.get("reason")})
            else:
                pending.append(it)
    closed = [c for c in book.closed]
    return {"returns": pd.Series(rets, index=dates), "pnl": pd.Series(pnls, index=dates), "trades": trades + closed,
            "events": book.events, "skips": skips, "pricing": "modeled options / real ETF prices", "volSource": surface.attrs.get("source")}


def _sharpe(r):
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std() * math.sqrt(TRADING_DAYS)) if len(r) > 5 and r.std() > 0 else 0.0


def perf(r):
    r = pd.Series(r).dropna()
    total = float((1 + r).prod() - 1)
    years = len(r) / TRADING_DAYS
    eq = (1 + r).cumprod()
    return {"totalReturn": total, "cagr": (1 + total) ** (1 / years) - 1 if years and total > -1 else None,
            "vol": float(r.std() * math.sqrt(TRADING_DAYS)), "sharpe": _sharpe(r), "maxDrawdown": float((eq / eq.cummax() - 1).min()),
            "days": int(len(r))}


def _block_p(diff, block=10, n=5000, seed=3):
    d = np.asarray(pd.Series(diff).dropna(), dtype=float)
    if len(d) < 20 or not d.std():
        return None
    rng = np.random.default_rng(seed)
    T, obs = len(d), d.mean()
    c = d - obs
    hits = sum(c[np.concatenate([(np.arange(s, s + block) % T) for s in rng.integers(0, T, size=math.ceil(T / block))])[:T]].mean() >= obs
               for _ in range(n))
    return float((1 + hits) / (1 + n))


def holm(ps):
    idx = [i for i, p in enumerate(ps) if p is not None]
    order = sorted(idx, key=lambda i: ps[i])
    adj, running = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(idx) - rank) * ps[i]))
        adj[i] = running
    return adj


def walk_forward_overlay(panel, base, signals, mode, start, fractional, surface, train_min=252 - 110, test_len=63):
    grid = OVERLAY_GRID[mode]
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    sims = [apply_overlay(panel, base, signals, {"mode": mode, **c}, start, fractional, surface=surface) for c in combos]
    dates = sims[0]["returns"].index
    stitched, folds = pd.Series(np.nan, index=dates), []
    for s in range(train_min, len(dates), test_len):
        e = min(s + test_len, len(dates))
        scores = [_sharpe(x["returns"].iloc[:s]) for x in sims]
        b = int(np.argmax(scores))
        stitched.iloc[s:e] = sims[b]["returns"].iloc[s:e].to_numpy()
        folds.append({"testStart": str(dates[s].date()), "params": combos[b], "trainSharpe": round(scores[b], 3),
                      "testSharpe": round(_sharpe(sims[b]["returns"].iloc[s:e]), 3)})
    oos = stitched.dropna()
    base_r = base.returns.reindex(oos.index)
    diff = oos - base_r
    trades = sum(len([t for t in x["trades"]]) for x in sims) / len(sims)
    skip_reasons = {}
    for x in sims:
        for sk in x["skips"]:
            key = (sk.get("why") or "")[:90]
            skip_reasons[key] = skip_reasons.get(key, 0) + 1
    exits = {}
    for x in sims:
        for t in x["trades"]:
            r = str(t.get("exitReason") or "")[:40]
            exits[r] = exits.get(r, 0) + 1
    return {"mode": mode, "sizing": "fractional (reference, not tradable at $10k)" if fractional else "account ($10k, integer contracts)",
            "combos": len(combos), "folds": folds, "oosStart": str(oos.index[0].date()) if len(oos) else None,
            "overlayOOS": perf(oos), "baseOOS": perf(base_r), "sharpeDiff": _sharpe(oos) - _sharpe(base_r),
            "meanDailyDiffBps": float(diff.mean() * 1e4), "pOverlayBeatsBase": _block_p(diff),
            "avgTradesPerCombo": trades, "topSkipReasons": dict(sorted(skip_reasons.items(), key=lambda kv: -kv[1])[:5]),
            "exitReasons": exits, "pricing": sims[0]["pricing"], "volSource": sims[0]["volSource"]}


def research(write=True):
    from market_data import load_panel
    from research import WARMUP, run_strategy
    from strategies import TimeSeriesMomentum
    import top_detector
    try:
        import vix_data
        vix = vix_data.load()["vix"]
    except SystemExit:
        vix = None
    panel = load_panel()
    start = panel.dates[WARMUP - 1]
    _, _, base = run_strategy(panel, TimeSeriesMomentum(), start=start, initial_equity=10_000.0, whole_shares=True,
                              min_order_value=100.0)
    signals = top_detector.build_signals(panel)
    surface = vol_surface(panel, vix)
    results = []
    for mode in OVERLAY_GRID:
        for fractional in (False, True):
            print(f"{datetime.now():%H:%M:%S} {mode} fractional={fractional}", file=sys.stderr, flush=True)
            results.append(walk_forward_overlay(panel, base, signals, mode, start, fractional, surface))
    acct = [r for r in results if r["sizing"].startswith("account")]
    adj = holm([r["pOverlayBeatsBase"] for r in acct])
    for r, a in zip(acct, adj):
        r["holmP"] = a
    frac = [r for r in results if not r["sizing"].startswith("account")]
    for r, a in zip(frac, holm([r["pOverlayBeatsBase"] for r in frac])):
        r["holmP"] = a
    summary = {"generated": datetime.now().isoformat(timespec="seconds"), "base": "core TSMOM, $10k, whole shares, Schwab costs",
               "results": results, "volSource": surface.attrs.get("source"),
               "honesty": ("Option P&L is MODELED (Black-Scholes on VIX-scaled vol with skew and spread models; no "
                           "historical option quotes). One year of out-of-sample data in a mostly rising market: hedges "
                           "are expected to cost money and few adverse episodes exist to pay them back. Account-mode "
                           "results show what $10k can actually trade (integer contracts; covered calls and cash-secured "
                           "puts need 100 shares / strike*100 of cash, so most are infeasible).")}
    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(OUT_PATH, json.dumps(summary, indent=1, default=str).encode())
    return summary


if __name__ == "__main__":
    from structured_log import log_run
    with log_run("options_research"):
        s = research()
        for r in s["results"]:
            print(f"{r['mode']:15s} {r['sizing'][:10]:10s} overlay Sharpe {r['overlayOOS']['sharpe']:.2f} CAGR {r['overlayOOS']['cagr'] or 0:.3f} "
                  f"DD {r['overlayOOS']['maxDrawdown']:.3f} | base Sharpe {r['baseOOS']['sharpe']:.2f} CAGR {r['baseOOS']['cagr'] or 0:.3f} "
                  f"DD {r['baseOOS']['maxDrawdown']:.3f} | p {r['pOverlayBeatsBase']} holm {r['holmP']} trades {r['avgTradesPerCombo']:.1f}")
