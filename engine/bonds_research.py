"""Walk-forward test of the config-gated bond-ETF sleeve against IEF, AGG and SHY.

Writes Massive/Research/bonds_report.md and bonds_report.json. Uses the same cost model as the
equity research. Individual cash Treasuries are not in this test (see treasury.py / paper_treasuries.py).
"""

from __future__ import annotations

import itertools
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from costs import CostModel
from fetch_and_model import DATA_DIR, write_atomically
from macro_data import PROXY_DISCLAIMER
from market_data import load_panel
from metrics import performance
from portfolio_sim import simulate
from strategies.bonds import BOND_ETFS, FixedIncomeStrategy

OUT_DIR = DATA_DIR / "Research"
WARMUP = 60
TRAIN_MIN = 252
TEST_LEN = 63

GRID = {
    "target_vol": (0.05, 0.08),
    "slope_steep": (0.25, 0.50),
    "hy_oas_risk_off_pct": (0.60, 0.80),
    "allow_hy": (True, False),
}


def _clean(value, digits=6):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else round(float(value), digits)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _clean(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, digits) for v in value]
    return value


def _perf(result, start=None):
    returns = result.returns
    if start is not None:
        returns = returns.loc[pd.Timestamp(start):]
    stats = performance(returns)
    stats["start"] = str(returns.dropna().index[0].date()) if returns.dropna().shape[0] else None
    stats["end"] = str(returns.dropna().index[-1].date()) if returns.dropna().shape[0] else None
    stats["sessions"] = int(returns.dropna().shape[0])
    return stats


def _buy_hold(panel, ticker, start):
    weights = pd.DataFrame(0.0, index=panel.dates, columns=panel.close.columns)
    if ticker in weights.columns:
        weights[ticker] = 0.999
    return simulate(panel, weights, start=start)


def _expand(grid):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def _sharpe(returns):
    r = pd.Series(returns).dropna()
    if len(r) < 20 or r.std(ddof=0) == 0:
        return float("-inf")
    return float(r.mean() / r.std(ddof=0) * math.sqrt(252))


def walk_forward(panel, start, cost_model):
    combos = list(_expand(GRID))
    dates = panel.dates
    n = len(dates)
    sims = []
    for params in combos:
        strategy = FixedIncomeStrategy(**params)
        weights = strategy.target_weights(panel)
        result = simulate(panel, weights, start=start, cost_model=cost_model)
        sims.append({"params": params, "weights": weights, "returns": result.returns, "result": result})
    starts = list(range(max(TRAIN_MIN, int(panel.dates.get_loc(start)) if start in panel.dates else TRAIN_MIN), n, TEST_LEN))
    # get_loc may fail if start is a timestamp; use search
    if not starts:
        first = next((i for i, d in enumerate(dates) if d >= pd.Timestamp(start)), TRAIN_MIN)
        starts = list(range(max(TRAIN_MIN, first + 1), n, TEST_LEN))
    trials, chosen = [], []
    stitched = pd.DataFrame(0.0, index=dates, columns=panel.close.columns)
    for fold, i0 in enumerate(starts):
        if i0 < 2 or i0 >= n:
            continue
        train = slice(dates[WARMUP], dates[i0 - 1])
        end = min(i0 + TEST_LEN, n)
        test = slice(dates[i0], dates[end - 1])
        scores = [_sharpe(sim["returns"].loc[train]) for sim in sims]
        tests = [_sharpe(sim["returns"].loc[test]) for sim in sims]
        best = int(np.argmax(scores))
        chosen.append(best)
        for number, sim in enumerate(sims):
            trials.append({"fold": fold, "trainEnd": str(dates[i0 - 1].date()),
                           "testStart": str(dates[i0].date()), "testEnd": str(dates[end - 1].date()),
                           "trial": number, "params": json.dumps(sim["params"]),
                           "trainSharpe": scores[number], "testSharpe": tests[number],
                           "selected": number == best})
        rows = dates[i0 - 1:end - 1]
        stitched.loc[rows] = sims[best]["weights"].reindex(index=rows, columns=stitched.columns).fillna(0.0).to_numpy()
    if not starts:
        return None, pd.DataFrame(), None
    first_test = dates[starts[0]]
    oos = simulate(panel, stitched, start=dates[starts[0] - 1], cost_model=cost_model)
    trial_frame = pd.DataFrame(trials)
    distinct = len(set(chosen))
    # Selection optimism: mean selected test Sharpe minus mean median trial test Sharpe
    if not trial_frame.empty:
        sel = trial_frame[trial_frame["selected"]]
        optimism = float(sel["testSharpe"].mean() - trial_frame.groupby("fold")["testSharpe"].median().mean())
    else:
        optimism = None
    return {
        "trials": len(combos), "folds": len(starts), "distinctSelections": distinct,
        "oosStart": str(first_test.date()), "selectionOptimismSharpe": optimism,
        "selectedParams": [json.dumps(sims[c]["params"]) for c in chosen],
    }, trial_frame, oos


def markdown(summary):
    def fmt(stats, key, pct=False):
        if not stats or stats.get(key) is None:
            return "n/a"
        return f"{100 * stats[key]:.1f}%" if pct else f"{stats[key]:.2f}"

    def row(name, stats):
        return (f"| {name} | {fmt(stats, 'cagr', True)} | {fmt(stats, 'sharpe')} | "
                f"{fmt(stats, 'maxDrawdown', True)} | {fmt(stats, 'volatility', True)} | "
                f"{stats.get('sessions', 'n/a')} |")

    lines = [
        "# Bond-ETF sleeve walk-forward",
        "",
        f"Generated {summary.get('generatedAt')}.",
        "",
        "The sleeve allocates across bond ETFs that are already in the local price database "
        f"({', '.join(summary.get('universe') or [])}). Duration stance uses the Treasury curve "
        "(or TLT vs SHY when the curve is missing); credit stance uses ICE BofA OAS proxies "
        "(or HYG vs IEF). " + PROXY_DISCLAIMER,
        "",
        f"Sample: {summary.get('data', {}).get('start')} to {summary.get('data', {}).get('end')} "
        f"({summary.get('data', {}).get('sessions')} sessions). Costs are the same Schwab-style model "
        "as the equity research (no commissions, plus SEC/TAF, spread and impact).",
        "",
        "## Verdict",
        "",
        summary.get("verdict") or "n/a",
        "",
        "## Full-sample (after warmup, after costs)",
        "",
        "| Book | CAGR | Sharpe | Max DD | Vol | Sessions |",
        "|---|---:|---:|---:|---:|---:|",
        row("Bond sleeve (defaults)", summary.get("fullSample", {}).get("bonds")),
        row("IEF buy & hold", summary.get("fullSample", {}).get("IEF")),
        row("AGG buy & hold", summary.get("fullSample", {}).get("AGG")),
        row("SHY buy & hold", summary.get("fullSample", {}).get("SHY")),
        "",
        "## Walk-forward out of sample",
        "",
    ]
    wf = summary.get("walkForward") or {}
    oos = wf.get("outOfSample") or {}
    if oos:
        lines += [
            f"Grid: {wf.get('trials')} parameter combinations, {wf.get('folds')} folds "
            f"(train ≥ {TRAIN_MIN} sessions, test {TEST_LEN}). "
            f"{wf.get('distinctSelections')} distinct selections. "
            f"Selection optimism (selected minus median test Sharpe): {wf.get('selectionOptimismSharpe')}.",
            "",
            "| Book | CAGR | Sharpe | Max DD | Vol | Sessions |",
            "|---|---:|---:|---:|---:|---:|",
            row("Bond sleeve OOS", oos),
            row("IEF OOS window", summary.get("oosBenchmarks", {}).get("IEF")),
            row("AGG OOS window", summary.get("oosBenchmarks", {}).get("AGG")),
            row("SHY OOS window", summary.get("oosBenchmarks", {}).get("SHY")),
            "",
        ]
    else:
        lines.append("Walk-forward could not be run (not enough history after warmup).")
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- Two years, one rates cycle. A duration or credit rule that looked good here can fail in a different cycle.",
        "- Average durations are static metadata, not live fund analytics.",
        "- The sleeve is off by default (`include_bond_sleeve=False`) until this record is stronger.",
        "- Individual cash Treasuries are paper-only and are not in these ETF results.",
        "",
    ]
    return "\n".join(lines)


def verdict_text(full, oos, benches, wf):
    parts = []
    b, ief = (full or {}).get("bonds") or {}, (full or {}).get("IEF") or {}
    if b.get("sharpe") is not None and ief.get("sharpe") is not None:
        better = "above" if b["sharpe"] > ief["sharpe"] else "below"
        parts.append(
            f"On the full sample after costs, the default bond sleeve Sharpe is {b['sharpe']:.2f} vs "
            f"{ief['sharpe']:.2f} for IEF ({better}) and a CAGR of {100 * (b.get('cagr') or 0):.1f}% vs "
            f"{100 * (ief.get('cagr') or 0):.1f}%, with a max drawdown of {100 * (b.get('maxDrawdown') or 0):.1f}% "
            f"vs {100 * (ief.get('maxDrawdown') or 0):.1f}%.")
    o = oos or {}
    ief_o = (benches or {}).get("IEF") or {}
    if o.get("sharpe") is not None and ief_o.get("sharpe") is not None:
        parts.append(
            f"Walk-forward out of sample, the selected sleeve Sharpe is {o['sharpe']:.2f} vs "
            f"{ief_o['sharpe']:.2f} for IEF over the same window.")
        if o["sharpe"] <= ief_o["sharpe"]:
            parts.append(
                "It does not beat a simple IEF hold on unseen folds in this sample, so it stays "
                "off by default. A longer rates history would be needed before trusting a duration or credit overlay.")
        else:
            parts.append(
                "It is ahead of IEF on the stitched OOS window, but two years and a small grid are not "
                "enough to treat that as a durable edge; the sleeve stays config-gated and off by default.")
    if wf and wf.get("selectionOptimismSharpe") is not None and wf["selectionOptimismSharpe"] > 0.2:
        parts.append(
            f"Selection optimism is {wf['selectionOptimismSharpe']:.2f} Sharpe points, which is a warning "
            "that the grid is fitting noise.")
    if not parts:
        return "Not enough overlapping history to judge the bond sleeve."
    return " ".join(parts)


def run(panel=None):
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = list(BOND_ETFS)
    panel = panel or load_panel(tickers=tickers)
    start = panel.dates[min(WARMUP, len(panel.dates) - 2)]
    costs = CostModel()
    strategy = FixedIncomeStrategy()
    weights = strategy.target_weights(panel)
    bonds = simulate(panel, weights, start=start, cost_model=costs)
    benches = {t: _buy_hold(panel, t, start) for t in ("IEF", "AGG", "SHY") if t in panel.close.columns}
    wf_meta, trials, oos = walk_forward(panel, start, costs)
    oos_start = None if oos is None else (wf_meta or {}).get("oosStart")
    oos_benches = {}
    if oos is not None and oos_start:
        oos_benches = {t: _perf(_buy_hold(panel, t, panel.dates[0]), oos_start)
                       for t in benches}
    full = {"bonds": _perf(bonds, start), **{t: _perf(r, start) for t, r in benches.items()}}
    wf = None
    if wf_meta and oos is not None:
        wf = {**wf_meta, "outOfSample": _perf(oos, oos_start)}
        if not trials.empty:
            path = OUT_DIR / "bonds_walkforward_trials.csv"
            write_atomically(path, trials.to_csv(index=False).encode())
    summary = {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data": {"start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date()),
                 "sessions": int(len(panel.dates))},
        "universe": [t for t in BOND_ETFS if t in panel.close.columns],
        "missing": [t for t in BOND_ETFS if t not in panel.close.columns],
        "costModel": costs.as_dict(),
        "fullSample": full,
        "walkForward": wf,
        "oosBenchmarks": oos_benches,
        "diagnostics": strategy.last_diagnostics,
        "proxyDisclaimer": PROXY_DISCLAIMER,
        "includeBondSleeveDefault": False,
        "runSeconds": None,
    }
    summary["verdict"] = verdict_text(full, None if wf is None else wf.get("outOfSample"), oos_benches, wf)
    summary["runSeconds"] = round(time.time() - started, 1)
    write_atomically(OUT_DIR / "bonds_report.json", json.dumps(_clean(summary), indent=2, default=str).encode())
    write_atomically(OUT_DIR / "bonds_report.md", markdown(_clean(summary)).encode())
    return summary


def main():
    summary = run()
    print(summary["verdict"])
    print(f"wrote {OUT_DIR / 'bonds_report.md'} in {summary['runSeconds']}s")
    return summary


if __name__ == "__main__":
    main()
