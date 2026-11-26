"""Research runner: full-period backtests, walk-forward out-of-sample tests, multi-horizon trend runs and return
normality diagnostics for 1-200 day horizons. All outputs go to Massive/Research (gitignored).

Usage (from engine/):
  python research.py all                    # everything below, writes research_summary.json + research_report.md
  python research.py backtest | walkforward | multihorizon | normality | attribution | tax | regime-allocation | tax | prob

Honesty notes: the database holds about two years (501 sessions). The 100-day signal needs ~100 sessions of
warm-up and walk-forward needs a training window, so out-of-sample evidence is about one year; 100- and 200-day
horizons have only a handful of independent periods and cannot be validated statistically.
"""

import argparse
import itertools
import json
import math
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from costs import CostModel
from fetch_and_model import DATA_DIR, write_atomically
from market_data import MARKET, load_panel
from schwab_daily import merge_caveats
from metrics import performance, summarize
from portfolio_sim import simulate
from risk import TRADING_DAYS, hold_between_rebalances, jarque_bera, moments
from strategies import CombinedStrategy, FixedIncomeStrategy, RegimeMeanReversion, TimeSeriesMomentum
from universe import CORE_ETFS

OUT_DIR = DATA_DIR / "Research"
WARMUP = 110          # sessions before metrics start (100-day lookback + buffer); same for every strategy
HORIZONS = (1, 5, 20, 50, 100, 200)
NORMALITY_HORIZONS = (1, 2, 5, 10, 20, 50, 100, 200)
GRIDS = {
    "tsmom": {"max_positions": [5, 8], "rebalance": ["W-FRI", "D"]},
    "meanrev": {"lookback": [5, 10], "entry_z": [-2.0, -2.5], "max_hold": [5, 10]},
    "combined": {"tsmom.max_positions": [5, 8], "meanrev.lookback": [5, 10], "meanrev.entry_z": [-2.0, -2.5],
                 "meanrev.max_hold": [5, 10]},
}


def log(message):
    print(f"{datetime.now():%H:%M:%S} {message}", file=sys.stderr, flush=True)


def make_strategy(family, params=None):
    params = dict(params or {})
    if family == "tsmom":
        return TimeSeriesMomentum(**params)
    if family == "meanrev":
        return RegimeMeanReversion(**params)
    if family == "bonds":
        return FixedIncomeStrategy(**params)
    if family == "combined":
        sleeves = {"tsmom": {}, "meanrev": {}}
        bonds_params = None
        top = {}
        for key, value in params.items():
            if "." in key:
                sleeve, name = key.split(".", 1)
                if sleeve == "bonds":
                    bonds_params = {**(bonds_params or {}), name: value}
                else:
                    sleeves.setdefault(sleeve, {})[name] = value
            else:
                top[key] = value
        objs = [TimeSeriesMomentum(**sleeves.get("tsmom", {})),
                RegimeMeanReversion(**sleeves.get("meanrev", {}))]
        if top.pop("include_bond_sleeve", False) or bonds_params is not None:
            objs.append(FixedIncomeStrategy(**(bonds_params or {})))
        return CombinedStrategy(sleeves=objs, **top)
    raise ValueError(family)


def run_strategy(panel, strategy, **sim_args):
    weights = strategy.target_weights(panel)
    stops = strategy.stops(panel)
    result = simulate(panel, weights, stops, sleeve_of=getattr(strategy, "sleeve_of", None), **sim_args)
    return weights, stops, result


def benchmark_weights(panel, kind):
    weights = pd.DataFrame(0.0, index=panel.dates, columns=panel.close.columns)
    if kind == "spy":
        weights[MARKET] = 0.999
    elif kind == "equal_weight_etfs":
        names = [t for t in CORE_ETFS if t in panel.close.columns]
        available = panel.close[names].notna()
        weights[names] = available.div(available.sum(axis=1), axis=0) * 0.999
        weights = hold_between_rebalances(weights, "M")
    return weights


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value.date())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _finite(value):
    """Recursively replace NaN/inf with None so the JSON is strict (browsers reject NaN)."""
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def write_json(path, payload):
    write_atomically(path, json.dumps(_finite(payload), indent=2, default=_json_default, allow_nan=False).encode())


def _frame_to_csv(frame, path):
    write_atomically(path, frame.to_csv(index=True).encode())


# ------------------------------------------------------------------ full-period backtests
def backtests(panel, out_dir):
    eval_start = panel.dates[WARMUP]
    results, curves, sim_by_label = {}, {}, {}
    for label, strategy in (("core_tsmom", TimeSeriesMomentum()), ("overlay_meanrev", RegimeMeanReversion()),
                            ("combined", CombinedStrategy())):
        log(f"backtest {label}")
        weights, stops, result = run_strategy(panel, strategy, start=panel.dates[WARMUP - 1])
        results[label] = {"config": strategy.describe(), **summarize(result, eval_start),
                          "latestDiagnostics": strategy.last_diagnostics}
        curves[label] = result.equity
        sim_by_label[label] = result
        _frame_to_csv(result.trades, out_dir / f"trades_{label}.csv")
        _frame_to_csv(result.decisions, out_dir / f"decisions_{label}.csv")
    for label in ("spy", "equal_weight_etfs"):
        result = simulate(panel, benchmark_weights(panel, label), start=panel.dates[WARMUP - 1])
        results[f"benchmark_{label}"] = summarize(result, eval_start)
        curves[f"benchmark_{label}"] = result.equity
        sim_by_label[f"benchmark_{label}"] = result
    _frame_to_csv(pd.DataFrame(curves), out_dir / "equity_curves.csv")
    return {"evaluationStart": str(eval_start.date()), "results": results}, sim_by_label


# ------------------------------------------------------------------ walk-forward
def _expand(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _sharpe(returns):
    returns = returns.dropna()
    return float(returns.mean() / returns.std() * math.sqrt(TRADING_DAYS)) if len(returns) > 5 and returns.std() > 0 else float("-inf")


def walk_forward(panel, family, grid=None, train_min=252, test_len=63):
    """Expanding-window walk-forward. Each parameter combination is simulated once over the whole history (the
    strategies are causal, so a row only depends on earlier data); for each fold the combination with the best
    training-window Sharpe is selected and its target weights are used for the next test window. The stitched
    out-of-sample weights are re-simulated end to end, so switching costs are included. Every trial is logged."""
    combos = _expand(grid or GRIDS[family])
    dates = panel.dates
    n = len(dates)
    sims = []
    for number, params in enumerate(combos):
        log(f"walk-forward {family} trial {number + 1}/{len(combos)} {params}")
        strategy = make_strategy(family, params)
        weights, stops, result = run_strategy(panel, strategy, start=dates[WARMUP - 1])
        sims.append({"params": params, "weights": weights, "stops": stops, "returns": result.returns,
                     "sleeve_of": getattr(strategy, "sleeve_of", None)})
    starts = list(range(train_min, n, test_len))
    trials, chosen = [], []
    stitched = pd.DataFrame(0.0, index=dates, columns=panel.close.columns)
    stitched_stops = pd.DataFrame(np.nan, index=dates, columns=panel.close.columns)
    for fold, start in enumerate(starts):
        end = min(start + test_len, n)
        train = slice(dates[WARMUP], dates[start - 1])
        test = slice(dates[start], dates[end - 1])
        scores = [_sharpe(sim["returns"].loc[train]) for sim in sims]
        tests = [_sharpe(sim["returns"].loc[test]) for sim in sims]
        best = int(np.argmax(scores))
        chosen.append(best)
        for number, sim in enumerate(sims):
            trials.append({"family": family, "fold": fold, "trainStart": str(dates[WARMUP].date()),
                           "trainEnd": str(dates[start - 1].date()), "testStart": str(dates[start].date()),
                           "testEnd": str(dates[end - 1].date()), "trial": number,
                           "params": json.dumps(sim["params"]), "trainSharpe": scores[number],
                           "testSharpe": tests[number],
                           "testReturn": float((1 + sim["returns"].loc[test]).prod() - 1), "selected": number == best})
        rows = dates[start - 1:end - 1]          # decisions at these closes fill inside the test window
        stitched.loc[rows] = sims[best]["weights"].reindex(index=rows, columns=stitched.columns).fillna(0.0).to_numpy()
        if sims[best]["stops"] is not None:
            stitched_stops.loc[rows] = sims[best]["stops"].reindex(index=rows, columns=stitched.columns).to_numpy()
    first_test = dates[starts[0]]
    oos = simulate(panel, stitched, stitched_stops, start=dates[starts[0] - 1])
    trial_frame = pd.DataFrame(trials)
    stability = {"selectedTrialByFold": [json.dumps(sims[c]["params"]) for c in chosen],
                 "distinctSelections": len(set(chosen))}
    rank_corr = []
    for fold, group in trial_frame.groupby("fold"):
        finite = group.replace([np.inf, -np.inf], np.nan).dropna(subset=["trainSharpe", "testSharpe"])
        if len(finite) > 2:
            rank_corr.append(float(finite["trainSharpe"].rank().corr(finite["testSharpe"].rank())))
    stability["trainTestRankCorrelationByFold"] = rank_corr
    value_freq = {}
    for c in chosen:
        for key, value in sims[c]["params"].items():
            value_freq.setdefault(key, {}).setdefault(str(value), 0)
            value_freq[key][str(value)] += 1
    stability["parameterValueSelections"] = value_freq
    median_test = trial_frame.groupby("fold")["testSharpe"].median().tolist()
    return {"family": family, "trials": len(combos), "folds": len(starts), "trainMinSessions": train_min,
            "testSessions": test_len, "oosStart": str(first_test.date()),
            "outOfSample": summarize(oos, first_test), "parameterStability": stability,
            "medianTrialTestSharpeByFold": median_test}, trial_frame, oos


# ------------------------------------------------------------------ multi-horizon trend runner
def multi_horizon(panel):
    rows = {}
    common_start = panel.dates[max(HORIZONS) + 60]
    for horizon in HORIZONS:
        strategy = TimeSeriesMomentum(horizons=(horizon,), rebalance="D" if horizon <= 5 else "W-FRI")
        log(f"multi-horizon trend {horizon}d")
        _, _, result = run_strategy(panel, strategy, start=panel.dates[min(horizon, 200) + 9])
        own_start = panel.dates[min(horizon + 60, len(panel.dates) - 2)]
        rows[str(horizon)] = {
            "ownPeriod": summarize(result, own_start)["performance"],
            "commonPeriod": performance(result.returns.loc[common_start:], result.equity,
                                        result.turnover, result.gross_exposure, result.costs),
            "independentSignalPeriods": int((len(panel.dates) - horizon) // horizon),
            "tradeStats": summarize(result, own_start)["tradeStats"],
        }
    return {"commonPeriodStart": str(common_start.date()), "byHorizon": rows,
            "note": "Single-horizon long/flat trend on the core ETF universe with the same sizing, caps and costs."}


# ------------------------------------------------------------------ normality diagnostics
def normality(panel, vol_window=60, min_dollar_volume=20e6, min_price=5.0):
    """Skew / excess kurtosis / Jarque-Bera by horizon for non-overlapping h-day log returns of liquid names,
    raw and standardized by each name's ex-ante (trailing) volatility, plus SPY alone."""
    classes = panel.classes.reindex(panel.close.columns)
    dv = (panel.close * panel.volume).median()
    liquid = [t for t in panel.close.columns if classes.get(t) in ("stock", "etf") and dv.get(t, 0) > min_dollar_volume
              and panel.close[t].median() > min_price]
    logs = panel.log_returns()[liquid]
    values = logs.to_numpy()
    sigma = logs.rolling(vol_window, min_periods=vol_window - 10).std().to_numpy()
    n = len(logs)
    table = []
    for horizon in NORMALITY_HORIZONS:
        starts = np.arange(vol_window, n - horizon + 1, horizon)
        raw, z = [], []
        for s in starts:
            block = values[s:s + horizon]           # returns on rows s .. s+h-1 (start row s-1 close -> s+h-1 close)
            total = block.sum(axis=0)
            ok = ~np.isnan(block).any(axis=0) & np.isfinite(sigma[s - 1]) & (sigma[s - 1] > 0)
            raw.append(total[ok])
            z.append(total[ok] / (sigma[s - 1][ok] * math.sqrt(horizon)))
        raw = np.concatenate(raw) if raw else np.array([])
        z = np.concatenate(z) if z else np.array([])
        if len(z) < 30:
            continue
        mz, mr = moments(z), moments(raw)
        jb, p = jarque_bera(z)
        spy_total = (np.log1p(panel.total_return()[MARKET])).rolling(horizon).sum().dropna()
        spy_moments = moments(spy_total.to_numpy()) if len(spy_total) > 10 else {}
        table.append({
            "horizonDays": horizon, "observations": int(len(z)), "independentPeriods": int(len(starts)),
            "names": len(liquid),
            "standardized": {"skew": mz["skew"], "excessKurtosis": mz["excessKurtosis"], "std": mz["std"],
                             "jarqueBera": jb, "jbPValue": p,
                             "beyond2SigmaPct": float((np.abs(z) > 2).mean() * 100),
                             "beyond3SigmaPct": float((np.abs(z) > 3).mean() * 100),
                             "q01": float(np.quantile(z, 0.01)), "q99": float(np.quantile(z, 0.99))},
            "rawPooled": {"skew": mr["skew"], "excessKurtosis": mr["excessKurtosis"], "std": mr["std"]},
            "spyOverlapping": {"windows": len(spy_total), "skew": spy_moments.get("skew"),
                               "excessKurtosis": spy_moments.get("excessKurtosis")},
        })
    return {"normalReference": {"beyond2SigmaPct": 4.55, "beyond3SigmaPct": 0.27, "q01": -2.326, "q99": 2.326},
            "table": table,
            "method": ("Non-overlapping h-day log total returns of liquid stocks and ETFs (median dollar volume > "
                       "$20M, median price > $5), standardized by each name's trailing 60-session daily volatility "
                       "x sqrt(h). Observations are pooled across names; 'independentPeriods' counts distinct time "
                       "blocks, which is the real sample size for market-wide effects. SPY uses overlapping windows "
                       "(descriptive only).")}


# ------------------------------------------------------------------ report
def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _num(x, digits=2):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{digits}f}"


def markdown(summary):
    lines = [f"# Research report ({summary['generatedAt']})", "",
             f"Data: {summary['data']['start']} to {summary['data']['end']}, {summary['data']['sessions']} sessions, "
             f"{summary['data']['tickers']} tickers loaded.", ""]
    if "backtests" in summary:
        b = summary["backtests"]
        lines += [f"## Full-period backtests (metrics from {b['evaluationStart']}, after warm-up; in-sample defaults)", "",
                  "| Strategy | CAGR | Vol | Sharpe | Sortino | Max DD | Calmar | Turnover/yr | Avg gross | Cost bps/yr | Trades | Win rate |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in b["results"].items():
            p, t = r["performance"], r["tradeStats"]
            lines.append(f"| {name} | {_pct(p.get('cagr'))} | {_pct(p.get('volatility'))} | {_num(p.get('sharpe'))} | "
                         f"{_num(p.get('sortino'))} | {_pct(p.get('maxDrawdown'))} | {_num(p.get('calmar'))} | "
                         f"{_num(p.get('turnoverPerYear'), 1)} | {_num(p.get('avgGrossExposure'))} | "
                         f"{_num(p.get('costBpsPerYear'), 0)} | {t.get('trades', 0)} | {_pct(t.get('winRate'))} |")
        lines.append("")
    if "walkForward" in summary:
        lines += ["## Walk-forward (out-of-sample only)", "",
                  "| Family | OOS start | Trials | Folds | CAGR | Sharpe | Max DD | Turnover/yr | Distinct picks | Train/test rank corr |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, w in summary["walkForward"].items():
            p = w["outOfSample"]["performance"]
            rc = w["parameterStability"]["trainTestRankCorrelationByFold"]
            lines.append(f"| {name} | {w['oosStart']} | {w['trials']} | {w['folds']} | {_pct(p.get('cagr'))} | "
                         f"{_num(p.get('sharpe'))} | {_pct(p.get('maxDrawdown'))} | {_num(p.get('turnoverPerYear'), 1)} | "
                         f"{w['parameterStability']['distinctSelections']} | {', '.join(_num(x) for x in rc)} |")
        lines.append("")
    if "attribution" in summary:
        from attribution import markdown_section
        section = markdown_section(summary["attribution"]).rstrip()
        if section:
            lines += section.splitlines() + [""]
    if "multiHorizon" in summary:
        m = summary["multiHorizon"]
        lines += [f"## Multi-horizon trend (common period from {m['commonPeriodStart']})", "",
                  "| Horizon | Independent periods | Sharpe (own) | Max DD (own) | Sharpe (common) | CAGR (common) | Turnover/yr |",
                  "|---|---|---|---|---|---|---|"]
        for h, r in m["byHorizon"].items():
            o, c = r["ownPeriod"], r["commonPeriod"]
            lines.append(f"| {h}d | {r['independentSignalPeriods']} | {_num(o.get('sharpe'))} | {_pct(o.get('maxDrawdown'))} | "
                         f"{_num(c.get('sharpe'))} | {_pct(c.get('cagr'))} | {_num(o.get('turnoverPerYear'), 1)} |")
        lines.append("")
    if "normality" in summary:
        lines += ["## Normality by horizon (vol-standardized, pooled)", "",
                  "| h | Obs | Indep. periods | Skew | Excess kurt | JB p | >2σ % (4.55) | >3σ % (0.27) | 1% quantile (-2.33) | SPY skew | SPY ex.kurt |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in summary["normality"]["table"]:
            s, spy = row["standardized"], row["spyOverlapping"]
            lines.append(f"| {row['horizonDays']} | {row['observations']} | {row['independentPeriods']} | {_num(s['skew'])} | "
                         f"{_num(s['excessKurtosis'])} | {s['jbPValue']:.2g} | {_num(s['beyond2SigmaPct'])} | "
                         f"{_num(s['beyond3SigmaPct'])} | {_num(s['q01'])} | {_num(spy['skew'])} | {_num(spy['excessKurtosis'])} |")
        lines.append("")
    lines += ["## Caveats", ""] + [f"- {c}" for c in summary["caveats"]]
    return "\n".join(lines) + "\n"


CAVEATS = [
    "Only ~2 years of data: Sharpe ratio standard errors are about +/-0.7 to 1.0 over one year of out-of-sample data.",
    "100- and 200-day horizons have 2-4 independent periods; their results are descriptive, not validated.",
    "Universe pre-screen uses average dollar volume over the loaded window (mild look-ahead in which tickers are "
    "loaded); daily trading eligibility is causal (trailing 20-day median dollar volume, price > $5).",
    "Tickers with unresolved jump-reversal / split-basis anomalies are excluded (survivorship-neutral but reduces breadth).",
    "Costs are modelled, not measured: Schwab profile ($0 commission on US-listed stocks/ETFs) + half-spread 1-100 bps "
    "by dollar volume + square-root impact + sell-side SEC fee ($20.60/$1M) and FINRA TAF ($0.000195/share, max $9.79).",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["all", "backtest", "walkforward", "multihorizon", "normality", "attribution", "tax", "regime-allocation", "prob", "dxy_forecast"])
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    from pathlib import Path
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "attribution":
        from attribution import main as attribution_main
        attribution_main(["--out", str(out_dir)])
        return
    if args.command == "prob":
        from prob_walkforward import main as prob_main
        return prob_main([])
    if args.command == "regime-allocation":
        from regime_allocation import research as alloc_research
        dest = out_dir if out_dir.name == "regime_allocation" else out_dir / "regime_allocation"
        alloc_research(start=args.start, end=args.end, out_dir=dest)
        return
    if args.command == "dxy_forecast":
        from dxy_forecast import run as dxy_forecast_run
        return dxy_forecast_run(write=True)
    started = time.time()
    log("loading panel")
    panel = load_panel(args.start, args.end)
    summary_path = out_dir / "research_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() and args.command != "all" else {}
    summary.update({"generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "data": {"start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date()),
                             "sessions": len(panel.dates), "tickers": int(panel.close.shape[1])},
                    "costModel": CostModel().as_dict(), "caveats": merge_caveats(CAVEATS, getattr(panel, "warnings", None))})
    sim_results, walkforward_oos = {}, {}
    if args.command in ("all", "backtest"):
        summary["backtests"], sim_results = backtests(panel, out_dir)
    if args.command in ("all", "walkforward"):
        summary["walkForward"] = {}
        frames = []
        for family in ("tsmom", "meanrev", "combined"):
            result, trials, oos = walk_forward(panel, family)
            summary["walkForward"][family] = result
            walkforward_oos[family] = oos
            frames.append(trials)
            _frame_to_csv(oos.decisions, out_dir / f"walkforward_{family}_oos_decisions.csv")
            _frame_to_csv(pd.DataFrame({"equity": oos.equity, "return": oos.returns}),
                          out_dir / f"walkforward_{family}_oos_equity.csv")
        _frame_to_csv(pd.concat(frames, ignore_index=True), out_dir / "walkforward_trials.csv")
    if args.command in ("all", "backtest", "walkforward"):
        from attribution import run as run_attribution
        log("return attribution")
        summary["attribution"] = run_attribution(out_dir, panel=panel)
    if args.command in ("all", "multihorizon"):
        summary["multiHorizon"] = multi_horizon(panel)
    if args.command in ("all", "normality"):
        log("normality diagnostics")
        summary["normality"] = normality(panel)
    if args.command in ("all", "tax"):
        log("pre-tax vs tax-aware comparison")
        from tax_research import compare as tax_compare
        summary["tax"] = tax_compare(panel, out_dir=out_dir)["comparison"]
    summary["runSeconds"] = round(time.time() - started, 1)
    write_json(summary_path, summary)
    write_atomically(out_dir / "research_report.md", markdown(summary).encode())
    log(f"wrote {summary_path} and research_report.md in {summary['runSeconds']}s")
    return summary


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("research"):
        main()
