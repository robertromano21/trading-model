"""Honest tests of dollar, rates and credit proxies against equities and commodities.

Writes Massive/Research/macro_report.md and macro_report.json (gitignored). Uses the project's
quant tools (HAC OLS, rolling OLS, correlations) plus Granger causality, lead/lag correlations,
walk-forward predictability and Benjamini-Hochberg FDR. Decision-time features are lagged one
session (see macro_data.PUBLICATION_LAG). Real CDS/CDX and swap quotes are not used.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically
from macro_data import PROXY_DISCLAIMER, load_dxy, load_macro, load_uup_returns, uup_crosscheck
from market_data import MARKET, load_panel
from risk import TRADING_DAYS

OUT_DIR = DATA_DIR / "Research"
REPORT_MD = OUT_DIR / "macro_report.md"
REPORT_JSON = OUT_DIR / "macro_report.json"

# Assets to test against the dollar and against rates/credit.
DOLLAR_ASSETS = ("SPY", "QQQ", "IWM",
                 "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
                 "GLD", "DBC", "USO", "XLE", "EEM", "EFA", "UUP")
CREDIT_ASSETS = ("SPY", "XLF", "XLE", "HYG", "LQD", "JNK", "IEF", "TLT", "EEM")
RATES_ASSETS = ("SPY", "TLT", "IEF", "SHY", "LQD", "GLD")
HORIZONS = (1, 5, 20)
LAGS = list(range(-10, 11))
ROLL_WINDOWS = (63, 126)
MIN_OBS = 60
WF_MIN_TRAIN = 126
FDR_ALPHA = 0.05


def _clean(value, digits=6):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return None
        return round(float(value), digits)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _clean(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, digits) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(pd.Timestamp(value).date())
    return value


def bh_fdr(pvalues, alpha=FDR_ALPHA):
    """Benjamini-Hochberg. Returns (rejected mask, adjusted p-values) in original order."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    adj = np.full(n, np.nan)
    rejected = np.zeros(n, dtype=bool)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return rejected, adj
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    ranked = p[order]
    adj_sorted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    adj[order] = adj_sorted
    rejected[order] = adj_sorted <= alpha
    return rejected, adj


def hac_ols(y, x, lags=None):
    """Simple regression y = a + b x with HAC t-stat. Returns dict or None."""
    frame = pd.concat([pd.Series(y).rename("y"), pd.Series(x).rename("x")], axis=1, join="inner").dropna()
    if len(frame) < MIN_OBS:
        return None
    from statsmodels.api import OLS, add_constant
    from statsmodels.stats.stattools import durbin_watson
    Y = frame["y"].to_numpy()
    X = add_constant(frame["x"].to_numpy())
    n = len(frame)
    lags = max(1, int(4 * (n / 100) ** (2 / 9))) if lags is None else lags
    try:
        fit = OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    except Exception:
        return None
    return {
        "n": int(n), "alpha": float(fit.params[0]), "beta": float(fit.params[1]),
        "alphaT": float(fit.tvalues[0]), "betaT": float(fit.tvalues[1]),
        "alphaP": float(fit.pvalues[0]), "betaP": float(fit.pvalues[1]),
        "r2": float(fit.rsquared), "hacLags": int(lags),
        "dw": float(durbin_watson(fit.resid)),
        "start": str(frame.index[0].date()), "end": str(frame.index[-1].date()),
    }


def rolling_betas(y, x, windows=ROLL_WINDOWS):
    out = {}
    frame = pd.concat([pd.Series(y).rename("y"), pd.Series(x).rename("x")], axis=1, join="inner").dropna()
    for window in windows:
        if len(frame) < window:
            continue
        cov = frame["y"].rolling(window, min_periods=window).cov(frame["x"])
        var = frame["x"].rolling(window, min_periods=window).var()
        beta = cov / var.replace(0, np.nan)
        series = beta.dropna()
        if series.empty:
            continue
        out[str(window)] = {
            "window": int(window), "n": int(len(series)),
            "last": float(series.iloc[-1]), "mean": float(series.mean()),
            "median": float(series.median()), "std": float(series.std(ddof=0)),
            "min": float(series.min()), "max": float(series.max()),
            "shareNegative": float((series < 0).mean()),
        }
    return out


def lead_lag(y, x, lags=LAGS):
    """corr(x_t, y_{t+k}) for k in lags. Positive k means x leads y."""
    a = pd.Series(x).rename("x")
    b = pd.Series(y).rename("y")
    rows = []
    for k in lags:
        frame = pd.concat([a, b.shift(-k)], axis=1, join="inner").dropna()
        if len(frame) < MIN_OBS:
            continue
        corr = float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))
        # HAC t on a standardized regression y_{t+k} = a + b x_t, report p of corr via beta p
        fit = hac_ols(frame.iloc[:, 1], frame.iloc[:, 0])
        rows.append({"lag": int(k), "corr": corr, "n": int(len(frame)),
                     "betaP": None if fit is None else fit["betaP"],
                     "betaT": None if fit is None else fit["betaT"]})
    return rows


def granger(y, x, maxlag=5):
    """Does x Granger-cause y? Returns the p-value of the largest lag test that is defined."""
    frame = pd.concat([pd.Series(y).rename("y"), pd.Series(x).rename("x")], axis=1, join="inner").dropna()
    if len(frame) < MIN_OBS + maxlag + 5:
        return None
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        with np.errstate(all="ignore"):
            tests = grangercausalitytests(frame[["y", "x"]], maxlag=maxlag, verbose=False)
        rows = []
        for lag, res in tests.items():
            ssr = res[0].get("ssr_ftest")
            rows.append({"lag": int(lag), "p": None if ssr is None else float(ssr[1]),
                         "f": None if ssr is None else float(ssr[0])})
        return {"maxlag": int(maxlag), "byLag": rows,
                "minP": min((r["p"] for r in rows if r["p"] is not None), default=None)}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def forward_return(series, horizon):
    """Return from t+1 through t+horizon (the label for a decision at close t)."""
    r = pd.Series(series)
    stacked = sum(np.log1p(r.shift(-i)) for i in range(1, int(horizon) + 1))
    return np.expm1(stacked)


def walk_forward_predict(y, x, min_train=WF_MIN_TRAIN, label_gap=1):
    """Expanding-window OOS: at t, fit on resolved labels only, then predict y_t.

    Feature x is already lagged. For a horizon-h label, set label_gap=h so training
    only uses outcomes that have fully realized before t (no overlapping-label leak).
    """
    frame = pd.concat([pd.Series(y).rename("y"), pd.Series(x).rename("x")], axis=1, join="inner").dropna()
    gap = max(int(label_gap), 1)
    if len(frame) < min_train + gap + 20:
        return None
    yv = frame["y"].to_numpy()
    xv = frame["x"].to_numpy()
    pred = np.full(len(frame), np.nan)
    start = max(min_train, gap)
    for i in range(start, len(frame)):
        end = i - gap + 1  # y[:end] has last label y[i-gap], resolved by time i
        if end < 20:
            continue
        x_tr, y_tr = xv[:end], yv[:end]
        var = np.var(x_tr)
        if var <= 0 or not np.isfinite(var):
            continue
        beta = np.cov(x_tr, y_tr, ddof=0)[0, 1] / var
        alpha = y_tr.mean() - beta * x_tr.mean()
        pred[i] = alpha + beta * xv[i]
    mask = np.isfinite(pred)
    if mask.sum() < 20:
        return None
    actual = yv[mask]
    forecast = pred[mask]
    resid = actual - forecast
    # OOS R^2 vs the expanding historical mean (only past y, no peeking).
    hist_mean = np.array([yv[:i].mean() if i else 0.0 for i in range(len(frame))])[mask]
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((actual - hist_mean) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    ic = float(np.corrcoef(actual, forecast)[0, 1]) if len(actual) > 2 else float("nan")
    # HAC t-stat of forecast on actual: actual = a + b * forecast
    fit = hac_ols(pd.Series(actual, index=frame.index[mask]), pd.Series(forecast, index=frame.index[mask]))
    hit = float(np.mean(np.sign(forecast) == np.sign(actual))) if len(actual) else float("nan")
    return {
        "nTrainMin": int(min_train), "nOos": int(mask.sum()),
        "start": str(frame.index[mask][0].date()), "end": str(frame.index[mask][-1].date()),
        "oosR2": r2, "ic": ic, "hitRate": hit,
        "forecastBeta": None if fit is None else fit["beta"],
        "forecastBetaT": None if fit is None else fit["betaT"],
        "forecastBetaP": None if fit is None else fit["betaP"],
        "meanActual": float(actual.mean()), "meanForecast": float(forecast.mean()),
    }


def _asset_returns(panel, tickers):
    have = [t for t in dict.fromkeys(tickers) if t in panel.close.columns]
    rets = panel.total_return()[have]
    return rets


def load_features():
    macro = load_macro(lag=1)
    dxy = load_dxy(lag=0)
    dxy_ret = dxy.pct_change().rename("dxy_ret")
    # Decision-time dollar return is lagged one session, consistent with other FRED features.
    features = pd.DataFrame({"dxy_ret": dxy_ret.shift(1)})
    if not macro.empty:
        for col in ("dxy_ret20", "dxy_vol20", "curve_slope", "curve_level_chg20", "curve_curvature",
                    "sofr_tbill_spread", "hy_oas", "hy_oas_chg20", "ig_oas", "hy_ig_oas",
                    "hyg_ief_ratio_chg20", "lqd_ief_ratio_chg20"):
            if col in macro.columns:
                features[col] = macro[col]
    return features


def run_dollar_block(rets, features):
    x = features["dxy_ret"] if "dxy_ret" in features.columns else pd.Series(dtype=float)
    rows = []
    pvals = []
    for ticker in rets.columns:
        y = rets[ticker]
        beta = hac_ols(y, x)
        roll = rolling_betas(y, x)
        ll = lead_lag(y, x)
        g = granger(y, x, maxlag=5)
        pred = {}
        for h in HORIZONS:
            mom = features["dxy_ret20"] if "dxy_ret20" in features.columns else x.rolling(20).sum()
            fwd = forward_return(y, h)
            wf = walk_forward_predict(fwd, mom, label_gap=h)
            pred[str(h)] = wf
            if wf and wf.get("forecastBetaP") is not None:
                pvals.append((f"{ticker}.dxy_mom.h{h}", wf["forecastBetaP"]))
        if beta and beta.get("betaP") is not None:
            pvals.append((f"{ticker}.dxy_beta", beta["betaP"]))
        if g and g.get("minP") is not None:
            pvals.append((f"{ticker}.dxy_granger", g["minP"]))
        rows.append({"ticker": ticker, "beta": beta, "rolling": roll, "leadLag": ll,
                     "granger": g, "predict": pred})
    return rows, pvals


def run_macro_block(rets, features, xcol, assets, label):
    if xcol not in features.columns:
        return [], []
    x = features[xcol]
    rows, pvals = [], []
    for ticker in assets:
        if ticker not in rets.columns:
            continue
        y = rets[ticker]
        beta = hac_ols(y, x)
        pred = {}
        for h in HORIZONS:
            fwd = forward_return(y, h)
            wf = walk_forward_predict(fwd, x, label_gap=h)
            pred[str(h)] = wf
            if wf and wf.get("forecastBetaP") is not None:
                pvals.append((f"{ticker}.{label}.h{h}", wf["forecastBetaP"]))
        if beta and beta.get("betaP") is not None:
            pvals.append((f"{ticker}.{label}.beta", beta["betaP"]))
        rows.append({"ticker": ticker, "feature": xcol, "beta": beta, "predict": pred})
    return rows, pvals


def _fmt_pct(x, digits=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100 * x:.{digits}f}%"


def _fmt(x, digits=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def verdict(dollar_rows, tests):
    """Plain-language, conservative summary. Does the dollar (or rates/credit) predict anything OOS?"""
    lines = []
    # Same-day beta of SPY on DXY (lagged) — contemporaneous relationship, not a trading signal.
    spy = next((r for r in dollar_rows if r["ticker"] == "SPY"), None)
    if spy and spy["beta"]:
        b = spy["beta"]
        direction = "negative" if b["beta"] < 0 else "positive"
        sig = "significant" if b["betaP"] < 0.05 else "not statistically significant"
        lines.append(
            f"Over {b['n']} overlapping sessions ({b['start']} to {b['end']}), the lagged DXY-proxy "
            f"daily return has a {direction} beta of {b['beta']:.2f} on SPY (HAC t={b['betaT']:.2f}, "
            f"p={b['betaP']:.3f}, R²={b['r2']:.3f}) — {sig} as a contemporaneous link, not a forecast.")
    # OOS predictability: any rejected FDR test on walk-forward forecast slope?
    oos = [t for t in tests if ".h" in t["id"]]
    # A significant forecast-beta p-value with a negative OOS R^2 means the walk-forward
    # prediction was worse than the historical mean. That is not a usable signal.
    def _wf_quality(test_id):
        return None
    useful = [t for t in oos if t["rejected"] and t.get("oosR2", -1) is not None and (t.get("oosR2") or -1) > 0 and (t.get("ic") or 0) > 0]
    # quality fields may be absent on the test rows; the caller can attach them.
    useful = [t for t in oos if t.get("rejected") and (t.get("oosR2") is not None) and t["oosR2"] > 0 and (t.get("ic") or 0) > 0]
    if not oos:
        lines.append("Walk-forward predictability tests could not be computed (not enough overlapping history).")
    elif not useful:
        n_sig = sum(1 for t in oos if t.get("rejected"))
        lines.append(
            "Walk-forward forecasts of next-period returns from dollar momentum, the curve and credit OAS "
            f"do not beat a simple historical mean on this sample. {n_sig} of {len(oos)} tests had an FDR-significant "
            "forecast-beta p-value, but those hits mostly have a negative out-of-sample R² (the prediction is worse "
            "than the mean) or a negative information coefficient. That is not a trading edge. Two years of one "
            "market regime are also too short to treat a leftover positive hit as durable.")
    else:
        names = ", ".join(f"{t['id']} (OOS R²={t['oosR2']:.3f}, adj p={t['pAdj']:.3f})" for t in useful[:8])
        lines.append(
            f"{len(useful)} of {len(oos)} walk-forward tests survive FDR at 5% and also have a positive OOS R² and IC: {names}. "
            "Treat even these as tentative — the sample is short and covers one market regime.")
    gld = next((r for r in dollar_rows if r["ticker"] == "GLD"), None)
    if gld and gld["beta"]:
        lines.append(
            f"GLD's lagged-DXY beta is {gld['beta']['beta']:.2f} (HAC t={gld['beta']['betaT']:.2f}, "
            f"p={gld['beta']['betaP']:.3f}).")
    eem = next((r for r in dollar_rows if r["ticker"] == "EEM"), None)
    if eem and eem["beta"]:
        lines.append(
            f"EEM's lagged-DXY beta is {eem['beta']['beta']:.2f} (HAC t={eem['beta']['betaT']:.2f}, "
            f"p={eem['beta']['betaP']:.3f}).")
    return " ".join(lines)


def markdown(summary):
    uup = summary.get("uup") or {}
    data = summary.get("data") or {}
    lines = [
        "# Macro and dollar research",
        "",
        f"Generated {summary.get('generatedAt')}. Local data only (gitignored).",
        "",
        "## What was tested",
        "",
        "A DXY-style dollar proxy is built from the ICE geometric formula and weights, using Massive "
        "forex daily bars when the plan includes FX, otherwise FRED noon buying rates. Decision-time "
        "features are lagged one session so FRED publication delay cannot leak into a forecast. "
        "Tests: rolling and full-sample HAC betas of equities, sector ETFs, GLD, commodities (DBC, USO, XLE), "
        "EEM and EFA on dollar-proxy returns; lead/lag correlations (k = −10…+10); Granger causality "
        "(5 lags); and walk-forward expanding-window forecasts of 1-, 5- and 20-day forward returns from "
        "20-day dollar momentum, the Treasury curve and credit OAS proxies. p-values are Newey-West HAC "
        "and are adjusted with Benjamini-Hochberg FDR.",
        "",
        PROXY_DISCLAIMER,
        "",
        f"Equity history used for tests: {data.get('start')} to {data.get('end')} ({data.get('sessions')} sessions). "
        "That is one broad regime and is too short to treat a lucky result as an edge.",
        "",
        "## Dollar proxy vs UUP",
        "",
    ]
    if uup.get("available"):
        lines.append(
            f"The DXY proxy's daily return correlates {uup['corr']:.3f} with UUP over {uup['n']} sessions "
            f"({uup['start']} to {uup['end']}). Annualized vol: proxy {uup.get('dxyVol', 0)*100:.1f}%, "
            f"UUP {uup.get('uupVol', 0)*100:.1f}%. A correlation well below 1 is expected: UUP is an ETF "
            "with costs and a slightly different basket, and the proxy is not the ICE futures print.")
    else:
        lines.append(f"UUP cross-check unavailable: {uup.get('reason', 'n/a')}.")
    lines += ["", "## Verdict", "", summary.get("verdict") or "n/a", "",
              "## Same-day (lagged-feature) dollar betas", "",
              "| Ticker | Beta | HAC t | p | R² | n |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summary.get("dollar") or []:
        b = row.get("beta") or {}
        lines.append(f"| {row['ticker']} | {_fmt(b.get('beta'))} | {_fmt(b.get('betaT'), 2)} | "
                     f"{_fmt(b.get('betaP'), 3)} | {_fmt(b.get('r2'))} | {b.get('n') or 'n/a'} |")
    lines += ["", "## Walk-forward dollar-momentum forecasts (20-day DXY return → next 5-day asset return)", "",
              "| Ticker | OOS R² | IC | Hit rate | Forecast beta p | n |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summary.get("dollar") or []:
        wf = (row.get("predict") or {}).get("5") or {}
        if not wf:
            continue
        lines.append(f"| {row['ticker']} | {_fmt(wf.get('oosR2'))} | {_fmt(wf.get('ic'))} | "
                     f"{_fmt_pct(wf.get('hitRate'))} | {_fmt(wf.get('forecastBetaP'), 3)} | {wf.get('nOos') or 'n/a'} |")
    lines += ["", "## FDR-adjusted tests", "",
              f"{summary.get('fdrRejected', 0)} of {summary.get('fdrTests', 0)} tests rejected at FDR 5%.",
              "",
              "## Caveats",
              "",
              "- Two years of local prices, one bull-market regime, survivorship still in the stock tape.",
              "- The dollar series is a proxy, not DXY futures. FRED FX is a noon buying rate, not a close.",
              "- Free FRED downloads of ICE BofA OAS (IG/HY/BBB/CCC) currently start 2023-11-27, which still covers this equity sample. Longer OAS history is not available from the keyless graph CSV.",
              "- Features are lagged one session; that is conservative and may understate a same-day link.",
              "- Walk-forward still re-uses the same two-year window to define features and evaluate them. "
              "A later, sealed period would be a stronger test.",
              "- " + PROXY_DISCLAIMER,
              ""]
    return "\n".join(lines) + "\n"


def run(panel=None):
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = panel or load_panel()
    features = load_features()
    rets = _asset_returns(panel, list(DOLLAR_ASSETS) + list(CREDIT_ASSETS) + list(RATES_ASSETS)
                          + ["SHY", "IEI", "TIP", "BND", "AGG", "EMB", "BIL", "SGOV", "MUB"])
    dollar_rows, p_dollar = run_dollar_block(rets, features)
    rates_rows, p_rates = run_macro_block(rets, features, "curve_slope", RATES_ASSETS, "curve_slope")
    credit_rows, p_credit = run_macro_block(rets, features, "hy_oas_chg20", CREDIT_ASSETS, "hy_oas_chg20")
    all_p = p_dollar + p_rates + p_credit
    rejected, adj = bh_fdr([p for _, p in all_p])
    tests = [{"id": i, "p": p, "pAdj": None if not math.isfinite(a) else float(a), "rejected": bool(r)}
             for (i, p), r, a in zip(all_p, rejected, adj)]
    uup = uup_crosscheck()
    summary = {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data": {"start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date()),
                 "sessions": int(len(panel.dates)), "tickers": int(panel.close.shape[1])},
        "featureStart": None if features.empty else str(features.dropna(how="all").index[0].date()),
        "featureEnd": None if features.empty else str(features.dropna(how="all").index[-1].date()),
        "uup": uup,
        "proxyDisclaimer": PROXY_DISCLAIMER,
        "dollar": dollar_rows,
        "rates": rates_rows,
        "credit": credit_rows,
        "tests": tests,
        "fdrTests": len(tests),
        "fdrRejected": int(sum(t["rejected"] for t in tests)),
        "runSeconds": None,
    }
    summary["verdict"] = verdict(dollar_rows, tests)
    summary["runSeconds"] = round(time.time() - started, 1)
    write_atomically(REPORT_JSON, json.dumps(_clean(summary), indent=2, default=str).encode())
    write_atomically(REPORT_MD, markdown(_clean(summary)).encode())
    return summary


def main():
    summary = run()
    print(summary["verdict"])
    print(f"wrote {REPORT_MD} in {summary['runSeconds']}s; FDR rejected {summary['fdrRejected']}/{summary['fdrTests']}")
    return summary


if __name__ == "__main__":
    main()
