"""Quant tools for the AI analyst: a fixed menu of statistics / econometrics / backtest operations.

JSON in -> JSON out. There is NO code execution: the request names one operation from OPS and passes plain
arguments, which are validated and bounded. Used by server/ai/tools.js (one short-lived process per call):

  echo '{"op": "describe", "args": {"tickers": ["SPY"], "start": "2025-01-01"}}' | python quant_tools.py
  python quant_tools.py --list                    # operations
  python quant_tools.py rule-studies              # refresh Massive/Research/rule_studies.json (prompt evidence)

Data: either explicit numbers (args.series = list or {name: list}, args.input = "returns" | "prices") or tickers read
from the local Massive daily database (args.tickers, start, end; dividends included in returns). Engine backtests
(backtest, walk_forward, daytrade_backtest) call the same code as research.py / daytrader.py; nothing here can place
an order or write to the paper ledgers. Output: {"ok": true, "op", "result", "seconds"} or {"ok": false, "error"}.
Libraries: numpy, pandas, scipy, statsmodels, arch (GARCH; EWMA fallback when missing).
"""

import inspect
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parent
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

TRADING_DAYS = 252
MAX_TICKERS = 60
MAX_POINTS = 50_000
MAX_SIMS = 20_000
MAX_BOOT = 20_000
ACCOUNT_EQUITY = 10_000.0


class ToolError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def bad(message):
    raise ToolError("BAD_ARGS", message)


# ------------------------------------------------------------------ argument helpers
def _num(args, key, default, lo=None, hi=None, integer=False):
    value = args.get(key, default)
    if value is None:
        value = default
    try:
        value = float(value)
    except (TypeError, ValueError):
        bad(f"{key} must be a number")
    if not math.isfinite(value):
        bad(f"{key} must be finite")
    if lo is not None and value < lo or hi is not None and value > hi:
        bad(f"{key} must be between {lo} and {hi}")
    return int(round(value)) if integer else value


def _choice(args, key, default, options):
    value = str(args.get(key) or default).lower()
    if value not in options:
        bad(f"{key} must be one of {sorted(options)}")
    return value


def _date(value):
    if value in (None, ""):
        return None
    try:
        return pd.Timestamp(str(value)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        bad(f"invalid date {value!r}")


def _tickers(args, required=True, maximum=MAX_TICKERS):
    raw = args.get("tickers") or []
    if isinstance(raw, str):
        raw = [t for t in raw.replace(",", " ").split() if t]
    out = []
    for t in raw:
        t = str(t).strip().upper()
        if not t or len(t) > 12 or not all(c.isalnum() or c in ".-" for c in t):
            bad(f"invalid ticker {t!r}")
        if t not in out:
            out.append(t)
    if required and not out and not args.get("series"):
        bad("give tickers or series")
    if len(out) > maximum:
        bad(f"at most {maximum} tickers")
    return out


def clean(value, digits=6):
    """JSON-safe: NaN/inf -> None, numpy -> python, timestamps -> ISO dates, floats rounded to `digits` significant."""
    if isinstance(value, dict):
        return {str(k): clean(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v, digits) for v in value]
    if isinstance(value, pd.Series):
        return clean(value.to_dict(), digits)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return float(f"{v:.{digits}g}") if math.isfinite(v) else None
    if isinstance(value, np.ndarray):
        return clean(value.tolist(), digits)
    return value


# ------------------------------------------------------------------ data
def fast_panel(tickers, start=None, end=None):
    """market_data.load_panel(tickers=...) with the ticker filter pushed into SQL (same sessions, anomaly exclusions,
    masked days and dividends; load_panel reads every ticker first, which takes ~30 s on the full database)."""
    from fetch_and_model import DATABASE
    from market_data import MARKET, Panel, _table_exists
    from price_history import complete_market_dates, open_history
    from universe import classify
    keep = sorted(set(tickers) | {MARKET})
    marks = ",".join("?" * len(keep))
    with open_history(DATABASE) as connection:
        sessions = sorted(complete_market_dates(connection))
        start, end = start or sessions[0], end or sessions[-1]
        sessions = [day for day in sessions if start <= day <= end]
        flagged, masked = set(), []
        if _table_exists(connection, "price_anomalies"):
            flagged = {row[0] for row in connection.execute(
                f"""SELECT DISTINCT ticker FROM price_anomalies WHERE status IN ('open', 'persists')
                    AND kind IN ('jump_reversal','unadjusted_split','split_basis_mismatch') AND ticker IN ({marks})""", keep)}
            masked = connection.execute(
                f"SELECT ticker, date FROM price_anomalies WHERE status = 'masked' AND ticker IN ({marks})", keep).fetchall()
        keep = [t for t in keep if t not in flagged or t == MARKET]
        marks = ",".join("?" * len(keep))
        frame = pd.read_sql_query(
            f"SELECT ticker, date, open, high, low, close, volume FROM daily_prices WHERE date BETWEEN ? AND ? AND ticker IN ({marks})",
            connection, params=[start, end, *keep])
        dividends = pd.DataFrame(columns=["ticker", "market_date", "cash"])
        if _table_exists(connection, "backtest_dividends"):
            dividends = pd.read_sql_query(
                f"""SELECT ticker, market_date, cash FROM backtest_dividends WHERE unsupported = 0
                    AND market_date BETWEEN ? AND ? AND ticker IN ({marks})""", connection, params=[start, end, *keep])
    index = pd.DatetimeIndex(pd.to_datetime(sessions))
    columns = sorted(frame.ticker.unique())
    frame["date"] = pd.to_datetime(frame["date"])
    pivot = lambda name: frame.pivot(index="date", columns="ticker", values=name).reindex(index=index, columns=columns).astype(float)
    dividends["market_date"] = pd.to_datetime(dividends["market_date"])
    cash = (dividends.pivot_table(index="market_date", columns="ticker", values="cash", aggfunc="sum")
            .reindex(index=index, columns=columns).fillna(0.0)) if len(dividends) else pd.DataFrame(0.0, index=index, columns=columns)
    mask = pd.DataFrame(False, index=index, columns=columns)
    for ticker, day in masked:
        stamp = pd.Timestamp(day)
        if ticker in mask.columns and stamp in mask.index:
            mask.loc[stamp, ticker] = True
    return Panel(pivot("open"), pivot("high"), pivot("low"), pivot("close"), pivot("volume"), cash, classify(columns), mask)


def load_panel_for(tickers, start=None, end=None):
    panel = fast_panel(tickers, start, end)
    missing = [t for t in tickers if t not in panel.close.columns]
    if len(missing) == len(tickers):
        raise ToolError("NO_DATA", f"no local price history for {', '.join(missing)}")
    return panel, missing


def aggregate(returns, horizon):
    """Non-overlapping `horizon`-bar compounded returns."""
    logr = np.log1p(returns.astype(float))
    n = len(logr) // horizon * horizon
    blocks = logr.iloc[len(logr) - n:].to_numpy().reshape(-1, horizon).sum(axis=1)
    index = logr.index[len(logr) - n:][horizon - 1::horizon]
    return pd.Series(np.expm1(blocks), index=index)


def series_from_args(args, want="returns", min_len=30):
    """Return (dict name -> pd.Series, info). want: 'returns' (simple, per bar), 'prices' or 'log_prices'."""
    info = {}
    raw = args.get("series")
    if raw is not None:
        kind = _choice(args, "input", "returns", {"returns", "prices"})
        items = raw.items() if isinstance(raw, dict) else [("series", raw)]
        out = {}
        for name, values in items:
            if not isinstance(values, (list, tuple)) or len(values) > MAX_POINTS:
                bad(f"series {name} must be a list of at most {MAX_POINTS} numbers")
            arr = pd.Series(pd.to_numeric(pd.Series(list(values), dtype=object), errors="coerce"), dtype=float).dropna().reset_index(drop=True)
            if kind == "prices":
                if (arr <= 0).any():
                    bad("prices must be positive")
                out[str(name)] = arr if want == "prices" else np.log(arr) if want == "log_prices" else arr.pct_change().dropna()
            else:
                if want != "returns":
                    arr = (1 + arr).cumprod()
                    arr = np.log(arr) if want == "log_prices" else arr
                out[str(name)] = arr
        info["source"] = "supplied series"
    else:
        tickers = _tickers(args)
        panel, missing = load_panel_for(tickers, _date(args.get("start")), _date(args.get("end")))
        out = {}
        for t in tickers:
            if t not in panel.close.columns:
                continue
            if want == "returns":
                out[t] = panel.total_return()[t].dropna()
            else:
                tr = panel.total_return()[t].fillna(0.0)
                idx = (1 + tr).cumprod().where(panel.close[t].notna()).dropna() * float(panel.close[t].dropna().iloc[0])
                out[t] = np.log(idx) if want == "log_prices" else idx
        info.update({"source": "Massive daily database (dividend-adjusted total return)", "missing": missing,
                     "start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date()), "sessions": len(panel.dates)})
    horizon = _num(args, "horizon", 1, 1, 260, integer=True)
    if horizon > 1 and want == "returns":
        out = {k: aggregate(v, horizon) for k, v in out.items()}
        info["horizonDays"] = horizon
    for k, v in out.items():
        if len(v) < min_len:
            raise ToolError("NO_DATA", f"{k}: only {len(v)} observations (need {min_len})")
    return out, info


def aligned(series_map):
    frame = pd.DataFrame(series_map).dropna()
    if len(frame) < 30:
        raise ToolError("NO_DATA", f"only {len(frame)} overlapping observations")
    return frame


def periods_per_year(args):
    return TRADING_DAYS / _num(args, "horizon", 1, 1, 260, integer=True)


# ------------------------------------------------------------------ descriptive statistics / tests
def _max_drawdown(returns):
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())


def op_describe(args):
    """Mean, vol, Sharpe, skew, excess kurtosis, quantiles, drawdown, lag-1 autocorrelation."""
    from scipy import stats
    data, info = series_from_args(args, min_len=10)
    ppy = periods_per_year(args)
    out = {}
    for name, r in data.items():
        x = r.to_numpy(dtype=float)
        sd = x.std(ddof=1)
        out[name] = {"n": len(x), "mean": x.mean(), "std": sd, "annMean": x.mean() * ppy, "annVol": sd * math.sqrt(ppy),
                     "sharpe": x.mean() / sd * math.sqrt(ppy) if sd > 0 else None,
                     "skew": stats.skew(x), "excessKurtosis": stats.kurtosis(x), "min": x.min(), "max": x.max(),
                     "quantiles": {str(q): float(np.quantile(x, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
                     "positiveShare": float((x > 0).mean()), "maxDrawdown": _max_drawdown(r),
                     "totalReturn": float(np.prod(1 + x) - 1), "autocorrLag1": float(pd.Series(x).autocorr(1))}
    return {"info": info, "series": out}


def op_normality(args):
    """Jarque-Bera, Shapiro-Wilk, Anderson-Darling, D'Agostino K2, skew/kurtosis and tail frequencies by horizon."""
    from scipy import stats
    horizons = args.get("horizons") or [1, 5, 20, 50]
    horizons = [int(h) for h in horizons if 1 <= int(h) <= 260][:8]
    data, info = series_from_args({k: v for k, v in args.items() if k != "horizon"}, min_len=30)
    rng = np.random.default_rng(7)
    out = {}
    for name, r in data.items():
        rows = []
        for h in horizons:
            x = np.log1p(aggregate(r, h).to_numpy() if h > 1 else r.to_numpy(dtype=float))
            if len(x) < 8:
                rows.append({"horizon": h, "n": len(x), "note": "too few independent periods"})
                continue
            z = (x - x.mean()) / x.std(ddof=1)
            sample = x if len(x) <= 5000 else rng.choice(x, 5000, replace=False)
            jb = stats.jarque_bera(x)
            sh = stats.shapiro(sample)
            ad = stats.anderson(x, dist="norm")
            levels = [float(s) for s in ad.significance_level]
            crit5 = float(ad.critical_values[levels.index(5.0)])
            row = {"horizon": h, "n": len(x), "skew": stats.skew(x), "excessKurtosis": stats.kurtosis(x),
                   "jarqueBera": {"stat": jb.statistic, "p": jb.pvalue},
                   "shapiroWilk": {"stat": sh.statistic, "p": sh.pvalue, "n": len(sample)},
                   "andersonDarling": {"stat": ad.statistic, "crit5pct": crit5, "rejectNormalAt5pct": bool(ad.statistic > crit5)},
                   "tails": {"over2sigmaPct": float((np.abs(z) > 2).mean() * 100), "normal2sigmaPct": 4.55,
                             "over3sigmaPct": float((np.abs(z) > 3).mean() * 100), "normal3sigmaPct": 0.27}}
            if len(x) >= 20:
                k2 = stats.normaltest(x)
                row["dagostinoK2"] = {"stat": k2.statistic, "p": k2.pvalue}
            rows.append(row)
        out[name] = rows
    return {"info": info, "byHorizon": out,
            "reading": "p < 0.05 rejects normality. Excess kurtosis > 0 = fat tails; size with ATR/vol and use t/historical VaR."}


def _block_bootstrap(x, n_boot, block, rng, length=None):
    n = len(x)
    length = length or n
    k = math.ceil(length / block)
    starts = rng.integers(0, max(1, n - block + 1), size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :length]
    return x[np.clip(idx, 0, n - 1)]


def op_bootstrap(args):
    """Moving-block bootstrap confidence interval for mean / median / Sharpe / vol / total return."""
    data, info = series_from_args(args)
    stat = _choice(args, "statistic", "sharpe", {"mean", "median", "sharpe", "vol", "total_return"})
    n_boot = _num(args, "n_boot", 5000, 200, MAX_BOOT, integer=True)
    block = _num(args, "block", 5, 1, 60, integer=True)
    conf = _num(args, "confidence", 0.95, 0.5, 0.999)
    ppy = periods_per_year(args)
    rng = np.random.default_rng(_num(args, "seed", 7, 0, 1e9, integer=True))
    funcs = {"mean": lambda m: m.mean(axis=1) * ppy, "median": lambda m: np.median(m, axis=1),
             "sharpe": lambda m: m.mean(axis=1) / m.std(axis=1, ddof=1) * math.sqrt(ppy),
             "vol": lambda m: m.std(axis=1, ddof=1) * math.sqrt(ppy), "total_return": lambda m: np.prod(1 + m, axis=1) - 1}
    out = {}
    for name, r in data.items():
        x = r.to_numpy(dtype=float)
        draws = funcs[stat](_block_bootstrap(x, n_boot, block, rng))
        lo, hi = np.quantile(draws, [(1 - conf) / 2, 1 - (1 - conf) / 2])
        out[name] = {"statistic": stat, "point": float(funcs[stat](x[None, :])[0]), "ciLow": lo, "ciHigh": hi,
                     "confidence": conf, "probPositive": float((draws > 0).mean()), "bootSE": float(draws.std(ddof=1)), "n": len(x)}
    return {"info": info, "method": f"moving-block bootstrap (block {block}, {n_boot} resamples)", "results": out}


def _hac_lags(n):
    return max(1, int(4 * (n / 100) ** (2 / 9)))


def op_t_test(args):
    """One-sample (plain and HAC/Newey-West), Welch two-sample or paired t-tests."""
    from scipy import stats
    import statsmodels.api as sm
    data, info = series_from_args(args)
    kind = _choice(args, "test", "one_sample", {"one_sample", "welch", "paired"})
    mu0 = _num(args, "mu0", 0.0)
    names = list(data)
    if kind == "one_sample":
        res = {}
        for n in names:
            x = data[n].to_numpy(dtype=float)
            t = stats.ttest_1samp(x, mu0)
            fit = sm.OLS(x - mu0, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": _hac_lags(len(x))})
            res[n] = {"mean": float(x.mean()), "t": t.statistic, "p": t.pvalue, "tHAC": float(fit.tvalues[0]),
                      "pHAC": float(fit.pvalues[0]), "n": len(x)}
        return {"info": info, "test": "one-sample t (H0: mean = mu0)", "mu0": mu0, "results": res}
    if len(names) < 2:
        bad("two series needed")
    a, b = data[names[0]], data[names[1]]
    if kind == "paired":
        frame = aligned({names[0]: a, names[1]: b})
        t, n = stats.ttest_rel(frame[names[0]], frame[names[1]]), len(frame)
    else:
        t, n = stats.ttest_ind(a, b, equal_var=False), [len(a), len(b)]
    return {"info": info, "test": f"{kind} t-test {names[0]} vs {names[1]}", "t": t.statistic, "p": t.pvalue, "n": n,
            "meanDiff": float(a.mean() - b.mean())}


def op_correlation(args):
    """Pearson / Spearman / Kendall correlation matrix (+ rolling correlation of the first pair)."""
    data, info = series_from_args(args)
    if len(data) < 2:
        bad("two or more series needed")
    frame = aligned(data)
    method = _choice(args, "method", "pearson", {"pearson", "spearman", "kendall"})
    out = {"info": info, "method": method, "n": len(frame), "matrix": frame.corr(method=method).to_dict()}
    if args.get("window"):
        w = _num(args, "window", 60, 10, 1000, integer=True)
        names = list(frame.columns)
        rolling = frame[names[0]].rolling(w).corr(frame[names[1]]).dropna()
        if len(rolling):
            out["rolling"] = {"pair": names[:2], "window": w, "latest": rolling.iloc[-1], "min": rolling.min(),
                              "max": rolling.max(), "mean": rolling.mean()}
    return out


def sharpe_stats(x, ppy=TRADING_DAYS, sr0=0.0, trials=1, trial_sr_std=None):
    from scipy import stats
    x = np.asarray(x, dtype=float)
    n = len(x)
    sr = x.mean() / x.std(ddof=1)
    sk, ku = stats.skew(x), stats.kurtosis(x, fisher=False)
    se = math.sqrt(max(1e-12, (1 - sk * sr + (ku - 1) / 4 * sr * sr) / (n - 1)))
    z = stats.norm.ppf(0.975)
    sr0p = sr0 / math.sqrt(ppy)
    out = {"n": n, "sharpeAnnual": sr * math.sqrt(ppy), "seAnnual": se * math.sqrt(ppy),
           "ci95Annual": [(sr - z * se) * math.sqrt(ppy), (sr + z * se) * math.sqrt(ppy)],
           "iidSeAnnual": math.sqrt((1 + 0.5 * sr * sr) / n) * math.sqrt(ppy),
           "probabilisticSharpe": float(stats.norm.cdf((sr - sr0p) / se)), "benchmarkSharpeAnnual": sr0,
           "skew": sk, "kurtosis": ku}
    if sr > sr0p:
        out["minTrackRecordObs"] = 1 + (1 - sk * sr + (ku - 1) / 4 * sr * sr) * (z / (sr - sr0p)) ** 2
    if trials and trials > 1:
        emc = 0.5772156649
        v = (trial_sr_std / math.sqrt(ppy)) ** 2 if trial_sr_std else se * se
        sr_star = math.sqrt(v) * ((1 - emc) * stats.norm.ppf(1 - 1 / trials) + emc * stats.norm.ppf(1 - 1 / (trials * math.e)))
        out["deflatedSharpe"] = {"trials": trials, "expectedMaxSharpeAnnual": sr_star * math.sqrt(ppy),
                                 "probability": float(stats.norm.cdf((sr - sr_star) / se))}
    return out


def op_sharpe_inference(args):
    """Sharpe with Lo/Mertens standard error and CI, probabilistic Sharpe, minimum track record, deflated Sharpe."""
    data, info = series_from_args(args)
    trials = _num(args, "trials", 1, 1, 100000, integer=True)
    sr0 = _num(args, "benchmark_sharpe", 0.0, -5, 10)
    std = args.get("trial_sharpe_std")
    ppy = periods_per_year(args)
    return {"info": info, "results": {k: sharpe_stats(v.to_numpy(), ppy, sr0, trials, float(std) if std else None)
                                      for k, v in data.items()},
            "reading": "Lo/Mertens SE (skew/kurtosis aware); PSR = P(true Sharpe > benchmark); deflated Sharpe corrects "
                       "for the number of trials tried (Bailey & Lopez de Prado)."}


# ------------------------------------------------------------------ econometrics
def _ols(y, X, hac=True):
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson
    lags = _hac_lags(len(y))
    Xc = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lags}) if hac else sm.OLS(y, Xc).fit()
    return fit, {"coef": fit.params.to_dict(), "se": fit.bse.to_dict(), "t": fit.tvalues.to_dict(),
                 "p": fit.pvalues.to_dict(), "r2": fit.rsquared, "adjR2": fit.rsquared_adj, "n": int(fit.nobs),
                 "durbinWatson": float(durbin_watson(fit.resid)), "residVol": float(np.std(fit.resid, ddof=1)),
                 "covariance": f"HAC Newey-West (lags {lags})" if hac else "OLS"}


def op_ols(args):
    """OLS of the first series on the others with HAC (Newey-West) standard errors."""
    data, info = series_from_args(args)
    if len(data) < 2:
        bad("first series/ticker = dependent variable, the rest = regressors")
    frame = aligned(data)
    names = list(frame.columns)
    fit, out = _ols(frame[names[0]], frame[names[1:]], hac=args.get("hac", True) is not False)
    out["alphaAnnualized"] = float(fit.params["const"] * periods_per_year(args))
    return {"info": info, "dependent": names[0], "regressors": names[1:], **out}


def op_rolling_ols(args):
    """Rolling-window OLS betas (summary + sampled path)."""
    from statsmodels.regression.rolling import RollingOLS
    import statsmodels.api as sm
    data, info = series_from_args(args)
    if len(data) < 2:
        bad("dependent + at least one regressor")
    frame = aligned(data)
    names = list(frame.columns)
    window = _num(args, "window", 60, 20, 1000, integer=True)
    if window >= len(frame):
        bad("window longer than the data")
    params = RollingOLS(frame[names[0]], sm.add_constant(frame[names[1:]]), window=window).fit().params.dropna()
    summary = {c: {"latest": params[c].iloc[-1], "mean": params[c].mean(), "min": params[c].min(), "max": params[c].max(),
                   "q10": params[c].quantile(0.1), "q90": params[c].quantile(0.9)} for c in params.columns}
    step = max(1, len(params) // 24)
    path = {(str(i.date()) if hasattr(i, "date") else str(i)): row.to_dict() for i, row in params.iloc[::step].iterrows()}
    return {"info": info, "dependent": names[0], "window": window, "betas": summary, "sampledPath": path}


DEFAULT_FACTORS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]


def op_factor_betas(args):
    """Multi-factor betas (default SPY, QQQ, IWM, TLT, GLD) with HAC t-stats, alpha, R2, idiosyncratic vol."""
    tickers = _tickers(args)
    factors = [str(f).upper() for f in (args.get("factors") or DEFAULT_FACTORS)][:8]
    panel, missing = load_panel_for(list(dict.fromkeys(tickers + factors)), _date(args.get("start")), _date(args.get("end")))
    rets = panel.total_return()
    factors = [f for f in factors if f in rets.columns]
    out = {}
    for t in tickers:
        cols = [f for f in factors if f != t]
        if t not in rets.columns or not cols:
            continue
        frame = rets[[t] + cols].dropna()
        if len(frame) < 60:
            continue
        fit, st = _ols(frame[t], frame[cols])
        out[t] = {"alphaAnnualized": float(fit.params["const"] * TRADING_DAYS), "alphaT_HAC": float(fit.tvalues["const"]),
                  "betas": {f: float(fit.params[f]) for f in cols}, "betaT_HAC": {f: float(fit.tvalues[f]) for f in cols},
                  "singleFactorBetas": {f: float(np.cov(frame[t], frame[f])[0, 1] / frame[f].var()) for f in cols},
                  "r2": st["r2"], "idioVolAnnualized": st["residVol"] * math.sqrt(TRADING_DAYS), "n": st["n"]}
    return {"info": {"start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date()), "missing": missing},
            "factors": factors, "model": "daily total returns, OLS with HAC errors", "results": out}


def op_adf_kpss(args):
    """ADF (H0 unit root) and KPSS (H0 stationary) on log prices or returns."""
    from statsmodels.tsa.stattools import adfuller, kpss
    target = _choice(args, "target", "log_prices", {"log_prices", "returns"})
    data, info = series_from_args(args, want=target)
    out = {}
    for name, x in data.items():
        x = x.dropna().to_numpy(dtype=float)
        adf = adfuller(x, autolag="AIC")
        kp = kpss(x, regression="c", nlags="auto")
        out[name] = {"adf": {"stat": adf[0], "p": adf[1], "lags": adf[2], "crit5pct": adf[4]["5%"]},
                     "kpss": {"stat": kp[0], "p": kp[1], "crit5pct": kp[3]["5%"], "note": "p clipped to [0.01, 0.10]"},
                     "verdict": ("stationary" if adf[1] < 0.05 and kp[1] > 0.05 else
                                 "unit root" if adf[1] >= 0.05 and kp[1] <= 0.05 else "inconclusive")}
    return {"info": info, "target": target, "results": out,
            "reading": "ADF H0 = unit root (p<0.05 rejects); KPSS H0 = stationary (p<0.05 rejects)."}


def _half_life(log_price):
    import statsmodels.api as sm
    lp = pd.Series(np.asarray(log_price, dtype=float))
    y = lp.diff().dropna()
    x = lp.shift().dropna().loc[y.index]
    fit = sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit()
    b = float(fit.params[1])
    return (-math.log(2) / math.log(1 + b) if -1 < b < 0 else None), b, float(fit.tvalues[1])


def op_engle_granger(args):
    """Engle-Granger cointegration for every pair (<= 10 tickers): p-value, hedge ratio, spread half-life, spread z."""
    from statsmodels.tsa.stattools import coint
    import statsmodels.api as sm
    data, info = series_from_args(args, want="log_prices")
    if len(data) < 2:
        bad("two or more tickers needed")
    frame = aligned(data)
    names = list(frame.columns)[:10]
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            stat, p, crit = coint(frame[a], frame[b], trend="c", autolag="AIC")
            fit = sm.OLS(frame[a], sm.add_constant(frame[b])).fit()
            spread = frame[a] - fit.params[b] * frame[b] - fit.params["const"]
            hl, _, _ = _half_life(spread)
            pairs.append({"y": a, "x": b, "egStat": stat, "p": p, "crit5pct": crit[1], "hedgeRatio": float(fit.params[b]),
                          "spreadHalfLifeDays": hl, "spreadZNow": float((spread.iloc[-1] - spread.mean()) / spread.std(ddof=1)),
                          "cointegratedAt5pct": bool(p < 0.05)})
    pairs.sort(key=lambda r: r["p"])
    return {"info": info, "test": "Engle-Granger on log prices (y = a + b*x + e; ADF on e)", "pairs": pairs,
            "note": "About two years of data: cointegration tests have low power; check half-life and stability."}


def op_johansen(args):
    """Johansen trace / max-eigenvalue tests and cointegration rank (2-12 tickers, log prices)."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    data, info = series_from_args(args, want="log_prices")
    if not 2 <= len(data) <= 12:
        bad("2 to 12 tickers")
    frame = aligned(data)
    k = _num(args, "lags", 1, 1, 10, integer=True)
    res = coint_johansen(frame.to_numpy(), det_order=0, k_ar_diff=k)
    rows = [{"rankAtMost": r, "trace": float(res.lr1[r]), "trace95": float(res.cvt[r, 1]), "maxEig": float(res.lr2[r]),
             "maxEig95": float(res.cvm[r, 1]), "rejectAt5pct": bool(res.lr1[r] > res.cvt[r, 1])} for r in range(len(frame.columns))]
    rank = next((r["rankAtMost"] for r in rows if not r["rejectAt5pct"]), len(rows))
    vec = res.evec[:, 0]
    return {"info": info, "tickers": list(frame.columns), "lagsInDifferences": k, "tests": rows, "cointegrationRank": rank,
            "firstVector": {c: float(v / vec[0]) for c, v in zip(frame.columns, vec)}}


def op_arima(args):
    """AR/ARMA/ARIMA fit (order given or AIC-selected over p,q <= 2), forecast with 95% bands, Ljung-Box."""
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.diagnostic import acorr_ljungbox
    data, info = series_from_args(args)
    name = list(data)[0]
    y = data[name].reset_index(drop=True).astype(float)
    steps = _num(args, "steps", 5, 1, 60, integer=True)
    order = args.get("order")
    if order is not None and (not isinstance(order, (list, tuple)) or len(order) != 3):
        bad("order must be [p, d, q]")
    candidates = [tuple(int(v) for v in order)] if order else [(p, 0, q) for p in range(3) for q in range(3)]
    best = None
    for o in candidates:
        if any(v < 0 or v > 5 for v in o):
            bad("ARIMA orders must be 0..5")
        try:
            fit = ARIMA(y, order=o).fit()
        except Exception:  # noqa: BLE001 - a failing candidate is skipped
            continue
        if best is None or fit.aic < best[1].aic:
            best = (o, fit)
    if best is None:
        raise ToolError("FIT_FAILED", "no ARIMA candidate converged")
    o, fit = best
    fc = fit.get_forecast(steps)
    ci = np.asarray(fc.conf_int(alpha=0.05))
    lb = acorr_ljungbox(fit.resid, lags=[10], return_df=True)
    return {"info": info, "series": name, "order": list(o), "selectedByAIC": order is None, "aic": fit.aic, "bic": fit.bic,
            "params": dict(fit.params), "pvalues": dict(fit.pvalues),
            "forecast": [{"step": i + 1, "mean": float(m), "lo95": float(ci[i, 0]), "hi95": float(ci[i, 1])}
                         for i, m in enumerate(np.asarray(fc.predicted_mean))],
            "ljungBox10": {"stat": float(lb["lb_stat"].iloc[0]), "p": float(lb["lb_pvalue"].iloc[0])},
            "note": "Daily return forecasts from ARMA models are tiny vs the noise; read them as evidence of (no) autocorrelation."}


def op_half_life_hurst(args):
    """Mean-reversion half-life (AR(1) on log price), Hurst exponent and Lo-MacKinlay variance ratios."""
    data, info = series_from_args(args, want="log_prices", min_len=60)
    out = {}
    for name, lp in data.items():
        lp = lp.dropna().reset_index(drop=True)
        hl, b, t = _half_life(lp)
        lags = list(range(2, min(100, len(lp) // 4)))
        tau = [np.std(lp.diff(l).dropna()) for l in lags]
        hurst = float(np.polyfit(np.log(lags), np.log(tau), 1)[0])
        r = lp.diff().dropna().to_numpy()
        vr = {}
        for q in (2, 5, 10):
            if len(r) > q * 10:
                rq = np.convolve(r, np.ones(q), mode="valid")
                ratio = rq.var(ddof=1) / (q * r.var(ddof=1))
                se = math.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * len(r)))
                vr[str(q)] = {"ratio": ratio, "z": (ratio - 1) / se}
        out[name] = {"halfLifeDays": hl, "ar1Coefficient": b, "ar1T": t, "hurst": hurst, "varianceRatio": vr,
                     "reading": "mean-reverting" if hurst < 0.45 else "trending" if hurst > 0.55 else "random-walk-like"}
    return {"info": info, "results": out,
            "reading": "Hurst < 0.5 mean-reverting, > 0.5 trending. Variance ratio < 1 (z < -2) = mean reversion at that horizon."}


# ------------------------------------------------------------------ volatility, risk, sizing, optimisation
def _ewma(r, steps=20, lam=0.94):
    x = np.asarray(r, dtype=float)
    var = np.var(x[:20]) if len(x) > 20 else np.var(x)
    for v in x:
        var = lam * var + (1 - lam) * v * v
    return {"currentAnnVol": math.sqrt(var * TRADING_DAYS), "forecastHorizonVol": math.sqrt(var * steps),
            "realized20AnnVol": float(np.std(x[-20:], ddof=1) * math.sqrt(TRADING_DAYS)),
            "realized60AnnVol": float(np.std(x[-60:], ddof=1) * math.sqrt(TRADING_DAYS))}


def op_garch(args):
    """GARCH(1,1) (Student-t by default) volatility forecast; deterministic EWMA fallback if the fit fails."""
    data, info = series_from_args(args, min_len=120)
    steps = _num(args, "steps", 20, 1, 260, integer=True)
    dist = _choice(args, "dist", "t", {"t", "normal", "skewt"})
    out = {}
    for name, r in list(data.items())[:10]:
        x = r.to_numpy(dtype=float) * 100
        try:
            from arch import arch_model
            res = arch_model(x, vol="GARCH", p=1, q=1, mean="Constant", dist=dist, rescale=False).fit(disp="off")
            omega, alpha, beta = (float(res.params[k]) for k in ("omega", "alpha[1]", "beta[1]"))
            var = np.asarray(res.forecast(horizon=steps, reindex=False).variance)[-1]
            persist = alpha + beta
            out[name] = {"model": f"GARCH(1,1) {dist}", "omega": omega, "alpha": alpha, "beta": beta, "persistence": persist,
                         "longRunAnnVol": math.sqrt(omega / (1 - persist) * TRADING_DAYS) / 100 if persist < 1 else None,
                         "currentAnnVol": float(np.asarray(res.conditional_volatility)[-1]) * math.sqrt(TRADING_DAYS) / 100,
                         "forecastAnnVol1d": math.sqrt(var[0] * TRADING_DAYS) / 100,
                         "forecastAnnVolAvg": math.sqrt(var.mean() * TRADING_DAYS) / 100,
                         "forecastHorizonVol": math.sqrt(var.sum()) / 100, "steps": steps,
                         "params": {k: float(v) for k, v in dict(res.params).items()}, "logLik": float(res.loglikelihood),
                         "aic": float(res.aic)}
            if "nu" in res.params:
                out[name]["tDegreesOfFreedom"] = float(res.params["nu"])
        except Exception as error:  # noqa: BLE001 - deterministic EWMA fallback
            out[name] = {"model": "EWMA (RiskMetrics lambda 0.94) fallback", "error": str(error)[:200], **_ewma(r, steps)}
    return {"info": info, "results": out}


def op_ewma(args):
    """RiskMetrics EWMA volatility (lambda 0.94) with 20/60-day realized vol."""
    data, info = series_from_args(args)
    steps = _num(args, "steps", 20, 1, 260, integer=True)
    return {"info": info, "results": {k: _ewma(v, steps) for k, v in data.items()}}


def _portfolio(data, weights):
    frame = aligned(data) if len(data) > 1 else pd.DataFrame(data).dropna()
    if weights:
        w = pd.Series({str(k).upper(): float(v) for k, v in weights.items()}).reindex(frame.columns).fillna(0.0)
    else:
        w = pd.Series(1 / len(frame.columns), index=frame.columns)
    return (frame * w).sum(axis=1), w


def op_var(args):
    """VaR and CVaR (expected shortfall): normal, Cornish-Fisher, fitted Student-t, historical; in $ for the account."""
    from scipy import stats
    data, info = series_from_args(args, min_len=60)
    horizon = _num(args, "horizon_days", 1, 1, 60, integer=True)
    confs = args.get("confidences") or [args.get("confidence") or 0.95, 0.99]
    confs = sorted({float(c) for c in confs if 0.5 < float(c) < 0.9999})
    equity = _num(args, "equity", ACCOUNT_EQUITY, 1, 1e10)
    targets = dict(data)
    if args.get("weights") or (len(data) > 1 and args.get("portfolio")):
        targets = {"portfolio": _portfolio(data, args.get("weights"))[0]}
    out = {}
    for name, r in targets.items():
        x = r.to_numpy(dtype=float)
        mu, sd = x.mean(), x.std(ddof=1)
        sk, ku = stats.skew(x), stats.kurtosis(x)
        df, loc, scale = stats.t.fit(x)
        sums = np.convolve(x, np.ones(horizon), mode="valid") if horizon > 1 else x
        rows = {}
        for c in confs:
            a = 1 - c
            z = stats.norm.ppf(a)
            zcf = z + (z * z - 1) * sk / 6 + (z ** 3 - 3 * z) * ku / 24 - (2 * z ** 3 - 5 * z) * sk * sk / 36
            hmu, hsd = mu * horizon, sd * math.sqrt(horizon)
            tq = stats.t.ppf(a, df) * scale * math.sqrt(horizon) + loc * horizon
            sample = stats.t.rvs(df, loc=loc * horizon, scale=scale * math.sqrt(horizon), size=200_000, random_state=7)
            hist = np.quantile(sums, a)
            tail = sums[sums <= hist]
            row = {"normalVaR": -(hmu + z * hsd), "normalCVaR": -(hmu - hsd * stats.norm.pdf(z) / a),
                   "cornishFisherVaR": -(hmu + zcf * hsd), "studentTVaR": -tq, "studentTCVaR": -sample[sample <= tq].mean(),
                   "historicalVaR": -hist, "historicalCVaR": -tail.mean() if len(tail) else None}
            row["dollars"] = {k: (v * equity if v is not None else None) for k, v in row.items()}
            rows[str(c)] = row
        out[name] = {"horizonDays": horizon, "n": len(x), "skew": sk, "excessKurtosis": ku, "tDegreesOfFreedom": df, "levels": rows}
    return {"info": info, "equity": equity, "results": out,
            "reading": "Losses as positive fractions of equity. Daily returns are fat-tailed: prefer Student-t / historical over normal."}


def op_sizing(args):
    """Kelly / half-Kelly, volatility-target weight (engine caps) and stop-distance share sizing for the $10k account."""
    data, info = series_from_args(args, min_len=60) if (args.get("tickers") or args.get("series")) else ({}, {})
    equity = _num(args, "equity", ACCOUNT_EQUITY, 1, 1e10)
    target_vol = _num(args, "target_vol", 0.10, 0.01, 1.0)
    cap = _num(args, "per_name_cap", 0.30, 0.01, 1.0)
    risk_per_trade = _num(args, "risk_per_trade", 0.005, 0.0001, 0.1)
    out = {}
    for name, r in data.items():
        x = r.to_numpy(dtype=float)
        mu, var = x.mean() * TRADING_DAYS, x.var(ddof=1) * TRADING_DAYS
        vol = math.sqrt(var)
        kelly = mu / var if var > 0 else None
        w_vol = min(cap, target_vol / vol) if vol > 0 else None
        price = float(info.get("lastPrice") or 0) or None
        out[name] = {"annMean": mu, "annVol": vol, "kellyFraction": kelly, "halfKelly": kelly / 2 if kelly is not None else None,
                     "volTargetWeight": w_vol, "volTargetDollars": w_vol * equity if w_vol else None,
                     "note": "Kelly uses noisy mean estimates; the engine sizes by volatility, never by Kelly."}
    stop_sizing = None
    if args.get("price") is not None and args.get("stop") is not None:
        price, stop = float(args["price"]), float(args["stop"])
        if not 0 < stop < price:
            bad("need 0 < stop < price for a long")
        shares = min(math.floor(risk_per_trade * equity / (price - stop)), math.floor(cap * equity / price))
        stop_sizing = {"price": price, "stop": stop, "riskPerShare": price - stop, "riskBudget": risk_per_trade * equity,
                       "shares": shares, "notional": shares * price, "weight": shares * price / equity}
    return {"info": info, "equity": equity, "targetVol": target_vol, "perNameCap": cap, "riskPerTrade": risk_per_trade,
            "results": out, "stopSizing": stop_sizing}


def op_monte_carlo(args):
    """Monte Carlo of equity paths (block bootstrap, normal or Student-t): terminal return, drawdown and loss odds."""
    from scipy import stats
    data, info = series_from_args(args, min_len=60)
    r = _portfolio(data, args.get("weights"))[0] if len(data) > 1 else list(data.values())[0]
    x = r.to_numpy(dtype=float)
    sims = _num(args, "sims", 5000, 100, MAX_SIMS, integer=True)
    days = _num(args, "days", 252, 5, 1260, integer=True)
    method = _choice(args, "method", "bootstrap", {"bootstrap", "normal", "student_t"})
    equity = _num(args, "equity", ACCOUNT_EQUITY, 1, 1e10)
    dd_limit = _num(args, "drawdown_threshold", 0.10, 0.01, 0.9)
    rng = np.random.default_rng(_num(args, "seed", 7, 0, 1e9, integer=True))
    if method == "bootstrap":
        paths = _block_bootstrap(x, sims, _num(args, "block", 5, 1, 60, integer=True), rng, days)
    elif method == "normal":
        paths = rng.normal(x.mean(), x.std(ddof=1), size=(sims, days))
    else:
        df, loc, scale = stats.t.fit(x)
        paths = stats.t.rvs(df, loc=loc, scale=scale, size=(sims, days), random_state=rng)
    eq = np.cumprod(1 + paths, axis=1)
    terminal = eq[:, -1] - 1
    mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    return {"info": info, "method": method, "sims": sims, "days": days, "equity": equity,
            "terminalReturn": {str(q): float(np.quantile(terminal, q)) for q in qs},
            "terminalEquity": {str(q): float(equity * (1 + np.quantile(terminal, q))) for q in qs},
            "probLoss": float((terminal < 0).mean()), "maxDrawdown": {str(q): float(np.quantile(mdd, q)) for q in qs},
            "drawdownThreshold": dd_limit, "probDrawdownWorse": float((mdd <= -dd_limit).mean())}


def _ledoit_wolf(X):
    X = X - X.mean(axis=0)
    n, p = X.shape
    S = X.T @ X / n
    F = np.trace(S) / p * np.eye(p)
    d2 = np.linalg.norm(S - F, "fro") ** 2
    b2 = sum(np.linalg.norm(np.outer(r, r) - S, "fro") ** 2 for r in X) / n ** 2
    shrink = min(1.0, b2 / d2) if d2 > 0 else 1.0
    return shrink * F + (1 - shrink) * S, shrink


def op_portfolio_opt(args):
    """Long-only optimisation with the engine's 30% name cap: risk parity, min variance, max Sharpe, mean-variance, equal."""
    from scipy.optimize import minimize
    data, info = series_from_args(args, min_len=60)
    if len(data) < 2:
        bad("two or more tickers")
    frame = aligned(data)
    names = list(frame.columns)
    method = _choice(args, "method", "risk_parity", {"mean_variance", "min_variance", "risk_parity", "equal_weight", "max_sharpe"})
    cap = max(_num(args, "per_name_cap", 0.30, 0.01, 1.0), 1 / len(names))
    rf = _num(args, "risk_free", 0.0, -0.1, 0.2)
    aversion = _num(args, "risk_aversion", 3.0, 0.1, 100)
    cov, shrink = _ledoit_wolf(frame.to_numpy())
    cov *= TRADING_DAYS
    mu = frame.mean().to_numpy() * TRADING_DAYS
    n = len(names)
    w0 = np.full(n, 1 / n)
    bounds = [(0.0, cap)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    vol = lambda w: math.sqrt(max(1e-16, w @ cov @ w))
    objective = {
        "min_variance": lambda w: w @ cov @ w,
        "max_sharpe": lambda w: -(w @ mu - rf) / vol(w),
        "mean_variance": lambda w: -(w @ mu) + aversion / 2 * (w @ cov @ w),
        "risk_parity": lambda w: (((w * (cov @ w) / vol(w)) - vol(w) / n) ** 2).sum() * 1e4,
    }
    w = w0 if method == "equal_weight" else minimize(objective[method], w0, bounds=bounds, constraints=cons, method="SLSQP").x
    w = np.clip(w, 0, None)
    w = w / w.sum()
    rc = w * (cov @ w) / vol(w)
    target_vol = args.get("target_vol")
    return {"info": info, "method": method, "perNameCap": cap, "covariance": f"Ledoit-Wolf shrinkage ({shrink:.2f})",
            "weights": dict(zip(names, w)), "expectedReturn": float(w @ mu), "volatility": vol(w),
            "sharpe": float((w @ mu - rf) / vol(w)), "riskContribution": dict(zip(names, rc / vol(w))),
            "volTargetScale": min(1.0, float(target_vol) / vol(w)) if target_vol else None,
            "note": "Expected returns from ~2 years of history are very noisy; risk parity / min variance are more robust."}


# ------------------------------------------------------------------ indicators
def op_indicators(args):
    """Engine indicator set (SMA/EMA, Donchian 20/55, Bollinger, z-scores, RSI14/RSI2, ATR, realized vol, momentum)."""
    import indicators as ind
    tickers = _tickers(args, maximum=10)
    panel, missing = load_panel_for(tickers, _date(args.get("start")), _date(args.get("end")))
    market = panel.close["SPY"] if "SPY" in panel.close.columns else None
    out = {}
    for t in tickers:
        if t not in panel.close.columns:
            continue
        frame = pd.DataFrame({"open": panel.open[t], "high": panel.high[t], "low": panel.low[t], "close": panel.close[t],
                              "volume": panel.volume[t]}).dropna()
        values = ind.compute(frame, market_close=market)
        latest = {k: (float(v.dropna().iloc[-1]) if hasattr(v, "dropna") and len(v.dropna()) else None) for k, v in values.items()}
        close = frame["close"]
        ret = lambda n: float(close.iloc[-1] / close.iloc[-1 - n] - 1) if len(close) > n else None
        rsi2 = ind.rsi(close, 2).dropna()
        out[t] = {"date": str(frame.index[-1].date()), "close": float(close.iloc[-1]), "latest": latest,
                  "rsi2": float(rsi2.iloc[-1]) if len(rsi2) else None,
                  "returns": {"5d": ret(5), "20d": ret(20), "63d": ret(63), "126d": ret(126), "252d": ret(252)},
                  "momentum12_1": float(close.iloc[-22] / close.iloc[-253] - 1) if len(close) > 253 else None,
                  "momentum6_1": float(close.iloc[-22] / close.iloc[-127] - 1) if len(close) > 127 else None,
                  "aboveSma200": bool(close.iloc[-1] > close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None}
    return {"missing": missing, "results": out}


# ------------------------------------------------------------------ engine backtests (read-only; same code as research.py)
def _typed(key, value, default):
    if isinstance(default, tuple):
        return tuple(int(float(v)) for v in (value if isinstance(value, (list, tuple)) else str(value).split(",")))
    if isinstance(default, bool):
        return str(value).lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    return str(value)


def _strategy_params(family, raw):
    from strategies import CombinedStrategy, RegimeMeanReversion, TimeSeriesMomentum
    sleeves = {"tsmom": TimeSeriesMomentum.defaults, "meanrev": RegimeMeanReversion.defaults}
    out = {}
    for key, value in dict(raw or {}).items():
        if family == "combined" and "." in key:
            sleeve, name = key.split(".", 1)
            if sleeve not in sleeves or name not in sleeves[sleeve]:
                bad(f"unknown sleeve parameter {key!r} (e.g. tsmom.max_positions, meanrev.entry_z)")
            out[key] = _typed(name, value, sleeves[sleeve][name])
            continue
        defaults = CombinedStrategy.defaults if family == "combined" else sleeves[family]
        if key not in defaults:
            bad(f"{family}: unknown parameter {key!r}; allowed: {sorted(defaults)}")
        out[key] = _typed(key, value, defaults[key])
    return out


def _panel_for_family(family, args):
    from market_data import load_panel
    from universe import CORE_ETFS
    start, end = _date(args.get("start")), _date(args.get("end"))
    tickers = _tickers(args, required=False, maximum=600)
    if tickers:
        return fast_panel(tickers, start, end), "restricted to the given tickers"
    if family == "tsmom":
        return fast_panel(list(CORE_ETFS), start, end), "CORE_ETFS (the tsmom universe)"
    return load_panel(start, end), "full screened universe (same as research.py)"


def _records(frame, n=12):
    if frame is None or len(frame) == 0:
        return []
    return json.loads(frame.tail(n).to_json(orient="records", date_format="iso"))


def _monthly(equity):
    if equity is None or len(equity) == 0:
        return {}
    return {str(i.date()): float(v) for i, v in equity.resample("ME").last().items()}


def op_backtest(args):
    """Engine backtest of tsmom / meanrev / combined with optional parameter overrides (research.py code path)."""
    import research as R
    from metrics import summarize
    from portfolio_sim import simulate
    family = _choice(args, "family", "combined", {"tsmom", "meanrev", "combined"})
    params = _strategy_params(family, args.get("params"))
    panel, universe = _panel_for_family(family, args)
    if len(panel.dates) <= R.WARMUP + 20:
        raise ToolError("NO_DATA", f"need more than {R.WARMUP + 20} sessions (warm-up {R.WARMUP})")
    strategy = R.make_strategy(family, params)
    eval_start = panel.dates[R.WARMUP]
    _, _, result = R.run_strategy(panel, strategy, start=panel.dates[R.WARMUP - 1])
    bench = None
    if "SPY" in panel.close.columns:
        bench = summarize(simulate(panel, R.benchmark_weights(panel, "spy"), start=panel.dates[R.WARMUP - 1]), eval_start)["performance"]
    return {"family": family, "overrides": params, "config": strategy.describe(), "universe": universe,
            "data": {"start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date()), "sessions": len(panel.dates),
                     "tickers": int(panel.close.shape[1])},
            "evaluationStart": str(eval_start.date()), "costModel": "schwab ($0 commission + spread/impact/SEC/TAF)",
            **summarize(result, eval_start), "benchmarkSPY": bench, "latestDiagnostics": strategy.last_diagnostics,
            "monthlyEquity": _monthly(result.equity), "lastTrades": _records(result.trades)}


def op_walk_forward(args):
    """Expanding-window walk-forward (research.walk_forward) with the default grid or a custom grid of <= 16 combos."""
    import research as R
    family = _choice(args, "family", "tsmom", {"tsmom", "meanrev", "combined"})
    grid = args.get("grid") or None
    if grid is not None:
        if not isinstance(grid, dict):
            bad("grid must be {param: [values]}")
        typed = {}
        for key, values in grid.items():
            if not isinstance(values, list) or not values:
                bad(f"grid {key} needs a list of values")
            typed[key] = [_strategy_params(family, {key: v})[key] for v in values]
        grid = typed
        combos = math.prod(len(v) for v in grid.values())
        if combos > 16:
            bad(f"grid has {combos} combinations; at most 16")
    train_min = _num(args, "train_min", 252, 120, 400, integer=True)
    test_len = _num(args, "test_len", 63, 20, 126, integer=True)
    panel, universe = _panel_for_family(family, args)
    if len(panel.dates) < train_min + test_len:
        raise ToolError("NO_DATA", "not enough history for one training window plus one test window")
    result, trials, _ = R.walk_forward(panel, family, grid=grid, train_min=train_min, test_len=test_len)
    table = trials.groupby("trial").agg(params=("params", "first"), trainSharpe=("trainSharpe", "mean"),
                                        testSharpe=("testSharpe", "mean"), timesSelected=("selected", "sum")).reset_index()
    return {**result, "universe": universe, "trialsTable": json.loads(table.to_json(orient="records"))}


ORB_INT = ("or_minutes", "entry_cutoff", "trend_window", "max_positions", "pdt_window", "pdt_max_day_trades")
ORB_NULLABLE = ("target_r", "pdt_max_day_trades")


def op_daytrade_backtest(args):
    """Opening-range-breakout backtest on cached Schwab intraday bars (intraday.backtest, Schwab costs, PDT, breaker)."""
    from contextlib import closing
    import intraday as it
    from costs import CostModel
    from schwab_data import connect_cache
    frequency = _num(args, "frequency", 5, 1, 5, integer=True)
    if frequency not in (1, 5):
        bad("frequency must be 1 or 5 minutes")
    defaults = it.ORBParams().as_dict()
    typed = {}
    for key, value in dict(args.get("params") or {}).items():
        if key not in defaults:
            bad(f"unknown ORB parameter {key!r}; allowed: {sorted(defaults)}")
        if value is None or str(value).lower() in ("none", "null"):
            if key not in ORB_NULLABLE:
                bad(f"{key} cannot be null")
            typed[key] = None
        elif isinstance(defaults[key], bool):
            typed[key] = str(value).lower() in ("1", "true", "yes")
        else:
            typed[key] = int(float(value)) if key in ORB_INT else float(value)
    params = it.ORBParams(**typed)
    symbols = _tickers(args, required=False, maximum=60) or None
    with closing(connect_cache()) as cache:
        bars = it.load_bars(cache, frequency, symbols=symbols, start=_date(args.get("start")), end=_date(args.get("end")))
    if bars.empty:
        raise ToolError("NO_DATA", "no cached Schwab intraday bars for that selection (python daytrader.py fetch)")
    panel = fast_panel(sorted(set(bars["symbol"]) | {"SPY"}))
    equity = _num(args, "initial_equity", ACCOUNT_EQUITY, 1000, 1e8)
    result, candidates = it.backtest(bars, panel, params, frequency, equity, CostModel.for_broker("schwab"))
    return {"params": params.as_dict(), "symbols": sorted(bars["symbol"].unique()), "initialEquity": equity,
            "candidates": int(len(candidates)), **it.summarize(result, frequency), "lastTrades": _records(result.trades)}


# ------------------------------------------------------------------ candidate rules (research studies, NOT engine strategies)
RULES = {
    "spy_200d_filter": "Hold the tickers equal-weight only while SPY closes above its 200-day SMA; otherwise cash.",
    "rsi2_dip": "Buy when RSI(2) < 10 and close > 200-day SMA; exit when close > 5-day SMA or after 10 sessions.",
    "donchian_breakout": "Enter when close > prior 55-day high; exit when close < prior 20-day low (equal weight).",
    "xs_momentum_12_1": "Month-end: hold the top N tickers by 12-1 momentum (t-252..t-21), only while SPY > 200-day SMA.",
    "xs_momentum_6_1": "Month-end: hold the top N tickers by 6-1 momentum (t-126..t-21), only while SPY > 200-day SMA.",
    "pairs_zscore": "Two tickers: spread = log A - b log B (rolling 120-day b); long A/short B when z < -2, reverse when "
                    "z > 2, exit at |z| < 0.5. Long/short: NOT tradable in the long-only account.",
}


def _state_machine(enter, leave, max_hold=None):
    held, age, out = False, 0, np.zeros(len(enter))
    for i in range(len(enter)):
        if held:
            age += 1
            if leave[i] or (max_hold and age >= max_hold):
                held = False
        elif enter[i]:
            held, age = True, 0
        out[i] = 1.0 if held else 0.0
    return out


def _rule_positions(rule, close, n_top, market):
    import indicators as ind
    pos = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    risk_on = (market > market.rolling(200, min_periods=200).mean()).reindex(close.index).fillna(False)
    if rule == "spy_200d_filter":
        pos[:] = 1.0
        pos = pos.mul(risk_on.astype(float), axis=0)
    elif rule == "rsi2_dip":
        for c in close.columns:
            s = close[c]
            enter = ((ind.rsi(s, 2) < 10) & (s > s.rolling(200, min_periods=200).mean())).to_numpy()
            pos[c] = _state_machine(enter, (s > s.rolling(5).mean()).to_numpy(), max_hold=10)
    elif rule == "donchian_breakout":
        for c in close.columns:
            s = close[c]
            pos[c] = _state_machine((s > s.rolling(55).max().shift()).to_numpy(), (s < s.rolling(20).min().shift()).to_numpy())
    else:
        look = 252 if rule.endswith("12_1") else 126
        mom = close.shift(21) / close.shift(look) - 1
        period = close.index.to_period("M")
        month_end = pd.Series(close.index, index=close.index).groupby(period).transform("max") == close.index
        current = pd.Series(0.0, index=close.columns)
        rows = []
        for day in close.index:
            if month_end.loc[day]:
                m = mom.loc[day].dropna()
                current = pd.Series(0.0, index=close.columns)
                if len(m) and risk_on.loc[day]:
                    current[m.nlargest(n_top).index] = 1.0
            rows.append(current.to_numpy().copy())
        pos = pd.DataFrame(rows, index=close.index, columns=close.columns)
    count = pos.sum(axis=1).replace(0, np.nan)
    return pos.div(count, axis=0).fillna(0.0)


def op_rule_study(args):
    """Quick vectorized study of a candidate rule not in the engine (SPY 200d filter, RSI2 dip, Donchian, XS momentum, pairs)."""
    from metrics import performance
    from universe import CORE_ETFS
    rule = _choice(args, "rule", "spy_200d_filter", set(RULES))
    tickers = _tickers(args, required=False)
    if not tickers:
        if rule == "pairs_zscore":
            bad("pairs_zscore needs two tickers")
        tickers = list(CORE_ETFS)
    cost_bps = _num(args, "cost_bps", 3.0, 0, 100)
    n_top = _num(args, "top_n", 3, 1, 20, integer=True)
    panel, missing = load_panel_for(list(dict.fromkeys(tickers + ["SPY"])), _date(args.get("start")), _date(args.get("end")))
    use = [t for t in tickers if t in panel.close.columns]
    close, rets = panel.close[use].ffill(), panel.total_return()[use].fillna(0.0)
    if rule == "pairs_zscore":
        if len(use) != 2:
            bad("pairs_zscore needs exactly two tickers with data")
        a, b = np.log(close[use[0]]), np.log(close[use[1]])
        beta = (a.rolling(120).cov(b) / b.rolling(120).var()).clip(-3, 3)
        spread = a - beta * b
        z = ((spread - spread.rolling(120).mean()) / spread.rolling(120).std()).fillna(0.0).to_numpy()
        state, states = 0, []
        for v in z:
            if state == 0 and v < -2:
                state = 1
            elif state == 0 and v > 2:
                state = -1
            elif state != 0 and abs(v) < 0.5:
                state = 0
            states.append(state)
        s = pd.Series(states, index=close.index, dtype=float)
        pos = pd.DataFrame({use[0]: 0.5 * s, use[1]: -0.5 * s * beta.fillna(1.0)})
    else:
        pos = _rule_positions(rule, close, n_top, panel.close["SPY"])
    held = pos.shift().fillna(0.0)                          # decided at close t, held from t+1 (no lookahead)
    turnover = pos.diff().abs().sum(axis=1).fillna(0.0)
    daily = (held * rets).sum(axis=1) - turnover.shift().fillna(0.0) * cost_bps / 1e4
    warm = 240 if rule == "pairs_zscore" else 200
    if len(daily) > warm + 60:
        daily = daily.iloc[warm:]
    perf = performance(daily)
    exposure = held.abs().sum(axis=1).loc[daily.index]
    return {"rule": rule, "definition": RULES[rule], "tickers": use, "missing": missing, "costBpsPerTurnover": cost_bps,
            "evaluationStart": str(daily.index[0].date()), "end": str(daily.index[-1].date()), "performance": perf,
            "turnoverPerYear": float(turnover.loc[daily.index].sum() / len(daily) * TRADING_DAYS),
            "timeInMarket": float((exposure > 0).mean()),
            "currentPositions": {k: float(v) for k, v in pos.iloc[-1].items() if v},
            "caveat": "Quick vectorized study (close-to-close, flat cost); not an engine strategy, not walk-forward validated."}


def refresh_rule_studies():
    """Run each candidate rule once (CORE_ETFS; pairs on XLK/QQQ) and store the results for the master prompt."""
    from fetch_and_model import DATA_DIR, write_atomically
    out = {"generatedAt": time.strftime("%Y-%m-%d %H:%M"), "studies": {}}
    for rule in RULES:
        args = {"rule": rule, "tickers": ["XLK", "QQQ"]} if rule == "pairs_zscore" else {"rule": rule}
        try:
            res = op_rule_study(args)
            perf = res["performance"]
            out["studies"][rule] = clean({"definition": res["definition"], "tickers": res["tickers"],
                                          "window": [res["evaluationStart"], res["end"]], "cagr": perf.get("cagr"),
                                          "sharpe": perf.get("sharpe"), "maxDrawdown": perf.get("maxDrawdown"),
                                          "volatility": perf.get("volatility"), "turnoverPerYear": res["turnoverPerYear"],
                                          "timeInMarket": res["timeInMarket"]}, 4)
        except Exception as error:  # noqa: BLE001
            out["studies"][rule] = {"error": str(error)[:300]}
    write_atomically(DATA_DIR / "Research" / "rule_studies.json", json.dumps(out, indent=2).encode())
    return out


# ------------------------------------------------------------------ strategy card (source of the analyst's master prompt)
def _doc(obj):
    return inspect.cleandoc(obj.__doc__ or "")


def _defaults_of(func):
    return {k: v.default for k, v in inspect.signature(func).parameters.items() if v.default is not inspect.Parameter.empty}


def op_strategy_card(args):
    """Strategy rules/parameters read from the engine code + measured results from Massive/Research (for the prompt)."""
    import costs
    import daytrader as dt
    import intraday as it
    import paper_trader as pt
    import research as R
    import risk
    from fetch_and_model import DATA_DIR
    from strategies import CombinedStrategy, RegimeMeanReversion, TimeSeriesMomentum
    from strategies import combined as cbmod, meanrev as mrmod, tsmom as tsmod
    from universe import CORE_ETFS

    def read(name):
        path = DATA_DIR / "Research" / name
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        except (OSError, ValueError):
            return None

    PERF = ("totalReturn", "cagr", "volatility", "sharpe", "sharpeStdErr", "sortino", "maxDrawdown", "calmar",
            "turnoverPerYear", "avgGrossExposure", "costBpsPerYear")
    pick = lambda perf: {k: perf.get(k) for k in PERF if perf and perf.get(k) is not None}
    lists = lambda d: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}
    summary, day, rules = read("research_summary.json"), read("daytrade_summary.json"), read("rule_studies.json")
    measured = None
    if summary:
        bt = summary.get("backtests") or {}
        rows = {}
        for name, v in (bt.get("results") or {}).items():
            perf = v.get("performance") or {}
            row = pick(perf)
            stats_ = v.get("tradeStats") or {}
            row.update({k: stats_.get(k) for k in ("trades", "winRate") if stats_.get(k) is not None})
            var = (perf.get("valueAtRisk") or {}).get("99_1d")
            if var:
                row["var99_1d"] = {m: var.get(m) for m in ("normal", "studentT", "historical", "bootstrap")}
            moments = perf.get("dailyMoments") or {}
            if moments:
                row["dailySkew"], row["dailyExcessKurtosis"] = moments.get("skew"), moments.get("excessKurtosis")
            rows[name] = row
        wf = {}
        for name, v in (summary.get("walkForward") or {}).items():
            stab = v.get("parameterStability") or {}
            wf[name] = {"oosStart": v.get("oosStart"), "trials": v.get("trials"), "folds": v.get("folds"),
                        **pick((v.get("outOfSample") or {}).get("performance") or {}),
                        "rankCorrByFold": stab.get("trainTestRankCorrelationByFold"),
                        "selectedByFold": stab.get("selectedTrialByFold"), "distinctSelections": stab.get("distinctSelections")}
        measured = {"generatedAt": summary.get("generatedAt"), "data": summary.get("data"),
                    "evaluationStart": bt.get("evaluationStart"), "backtests": rows, "walkForward": wf,
                    "multiHorizon": summary.get("multiHorizon"), "normality": summary.get("normality"),
                    "caveats": summary.get("caveats")}
    daytrade = None
    if day:
        def keep(s):
            s = s or {}
            return {**{k: s.get(k) for k in ("sessions", "first", "last", "trades", "winRate", "avgR", "totalPnl", "skipped")},
                    **{k: (s.get("performance") or {}).get(k) for k in ("totalReturn", "sharpe", "maxDrawdown", "volatility")}}
        wfd = day.get("walkForward") or {}
        data = day.get("data") or {}
        daytrade = {"generatedAt": day.get("generatedAt"), "data": {k: v for k, v in data.items() if k != "symbols"},
                    "symbols": data.get("symbols"), "fullPeriodDefaults": keep(day.get("fullPeriodDefaults")),
                    "fullPeriodNoPdt": keep(day.get("fullPeriodDefaultsNoPdt")),
                    "walkForward": {"train": wfd.get("trainSessions"), "test": wfd.get("testSessions"), "trials": wfd.get("trials"),
                                    "chosen": wfd.get("chosen"), "outOfSample": keep(wfd.get("outOfSample")),
                                    "rankCorr": wfd.get("trainTestRankCorrelation")},
                    "caveats": day.get("caveats")}
    return {
        "generatedAt": time.strftime("%Y-%m-%d %H:%M"),
        "strategies": {
            "tsmom": {"doc": _doc(tsmod), "defaults": lists(TimeSeriesMomentum.defaults), "universe": list(CORE_ETFS)},
            "meanrev": {"doc": _doc(mrmod), "defaults": lists(RegimeMeanReversion.defaults)},
            "combined": {"doc": _doc(cbmod), "defaults": lists(CombinedStrategy.defaults), "paperConfig": pt.DEFAULT_CONFIG},
        },
        "research": {"warmupSessions": R.WARMUP, "grids": R.GRIDS, "walkForwardDefaults": _defaults_of(R.walk_forward),
                     "horizons": list(R.HORIZONS), "normalityHorizons": list(R.NORMALITY_HORIZONS)},
        "daytrade": {"doc": _doc(it), "params": it.ORBParams().as_dict(), "grid": dt.GRID, "pdtEquity": it.PDT_EQUITY},
        "paper": {"doc": _doc(pt).split("\n\n")[0], "defaults": pt.DEFAULTS},
        "risk": {"doc": _doc(risk), "blendedVol": _defaults_of(risk.blended_vol),
                 "portfolioVolTarget": _defaults_of(risk.portfolio_vol_target),
                 "circuitBreaker": {"daily_loss_limit": risk.CircuitBreaker.daily_loss_limit, "cooldown": risk.CircuitBreaker.cooldown},
                 "valueAtRisk": _defaults_of(risk.value_at_risk)},
        "costs": {"doc": _doc(costs).split("\n\n")[0], "schwab": costs.CostModel.for_broker("schwab").as_dict()},
        "candidateRules": {"definitions": RULES, "studies": (rules or {}).get("studies"), "generatedAt": (rules or {}).get("generatedAt")},
        "measured": measured, "daytradeMeasured": daytrade,
    }


# ------------------------------------------------------------------ registry / runner
def op_macro(args):
    """Dollar, rates and credit-proxy snapshot or test. See macro_tools.op_macro."""
    from macro_tools import op_macro as _run
    return _run(args)


def op_attribute_returns(args):
    """Break strategy or paper-account returns into market beta, sector/bond bets and residual alpha."""
    import attribution
    return attribution.tool_attribute_returns(args)


def op_prob_forecast(args):
    """Latest walk-forward calibrated probabilities (purged OOS tape + frozen pack)."""
    import prob_walkforward
    return prob_walkforward.latest_for_tool(args)


OPS = {
    "describe": op_describe, "normality": op_normality, "bootstrap": op_bootstrap, "t_test": op_t_test,
    "correlation": op_correlation, "sharpe_inference": op_sharpe_inference,
    "ols": op_ols, "rolling_ols": op_rolling_ols, "factor_betas": op_factor_betas, "adf_kpss": op_adf_kpss,
    "engle_granger": op_engle_granger, "johansen": op_johansen, "arima": op_arima, "half_life_hurst": op_half_life_hurst,
    "garch": op_garch, "ewma": op_ewma, "var": op_var, "sizing": op_sizing, "monte_carlo": op_monte_carlo,
    "portfolio_opt": op_portfolio_opt, "indicators": op_indicators,
    "backtest": op_backtest, "walk_forward": op_walk_forward, "daytrade_backtest": op_daytrade_backtest,
    "rule_study": op_rule_study, "strategy_card": op_strategy_card,
    "macro": op_macro, "attribute_returns": op_attribute_returns, "prob_forecast": op_prob_forecast,
}
try:  # options / downside / vol-sleeve analyst ops (engine/options_tools.py; market data only, no order paths)
    import options_tools as _options_tools
    _options_tools.install(OPS, ToolError)
except ImportError:
    pass
try:
    import exhaustion as _exhaustion
    _exhaustion.install(OPS, ToolError)
except ImportError:
    pass


def run(request):
    started = time.time()
    op = str((request or {}).get("op", ""))
    if op not in OPS:
        return {"ok": False, "op": op, "error": {"code": "UNKNOWN_OP", "message": f"unknown op {op!r}; one of {sorted(OPS)}"}}
    args = request.get("args") or {}
    if not isinstance(args, dict):
        return {"ok": False, "op": op, "error": {"code": "BAD_ARGS", "message": "args must be an object"}}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                result = OPS[op](args)
        return {"ok": True, "op": op, "result": clean(result), "seconds": round(time.time() - started, 2)}
    except ToolError as error:
        return {"ok": False, "op": op, "error": {"code": error.code, "message": str(error)}}
    except SystemExit as error:   # engine helpers raise SystemExit for missing caches
        return {"ok": False, "op": op, "error": {"code": "NO_DATA", "message": str(error)[:500]}}
    except Exception as error:  # noqa: BLE001 - never crash the caller; report a typed failure
        return {"ok": False, "op": op, "error": {"code": "COMPUTE_ERROR", "message": f"{type(error).__name__}: {error}"[:500]}}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--list"]:
        print(json.dumps({k: (v.__doc__ or "").strip() for k, v in OPS.items()}, indent=2))
        return 0
    if argv[:1] == ["rule-studies"]:
        print(json.dumps(clean(refresh_rule_studies()), indent=2))
        return 0
    raw = sys.stdin.buffer.read().decode("utf-8")
    try:
        request = json.loads(raw)
    except ValueError as error:
        out = {"ok": False, "error": {"code": "BAD_JSON", "message": str(error)}}
    else:
        out = run(request)
    sys.stdout.write(json.dumps(out, allow_nan=False, default=str))
    sys.stdout.flush()
    return 0 if out.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
