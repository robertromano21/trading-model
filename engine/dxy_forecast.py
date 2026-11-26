"""Walk-forward forecast of the FRED DXY-style dollar proxy (1999-2026).

The target is the ICE-formula DXY proxy in Massive/Macro/dxy_proxy.csv, not ICE futures.
UUP is a tradable cross-check on the local price history (about two years). Real-time
CDS/CDX and swap quotes are not used. All features are point-in-time with lag >= 1
session; monthly OECD/FRED series are only used after a conservative publication lag.

Usage (from engine/):
  python dxy_forecast.py update      # incremental FRED refresh for forecast series
  python dxy_forecast.py research    # walk-forward, usefulness tests, write the report
  python dxy_forecast.py outlook     # latest recommend-only dollar outlook (JSON)
  python research.py dxy_forecast    # same as research

Trading use is off by default. The outlook is a recommendation only, and only if it
beats the random walk on the pre-declared gates.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from fetch_and_model import DATA_DIR, write_atomically
from macro_data import (
    FRED_START,
    ICE_EXPONENTS,
    MACRO_DIR,
    PROXY_DISCLAIMER,
    download_fred,
    load_dxy,
    load_macro,
    load_uup_returns,
)

OUT_DIR = DATA_DIR / "Research"
FRED_DIR = MACRO_DIR / "dxy_forecast" / "fred"
OUTLOOK_PATH = OUT_DIR / "dxy_forecast_outlook.json"
MACRO_OUTLOOK_PATH = MACRO_DIR / "dxy_outlook.json"
REPORT_MD = OUT_DIR / "dxy_forecast_report.md"
REPORT_JSON = OUT_DIR / "dxy_forecast_report.json"
LEDGER_PATH = OUT_DIR / "dxy_forecast_holdout_ledger.json"
TRIALS_CSV = OUT_DIR / "dxy_forecast_trials.csv"

HORIZONS = (5, 20, 63)
MIN_TRAIN = 1260
REFIT_EVERY = 21
HOLDOUT_START = pd.Timestamp("2023-01-03")
MONTHLY_LAG_DAYS = 45
DAILY_LAG = 1
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
PRIMARY_HORIZON = 20
TRADE_COST_BPS = 2.0
UUP_COST_BPS = 5.0
TRADE_THRESHOLDS = (0.0, 0.002, 0.005)
MAX_FEATURE_NAN = 0.35
EMBARGO_EXTRA = 5
TRADING_DAYS = 252

ICE_WEIGHTS = {pair: abs(exp) for pair, exp in ICE_EXPONENTS.items()}

FORECAST_FRED = {
    "DFF": "dff",
    "VIXCLS": "vix",
    "DCOILWTICO": "oil",
    "SP500": "spx",
    "NASDAQCOM": "nasdaq",
    "BAA10Y": "baa10y",
    "RBUSBIS": "bis_real",
    "IUDSOIA": "sonia",
    "CPIAUCSL": "cpi_us",
    "CP0000DEM086NEST": "cpi_de",
    "CP0000EZ19M086NEST": "cpi_ez",
    "IRLTLT01DEM156N": "lt_de",
    "IRLTLT01JPM156N": "lt_jp",
    "IRLTLT01GBM156N": "lt_gb",
    "IRLTLT01CAM156N": "lt_ca",
    "IRLTLT01SEM156N": "lt_se",
    "IRLTLT01CHM156N": "lt_ch",
    "IR3TIB01DEM156N": "st_de",
    "IR3TIB01JPM156N": "st_jp",
    "IR3TIB01GBM156N": "st_gb",
    "IR3TIB01CAM156N": "st_ca",
    "IR3TIB01SEM156N": "st_se",
    "IR3TIB01CHM156N": "st_ch",
}

LT_MAP = {"EURUSD": "lt_de", "USDJPY": "lt_jp", "GBPUSD": "lt_gb",
          "USDCAD": "lt_ca", "USDSEK": "lt_se", "USDCHF": "lt_ch"}
ST_MAP = {"EURUSD": "st_de", "USDJPY": "st_jp", "GBPUSD": "st_gb",
          "USDCAD": "st_ca", "USDSEK": "st_se", "USDCHF": "st_ch"}

FEATURE_GROUPS = {
    "all": None,
    "no_ppp": ("ppp_dev", "cpi_ppp_dev"),
    "no_risk": ("vix", "d_vix_21", "credit_baa10y", "d_credit_21", "eq_ret_21", "eq_dd_252"),
    "mom_carry": ("mom_21", "mom_63", "mom_126", "mom_252", "carry_3m", "diff_3m", "diff_10y"),
}

# Pre-specified signs for a stronger-dollar forecast (+1 helps the name, -1 drags).
# Textbook / long-run priors only. The earlier XLF.dxy_mom.h20 result used 2024-26
# data and is not used for selection (see the holdout ledger).
SECTOR_SIGNS = {
    "EEM": -1, "EFA": -1, "XLE": -1, "XLB": -1, "XLK": -1, "SPY": -1, "XLF": 0,
}

PROXY_NOTE = (
    "The forecast target is a DXY-style ICE-formula proxy from FRED FX, not the ICE "
    "futures print. Foreign 2-year yields are not on free FRED for the full basket; "
    "the short differential uses 3-month rates (OECD monthly, plus SONIA for GBP). "
    "The long differential uses OECD 10-year government yields. Monthly series are "
    f"first used {MONTHLY_LAG_DAYS} days after the observation date (no ALFRED vintages). "
    "Japan CPI on free FRED stops in 2021, so PPP valuation uses the BIS real broad "
    "dollar (RBUSBIS) as the primary slow measure, with a CPI-relative deviation when "
    "both US and German/Euro HICP are available. "
    + PROXY_DISCLAIMER
)


def _today():
    return datetime.now().strftime("%Y-%m-%d")


def _clean(value, digits=6):
    if value is None:
        return None
    try:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return round(value, digits)


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)[:19]
    if isinstance(obj, (np.floating, float)):
        return _clean(float(obj))
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def log_holdout(purpose, looked=False, extra=None):
    """Append a record whenever research touches (or could touch) the holdout window."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger = []
    if LEDGER_PATH.is_file():
        try:
            ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8")) or []
        except json.JSONDecodeError:
            ledger = []
    entry = {
        "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tz": "America/New_York",
        "purpose": purpose,
        "lookedAtHoldout": bool(looked),
        "holdoutStart": str(HOLDOUT_START.date()),
    }
    if extra:
        entry["extra"] = _json_ready(extra)
    ledger.append(entry)
    write_atomically(LEDGER_PATH, json.dumps(ledger, indent=2).encode())
    return entry


def update_forecast_fred(full=False, pause=1.0):
    """Incremental, paced FRED download. Never touches Massive FX."""
    FRED_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    ids = list(FORECAST_FRED)
    for i in range(0, len(ids), 6):
        batch = ids[i:i + 6]
        start = FRED_START
        if not full:
            existing = []
            for sid in batch:
                path = FRED_DIR / f"{sid}.csv"
                if path.is_file():
                    try:
                        idx = pd.read_csv(path, parse_dates=[0], index_col=0).index
                        if len(idx):
                            existing.append((idx.max() - pd.Timedelta(days=15)).strftime("%Y-%m-%d"))
                    except (OSError, ValueError, pd.errors.EmptyDataError):
                        pass
            if existing and len(existing) == len(batch):
                start = min(existing)
        try:
            frame = download_fred(batch, start=start, timeout=90)
        except Exception as error:  # noqa: BLE001
            frame = None
            written.setdefault("_errors", []).append({"batch": batch, "error": f"{type(error).__name__}: {error}"})
        if frame is not None:
            for sid in batch:
                if sid not in frame.columns:
                    continue
                col = frame[[sid]].dropna()
                path = FRED_DIR / f"{sid}.csv"
                if path.is_file() and not full:
                    prev = pd.read_csv(path, parse_dates=[0], index_col=0)
                    col = pd.concat([prev, col]).sort_index()
                    col = col[~col.index.duplicated(keep="last")]
                col.to_csv(path, index_label="date")
                written[sid] = {"rows": int(len(col)), "start": str(col.index[0].date()),
                                "end": str(col.index[-1].date())}
        time.sleep(pause)
    return written


def _read_series(path):
    path = Path(path)
    if not path.is_file():
        return None
    frame = pd.read_csv(path, parse_dates=[0], index_col=0)
    if frame.empty:
        return None
    return frame.iloc[:, 0].replace(".", np.nan).astype(float).dropna()


def load_forecast_fred():
    out = {}
    for sid, name in FORECAST_FRED.items():
        series = _read_series(FRED_DIR / f"{sid}.csv")
        if series is not None:
            series.name = name
            out[name] = series
    return out


def apply_availability_lag(series, lag_days, monthly=False):
    """Shift a dated observation so it is only visible lag_days after its observation date.

    For monthly OECD/FRED series the observation is typically the first of the month.
    A lag of 45 days means January is first used around mid-March. This is a
    conservative calendar rule, not an ALFRED vintage.
    """
    if series is None or series.empty:
        return series
    s = series.sort_index().copy()
    s.index = pd.to_datetime(s.index)
    s.index = s.index + pd.DateOffset(days=int(lag_days))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def to_daily(series, calendar, lag_days=0, monthly=False):
    """Reindex a (possibly lagged) series onto a daily calendar with no lookahead."""
    if series is None or series.empty:
        return pd.Series(index=calendar, dtype=float)
    s = apply_availability_lag(series, lag_days, monthly=monthly) if lag_days else series.sort_index()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    aligned = s.reindex(s.index.union(calendar)).sort_index().ffill()
    return aligned.reindex(calendar)


def weighted_composite(parts, weights):
    """DXY-weighted average of available legs; renormalize over non-null legs each day."""
    frame = pd.DataFrame(parts)
    if frame.empty:
        return pd.Series(dtype=float)
    w = pd.Series({k: float(weights[k]) for k in frame.columns if k in weights}, dtype=float)
    if w.empty:
        return pd.Series(index=frame.index, dtype=float)
    available = frame.notna().astype(float).mul(w, axis=1)
    numer = frame.mul(w, axis=1).sum(axis=1, min_count=1)
    denom = available.sum(axis=1)
    out = numer / denom.replace(0, np.nan)
    out.name = "composite"
    return out


def forward_return(level, horizon):
    """Close-to-close return over the next horizon sessions. Label at t uses level[t+h]."""
    level = level.astype(float)
    return (level.shift(-int(horizon)) / level) - 1.0


def _pct(series, window):
    return series.pct_change(int(window), fill_method=None)


def build_features(calendar=None, lag=DAILY_LAG):
    """Point-in-time feature panel aligned to the DXY proxy calendar.

    Daily FRED/DXY values dated t are shifted by ``lag`` sessions (default 1). Monthly
    series use MONTHLY_LAG_DAYS before being joined and then receive the same session lag.
    """
    dxy = load_dxy(lag=0)
    if dxy is None or dxy.empty:
        raise RuntimeError("DXY proxy is missing. Run python macro_data.py update first.")
    dxy = dxy.astype(float).dropna().sort_index()
    calendar = pd.DatetimeIndex(calendar if calendar is not None else dxy.index)
    calendar = calendar.sort_values().unique()

    raw = load_forecast_fred()
    daily_keys = ("dff", "vix", "oil", "spx", "nasdaq", "baa10y", "sonia")
    daily = {}
    for key in daily_keys:
        if key in raw:
            daily[key] = to_daily(raw[key], calendar, lag_days=0)

    monthly_keys = [k for k in raw if k not in daily_keys]
    monthly = {k: to_daily(raw[k], calendar, lag_days=MONTHLY_LAG_DAYS, monthly=True) for k in monthly_keys}

    macro = load_macro(lag=0)
    dgs2 = dgs10 = dgs3m = hy_oas = None
    if macro is not None and not macro.empty:
        macro = macro.reindex(calendar).ffill()
        dgs2 = macro["dgs2"] if "dgs2" in macro.columns else None
        dgs10 = macro["dgs10"] if "dgs10" in macro.columns else None
        dgs3m = macro["dgs3mo"] if "dgs3mo" in macro.columns else None
        hy_oas = macro["hy_oas"] if "hy_oas" in macro.columns else None

    dxy_level = dxy.reindex(calendar).ffill()
    lt_parts = {pair: monthly[col] for pair, col in LT_MAP.items() if col in monthly}
    st_parts = {pair: monthly[col] for pair, col in ST_MAP.items() if col in monthly}
    if "sonia" in daily:
        st_parts["GBPUSD"] = daily["sonia"]
    cpi_parts = {}
    if "cpi_de" in monthly:
        cpi_parts["EURUSD"] = monthly["cpi_de"]
    elif "cpi_ez" in monthly:
        cpi_parts["EURUSD"] = monthly["cpi_ez"]

    foreign_10y = weighted_composite(lt_parts, ICE_WEIGHTS)
    foreign_3m = weighted_composite(st_parts, ICE_WEIGHTS)
    foreign_cpi = weighted_composite(cpi_parts, ICE_WEIGHTS) if cpi_parts else None

    us_10y = dgs10 if dgs10 is not None else pd.Series(index=calendar, dtype=float)
    us_3m = dgs3m if dgs3m is not None else pd.Series(index=calendar, dtype=float)
    us_2y = dgs2 if dgs2 is not None else pd.Series(index=calendar, dtype=float)
    dff = daily.get("dff", pd.Series(index=calendar, dtype=float))

    diff_10y = us_10y - foreign_10y
    diff_3m = us_3m - foreign_3m
    fed_path = us_2y - dff
    carry_3m = diff_3m / 100.0

    equity = daily.get("spx")
    nasdaq = daily.get("nasdaq")
    eq = equity.copy() if equity is not None else None
    if nasdaq is not None:
        eq = nasdaq.copy() if eq is None else eq.combine_first(nasdaq)

    vix = daily.get("vix")
    credit = daily.get("baa10y")
    if credit is None and hy_oas is not None:
        credit = hy_oas
    oil = daily.get("oil")
    bis = monthly.get("bis_real")

    feats = pd.DataFrame(index=calendar)
    feats["diff_10y"] = diff_10y
    feats["d_diff_10y_63"] = diff_10y.diff(63)
    feats["diff_3m"] = diff_3m
    feats["d_diff_3m_63"] = diff_3m.diff(63)
    feats["fed_path"] = fed_path
    feats["d_fed_path_21"] = fed_path.diff(21)
    feats["mom_21"] = _pct(dxy_level, 21)
    feats["mom_63"] = _pct(dxy_level, 63)
    feats["mom_126"] = _pct(dxy_level, 126)
    feats["mom_252"] = _pct(dxy_level, 252)
    feats["carry_3m"] = carry_3m
    if vix is not None:
        feats["vix"] = vix
        feats["d_vix_21"] = vix.diff(21)
    if credit is not None:
        feats["credit_baa10y"] = credit
        feats["d_credit_21"] = credit.diff(21)
    if eq is not None:
        feats["eq_ret_21"] = _pct(eq, 21)
        peak = eq.rolling(252, min_periods=60).max()
        feats["eq_dd_252"] = eq / peak - 1.0
    if oil is not None:
        feats["oil_ret_21"] = _pct(oil, 21)
        feats["oil_ret_5"] = _pct(oil, 5)
    if bis is not None:
        mu = bis.rolling(1260, min_periods=252).mean()
        sd = bis.rolling(1260, min_periods=252).std()
        feats["ppp_dev"] = (bis - mu) / sd.replace(0, np.nan)
    if foreign_cpi is not None and "cpi_us" in monthly:
        rel = (monthly["cpi_us"] / foreign_cpi).replace(0, np.nan)
        fair = np.log(rel)
        dev = np.log(dxy_level) - fair
        mu = dev.rolling(1260, min_periods=252).mean()
        sd = dev.rolling(1260, min_periods=252).std()
        feats["cpi_ppp_dev"] = (dev - mu) / sd.replace(0, np.nan)

    surprise = _surprise_feature(calendar)
    if surprise is not None:
        feats["surprise_z"] = surprise

    if lag and lag > 0:
        feats = feats.shift(int(lag))
    feats.index = pd.DatetimeIndex(calendar)
    feats.attrs["notes"] = {
        "dailyLagSessions": int(lag),
        "monthlyLagDays": MONTHLY_LAG_DAYS,
        "foreign10y": sorted(lt_parts),
        "foreign3m": sorted(st_parts),
        "cpiLegs": sorted(cpi_parts),
        "equityProxy": "SP500 when available, else NASDAQCOM",
        "pppPrimary": "RBUSBIS 5y z-score" if "ppp_dev" in feats.columns else None,
    }
    return feats, dxy_level.reindex(calendar)


def _surprise_feature(calendar):
    """Optional GB-14 standardized surprise series, lagged to the session after the event."""
    try:
        from macro_surprise import latest_surprises
        items = latest_surprises(n=10_000)
    except Exception:  # noqa: BLE001
        return None
    if not items:
        return None
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        day = item.get("date") or item.get("asOf") or item.get("released") or item.get("when")
        z = item.get("z") or item.get("standardized") or item.get("surpriseZ") or item.get("surprise")
        if day is None or z is None:
            continue
        try:
            rows.append((pd.Timestamp(str(day)[:10]), float(z)))
        except (TypeError, ValueError):
            continue
    if not rows:
        return None
    series = pd.Series({d: z for d, z in rows}).sort_index().groupby(level=0).mean()
    aligned = series.reindex(series.index.union(calendar)).sort_index().ffill()
    return aligned.reindex(calendar).shift(1)


def _as_xy(y, X, dropna=True):
    frame = pd.concat([y.rename("y"), X], axis=1, join="inner")
    if dropna:
        frame = frame.dropna()
    if frame.empty or frame.drop(columns=["y"]).empty:
        return None, None, None
    return frame["y"].astype(float), frame.drop(columns=["y"]).astype(float), frame.index


def _standardize(train, test, eps=1e-8):
    mu = train.mean(axis=0)
    sd = train.std(axis=0, ddof=1).replace(0, np.nan).fillna(1.0).clip(lower=eps)
    return (train - mu) / sd, (test - mu) / sd, mu, sd


def fit_ridge(y, X, alpha=1.0):
    """Ridge with an unpenalized intercept. X should already be standardized."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    X1 = np.column_stack([np.ones(n), X])
    A = X1.T @ X1
    if p:
        A[1:, 1:] = A[1:, 1:] + float(alpha) * np.eye(p)
    try:
        beta = np.linalg.solve(A, X1.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, X1.T @ y, rcond=None)[0]
    return beta


def predict_ridge(X, beta):
    X = np.asarray(X, dtype=float)
    X1 = np.column_stack([np.ones(len(X)), X])
    return X1 @ np.asarray(beta, dtype=float)


def fit_logit(y01, X):
    y01 = np.asarray(y01, dtype=float)
    X = np.asarray(X, dtype=float)
    if len(y01) < 40 or y01.min() == y01.max():
        return None
    X1 = sm.add_constant(X, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.Logit(y01, X1).fit(disp=False, maxiter=200, method="lbfgs")
            return np.asarray(res.params, dtype=float)
        except Exception:  # noqa: BLE001
            try:
                res = sm.Logit(y01, X1).fit(disp=False, maxiter=200)
                return np.asarray(res.params, dtype=float)
            except Exception:  # noqa: BLE001
                return None


def predict_logit(X, beta):
    if beta is None:
        return np.full(len(X), 0.5)
    X = np.asarray(X, dtype=float)
    X1 = np.column_stack([np.ones(len(X)), X])
    z = np.clip(X1 @ np.asarray(beta, dtype=float), -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def platt_calibrate(scores, y01):
    """Fit a logistic map from raw probabilities to labels. Returns (a, b) for a + b*logit(p)."""
    scores = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    y01 = np.asarray(y01, dtype=float)
    if len(scores) < 40 or y01.min() == y01.max():
        return 0.0, 1.0
    logits = np.log(scores / (1.0 - scores)).reshape(-1, 1)
    beta = fit_logit(y01, logits)
    if beta is None or len(beta) < 2:
        return 0.0, 1.0
    return float(beta[0]), float(beta[1])


def apply_platt(scores, a, b):
    scores = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(scores / (1.0 - scores))
    z = np.clip(a + b * logits, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def oos_r2(y, pred, bench=None):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    bench = np.zeros_like(y) if bench is None else np.asarray(bench, dtype=float)
    denom = float(np.sum((y - bench) ** 2))
    if denom <= 0:
        return None
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def clark_west(y, pred, bench=None, lags=None):
    """Clark-West (2007) test that model MSE is below a nested benchmark (default: RW=0)."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    bench = np.zeros_like(y) if bench is None else np.asarray(bench, dtype=float)
    if len(y) < 20:
        return {"t": None, "pValue": None, "n": int(len(y))}
    e_b = y - bench
    e_m = y - pred
    adj = e_b ** 2 - e_m ** 2 + (pred - bench) ** 2
    lags = int(lags if lags is not None else max(1, int(round(1.5 * (len(y) ** (1 / 3))))))
    try:
        res = sm.OLS(adj, np.ones(len(adj))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        t = float(res.tvalues[0])
        p = float(1.0 - stats.norm.cdf(t))
        return {"t": _clean(t, 4), "pValue": _clean(p, 6), "n": int(len(y)), "meanAdj": _clean(float(adj.mean()), 8)}
    except Exception as error:  # noqa: BLE001
        return {"t": None, "pValue": None, "n": int(len(y)), "error": f"{type(error).__name__}: {error}"}


def hit_rate(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = (y != 0) & (pred != 0)
    if mask.sum() < 5:
        return None
    return float(np.mean(np.sign(y[mask]) == np.sign(pred[mask])))


def information_coefficient(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y) < 10 or np.std(y) == 0 or np.std(pred) == 0:
        return None
    rho, p = stats.spearmanr(y, pred)
    if not math.isfinite(float(rho)):
        return None
    return {"spearman": _clean(float(rho), 4), "pValue": _clean(float(p), 6)}


def brier_score(y01, prob):
    y01 = np.asarray(y01, dtype=float)
    prob = np.asarray(prob, dtype=float)
    if len(y01) < 5:
        return None
    return float(np.mean((prob - y01) ** 2))


def deflated_sharpe(returns, n_trials, periods=TRADING_DAYS):
    """Bailey & Lopez de Prado deflated Sharpe on a session return series."""
    r = pd.Series(returns).dropna().astype(float)
    n = int(len(r))
    if n < 30 or r.std(ddof=1) == 0 or n_trials < 1:
        return {"deflatedSharpeProb": None, "sharpe": None, "n": n, "nTrials": int(n_trials)}
    sr = float(r.mean() / r.std(ddof=1) * math.sqrt(periods))
    sr_daily = sr / math.sqrt(periods)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0
    gamma = 0.5772156649
    n_trials = max(int(n_trials), 1)
    if n_trials == 1:
        sr0 = 0.0
    else:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(1.0 / n) * ((1.0 - gamma) * z1 + gamma * z2)
    se_sq = (1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * sr_daily ** 2) / max(n - 1, 1)
    se = math.sqrt(max(se_sq, 1e-18))
    z = (sr_daily - sr0) / se
    return {
        "sharpe": _clean(sr, 4),
        "deflatedSharpeProb": _clean(float(stats.norm.cdf(z)), 4),
        "expectedMaxSharpeDaily": _clean(sr0, 6),
        "n": n,
        "nTrials": int(n_trials),
        "skew": _clean(skew, 4),
        "excessKurtosis": _clean(kurt - 3.0, 4),
    }


def _feature_subset(X, group):
    drop = FEATURE_GROUPS.get(group)
    if drop is None:
        return X
    if group == "mom_carry":
        keep = [c for c in FEATURE_GROUPS["mom_carry"] if c in X.columns]
        return X[keep] if keep else X
    keep = [c for c in X.columns if c not in drop]
    return X[keep] if keep else X


def walk_forward_ridge(y, X, horizon, alpha=1.0, min_train=MIN_TRAIN, refit_every=REFIT_EVERY,
                       start=None, end=None, embargo_extra=EMBARGO_EXTRA):
    """Expanding, purged walk-forward. Labels must be fully resolved before they enter the fit.

    A prediction at date t uses a model fit only on dates s with s + horizon + embargo_extra < t.
    """
    y, X, idx = _as_xy(y, X, dropna=False)
    if y is None:
        return pd.Series(dtype=float), {"error": "not enough overlap", "nFits": 0, "meanAbsBeta": {}}
    if start is not None:
        mask = idx >= pd.Timestamp(start)
        y, X, idx = y.loc[mask], X.loc[mask], idx[mask]
    if end is not None:
        mask = idx <= pd.Timestamp(end)
        y, X, idx = y.loc[mask], X.loc[mask], idx[mask]
    pred = pd.Series(index=idx, dtype=float)
    beta_hist = []
    n_fits = 0
    last_pack = None
    last_fit_i = -10**9
    embargo = int(horizon) + int(embargo_extra)
    dates = list(idx)
    for i, t in enumerate(dates):
        j = i - embargo
        if j < min_train:
            continue
        if last_pack is None or (i - last_fit_i) >= refit_every:
            y_tr = y.iloc[:j]
            X_tr = X.iloc[:j]
            keep = [c for c in X_tr.columns if X_tr[c].notna().mean() >= (1.0 - MAX_FEATURE_NAN)]
            if not keep:
                continue
            pack = pd.concat([y_tr, X_tr[keep]], axis=1).dropna()
            if len(pack) < min_train:
                continue
            y_fit = pack.iloc[:, 0]
            X_fit = pack.iloc[:, 1:]
            X_std, _, mu, sd = _standardize(X_fit, X_fit)
            beta = fit_ridge(y_fit.values, X_std.values, alpha=alpha)
            last_pack = {"beta": beta, "mu": mu, "sd": sd, "cols": list(X_fit.columns)}
            last_fit_i = i
            n_fits += 1
            beta_hist.append(pd.Series(beta[1:], index=X_fit.columns))
        if last_pack is None:
            continue
        row = X.iloc[i].reindex(last_pack["cols"])
        if row.isna().any():
            continue
        z = (row - last_pack["mu"]) / last_pack["sd"]
        pred.iloc[i] = float(predict_ridge(z.values.reshape(1, -1), last_pack["beta"])[0])
    mean_abs = {}
    if beta_hist:
        mat = pd.concat(beta_hist, axis=1).T.abs().mean()
        mean_abs = {k: _clean(v, 5) for k, v in mat.items()}
    return pred, {"nFits": n_fits, "meanAbsBeta": mean_abs, "nPred": int(pred.notna().sum())}


def walk_forward_logit(y, X, horizon, min_train=MIN_TRAIN, refit_every=REFIT_EVERY,
                       start=None, end=None, embargo_extra=EMBARGO_EXTRA):
    """Expanding, purged logistic for P(forward return > 0), with Platt calibration on a purged train tail."""
    y, X, idx = _as_xy(y, X, dropna=False)
    if y is None:
        return pd.Series(dtype=float), {"nFits": 0}
    if start is not None:
        mask = idx >= pd.Timestamp(start)
        y, X, idx = y.loc[mask], X.loc[mask], idx[mask]
    if end is not None:
        mask = idx <= pd.Timestamp(end)
        y, X, idx = y.loc[mask], X.loc[mask], idx[mask]
    pred = pd.Series(index=idx, dtype=float)
    n_fits = 0
    last_pack = None
    last_fit_i = -10**9
    embargo = int(horizon) + int(embargo_extra)
    dates = list(idx)
    for i, t in enumerate(dates):
        j = i - embargo
        if j < min_train:
            continue
        if last_pack is None or (i - last_fit_i) >= refit_every:
            keep = [c for c in X.columns if X.iloc[:j][c].notna().mean() >= (1.0 - MAX_FEATURE_NAN)]
            if not keep:
                continue
            pack = pd.concat([y.iloc[:j], X.iloc[:j][keep]], axis=1).dropna()
            if len(pack) < min_train:
                continue
            cut = int(len(pack) * 0.8)
            if cut < 80 or (len(pack) - cut) < 40:
                y_fit, X_fit = pack.iloc[:, 0], pack.iloc[:, 1:]
                y_cal, X_cal = None, None
            else:
                y_fit, X_fit = pack.iloc[:cut, 0], pack.iloc[:cut, 1:]
                y_cal, X_cal = pack.iloc[cut:, 0], pack.iloc[cut:, 1:]
            X_std, Xc_std, mu, sd = _standardize(X_fit, X_cal if X_cal is not None else X_fit)
            beta = fit_logit((y_fit > 0).astype(float).values, X_std.values)
            if beta is None:
                continue
            raw = predict_logit(Xc_std.values, beta) if X_cal is not None else predict_logit(X_std.values, beta)
            labels = (y_cal > 0).astype(float).values if X_cal is not None else (y_fit > 0).astype(float).values
            a, b = platt_calibrate(raw, labels)
            last_pack = {"beta": beta, "mu": mu, "sd": sd, "cols": list(X_fit.columns), "a": a, "b": b}
            last_fit_i = i
            n_fits += 1
        if last_pack is None:
            continue
        row = X.iloc[i].reindex(last_pack["cols"])
        if row.isna().any():
            continue
        z = (row - last_pack["mu"]) / last_pack["sd"]
        raw = predict_logit(z.values.reshape(1, -1), last_pack["beta"])[0]
        pred.iloc[i] = float(apply_platt([raw], last_pack["a"], last_pack["b"])[0])
    return pred, {"nFits": n_fits, "nPred": int(pred.notna().sum())}


def naive_momentum(level, horizon, lag=DAILY_LAG):
    return _pct(level, int(horizon)).shift(int(lag))


def naive_carry(carry_annual, horizon):
    return carry_annual.astype(float) * (int(horizon) / TRADING_DAYS)


def score_predictions(y, pred, horizon, label, window=None):
    both = pd.concat([y.rename("y"), pred.rename("p")], axis=1).dropna()
    if window == "pre":
        both = both.loc[both.index < HOLDOUT_START]
    elif window == "holdout":
        both = both.loc[both.index >= HOLDOUT_START]
    if both.empty:
        return {"label": label, "horizon": int(horizon), "window": window, "n": 0}
    yw, pw = both["y"].values, both["p"].values
    ic = information_coefficient(yw, pw)
    is_prob = str(label).startswith("logit")
    # Logistic outputs are probabilities, not expected returns, so OOS R2 vs a
    # return-space random walk is not a comparable metric.
    cw = None if is_prob else clark_west(yw, pw, lags=max(int(horizon), 1))
    out = {
        "label": label,
        "horizon": int(horizon) if horizon else None,
        "window": window,
        "n": int(len(both)),
        "start": str(both.index[0].date()),
        "end": str(both.index[-1].date()),
        "oosR2": None if is_prob else _clean(oos_r2(yw, pw), 4),
        "clarkWest": cw,
        "hitRate": _clean(hit_rate(yw, pw), 4),
        "ic": ic,
        "predStd": _clean(float(np.std(pw, ddof=1)), 6),
        "yStd": _clean(float(np.std(yw, ddof=1)), 6),
        "meanPred": _clean(float(np.mean(pw)), 6),
        "meanY": _clean(float(np.mean(yw)), 6),
    }
    # Brier when pred looks like a probability.
    if np.nanmin(pw) >= 0 and np.nanmax(pw) <= 1:
        out["brier"] = _clean(brier_score((yw > 0).astype(float), pw), 4)
    return out


def score_by_decade(y, pred):
    both = pd.concat([y.rename("y"), pred.rename("p")], axis=1).dropna()
    rows = []
    if both.empty:
        return rows
    decade = (both.index.year // 10) * 10
    for dec, part in both.groupby(decade):
        row = score_predictions(part["y"], part["p"], horizon=0, label=f"{int(dec)}s", window="decade")
        row["decade"] = int(dec)
        row["horizon"] = None
        rows.append(row)
    return rows


def score_by_vix_regime(y, pred, vix):
    both = pd.concat([y.rename("y"), pred.rename("p"), vix.rename("vix")], axis=1).dropna()
    rows = []
    if both.empty or both["vix"].notna().sum() < 40:
        return rows
    thresh = float(both["vix"].median())
    for name, part in (("vix_low", both[both["vix"] <= thresh]), ("vix_high", both[both["vix"] > thresh])):
        if len(part) < 20:
            continue
        row = score_predictions(part["y"], part["p"], horizon=0, label=name, window="regime")
        row["horizon"] = None
        row["vixSplit"] = _clean(thresh, 2)
        rows.append(row)
    return rows


def trading_returns(signal, asset_ret, threshold=0.0, cost_bps=TRADE_COST_BPS, lag=1):
    """Long/short/flat the asset from a forecast. Position decided at t, return earned at t+lag."""
    sig = signal.astype(float)
    pos = pd.Series(0.0, index=sig.index)
    pos[sig > threshold] = 1.0
    pos[sig < -threshold] = -1.0
    pos = pos.shift(int(lag)).reindex(asset_ret.index)
    r = asset_ret.astype(float).reindex(pos.index)
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (float(cost_bps) / 10_000.0)
    pnl = pos * r - cost
    return pnl.rename("pnl"), pos.rename("pos")


def _panel_returns(tickers):
    try:
        from market_data import load_panel
        panel = load_panel(tickers=list(tickers), use_long_history=True)
        return panel.total_return(), panel, None
    except Exception as error:  # noqa: BLE001
        return None, None, error


def usefulness_sectors(forecast, horizon=PRIMARY_HORIZON):
    """Holdout-only evaluation (local ETF history is 2024-11-25 onward, inside the holdout)."""
    tickers = list(dict.fromkeys(["SPY", "XLF", "XLK", "XLE", "XLB", "EEM", "EFA", "QQQ", "UUP"]))
    rets, panel, err = _panel_returns(tickers)
    if rets is None:
        return {"available": False, "reason": f"price panel unavailable: {err}"}
    log_holdout(
        "sector usefulness: local ETF history 2024-11-25 onward sits inside the 2023+ holdout; "
        "signs were pre-specified and no threshold was tuned on this window. The earlier "
        "XLF.dxy_mom.h20 finding is not used for selection.",
        looked=True,
        extra={"tickers": tickers, "sessions": int(len(rets))},
    )
    dxy = load_dxy(lag=1)
    dxy_mom = _pct(dxy, int(horizon)) if dxy is not None else None
    fwd_forecast = forecast.reindex(rets.index)
    rows = []
    for ticker in tickers:
        if ticker not in rets.columns:
            continue
        fwd = forward_return((1 + rets[ticker]).cumprod(), horizon)
        both = pd.concat([fwd.rename("y"), fwd_forecast.rename("f")], axis=1).dropna()
        if dxy_mom is not None:
            both = both.join(dxy_mom.rename("mom"), how="left")
        if len(both) < 40:
            continue
        row = {
            "ticker": ticker,
            "preSpecifiedSign": SECTOR_SIGNS.get(ticker),
            "n": int(len(both)),
            "forecast": {
                "oosR2": _clean(oos_r2(both["y"].values, both["f"].values), 4),
                "clarkWest": clark_west(both["y"].values, both["f"].values, lags=horizon),
                "hitRate": _clean(hit_rate(both["y"].values, both["f"].values), 4),
                "ic": information_coefficient(both["y"].values, both["f"].values),
            },
        }
        if "mom" in both.columns and both["mom"].notna().sum() > 40:
            m = both.dropna()
            row["laggedDxyMom"] = {
                "oosR2": _clean(oos_r2(m["y"].values, m["mom"].values), 4),
                "clarkWest": clark_west(m["y"].values, m["mom"].values, lags=horizon),
                "ic": information_coefficient(m["y"].values, m["mom"].values),
            }
        rows.append(row)

    names = [t for t, s in SECTOR_SIGNS.items() if s == -1 and t in rets.columns]
    tilt = None
    if names and fwd_forecast.notna().any():
        pos = np.sign(fwd_forecast.fillna(0.0))
        basket = rets[names].mean(axis=1)
        pnl, _ = trading_returns(-pos, basket, threshold=0.0, cost_bps=UUP_COST_BPS, lag=1)
        spy = rets["SPY"] if "SPY" in rets.columns else None
        tilt = {
            "names": names,
            "rule": "pre-specified: when the locked 20d dollar forecast is positive, short the equal-weight "
                    "EEM/EFA/XLE/XLB/XLK/SPY basket (and the reverse when negative); 1-session execution lag, "
                    f"{UUP_COST_BPS} bps one-way.",
            "pnl": _perf_brief(pnl.dropna()),
            "spy": _perf_brief(spy.reindex(pnl.dropna().index).dropna()) if spy is not None else None,
        }
    return {"available": True, "horizon": int(horizon), "rows": rows, "tilt": tilt,
            "sample": {"start": str(rets.index[0].date()), "end": str(rets.index[-1].date()),
                       "sessions": int(len(rets))}}


def usefulness_hedge(forecast, horizon=PRIMARY_HORIZON):
    """Forecast-sized dollar hedge on SPY, in the same next-session / after-cost spirit as GB-18.

    Does not edit vol_sleeve.py. Compares SPY, a constant-sign 10% UUP overlay, and a
    forecast-sized UUP overlay (long UUP when the dollar is forecast to strengthen).
    """
    rets, panel, err = _panel_returns(["SPY", "UUP"])
    if rets is None or "SPY" not in getattr(rets, "columns", []):
        return {"available": False, "reason": f"SPY/UUP panel unavailable: {err}"}
    log_holdout(
        "hedge usefulness on local SPY/UUP (2024-11-25 onward, inside the holdout). "
        "Hedge sign is pre-specified (long UUP when the locked dollar forecast is positive). "
        "No weight grid was searched on this window; weight is 10% * sign, or 10% * clipped z.",
        looked=True,
    )
    spy = rets["SPY"]
    uup = rets["UUP"] if "UUP" in rets.columns else None
    f = forecast.reindex(spy.index)
    normalizer = 0.02
    w = (0.10 * (f / normalizer)).clip(-0.10, 0.10).shift(1)
    const = pd.Series(0.10 * np.sign(f.fillna(0.0)), index=spy.index).shift(1)
    out = {"available": True, "spy": _perf_brief(spy.dropna()), "rule": {
        "constantUup": "10% long UUP when forecast>0, 10% short UUP when forecast<0, next session, "
                       f"{UUP_COST_BPS} bps one-way on the UUP leg only",
        "sizedUup": "UUP weight = clip(0.10 * forecast_20d / 0.02, -0.10, 0.10), same lag and costs. "
                    "0.02 normalizer and 10% cap were pre-specified.",
    }}
    if uup is None:
        out["reason"] = "UUP missing; cannot form the dollar hedge."
        return out

    def _book(weight):
        turnover = weight.diff().abs().fillna(weight.abs())
        cost = turnover * (UUP_COST_BPS / 10_000.0)
        return spy + weight.reindex(spy.index).fillna(0.0) * uup - cost

    out["constantHedge"] = _perf_brief(_book(const).dropna())
    out["sizedHedge"] = _perf_brief(_book(w).dropna())
    sleeve_path = OUT_DIR / "vol_sleeve_oos_returns.csv"
    if sleeve_path.is_file():
        try:
            sleeve = pd.read_csv(sleeve_path, parse_dates=[0], index_col=0).iloc[:, 0].astype(float)
            scale = (1.0 - 0.5 * w.reindex(sleeve.index).fillna(0.0) / 0.10).clip(0.5, 1.5)
            mixed = sleeve * scale.reindex(sleeve.index).fillna(1.0)
            out["volSleeveOverlay"] = {
                "note": "Optional overlay on the saved GB-18 OOS vol-sleeve returns (file not refit). "
                        "Scale = clip(1 - 0.5 * sized_UUP_weight / 0.10, 0.5, 1.5). Pre-specified, not tuned.",
                "sleeve": _perf_brief(sleeve.dropna()),
                "scaled": _perf_brief(mixed.dropna()),
            }
        except Exception as error:  # noqa: BLE001
            out["volSleeveOverlay"] = {"error": f"{type(error).__name__}: {error}"}
    return out


def usefulness_hmm(forecast, horizon=PRIMARY_HORIZON):
    """Does the locked forecast improve the macro-HMM / combined book, without editing regime_hmm.py?"""
    result = {"available": False}
    try:
        rets, panel, err = _panel_returns(["SPY", "IEF", "TLT", "HYG"])
        if rets is None or "SPY" not in rets.columns:
            result["incremental"] = {"available": False, "reason": f"panel: {err}"}
        else:
            log_holdout(
                "macro-HMM usefulness on the local 2-year equity panel (inside the holdout). "
                "The forecast is the locked OOS dollar forecast; no HMM hyperparameter search "
                "was run on this window.",
                looked=True,
            )
            macro = load_macro(lag=1)
            fwd = forward_return((1 + rets["SPY"]).cumprod(), horizon)
            cols = [c for c in ("dxy_ret20", "dxy_vol20", "curve_slope", "curve_level_chg20",
                                "hy_oas", "sofr_tbill_spread") if macro is not None and c in macro.columns]
            design = macro[cols].reindex(fwd.index) if cols else pd.DataFrame(index=fwd.index)
            design = design.join(forecast.rename("dxy_forecast"), how="left")
            both = pd.concat([fwd.rename("y"), design], axis=1).dropna()
            inc = {"n": int(len(both)), "features": cols}
            if len(both) >= 80 and cols:
                y = both["y"]
                X0 = both[cols]
                X1 = both[cols + ["dxy_forecast"]]
                min_tr = min(80, max(40, len(both) // 3))
                p0, _ = walk_forward_ridge(y, X0, horizon, alpha=1.0, min_train=min_tr,
                                           refit_every=10, embargo_extra=1)
                p1, _ = walk_forward_ridge(y, X1, horizon, alpha=1.0, min_train=min_tr,
                                           refit_every=10, embargo_extra=1)
                cmp = pd.concat([y.rename("y"), p0.rename("m0"), p1.rename("m1")], axis=1).dropna()
                if len(cmp) >= 30:
                    inc["macroOnly"] = {"oosR2": _clean(oos_r2(cmp["y"].values, cmp["m0"].values), 4),
                                        "clarkWest": clark_west(cmp["y"].values, cmp["m0"].values, lags=horizon)}
                    inc["macroPlusForecast"] = {"oosR2": _clean(oos_r2(cmp["y"].values, cmp["m1"].values), 4),
                                                "clarkWest": clark_west(cmp["y"].values, cmp["m1"].values, lags=horizon)}
                    inc["forecastAdds"] = bool((inc["macroPlusForecast"]["oosR2"] or -1) > (inc["macroOnly"]["oosR2"] or -1))
            result["incremental"] = inc
            result["available"] = True
    except Exception as error:  # noqa: BLE001
        result["incremental"] = {"error": f"{type(error).__name__}: {error}"}

    try:
        # Reuse a saved combined walk-forward equity curve if present (no HMM refit, no
        # edit to regime_hmm.py). Overlay is pre-specified and lagged one session.
        path = OUT_DIR / "walkforward_combined_oos_equity.csv"
        if path.is_file():
            frame = pd.read_csv(path, parse_dates=[0], index_col=0)
            col = "return" if "return" in frame.columns else ("equity" if "equity" in frame.columns else frame.columns[0])
            series = frame[col].astype(float)
            if col == "equity" or (series.min() > 0 and series.max() > 2):
                rets = series.pct_change(fill_method=None)
            else:
                rets = series
            f = forecast.reindex(rets.index)
            scale = (1.0 - 0.30 * (f / 0.02).clip(0.0, 1.0)).clip(0.70, 1.0)
            overlay = rets * scale.shift(1).reindex(rets.index).fillna(1.0)
            result["combinedMacroHmm"] = _perf_brief(rets.dropna())
            result["combinedMacroHmmForecastOverlay"] = _perf_brief(overlay.dropna())
            result["combinedNote"] = (
                "Overlay on the saved combined walk-forward OOS equity (file reused, HMM not refit): "
                "a pre-specified 0-30% gross cut is applied next session when the locked 20d dollar "
                "forecast is positive (cut = 0.30 * clip(forecast/0.02, 0, 1)). The saved book is the "
                "default combined walk-forward, not necessarily the macro-HMM variant."
            )
        result["available"] = True
    except Exception as error:  # noqa: BLE001
        result["combinedError"] = f"{type(error).__name__}: {error}"

    return result


def _perf_brief(r):
    r = pd.Series(r).dropna().astype(float)
    if r.empty:
        return {"n": 0}
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = float((eq / peak - 1.0).min())
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if r.std(ddof=1) else None
    years = len(r) / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 and eq.iloc[-1] > 0 else None
    return {
        "n": int(len(r)),
        "start": str(r.index[0].date()) if hasattr(r.index[0], "date") else str(r.index[0]),
        "end": str(r.index[-1].date()) if hasattr(r.index[-1], "date") else str(r.index[-1]),
        "totalReturn": _clean(float(eq.iloc[-1] - 1.0), 4),
        "cagr": _clean(cagr, 4),
        "sharpe": _clean(sharpe, 4),
        "maxDrawdown": _clean(dd, 4),
        "hitRate": _clean(float((r > 0).mean()), 4),
    }


def _evaluate_window(y, pred_map, horizon, window):
    return [score_predictions(y, pred, horizon, name, window=window) for name, pred in pred_map.items()]


def run(write=True, update_data=False):
    started = time.time()
    if update_data:
        update_forecast_fred(full=False)
    features, dxy = build_features()
    log_holdout(
        "research run started: feature build and model selection are restricted to dates "
        f"before {HOLDOUT_START.date()}; holdout predictions are scored once after selection.",
        looked=False,
    )
    vix = features["vix"] if "vix" in features.columns else pd.Series(index=features.index, dtype=float)
    carry = features["carry_3m"] if "carry_3m" in features.columns else pd.Series(0.0, index=features.index)

    selection = []
    all_scores = []
    pred_store = {}
    fit_info = {}
    n_trials = 0

    for horizon in HORIZONS:
        y = forward_return(dxy, horizon)
        mom = naive_momentum(dxy, horizon)
        car = naive_carry(carry, horizon)
        rw = pd.Series(0.0, index=y.index)
        pred_map = {"random_walk": rw, "momentum_only": mom, "carry_only": car}
        pred_store[(horizon, "random_walk")] = rw
        pred_store[(horizon, "momentum_only")] = mom
        pred_store[(horizon, "carry_only")] = car

        for group in FEATURE_GROUPS:
            X = _feature_subset(features, group)
            for alpha in RIDGE_ALPHAS:
                n_trials += 1
                label = f"ridge_{group}_a{alpha}"
                pred, info = walk_forward_ridge(y, X, horizon, alpha=alpha)
                pred_map[label] = pred
                pred_store[(horizon, label)] = pred
                fit_info[(horizon, label)] = info
        n_trials += 1
        logit_pred, logit_info = walk_forward_logit(y, features, horizon)
        pred_map["logit_all"] = logit_pred
        pred_store[(horizon, "logit_all")] = logit_pred
        fit_info[(horizon, "logit_all")] = logit_info

        pre_rows = _evaluate_window(y, pred_map, horizon, "pre")
        hol_rows = _evaluate_window(y, pred_map, horizon, "holdout")
        all_scores.extend(pre_rows)
        all_scores.extend(hol_rows)
        selection.extend(pre_rows)

    def _r2(row):
        v = row.get("oosR2")
        return v if isinstance(v, (int, float)) else -9e9

    primary_pre = [r for r in selection if r.get("horizon") == PRIMARY_HORIZON
                   and r.get("window") == "pre" and str(r.get("label", "")).startswith("ridge_")]
    if not primary_pre:
        primary_pre = [r for r in selection if str(r.get("label", "")).startswith("ridge_")]
    selected = max(primary_pre, key=_r2) if primary_pre else None
    selected_label = selected["label"] if selected else "ridge_all_a1.0"
    selected_pred = pred_store.get((PRIMARY_HORIZON, selected_label))
    y20 = forward_return(dxy, PRIMARY_HORIZON)

    holdout_score = next((r for r in all_scores if r.get("label") == selected_label
                          and r.get("horizon") == PRIMARY_HORIZON and r.get("window") == "holdout"), None)
    pre_score = next((r for r in all_scores if r.get("label") == selected_label
                      and r.get("horizon") == PRIMARY_HORIZON and r.get("window") == "pre"), None)
    log_holdout(
        "locked-model holdout scoring (one look): selected on pre-holdout OOS R2 at the 20-session "
        f"horizon, label={selected_label}. Holdout metrics were read after the choice was fixed.",
        looked=True,
        extra={"selected": selected_label, "preOosR2": (pre_score or {}).get("oosR2"),
               "holdoutOosR2": (holdout_score or {}).get("oosR2")},
    )

    decade_sel = score_by_decade(y20, selected_pred) if selected_pred is not None else []
    regime_sel = score_by_vix_regime(y20, selected_pred, vix) if selected_pred is not None else []

    dxy_ret = dxy.pct_change(fill_method=None)
    trade_trials = []
    best = None
    if selected_pred is not None:
        for thresh in TRADE_THRESHOLDS:
            n_trials += 1
            pnl, pos = trading_returns(selected_pred, dxy_ret, threshold=thresh,
                                       cost_bps=TRADE_COST_BPS, lag=1)
            pre_pnl = pnl.loc[pnl.index < HOLDOUT_START].dropna()
            hol_pnl = pnl.loc[pnl.index >= HOLDOUT_START].dropna()
            trial = {"threshold": thresh, "pre": _perf_brief(pre_pnl), "holdout": _perf_brief(hol_pnl),
                     "avgAbsPosPre": _clean(float(pos.loc[pos.index < HOLDOUT_START].abs().mean() or 0), 4)}
            trade_trials.append(trial)
            score = (trial["pre"] or {}).get("sharpe") or -9e9
            if best is None or score > best[0]:
                best = (score, thresh, pnl, pos)
    trade_thresh = best[1] if best else 0.0
    log_holdout(
        f"trading-rule threshold selected on pre-holdout Sharpe only (threshold={trade_thresh}); "
        "holdout PnL scored once after that choice.",
        looked=True,
        extra={"threshold": trade_thresh},
    )
    trade_pnl = best[2] if best else pd.Series(dtype=float)
    pre_trade = trade_pnl.loc[trade_pnl.index < HOLDOUT_START].dropna() if len(trade_pnl) else pd.Series(dtype=float)
    hol_trade = trade_pnl.loc[trade_pnl.index >= HOLDOUT_START].dropna() if len(trade_pnl) else pd.Series(dtype=float)
    dsr_pre = deflated_sharpe(pre_trade, n_trials=max(n_trials, 1))
    dsr_hol = deflated_sharpe(hol_trade, n_trials=max(n_trials, 1))

    uup_block = None
    try:
        uup = load_uup_returns()
        if uup is not None and selected_pred is not None and len(uup) > 40:
            log_holdout(
                "UUP trading-rule evaluation on the local UUP series (about two years, inside the holdout). "
                "The threshold was already locked from the long-history DXY-proxy pre-holdout Sharpe.",
                looked=True,
            )
            uup = uup.astype(float)
            if isinstance(uup, pd.DataFrame):
                uup = uup.iloc[:, 0]
            pnl_u, _ = trading_returns(selected_pred.reindex(uup.index), uup,
                                       threshold=trade_thresh, cost_bps=UUP_COST_BPS, lag=1)
            uup_block = {
                "sample": {"start": str(uup.index[0].date()), "end": str(uup.index[-1].date()),
                           "sessions": int(uup.dropna().shape[0])},
                "pnl": _perf_brief(pnl_u.dropna()),
                "deflatedSharpe": deflated_sharpe(pnl_u.dropna(), n_trials=max(n_trials, 1)),
                "costBpsOneWay": UUP_COST_BPS,
                "note": "UUP history in prices.sqlite3 is only the local ~2-year window, all inside the holdout. "
                        "The long-history rule is the DXY proxy (not a listed ETF).",
            }
    except Exception as error:  # noqa: BLE001
        uup_block = {"error": f"{type(error).__name__}: {error}"}

    usefulness = {"sectors": None, "hedge": None, "hmm": None}
    if selected_pred is not None:
        for key, fn in (("sectors", usefulness_sectors), ("hedge", usefulness_hedge), ("hmm", usefulness_hmm)):
            try:
                usefulness[key] = fn(selected_pred, PRIMARY_HORIZON)
            except Exception as error:  # noqa: BLE001
                usefulness[key] = {"error": f"{type(error).__name__}: {error}"}

    gates = _gates(pre_score, holdout_score)
    drivers = (fit_info.get((PRIMARY_HORIZON, selected_label)) or {}).get("meanAbsBeta") or {}
    outlook = build_outlook(
        dxy, features, selected_pred, y20, selected_label, pre_score, holdout_score, drivers, gates,
    )
    outlook = attach_alert(outlook, _read_outlook())

    summary = {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "timezone": "America/New_York",
        "target": "FRED DXY-style ICE-formula proxy (not ICE futures)",
        "sample": {
            "dxyStart": str(dxy.dropna().index[0].date()),
            "dxyEnd": str(dxy.dropna().index[-1].date()),
            "sessions": int(dxy.dropna().shape[0]),
            "holdoutStart": str(HOLDOUT_START.date()),
            "minTrainSessions": MIN_TRAIN,
            "refitEvery": REFIT_EVERY,
            "monthlyLagDays": MONTHLY_LAG_DAYS,
            "dailyLagSessions": DAILY_LAG,
            "featureNotes": features.attrs.get("notes"),
        },
        "dataNotes": PROXY_NOTE,
        "nTrials": int(n_trials),
        "selected": {
            "label": selected_label,
            "horizon": PRIMARY_HORIZON,
            "rule": "max pre-holdout OOS R2 among ridge variants at the 20-session horizon",
            "preHoldout": pre_score,
            "holdout": holdout_score,
            "meanAbsBeta": drivers,
        },
        "scores": all_scores,
        "stability": {"byDecadeSelected": decade_sel, "byVixSelected": regime_sel},
        "tradingRule": {
            "instrument": "DXY proxy (not a listed ETF); long/short/flat, 1-session lag, "
                          f"{TRADE_COST_BPS} bps one-way, threshold selected on pre-holdout Sharpe",
            "threshold": trade_thresh,
            "costBpsOneWay": TRADE_COST_BPS,
            "preHoldout": _perf_brief(pre_trade),
            "holdout": _perf_brief(hol_trade),
            "deflatedSharpePre": dsr_pre,
            "deflatedSharpeHoldout": dsr_hol,
            "trials": trade_trials,
            "uup": uup_block,
        },
        "usefulness": usefulness,
        "gates": gates,
        "outlook": outlook,
        "runSeconds": round(time.time() - started, 1),
    }
    if write:
        _write_outputs(summary)
    return summary


def _gates(pre_score, holdout_score):
    pre_r2 = (pre_score or {}).get("oosR2")
    pre_p = ((pre_score or {}).get("clarkWest") or {}).get("pValue")
    hol_r2 = (holdout_score or {}).get("oosR2")
    hol_p = ((holdout_score or {}).get("clarkWest") or {}).get("pValue")
    pre_ok = isinstance(pre_r2, (int, float)) and pre_r2 > 0 and isinstance(pre_p, (int, float)) and pre_p < 0.05
    hol_ok = isinstance(hol_r2, (int, float)) and hol_r2 > 0 and isinstance(hol_p, (int, float)) and hol_p < 0.05
    passed = bool(pre_ok and hol_ok)
    return {
        "passed": passed,
        "preHoldoutOosR2Positive": bool(isinstance(pre_r2, (int, float)) and pre_r2 > 0),
        "preHoldoutClarkWestP05": bool(isinstance(pre_p, (int, float)) and pre_p < 0.05),
        "holdoutOosR2Positive": bool(isinstance(hol_r2, (int, float)) and hol_r2 > 0),
        "holdoutClarkWestP05": bool(isinstance(hol_p, (int, float)) and hol_p < 0.05),
        "tradingDefault": "off",
        "reason": (
            "Gates require pre-holdout and holdout OOS R2 > 0 versus the random walk and a "
            "Clark-West p-value below 0.05 at the 20-session horizon. Trading use stays off "
            "regardless; the outlook is recommend-only."
            + (" Gates passed: the model beat the random walk in both windows." if passed
               else " Gates did not pass; do not size paper or live trades from this outlook.")
        ),
        "preOosR2": pre_r2,
        "preClarkWestP": pre_p,
        "holdoutOosR2": hol_r2,
        "holdoutClarkWestP": hol_p,
    }


def _residual_vol(y, pred, window="pre"):
    both = pd.concat([y.rename("y"), pred.rename("p")], axis=1).dropna()
    if window == "pre":
        both = both.loc[both.index < HOLDOUT_START]
    if len(both) < 20:
        return None
    return float((both["y"] - both["p"]).std(ddof=1))


def build_outlook(dxy, features, pred, y, selected_label, pre_score, holdout_score, drivers, gates):
    asof = None
    expected = None
    if pred is not None and pred.notna().any():
        last = pred.dropna()
        asof = last.index[-1]
        expected = float(last.iloc[-1])
    p_stronger = None
    try:
        y_pre = y.loc[y.index < HOLDOUT_START] if y is not None else None
        X_pre = features.loc[features.index < HOLDOUT_START]
        if y_pre is not None and asof is not None:
            pack = pd.concat([y_pre.rename("y"), X_pre], axis=1).dropna()
            if len(pack) > MIN_TRAIN:
                if len(pack) > PRIMARY_HORIZON + MIN_TRAIN:
                    pack = pack.iloc[:-PRIMARY_HORIZON]
                keep = [c for c in pack.columns if c != "y" and pack[c].notna().mean() >= (1.0 - MAX_FEATURE_NAN)]
                pack2 = pack[["y"] + keep].dropna()
                X_last = features.loc[[asof], keep] if asof in features.index else pack2[keep].iloc[[-1]]
                X_std, X_now, mu, sd = _standardize(pack2[keep], X_last)
                beta = fit_logit((pack2["y"] > 0).astype(float).values, X_std.values)
                if beta is not None and not X_now.isna().any().any():
                    raw = predict_logit(X_now.values, beta)
                    a, b = platt_calibrate(predict_logit(X_std.values, beta),
                                           (pack2["y"] > 0).astype(float).values)
                    p_stronger = float(apply_platt(raw, a, b)[0])
    except Exception:  # noqa: BLE001
        p_stronger = None
    if p_stronger is None and expected is not None:
        vol = _residual_vol(y, pred) or 0.015
        p_stronger = float(stats.norm.cdf(expected / max(vol, 1e-6)))

    vol = _residual_vol(y, pred) or 0.015
    ci = None
    if expected is not None and vol:
        ci = [_clean(expected - 1.28155 * vol, 5), _clean(expected + 1.28155 * vol, 5)]
    direction = "uncertain"
    if expected is not None and p_stronger is not None:
        if expected > 0 and p_stronger >= 0.55:
            direction = "stronger"
        elif expected < 0 and p_stronger <= 0.45:
            direction = "weaker"
    top_drivers = sorted(drivers.items(), key=lambda kv: abs(kv[1] or 0), reverse=True)[:6]
    confidence = "low"
    if gates.get("passed"):
        confidence = "medium" if (pre_score or {}).get("oosR2", 0) < 0.03 else "high"

    level = None
    if asof is not None and asof in dxy.index and pd.notna(dxy.loc[asof]):
        level = _clean(float(dxy.loc[asof]), 4)

    return {
        "asOf": str(asof.date()) if asof is not None else None,
        "horizonSessions": PRIMARY_HORIZON,
        "expectedMove": _clean(expected, 5),
        "ci80": ci,
        "pStronger": _clean(p_stronger, 4),
        "direction": direction,
        "confidence": confidence,
        "model": selected_label,
        "drivers": [{"name": k, "meanAbsStandardizedBeta": v} for k, v in top_drivers],
        "gatesPassed": bool(gates.get("passed")),
        "tradingDefault": "off",
        "recommendation": (
            "Recommend only. Do not size live or paper trades from this outlook unless the gates "
            "pass and a human turns the feature on. " + (gates.get("reason") or "")
        ),
        "proxyNote": PROXY_NOTE,
        "level": level,
    }


def _read_outlook():
    for path in (OUTLOOK_PATH, MACRO_OUTLOOK_PATH):
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return None


def attach_alert(outlook, previous, prob_jump=0.15):
    alert = None
    if previous and outlook.get("pStronger") is not None and previous.get("pStronger") is not None:
        delta_p = float(outlook["pStronger"]) - float(previous["pStronger"])
        flipped = (outlook.get("direction") in ("stronger", "weaker")
                   and previous.get("direction") in ("stronger", "weaker")
                   and outlook.get("direction") != previous.get("direction"))
        e0, e1 = previous.get("expectedMove"), outlook.get("expectedMove")
        moved = (isinstance(e0, (int, float)) and isinstance(e1, (int, float)) and abs(e1 - e0) >= 0.01)
        if flipped or abs(delta_p) >= prob_jump or moved:
            alert = {
                "flag": True,
                "reason": ("direction flipped" if flipped
                           else (f"|dP(stronger)|={delta_p:.2f}" if abs(delta_p) >= prob_jump
                                 else "expected 20d move changed by at least 1 percentage point")),
                "previous": {"asOf": previous.get("asOf"), "direction": previous.get("direction"),
                             "pStronger": previous.get("pStronger"),
                             "expectedMove": previous.get("expectedMove")},
            }
    outlook = dict(outlook)
    outlook["alert"] = alert
    return outlook


def _write_outputs(summary):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    write_atomically(REPORT_JSON, json.dumps(_json_ready(summary), indent=2).encode())
    write_atomically(REPORT_MD, markdown(summary).encode())
    outlook = summary.get("outlook") or {}
    write_atomically(OUTLOOK_PATH, json.dumps(_json_ready(outlook), indent=2).encode())
    write_atomically(MACRO_OUTLOOK_PATH, json.dumps(_json_ready(outlook), indent=2).encode())
    rows = []
    for row in summary.get("scores") or []:
        cw = row.get("clarkWest") or {}
        ic = row.get("ic") or {}
        rows.append({
            "label": row.get("label"), "horizon": row.get("horizon"), "window": row.get("window"),
            "n": row.get("n"), "oosR2": row.get("oosR2"), "hitRate": row.get("hitRate"),
            "cwP": cw.get("pValue"), "cwT": cw.get("t"),
            "ic": ic.get("spearman") if isinstance(ic, dict) else None,
            "brier": row.get("brier"),
            "start": row.get("start"), "end": row.get("end"),
        })
    if rows:
        pd.DataFrame(rows).to_csv(TRIALS_CSV, index=False)


def _fmt(value, digits=3, pct=False):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(value):
        return "-"
    if pct:
        return f"{value * 100:.{digits}f}%"
    return f"{value:.{digits}f}"


def _fmt_p(value):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def markdown(summary):
    sample = summary.get("sample") or {}
    sel = summary.get("selected") or {}
    gates = summary.get("gates") or {}
    trade = summary.get("tradingRule") or {}
    outlook = summary.get("outlook") or {}
    lines = [
        "# Dollar (DXY proxy) forecast",
        "",
        f"Generated {summary.get('generatedAt')} ({summary.get('timezone')}).",
        "",
        "## Data",
        "",
        f"Target: {summary.get('target')}. Sample {sample.get('dxyStart')} to {sample.get('dxyEnd')} "
        f"({sample.get('sessions')} sessions). Holdout starts {sample.get('holdoutStart')}. "
        f"Expanding purged fits, minimum train {sample.get('minTrainSessions')} sessions, refit every "
        f"{sample.get('refitEvery')} sessions, daily lag {sample.get('dailyLagSessions')} session, "
        f"monthly publication lag {sample.get('monthlyLagDays')} days. Trials counted: "
        f"{summary.get('nTrials')}.",
        "",
        summary.get("dataNotes") or PROXY_NOTE,
        "",
        "## Results versus the random walk",
        "",
        f"Selected model: `{sel.get('label')}` at a {sel.get('horizon')}-session horizon "
        f"({sel.get('rule')}).",
        "",
        "Window | Model | Horizon | N | OOS R2 vs RW | Clark-West p | Hit rate | IC",
        "---|---|---:|---:|---:|---:|---:|---:",
    ]
    focus = []
    for row in summary.get("scores") or []:
        if row.get("horizon") != PRIMARY_HORIZON:
            continue
        if row.get("label") in {sel.get("label"), "random_walk", "momentum_only", "carry_only", "logit_all"} or (
            row.get("window") in ("pre", "holdout") and str(row.get("label", "")).startswith("ridge_all_")
        ):
            focus.append(row)
    seen = set()
    for row in focus + [r for r in (summary.get("scores") or []) if r.get("label") == sel.get("label")]:
        key = (row.get("window"), row.get("label"), row.get("horizon"))
        if key in seen or row.get("horizon") != PRIMARY_HORIZON:
            continue
        seen.add(key)
        ic = (row.get("ic") or {}).get("spearman") if isinstance(row.get("ic"), dict) else None
        p = (row.get("clarkWest") or {}).get("pValue")
        lines.append(
            f"{row.get('window')} | {row.get('label')} | {row.get('horizon')} | {row.get('n')} | "
            f"{_fmt(row.get('oosR2'), 3)} | {_fmt_p(p)} | {_fmt(row.get('hitRate'), 3, pct=True)} | "
            f"{_fmt(ic, 3)}"
        )
    lines += [
        "",
        "Other horizons (selected ridge family and benchmarks) are in "
        "`Massive/Research/dxy_forecast_report.json` and `dxy_forecast_trials.csv`.",
        "",
        "### Gates",
        "",
        f"Passed: **{'yes' if gates.get('passed') else 'no'}**. "
        f"Pre-holdout OOS R2 {_fmt(gates.get('preOosR2'), 3)} "
        f"(Clark-West p {_fmt_p(gates.get('preClarkWestP'))}); "
        f"holdout OOS R2 {_fmt(gates.get('holdoutOosR2'), 3)} "
        f"(Clark-West p {_fmt_p(gates.get('holdoutClarkWestP'))}). "
        f"Trading default remains **{gates.get('tradingDefault') or 'off'}**.",
        "",
        gates.get("reason") or "",
        "",
        "## Which drivers matter",
        "",
    ]
    betas = sel.get("meanAbsBeta") or {}
    if not betas:
        lines.append("No stored standardized coefficients.")
    else:
        lines += [
            "Mean absolute standardized ridge coefficient across expanding refits (20-session model):",
            "",
            "Feature | Mean |beta|",
            "---|---:",
        ]
        for name, value in sorted(betas.items(), key=lambda kv: abs(kv[1] or 0), reverse=True):
            lines.append(f"{name} | {_fmt(value, 4)}")
    lines.append("")
    if summary.get("stability", {}).get("byDecadeSelected"):
        lines += [
            "Stability of the selected 20-session model by decade (full OOS, including holdout):",
            "",
            "Decade | N | OOS R2 | Clark-West p | Hit rate | IC",
            "---|---:|---:|---:|---:|---:",
        ]
        for row in summary["stability"]["byDecadeSelected"]:
            ic = (row.get("ic") or {}).get("spearman") if isinstance(row.get("ic"), dict) else None
            p = (row.get("clarkWest") or {}).get("pValue")
            lines.append(
                f"{row.get('decade')}s | {row.get('n')} | {_fmt(row.get('oosR2'), 3)} | "
                f"{_fmt_p(p)} | {_fmt(row.get('hitRate'), 3, pct=True)} | {_fmt(ic, 3)}"
            )
        lines.append("")
    pre, hol = trade.get("preHoldout") or {}, trade.get("holdout") or {}
    dsr_p, dsr_h = trade.get("deflatedSharpePre") or {}, trade.get("deflatedSharpeHoldout") or {}
    lines += [
        "## Trading rule (DXY proxy, not a live order)",
        "",
        trade.get("instrument") or "",
        "",
        f"Threshold {trade.get('threshold')}. Pre-holdout Sharpe {_fmt(pre.get('sharpe'), 2)}, "
        f"CAGR {_fmt(pre.get('cagr'), 2, pct=True)}, max DD {_fmt(pre.get('maxDrawdown'), 2, pct=True)}, "
        f"deflated Sharpe probability {_fmt(dsr_p.get('deflatedSharpeProb'), 3)} "
        f"({dsr_p.get('nTrials')} trials). Holdout Sharpe {_fmt(hol.get('sharpe'), 2)}, "
        f"CAGR {_fmt(hol.get('cagr'), 2, pct=True)}, max DD {_fmt(hol.get('maxDrawdown'), 2, pct=True)}, "
        f"deflated Sharpe probability {_fmt(dsr_h.get('deflatedSharpeProb'), 3)}.",
    ]
    if trade.get("uup") and "pnl" in (trade.get("uup") or {}):
        u = trade["uup"]
        lines.append(
            f"UUP (local ~2-year sample, inside the holdout): Sharpe {_fmt((u['pnl'] or {}).get('sharpe'), 2)}, "
            f"total {_fmt((u['pnl'] or {}).get('totalReturn'), 2, pct=True)}. {u.get('note') or ''}"
        )
    lines += ["", "## Does it help stock and hedge decisions?", "", _usefulness_md(summary.get("usefulness") or {}), ""]
    alert = outlook.get("alert") or {}
    lines += [
        "## Current dollar outlook (recommend only)",
        "",
        f"As of {outlook.get('asOf')}: 20-session expected move {_fmt(outlook.get('expectedMove'), 2, pct=True)} "
        f"(80% band {outlook.get('ci80')}), P(stronger dollar) {_fmt(outlook.get('pStronger'), 3, pct=True)}, "
        f"direction **{outlook.get('direction')}**, confidence {outlook.get('confidence')}. "
        f"Gates passed: {outlook.get('gatesPassed')}. Trading default: {outlook.get('tradingDefault')}.",
    ]
    if alert.get("flag"):
        lines.append(f"**Alert:** {alert.get('reason')}.")
    lines += ["", outlook.get("recommendation") or "", "", f"Run time {summary.get('runSeconds')}s.", ""]
    return "\n".join(lines) + "\n"


def _usefulness_md(block):
    parts = []
    sectors = block.get("sectors") or {}
    if sectors.get("rows"):
        parts += [
            "Sector / asset tests use the locked 20-session forecast (not lagged dollar returns) "
            "on the local ETF history (2024-11-25 onward, inside the holdout). Signs were pre-specified.",
            "",
            "Ticker | Prior sign | N | Forecast OOS R2 | CW p | IC | Lagged DXY mom OOS R2",
            "---|---:|---:|---:|---:|---:|---:",
        ]
        for row in sectors["rows"]:
            f = row.get("forecast") or {}
            m = row.get("laggedDxyMom") or {}
            ic = (f.get("ic") or {}).get("spearman") if isinstance(f.get("ic"), dict) else None
            p = (f.get("clarkWest") or {}).get("pValue")
            parts.append(
                f"{row.get('ticker')} | {row.get('preSpecifiedSign')} | {row.get('n')} | "
                f"{_fmt(f.get('oosR2'), 3)} | {_fmt_p(p)} | {_fmt(ic, 3)} | {_fmt(m.get('oosR2'), 3)}"
            )
        tilt = sectors.get("tilt") or {}
        if tilt.get("pnl"):
            parts += [
                "",
                f"Pre-specified tilt book ({', '.join(tilt.get('names') or [])}): "
                f"Sharpe {_fmt(tilt['pnl'].get('sharpe'), 2)} vs SPY {_fmt((tilt.get('spy') or {}).get('sharpe'), 2)}. "
                f"{tilt.get('rule')}",
            ]
    elif sectors.get("error") or sectors.get("reason"):
        parts.append(f"Sector tests: {sectors.get('error') or sectors.get('reason')}")
    hedge = block.get("hedge") or {}
    if hedge.get("available"):
        parts += [
            "",
            f"Hedge (SPY vs constant 10% UUP overlay vs forecast-sized UUP overlay, next session, "
            f"pre-specified weights): SPY Sharpe {_fmt((hedge.get('spy') or {}).get('sharpe'), 2)}, "
            f"constant hedge {_fmt((hedge.get('constantHedge') or {}).get('sharpe'), 2)}, "
            f"sized hedge {_fmt((hedge.get('sizedHedge') or {}).get('sharpe'), 2)}.",
        ]
        vs = hedge.get("volSleeveOverlay") or {}
        if vs.get("sleeve"):
            parts.append(
                f"GB-18 vol-sleeve OOS overlay (file reused, not refit): sleeve Sharpe "
                f"{_fmt(vs['sleeve'].get('sharpe'), 2)} vs scaled {_fmt(vs['scaled'].get('sharpe'), 2)}."
            )
    hmm = block.get("hmm") or {}
    inc = hmm.get("incremental") or {}
    if inc.get("macroOnly") or inc.get("macroPlusForecast"):
        parts += [
            "",
            f"Macro-HMM features predicting {PRIMARY_HORIZON}-session SPY returns, expanding OOS on the "
            f"local panel: macro-only OOS R2 {_fmt((inc.get('macroOnly') or {}).get('oosR2'), 3)} vs "
            f"macro + locked dollar forecast {_fmt((inc.get('macroPlusForecast') or {}).get('oosR2'), 3)} "
            f"(forecast adds incremental R2: {inc.get('forecastAdds')}).",
        ]
    if hmm.get("combinedMacroHmm"):
        parts.append(
            f"Combined + 3-state macro HMM overlay: baseline Sharpe "
            f"{_fmt(hmm['combinedMacroHmm'].get('sharpe'), 2)} vs forecast gross-cut overlay "
            f"{_fmt((hmm.get('combinedMacroHmmForecastOverlay') or {}).get('sharpe'), 2)}. "
            f"{hmm.get('combinedNote') or ''}"
        )
    return "\n".join(parts) if parts else "Usefulness tests did not run (see the JSON for errors)."


def load_outlook():
    data = _read_outlook()
    if data:
        return data
    if REPORT_JSON.is_file():
        try:
            return json.loads(REPORT_JSON.read_text(encoding="utf-8")).get("outlook")
        except (OSError, json.JSONDecodeError):
            return None
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["update", "research", "outlook", "status"], nargs="?", default="outlook")
    parser.add_argument("--full", action="store_true", help="re-pull the full FRED history for forecast series")
    args = parser.parse_args(argv)
    if args.command == "update":
        written = update_forecast_fred(full=args.full)
        print(json.dumps(_json_ready(written), indent=2))
        return written
    if args.command == "research":
        summary = run(write=True, update_data=False)
        print(json.dumps(_json_ready({
            "selected": summary.get("selected"),
            "gates": summary.get("gates"),
            "outlook": summary.get("outlook"),
            "tradingRule": {
                "threshold": (summary.get("tradingRule") or {}).get("threshold"),
                "preHoldout": (summary.get("tradingRule") or {}).get("preHoldout"),
                "holdout": (summary.get("tradingRule") or {}).get("holdout"),
                "deflatedSharpeHoldout": (summary.get("tradingRule") or {}).get("deflatedSharpeHoldout"),
            },
            "nTrials": summary.get("nTrials"),
            "runSeconds": summary.get("runSeconds"),
        }), indent=2))
        return summary
    if args.command == "status":
        payload = {
            "fredDir": str(FRED_DIR),
            "series": sorted(p.stem for p in FRED_DIR.glob("*.csv")) if FRED_DIR.is_dir() else [],
            "outlook": load_outlook(),
            "report": str(REPORT_MD) if REPORT_MD.is_file() else None,
        }
        print(json.dumps(_json_ready(payload), indent=2))
        return payload
    existing = load_outlook()
    if existing:
        print(json.dumps(_json_ready(existing), indent=2))
        return existing
    summary = run(write=True, update_data=False)
    print(json.dumps(_json_ready(summary.get("outlook")), indent=2))
    return summary.get("outlook")


if __name__ == "__main__":
    main()
