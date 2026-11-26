"""Break strategy returns into market exposure, sector/bond bets, and residual skill (alpha).

Return-based attribution uses full-sample and rolling OLS with HAC/Newey-West standard errors.
Holdings-based attribution is Brinson-Fachler, built from the backtest's target weights.

Outputs go to Massive/Research/attribution.json (gitignored). This module never touches
broker/order endpoints and does not print secrets.

Usage (from engine/):
  python attribution.py              # score saved research equity/decisions + paper + day trading
  python research.py attribution     # same, also refreshes research_summary.json and the report

AI analyst tool
---------------
Register tool_attribute_returns (see ANALYST_TOOL below) in the Retort/quant_tools enum loop
the same way other engine tools are registered. Do not import this module from quant_tools at
import time if that file is mid-edit; call attribution.tool_attribute_returns(args) from the
dispatcher. No character limit is imposed on the JSON result.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically
from market_data import MARKET, load_panel
from metrics import performance
from risk import TRADING_DAYS

OUT_DIR = DATA_DIR / "Research"
PAPER_DB = DATA_DIR / "paper_trading.sqlite3"
MARKET_FACTOR = MARKET  # SPY
SECTOR_ETFS = ("XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC")
BOND_ETFS = ("IEF", "TLT")
STYLE_ETFS = ("QQQ", "IWM", "DIA")
OTHER_ETFS = ("GLD", "EFA", "EEM")
MACRO_FACTORS = ("DXY", "UST10Y", "CURVE", "HY_OAS", "SOFR_TBILL")
ALL_FACTORS = (MARKET_FACTOR,) + SECTOR_ETFS + BOND_ETFS + STYLE_ETFS + OTHER_ETFS + MACRO_FACTORS

GROUP_OF = {
    "SPY": "US Market", "QQQ": "US Nasdaq/Growth", "DIA": "US Dow", "IWM": "US Small Cap",
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
    "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication Services",
    "IEF": "Intermediate Treasuries", "TLT": "Long Treasuries",
    "GLD": "Gold", "EFA": "International Developed", "EEM": "Emerging Markets",
    "DXY": "US Dollar proxy",
    "UST10Y": "Treasury 10y yield change (negated, not a bond total return)",
    "CURVE": "Treasury 10y-2y slope change (proxy)",
    "HY_OAS": "HY OAS tightening (ICE BofA OAS, CDS/CDX proxy)",
    "SOFR_TBILL": "SOFR minus 3M Treasury change (funding proxy, not a swap spread)",
}
GROUP_BENCHMARK = {
    "US Market": "SPY", "US Nasdaq/Growth": "QQQ", "US Dow": "DIA", "US Small Cap": "IWM",
    "Technology": "XLK", "Financials": "XLF", "Energy": "XLE", "Health Care": "XLV",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Industrials": "XLI",
    "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE", "Communication Services": "XLC",
    "Intermediate Treasuries": "IEF", "Long Treasuries": "TLT",
    "Gold": "GLD", "International Developed": "EFA", "Emerging Markets": "EEM",
}
ROLLING_WINDOWS = (63, 126)
MIN_OBS = 40
CORR_ASSIGN_MIN = 0.25

# Documented entry point for the AI analyst tool loop (Retort enums, no character limit).
ANALYST_TOOL = {
    "name": "attribute_returns",
    "module": "attribution",
    "function": "tool_attribute_returns",
    "summary": "Break strategy or paper-account returns into market beta, sector/bond bets and residual alpha.",
    "args": {
        "strategy": "optional: core_tsmom | overlay_meanrev | combined | daytrade | paper | all (default all)",
        "source": "optional: backtest | walkforward | paper | daytrade | all (default all)",
        "recompute": "optional bool, default false: rerun from prices/equity files instead of the saved JSON",
        "window": "optional rolling window in sessions (default 63 and 126 are always included)",
    },
    "returns": "JSON: CAPM and multi-factor OLS with HAC t-stats, rolling alpha/beta, beta-matched SPY+bonds "
               "blend, Brinson holdings attribution, and a written verdict on whether the strategy is just a "
               "cheaper index fund. Notes the short (~2 year) sample.",
}


def hac_lags(n):
    """Newey-West lag length 4*(n/100)^(2/9), matching engine/quant_tools.py."""
    return max(1, int(4 * (n / 100) ** (2 / 9)))


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value.date()) if hasattr(value, "date") else str(value)
    return value


def write_json(path, payload):
    write_atomically(path, json.dumps(_finite(payload), indent=2, allow_nan=False).encode())


def _clean_num(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def returns_from_equity(equity):
    """Daily simple returns from a level series; drops the opening NaN only."""
    equity = pd.Series(equity).astype(float).sort_index()
    equity.index = pd.to_datetime(equity.index)
    returns = equity.pct_change()
    return returns.iloc[1:].replace([np.inf, -np.inf], np.nan).dropna()


def align(strategy, factors):
    """Inner-join strategy returns with a factor return frame; drop NaNs."""
    if isinstance(factors, pd.Series):
        factors = factors.to_frame()
    frame = pd.concat([strategy.rename("strategy"), factors], axis=1, join="inner").dropna()
    return frame["strategy"], frame.drop(columns=["strategy"])


def ols_hac(y, X, hac=True):
    """OLS of y on X (adds intercept) with Newey-West HAC standard errors."""
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson

    y = y.astype(float)
    if isinstance(X, pd.DataFrame):
        X = X.astype(float)
    else:
        name = getattr(X, "name", None) or "x"
        X = pd.DataFrame({name: X.astype(float)})
    frame = pd.concat([y.rename("y"), X], axis=1, join="inner").dropna()
    if len(frame) < MIN_OBS or frame.drop(columns=["y"]).std().lt(1e-12).all():
        return None
    yv, Xv = frame["y"], frame.drop(columns=["y"])
    Xv = Xv.loc[:, ~Xv.T.duplicated()]
    lags = hac_lags(len(frame))
    Xc = sm.add_constant(Xv, has_constant="add")
    try:
        fit = sm.OLS(yv, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lags}) if hac else sm.OLS(yv, Xc).fit()
    except Exception:
        return None
    params = {str(k): _clean_num(v) for k, v in fit.params.items()}
    tstats = {str(k): _clean_num(v) for k, v in fit.tvalues.items()}
    pvalues = {str(k): _clean_num(v) for k, v in fit.pvalues.items()}
    ses = {str(k): _clean_num(v) for k, v in fit.bse.items()}
    resid = fit.resid.astype(float)
    resid_vol = float(resid.std(ddof=1)) if len(resid) > 1 else None
    return {
        "n": int(fit.nobs),
        "hacLags": lags if hac else 0,
        "covariance": f"HAC Newey-West (lags {lags})" if hac else "OLS",
        "alphaDaily": params.get("const"),
        "alphaAnnual": None if params.get("const") is None else params["const"] * TRADING_DAYS,
        "alphaT": tstats.get("const"),
        "alphaP": pvalues.get("const"),
        "alphaSE": ses.get("const"),
        "betas": {k: v for k, v in params.items() if k != "const"},
        "betaT": {k: v for k, v in tstats.items() if k != "const"},
        "betaP": {k: v for k, v in pvalues.items() if k != "const"},
        "betaSE": {k: v for k, v in ses.items() if k != "const"},
        "r2": _clean_num(fit.rsquared),
        "adjR2": _clean_num(fit.rsquared_adj),
        "residVolAnnual": None if resid_vol is None else resid_vol * math.sqrt(TRADING_DAYS),
        "durbinWatson": _clean_num(durbin_watson(resid)),
        "residual": resid,
        "fitted": fit,
    }


def tracking_error(strategy, benchmark):
    y, X = align(strategy, benchmark)
    if len(y) < 2:
        return None
    excess = y - X.iloc[:, 0]
    te = float(excess.std(ddof=1) * math.sqrt(TRADING_DAYS)) if excess.std(ddof=1) > 0 else 0.0
    ir = float(excess.mean() / excess.std(ddof=1) * math.sqrt(TRADING_DAYS)) if excess.std(ddof=1) > 0 else None
    return {"trackingError": te, "informationRatio": ir, "excessMeanAnnual": float(excess.mean() * TRADING_DAYS),
            "corr": _clean_num(y.corr(X.iloc[:, 0])), "n": int(len(y))}


def capm_report(strategy, spy):
    named = spy.rename(MARKET_FACTOR) if isinstance(spy, pd.Series) else spy
    fit = ols_hac(strategy, named)
    if not fit:
        return None
    te = tracking_error(strategy, spy)
    beta = fit["betas"].get(MARKET_FACTOR)
    out = {k: v for k, v in fit.items() if k not in ("residual", "fitted")}
    out["beta"] = beta
    out["betaT"] = fit["betaT"].get(MARKET_FACTOR)
    out["betaP"] = fit["betaP"].get(MARKET_FACTOR)
    if te:
        out.update(te)
    if fit["alphaAnnual"] is not None and fit["residVolAnnual"]:
        out["appraisalRatio"] = fit["alphaAnnual"] / fit["residVolAnnual"]
    return out


def multifactor_report(strategy, factors, label):
    y, X = align(strategy, factors)
    if X.empty:
        return None
    fit = ols_hac(y, X)
    if not fit:
        return None
    out = {k: v for k, v in fit.items() if k not in ("residual", "fitted")}
    out["label"] = label
    out["factors"] = list(X.columns)
    return out


def residualize(series, market):
    """OLS residual of series on market (plus intercept), aligned to series index."""
    frame = pd.concat([series.rename("y"), market.rename("m")], axis=1, join="inner").dropna()
    if len(frame) < MIN_OBS or frame["m"].std() <= 1e-12:
        return series.reindex(series.index) * np.nan
    beta = float(np.cov(frame["y"], frame["m"], ddof=1)[0, 1] / frame["m"].var(ddof=1))
    alpha = float(frame["y"].mean() - beta * frame["m"].mean())
    resid = frame["y"] - (alpha + beta * frame["m"])
    return resid.reindex(series.index)


def _drop_weak_factors(design, max_corr=0.97):
    design = design.dropna(axis=1, how="all")
    keep = []
    for col in design.columns:
        series = design[col]
        if series.notna().sum() < MIN_OBS or series.std(skipna=True) < 1e-8:
            continue
        if keep:
            corr = design[keep].corrwith(series).abs().max()
            if corr is not None and corr > max_corr:
                continue
        keep.append(col)
    return design[keep] if keep else design.iloc[:, 0:0]


def sources_model(strategy, factor_returns):
    """Hierarchical decomposition: market (SPY) + sector residuals vs SPY + bonds + other.

    Sector ETF returns are residualized against SPY so betas are incremental tilts, not a
    second copy of market exposure. Bonds are left raw.
    """
    if MARKET_FACTOR not in factor_returns.columns:
        return None
    spy = factor_returns[MARKET_FACTOR]
    parts = {"market": spy.rename(MARKET_FACTOR)}
    sector_resid = {}
    for ticker in SECTOR_ETFS:
        if ticker in factor_returns.columns:
            sector_resid[ticker] = residualize(factor_returns[ticker], spy)
    if sector_resid:
        parts["sectors"] = pd.DataFrame(sector_resid)
    bond_cols = [t for t in BOND_ETFS if t in factor_returns.columns]
    if bond_cols:
        parts["bonds"] = factor_returns[bond_cols]
    other_cols = [t for t in OTHER_ETFS if t in factor_returns.columns]
    if other_cols:
        parts["other"] = pd.DataFrame({t: residualize(factor_returns[t], spy) for t in other_cols})
    macro_cols = [t for t in MACRO_FACTORS if t in factor_returns.columns]
    if macro_cols:
        parts["macro"] = pd.DataFrame({t: residualize(factor_returns[t], spy) for t in macro_cols})

    capm = capm_report(strategy, spy)
    spy_bonds = multifactor_report(
        strategy, pd.concat([parts["market"], parts.get("bonds", pd.DataFrame())], axis=1),
        "SPY + bonds (IEF, TLT)")
    design = pd.concat([parts["market"], parts.get("bonds", pd.DataFrame()),
                        parts.get("sectors", pd.DataFrame()), parts.get("other", pd.DataFrame()),
                        parts.get("macro", pd.DataFrame())], axis=1)
    design = _drop_weak_factors(design)
    full = multifactor_report(strategy, design, "market + residualized sectors + bonds + other + macro proxies")
    if not full:
        return {"capm": capm, "spyBonds": spy_bonds, "full": None}

    y, X = align(strategy, design)
    fit = ols_hac(y, X)
    contrib = {}
    if fit:
        resid = fit["residual"]
        aligned_X = X.loc[resid.index]
        market_beta = fit["betas"].get(MARKET_FACTOR) or 0.0
        market_ret = (market_beta * aligned_X[MARKET_FACTOR]).mean() * TRADING_DAYS if MARKET_FACTOR in aligned_X else 0.0
        bond_names = [c for c in bond_cols if c in aligned_X.columns]
        sector_names = [c for c in SECTOR_ETFS if c in aligned_X.columns]
        other_names = [c for c in other_cols if c in aligned_X.columns]
        macro_names = [c for c in macro_cols if c in aligned_X.columns]

        def _block(names):
            if not names:
                return 0.0
            return float(sum((fit["betas"].get(n) or 0.0) * aligned_X[n] for n in names).mean() * TRADING_DAYS)

        contrib = {
            "alphaAnnual": fit["alphaAnnual"],
            "marketAnnual": _clean_num(market_ret),
            "sectorTiltsAnnual": _clean_num(_block(sector_names)),
            "bondsAnnual": _clean_num(_block(bond_names)),
            "otherAnnual": _clean_num(_block(other_names)),
            "macroAnnual": _clean_num(_block(macro_names)),
            "strategyMeanAnnual": _clean_num(y.loc[resid.index].mean() * TRADING_DAYS),
        }
        r2_m = (capm or {}).get("r2") or 0.0
        r2_b = (spy_bonds or {}).get("r2") or r2_m
        r2_f = full.get("r2") or r2_b
        contrib["varianceShare"] = {
            "market": _clean_num(r2_m),
            "bondsIncremental": _clean_num(max(0.0, r2_b - r2_m)),
            "sectorAndOtherIncremental": _clean_num(max(0.0, r2_f - r2_b)),
            "residual": _clean_num(max(0.0, 1.0 - r2_f)),
        }
    return {"capm": capm, "spyBonds": spy_bonds, "full": full, "contributions": contrib}


def rolling_capm(strategy, spy, windows=ROLLING_WINDOWS, sample_points=24):
    from statsmodels.regression.rolling import RollingOLS
    import statsmodels.api as sm

    y, X = align(strategy, spy.rename(MARKET_FACTOR))
    if len(y) < min(windows) + 5:
        return {}
    out = {}
    for window in windows:
        if window >= len(y):
            continue
        params = RollingOLS(y, sm.add_constant(X), window=window).fit().params.dropna()
        if params.empty:
            continue
        alpha = params["const"] * TRADING_DAYS
        beta = params[MARKET_FACTOR]
        latest = None
        tail_y, tail_X = y.iloc[-window:], X.iloc[-window:]
        latest_fit = ols_hac(tail_y, tail_X)
        if latest_fit:
            latest = {"alphaAnnual": latest_fit["alphaAnnual"], "alphaT": latest_fit["alphaT"],
                      "alphaP": latest_fit["alphaP"], "beta": latest_fit["betas"].get(MARKET_FACTOR),
                      "betaT": latest_fit["betaT"].get(MARKET_FACTOR), "r2": latest_fit["r2"],
                      "end": str(y.index[-1].date())}
        step = max(1, len(params) // sample_points)
        path = [{"date": str(idx.date()), "alphaAnnual": _clean_num(alpha.loc[idx]),
                 "beta": _clean_num(beta.loc[idx])} for idx in params.index[::step]]
        if len(params.index[::step]) == 0 or params.index[-1] != params.index[::step][-1]:
            path.append({"date": str(params.index[-1].date()),
                         "alphaAnnual": _clean_num(alpha.iloc[-1]), "beta": _clean_num(beta.iloc[-1])})
        out[str(window)] = {
            "window": window,
            "alphaAnnual": _series_summary(alpha),
            "beta": _series_summary(beta),
            "latestWindow": latest,
            "sampledPath": path,
        }
    return out


def _series_summary(series):
    series = series.dropna()
    if series.empty:
        return None
    return {"latest": _clean_num(series.iloc[-1]), "mean": _clean_num(series.mean()),
            "min": _clean_num(series.min()), "max": _clean_num(series.max()),
            "q10": _clean_num(series.quantile(0.1)), "q90": _clean_num(series.quantile(0.9))}


def beta_matched_blend(strategy, spy, ief=None, tlt=None):
    """Cheaper comparison: CAPM-beta in SPY, the rest in intermediate Treasuries (or cash).

    Also builds a replicating mix from a regression on SPY+IEF+TLT, weights clipped to [0, 1]
    and scaled down if they exceed 1 (the live book does not use leverage).
    """
    capm = capm_report(strategy, spy)
    if not capm or capm.get("beta") is None:
        return None
    beta = float(capm["beta"])
    pieces = [spy.rename("SPY")]
    if ief is not None:
        pieces.append(ief.rename("IEF"))
    if tlt is not None:
        pieces.append(tlt.rename("TLT"))
    y, cols = align(strategy, pd.concat(pieces, axis=1))
    spy_a = cols["SPY"]
    ief_a = cols["IEF"] if "IEF" in cols else pd.Series(0.0, index=y.index)
    w_spy = min(max(beta, 0.0), 1.0)
    w_ief = (1.0 - w_spy) if "IEF" in cols else 0.0
    w_cash_simple = 1.0 - w_spy - w_ief
    simple = w_spy * spy_a + w_ief * ief_a
    simple_stats = _blend_stats(y, simple, {"SPY": w_spy, "IEF": w_ief, "cash": w_cash_simple},
                                "CAPM beta in SPY, remainder in IEF (weights clipped to [0, 1])")

    replicating = None
    factor_cols = [c for c in ("SPY", "IEF", "TLT") if c in cols]
    if factor_cols:
        fit = ols_hac(y, cols[factor_cols])
        if fit:
            raw = {k: max(float(fit["betas"].get(k) or 0.0), 0.0) for k in factor_cols}
            total = sum(raw.values())
            if total > 1.0:
                raw = {k: v / total for k, v in raw.items()}
            raw.setdefault("IEF", 0.0)
            raw.setdefault("TLT", 0.0)
            raw["cash"] = max(0.0, 1.0 - sum(v for k, v in raw.items() if k != "cash"))
            mix = sum(raw.get(c, 0.0) * cols[c] for c in factor_cols)
            replicating = _blend_stats(
                y, mix, raw,
                "Regression on SPY+IEF+TLT; negative weights zeroed, sum above 1 scaled down, residual is cash")
    return {"capmBeta": beta, "simple": simple_stats, "replicating": replicating}


def _blend_stats(strategy, blend, weights, method):
    te = tracking_error(strategy, blend)
    vs = ols_hac(strategy, blend.rename("blend"))
    strat_perf = performance(strategy)
    blend_perf = performance(blend)
    keys = ("cagr", "volatility", "sharpe", "maxDrawdown", "totalReturn", "sessions")
    return {
        "method": method,
        "weights": {k: _clean_num(v) for k, v in weights.items()},
        "strategy": {k: _clean_num(strat_perf.get(k)) for k in keys},
        "blend": {k: _clean_num(blend_perf.get(k)) for k in keys},
        "trackingError": None if not te else te["trackingError"],
        "informationRatio": None if not te else te["informationRatio"],
        "alphaVsBlendAnnual": None if not vs else vs["alphaAnnual"],
        "alphaVsBlendT": None if not vs else vs["alphaT"],
        "alphaVsBlendP": None if not vs else vs["alphaP"],
        "r2VsBlend": None if not vs else vs["r2"],
    }


def weights_from_decisions(decisions, dates):
    """Rebuild a date x ticker target-weight frame from fill-dated decisions (forward-filled)."""
    if decisions is None or len(decisions) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(dates)))
    frame = decisions.copy()
    if "status" in frame.columns:
        frame = frame[frame["status"].fillna("filled").astype(str).str.lower().ne("rejected")]
    date_col = "fill_date" if "fill_date" in frame.columns and frame["fill_date"].notna().any() else "decision_date"
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.dropna(subset=[date_col, "ticker"])
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    tickers = sorted(frame["ticker"].astype(str).unique())
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    for day, group in frame.sort_values(date_col).groupby(date_col, sort=True):
        last = group.groupby("ticker", sort=False).tail(1)
        loc = dates[dates >= pd.Timestamp(day)]
        if len(loc) == 0:
            continue
        for _, row in last.iterrows():
            weights.loc[loc, str(row["ticker"])] = float(row["target_weight"] or 0.0)
    return weights


def assign_groups(tickers, factor_returns, asset_returns=None):
    """Map each ticker to a Brinson group. Known ETFs are hard-coded; stocks go to the sector
    ETF they correlate with most over the sample (a mild look-ahead on the label only).
    """
    groups = {}
    for ticker in tickers:
        if ticker in GROUP_OF:
            groups[ticker] = GROUP_OF[ticker]
            continue
        if asset_returns is None or ticker not in getattr(asset_returns, "columns", []):
            groups[ticker] = "Unclassified stocks"
            continue
        best, best_corr = "Unclassified stocks", CORR_ASSIGN_MIN
        series = asset_returns[ticker]
        for etf in SECTOR_ETFS:
            if etf not in factor_returns.columns:
                continue
            corr = series.corr(factor_returns[etf])
            if corr is not None and math.isfinite(corr) and corr > best_corr:
                best, best_corr = GROUP_OF[etf], float(corr)
        groups[ticker] = best
    return groups


def brinson(weights, asset_returns, groups, factor_returns, benchmark="spy"):
    """Daily Brinson-Fachler attribution, summed over the overlap of weights and returns.

    benchmark='spy': 100% US Market (SPY). benchmark='equal_sectors': equal weight in the 11
    sector ETFs (renormalized when a sector ETF is missing). Cash is the residual of portfolio
    weights and earns 0 (the research book uses a 0 risk-free rate).
    """
    if weights is None or weights.empty or asset_returns is None or asset_returns.empty:
        return None
    dates = weights.index.intersection(asset_returns.index).intersection(factor_returns.index)
    if len(dates) < MIN_OBS:
        return None
    weights = weights.reindex(dates).fillna(0.0)
    rets = asset_returns.reindex(index=dates, columns=weights.columns).fillna(0.0)
    port = (weights * rets).sum(axis=1)
    group_names = sorted(set(groups.values()) | {"Cash"})
    wp = pd.DataFrame(0.0, index=dates, columns=group_names)
    rp = pd.DataFrame(0.0, index=dates, columns=group_names)
    for ticker, group in groups.items():
        if ticker not in weights.columns:
            continue
        wp[group] = wp[group] + weights[ticker]
        rp[group] = rp[group] + weights[ticker] * rets[ticker]
    with np.errstate(divide="ignore", invalid="ignore"):
        rp = rp.div(wp.replace(0.0, np.nan), axis=0).fillna(0.0)
    wp["Cash"] = (1.0 - wp.drop(columns=["Cash"]).sum(axis=1)).clip(lower=0.0)
    rp["Cash"] = 0.0

    wb = pd.DataFrame(0.0, index=dates, columns=group_names)
    rb = pd.DataFrame(0.0, index=dates, columns=group_names)
    if benchmark == "spy":
        if "US Market" in wb.columns:
            wb["US Market"] = 1.0
            rb["US Market"] = factor_returns.reindex(dates)[MARKET_FACTOR].fillna(0.0)
        for group, etf in GROUP_BENCHMARK.items():
            if group == "US Market" or group not in rb.columns:
                continue
            if etf in factor_returns.columns:
                rb[group] = factor_returns.reindex(dates)[etf].fillna(0.0)
    else:
        present = [GROUP_OF[t] for t in SECTOR_ETFS if GROUP_OF[t] in wb.columns]
        for group in present:
            etf = GROUP_BENCHMARK[group]
            if etf in factor_returns.columns:
                rb[group] = factor_returns.reindex(dates)[etf].fillna(0.0)
        available = pd.DataFrame({
            g: factor_returns.reindex(dates)[GROUP_BENCHMARK[g]].notna()
            if GROUP_BENCHMARK[g] in factor_returns.columns else False
            for g in present
        })
        denom = available.sum(axis=1).replace(0, np.nan)
        for group in present:
            wb[group] = available[group].astype(float).div(denom, axis=0).fillna(0.0)
        wb["Cash"] = (1.0 - wb.drop(columns=["Cash"]).sum(axis=1)).clip(lower=0.0)

    alloc = ((wp - wb) * rb).sum(axis=1)
    selec = (wb * (rp - rb)).sum(axis=1)
    inter = ((wp - wb) * (rp - rb)).sum(axis=1)
    bench = (wb * rb).sum(axis=1)
    excess = port - bench
    by_group = []
    for group in group_names:
        if wp[group].mean() < 1e-6 and wb[group].mean() < 1e-6:
            continue
        a = float(((wp[group] - wb[group]) * rb[group]).sum())
        s = float((wb[group] * (rp[group] - rb[group])).sum())
        i = float(((wp[group] - wb[group]) * (rp[group] - rb[group])).sum())
        by_group.append({
            "group": group,
            "avgPortfolioWeight": _clean_num(wp[group].mean()),
            "avgBenchmarkWeight": _clean_num(wb[group].mean()),
            "avgPortfolioReturnAnnual": _clean_num(rp[group].mean() * TRADING_DAYS),
            "avgBenchmarkReturnAnnual": _clean_num(rb[group].mean() * TRADING_DAYS),
            "allocation": _clean_num(a),
            "selection": _clean_num(s),
            "interaction": _clean_num(i),
            "total": _clean_num(a + s + i),
        })
    by_group.sort(key=lambda row: abs(row["total"] or 0.0), reverse=True)
    residual = float((excess - (alloc + selec + inter)).abs().mean())
    return {
        "benchmark": "100% SPY" if benchmark == "spy" else "Equal-weight sector ETFs (XLK…XLC)",
        "sessions": int(len(dates)),
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "portfolioReturn": _clean_num(float((1 + port).prod() - 1)),
        "benchmarkReturn": _clean_num(float((1 + bench).prod() - 1)),
        "excessReturn": _clean_num(float((1 + port).prod() - 1) - float((1 + bench).prod() - 1)),
        "allocationSum": _clean_num(float(alloc.sum())),
        "selectionSum": _clean_num(float(selec.sum())),
        "interactionSum": _clean_num(float(inter.sum())),
        "arithmeticExcessSum": _clean_num(float(excess.sum())),
        "identityResidualMeanAbs": _clean_num(residual),
        "avgGross": _clean_num(float(weights.abs().sum(axis=1).mean())),
        "avgCash": _clean_num(float(wp["Cash"].mean())),
        "byGroup": by_group,
    }


def verdict(label, capm, blend, contributions=None, sample_note=""):
    """Plain-language answer to: is this just a cheaper index fund?"""
    if not capm or capm.get("alphaT") is None or capm.get("beta") is None:
        return {"label": "unavailable", "isCheaperIndexFund": None,
                "summary": f"{label}: not enough overlapping returns to estimate alpha and beta.",
                "reasons": []}
    t = capm["alphaT"]
    alpha = capm.get("alphaAnnual")
    r2 = capm.get("r2") or 0.0
    beta = capm["beta"]
    te = capm.get("trackingError")
    ir = capm.get("informationRatio")
    sig5 = abs(t) >= 1.96
    sig10 = abs(t) >= 1.65
    simple = (blend or {}).get("simple") or {}
    blend_sharpe = (simple.get("blend") or {}).get("sharpe")
    strat_sharpe = (simple.get("strategy") or {}).get("sharpe")
    reasons = []
    if alpha is not None and capm.get("alphaP") is not None:
        reasons.append(f"CAPM alpha {alpha * 100:.2f}%/yr, HAC t={t:.2f}, p={capm.get('alphaP'):.3f}")
    elif alpha is not None:
        reasons.append(f"CAPM alpha {alpha * 100:.2f}%/yr, HAC t={t:.2f}")
    if capm.get("betaT") is not None:
        reasons.append(f"beta vs SPY {beta:.2f} (t={capm.get('betaT'):.2f})")
    else:
        reasons.append(f"beta vs SPY {beta:.2f}")
    reasons.append(f"R² {r2:.2f}")
    if te is not None:
        reasons.append(f"tracking error vs SPY {te * 100:.1f}%/yr, information ratio {ir:.2f}" if ir is not None else
                       f"tracking error vs SPY {te * 100:.1f}%/yr")
    if contributions and contributions.get("varianceShare"):
        vs = contributions["varianceShare"]
        reasons.append(
            "variance shares — market {:.0f}%, bonds {:.0f}%, sector/other tilts {:.0f}%, residual {:.0f}%".format(
                (vs.get("market") or 0) * 100, (vs.get("bondsIncremental") or 0) * 100,
                (vs.get("sectorAndOtherIncremental") or 0) * 100, (vs.get("residual") or 0) * 100))
    if strat_sharpe is not None and blend_sharpe is not None:
        reasons.append(f"Sharpe {strat_sharpe:.2f} vs beta-matched SPY+IEF blend {blend_sharpe:.2f}")
    if simple.get("weights"):
        w = simple["weights"]
        reasons.append("blend weights " + ", ".join(
            f"{k} {v * 100:.0f}%" for k, v in w.items() if v is not None and abs(v) > 0.005))

    alpha_txt = f"{alpha * 100:.2f}" if alpha is not None else "n/a"
    if sig5 and alpha is not None and alpha > 0:
        tag, cheaper = "significant_alpha", False
        summary = (f"{label} is not just a cheaper index fund: annualized CAPM alpha is {alpha_txt}% "
                   f"(HAC t={t:.2f}), significant at 5%, with beta {beta:.2f} and R² {r2:.2f}.")
    elif sig5 and alpha is not None and alpha < 0:
        tag, cheaper = "significant_negative_alpha", False
        summary = (f"{label} underperformed a beta-matched market exposure: annualized CAPM alpha is "
                   f"{alpha_txt}% (HAC t={t:.2f}), significant at 5%. Beta {beta:.2f}, R² {r2:.2f}.")
    elif r2 >= 0.65 and not sig10 and abs(beta) >= 0.4:
        tag, cheaper = "mostly_market", True
        summary = (f"{label} looks mostly like a lower-beta market (and bond) mix, not a distinct skill engine: "
                   f"beta {beta:.2f}, R² {r2:.2f}, and alpha {alpha_txt}%/yr is not distinguishable from zero "
                   f"(HAC t={t:.2f}) on this short sample.")
    elif not sig10:
        tag, cheaper = "inconclusive", None
        summary = (f"{label} does not show statistically significant CAPM alpha on this sample "
                   f"(alpha {alpha_txt}%/yr, HAC t={t:.2f}, beta {beta:.2f}, R² {r2:.2f}). "
                   f"The series is too short to treat residual returns as proven skill — or to rule them out.")
    else:
        tag, cheaper = "marginal", None
        summary = (f"{label} has marginally significant CAPM alpha ({alpha_txt}%/yr, HAC t={t:.2f}, 10% but "
                   f"not 5% level), beta {beta:.2f}, R² {r2:.2f}. Treat this as a hint, not a result.")
    if sample_note:
        summary += " " + sample_note
    return {"label": tag, "isCheaperIndexFund": cheaper, "summary": summary, "reasons": reasons}


# ------------------------------------------------------------------ data loaders

def attach_macro_factors(rets):
    """Join lagged free-data macro factor proxies. Real CDS/CDX and swap quotes are proprietary.

    DXY is the ICE-formula dollar-index proxy return. UST10Y is -Delta DGS10 / 100 (1-year
    duration units, not a Treasury total return). CURVE is the daily change in the 10y-2y
    slope. HY_OAS is -Delta ICE BofA HY OAS / 100 (tightening is positive; a CDS/CDX proxy).
    SOFR_TBILL is the change in SOFR minus 3M Treasury (a funding proxy, not a swap spread).
    Publication lag is 1 session. Missing series are left off.
    """
    extras = {}
    try:
        from macro_data import load_macro
        panel = load_macro(lag=1)
    except Exception:
        panel = None
    if panel is None or getattr(panel, "empty", True):
        try:
            from macro_data import load_dxy
            dxy = load_dxy(lag=1)
            if dxy is not None and not dxy.dropna().empty:
                extras["DXY"] = dxy.pct_change(fill_method=None)
        except Exception:
            extras = extras
    else:
        if "dxy" in panel.columns:
            extras["DXY"] = panel["dxy"].pct_change(fill_method=None)
        elif "dxy_ret" in panel.columns:
            extras["DXY"] = panel["dxy_ret"]
        if "dgs10" in panel.columns:
            extras["UST10Y"] = -panel["dgs10"].diff() / 100.0
        slope = panel["curve_slope"] if "curve_slope" in panel.columns else panel["t10y2y"] if "t10y2y" in panel.columns else None
        if slope is not None:
            extras["CURVE"] = slope.diff() / 100.0
        if "hy_oas" in panel.columns:
            extras["HY_OAS"] = -panel["hy_oas"].diff() / 100.0
        if "sofr_tbill_spread" in panel.columns:
            extras["SOFR_TBILL"] = panel["sofr_tbill_spread"].diff() / 100.0
    if not extras:
        return rets
    frame = pd.concat([rets] + [s.rename(name) for name, s in extras.items()], axis=1)
    # Keep the original equity index; macro series may have extra FX or FRED days.
    return frame.reindex(rets.index)


def load_factor_returns(panel=None, tickers=None, start=None, end=None):
    names = list(dict.fromkeys(list(tickers or []) + list(ALL_FACTORS)))
    if panel is not None:
        have = [t for t in names if t in panel.close.columns]
        if MARKET_FACTOR in have and len(have) >= 8:
            sliced = panel
            if end:
                sliced = panel.upto(end)
            rets = sliced.total_return()[have]
            if start:
                rets = rets.loc[pd.Timestamp(start):]
            return attach_macro_factors(rets)
    loaded = load_panel(start=start, end=end, tickers=names)
    return attach_macro_factors(loaded.total_return())


def load_equity_curves(out_dir=OUT_DIR):
    path = Path(out_dir) / "equity_curves.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return {col: returns_from_equity(frame[col]) for col in frame.columns}


def load_walkforward_returns(out_dir=OUT_DIR):
    out = {}
    for family in ("tsmom", "meanrev", "combined"):
        path = Path(out_dir) / f"walkforward_{family}_oos_equity.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        col = "equity" if "equity" in frame.columns else frame.columns[0]
        out[family] = returns_from_equity(frame[col])
    return out


def load_decisions(out_dir, stem):
    path = Path(out_dir) / f"decisions_{stem}.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def load_walkforward_decisions(out_dir, family):
    path = Path(out_dir) / f"walkforward_{family}_oos_decisions.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def load_paper_returns(db_path=PAPER_DB, account="default"):
    path = Path(db_path)
    if not path.is_file():
        return None, "No paper-trading database yet."
    con = None
    try:
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        history = pd.read_sql_query(
            "SELECT date, equity, daily_return FROM equity_history WHERE account = ? ORDER BY date",
            con, params=(account,))
        fills = pd.read_sql_query(
            "SELECT fill_date, ticker, shares, price, notional FROM fills WHERE account = ? ORDER BY fill_date",
            con, params=(account,))
        signal = pd.read_sql_query(
            "SELECT asof, ticker, target_weight, sleeve FROM signal_state WHERE account = ? ORDER BY asof",
            con, params=(account,))
    except sqlite3.Error as error:
        return None, f"Could not read the paper database ({error})."
    finally:
        if con is not None:
            con.close()
    if history is None or history.empty:
        return {"available": False,
                "reason": "No paper equity history yet. Run the swing paper trader and save a session.",
                "fills": int(0 if fills is None else len(fills)),
                "signalRows": int(0 if signal is None else len(signal))}, None
    history["date"] = pd.to_datetime(history["date"])
    history = history.set_index("date").sort_index()
    if history["daily_return"].notna().sum() >= MIN_OBS:
        returns = history["daily_return"].astype(float)
    else:
        returns = returns_from_equity(history["equity"])
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    weights = None
    if signal is not None and not signal.empty:
        signal = signal.rename(columns={"asof": "fill_date"})
        weights = weights_from_decisions(signal, returns.index)
    return {"available": True, "returns": returns, "weights": weights, "account": account,
            "start": str(returns.index[0].date()) if len(returns) else None,
            "end": str(returns.index[-1].date()) if len(returns) else None,
            "sessions": int(len(returns))}, None


def load_daytrade_returns(out_dir=OUT_DIR, summary=None):
    """Rebuild a daily return series from the saved default-parameter trades + the session calendar."""
    from price_history import complete_market_dates, open_history
    from fetch_and_model import DATABASE

    trades_path = Path(out_dir) / "daytrade_trades_defaults.csv"
    summary_path = Path(out_dir) / "daytrade_summary.json"
    if summary is None and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not trades_path.is_file() or not summary:
        return None, "No day-trading research trades saved yet. Run engine/daytrader.py research."
    block = summary.get("fullPeriodDefaults") or {}
    first, last = block.get("first"), block.get("last")
    initial = float(summary.get("initialEquity") or 10_000.0)
    if not first or not last:
        return None, "Day-trading summary is missing a session range."
    trades = pd.read_csv(trades_path)
    trades["session_date"] = pd.to_datetime(trades["session_date"])
    daily_pnl = trades.groupby("session_date")["pnl"].sum()
    with open_history(DATABASE) as connection:
        sessions = [pd.Timestamp(day) for day in complete_market_dates(connection)
                    if first <= str(day) <= last]
    if not sessions:
        sessions = sorted(daily_pnl.index)
    equity = initial
    records = []
    for day in sessions:
        pnl = float(daily_pnl.get(day, 0.0) or 0.0)
        start_eq = equity
        ret = pnl / start_eq if start_eq else 0.0
        equity = start_eq + pnl
        records.append((day, ret, equity, pnl))
    frame = pd.DataFrame(records, columns=["date", "return", "equity", "pnl"]).set_index("date")
    saved = (block.get("performance") or {}).get("totalReturn")
    got = float(frame["equity"].iloc[-1] / initial - 1) if len(frame) else None
    note = None
    if saved is not None and got is not None and abs(got - saved) > 0.005:
        note = (f"Reconstructed day-trade total return {got:.4f} differs from the saved {saved:.4f}; "
                "daily PnL is taken from daytrade_trades_defaults.csv and zero-filled on no-trade sessions.")
    return {"returns": frame["return"], "initialEquity": initial, "reconstructedTotalReturn": got,
            "savedTotalReturn": saved, "sessions": int(len(frame)), "note": note,
            "daysWithTrades": int((frame["pnl"] != 0).sum())}, None


# ------------------------------------------------------------------ orchestration

SAMPLE_NOTE = ("These figures use about two years of daily data (roughly 1.5 years after the 100-day warm-up, "
               "and about one year out of sample). HAC t-stats on daily alpha have more power than a Sharpe "
               "comparison but still cannot prove or disprove a modest edge.")


def _data_caveats(lines, panel=None):
    """Prepend Schwab long-history / survivorship warnings when that data is in use."""
    from schwab_daily import ADJUSTMENT_NOTE, SURVIVORSHIP_WARNING, merge_caveats, read_summary
    warnings = list(getattr(panel, "warnings", None) or [])
    if not warnings:
        coverage = ((read_summary() or {}).get("coverage") or {})
        if coverage.get("rows"):
            warnings = [SURVIVORSHIP_WARNING, ADJUSTMENT_NOTE]
    return merge_caveats(lines, warnings)


def attribute_series(name, label, returns, factor_returns, weights=None, asset_returns=None, groups=None,
                     include_brinson=True):
    """Run the full attribution stack on one return series."""
    y, X = align(returns, factor_returns)
    if len(y) < MIN_OBS:
        return {"name": name, "label": label, "available": False,
                "reason": f"Only {len(y)} overlapping sessions (need {MIN_OBS})."}
    spy = X[MARKET_FACTOR] if MARKET_FACTOR in X.columns else None
    ief = X["IEF"] if "IEF" in X.columns else None
    tlt = X["TLT"] if "TLT" in X.columns else None
    if spy is None:
        return {"name": name, "label": label, "available": False,
                "reason": "SPY returns are missing from the factor panel."}
    sources = sources_model(y, X)
    capm = (sources or {}).get("capm") or capm_report(y, spy)
    blend = beta_matched_blend(y, spy, ief, tlt)
    rolling = rolling_capm(y, spy)
    style_cols = [c for c in (MARKET_FACTOR,) + STYLE_ETFS + BOND_ETFS + OTHER_ETFS if c in X.columns]
    style = multifactor_report(y, X[style_cols], "style: SPY + size/growth + bonds + gold + international") if style_cols else None
    brinson_out = {}
    if include_brinson and weights is not None and not getattr(weights, "empty", True):
        names = list(weights.columns)
        held = asset_returns
        missing = []
        if held is None:
            missing = [t for t in names if t not in X.columns]
            held = X.reindex(columns=[t for t in names if t in X.columns])
        else:
            missing = [t for t in names if t not in held.columns]
            extra = [t for t in names if t in X.columns and t not in held.columns]
            if extra:
                held = held.join(X[extra], how="outer")
        group_map = groups or assign_groups(names, X, held)
        brinson_out["vsSpy"] = brinson(weights, held, group_map, X, "spy")
        brinson_out["vsEqualSectors"] = brinson(weights, held, group_map, X, "equal_sectors")
        brinson_out["groupAssignment"] = group_map
        if missing:
            brinson_out["missingAssetReturns"] = missing
    perf = performance(y)
    return {
        "name": name,
        "label": label,
        "available": True,
        "period": {"start": str(y.index[0].date()), "end": str(y.index[-1].date()), "sessions": int(len(y))},
        "performance": {k: _clean_num(perf.get(k)) for k in
                        ("cagr", "volatility", "sharpe", "maxDrawdown", "totalReturn", "sessions")},
        "capm": capm,
        "spyBonds": (sources or {}).get("spyBonds"),
        "sources": {k: (sources or {}).get(k) for k in ("full", "contributions")},
        "styleFactors": style,
        "rollingCapm": rolling,
        "betaMatchedBlend": blend,
        "brinson": brinson_out or None,
        "verdict": verdict(label, capm, blend, (sources or {}).get("contributions"), SAMPLE_NOTE),
    }


def _prep_holdings(out_dir, stem, dates, factor_returns, extra_panel=None):
    if stem.startswith("wf_"):
        decisions = load_walkforward_decisions(out_dir, stem[3:])
    else:
        decisions = load_decisions(out_dir, stem)
    if decisions is None:
        return None, None, None
    weights = weights_from_decisions(decisions, dates)
    tickers = list(weights.columns)
    asset = None
    if extra_panel is not None:
        have = [t for t in tickers if t in extra_panel.columns]
        if have:
            asset = extra_panel[have]
    missing = [t for t in tickers if asset is None or t not in asset.columns]
    if missing:
        try:
            extra = load_panel(tickers=missing).total_return()
            if asset is None:
                asset = extra
            else:
                extra = extra.drop(columns=[c for c in extra.columns if c in asset.columns], errors="ignore")
                if len(extra.columns):
                    asset = asset.join(extra, how="outer")
        except Exception as error:
            print(f"warning: could not load asset returns for {len(missing)} names: {error}", file=sys.stderr)
    groups = assign_groups(tickers, factor_returns, asset)
    return weights, asset, groups


def run(out_dir=None, panel=None, include_paper=True, include_daytrade=True):
    """Compute attribution for every available strategy series and write attribution.json."""
    started = time.time()
    out_dir = Path(out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    factor_returns = load_factor_returns(panel)
    curves = load_equity_curves(out_dir)
    walk = load_walkforward_returns(out_dir)
    decision_stems = {
        "core_tsmom": "core_tsmom",
        "overlay_meanrev": "overlay_meanrev",
        "combined": "combined",
    }
    extra_tickers = set()
    for stem in decision_stems.values():
        dec = load_decisions(out_dir, stem)
        if dec is not None and "ticker" in dec.columns:
            extra_tickers.update(dec["ticker"].dropna().astype(str))
    for family in ("tsmom", "meanrev", "combined"):
        dec = load_walkforward_decisions(out_dir, family)
        if dec is not None and "ticker" in dec.columns:
            extra_tickers.update(dec["ticker"].dropna().astype(str))
    extra_tickers -= set(factor_returns.columns)
    holdings_panel = None
    if extra_tickers:
        print(f"loading {len(extra_tickers)} held-name return series for Brinson", file=sys.stderr, flush=True)
        try:
            holdings_panel = load_panel(tickers=sorted(extra_tickers)).total_return()
        except Exception as error:
            print(f"warning: holdings panel failed: {error}", file=sys.stderr)

    series = {}
    labels = {
        "core_tsmom": "TSMOM core (full-period backtest)",
        "overlay_meanrev": "Mean-reversion overlay (full-period backtest)",
        "combined": "Combined TSMOM + overlay (full-period backtest)",
        "benchmark_spy": "Buy-and-hold SPY (sanity check)",
        "benchmark_equal_weight_etfs": "Equal-weight core ETFs (benchmark)",
    }
    for name, rets in curves.items():
        weights = asset = groups = None
        if name in decision_stems:
            weights, asset, groups = _prep_holdings(
                out_dir, decision_stems[name], rets.index, factor_returns, holdings_panel)
        series[f"backtest.{name}"] = attribute_series(
            f"backtest.{name}", labels.get(name, name), rets, factor_returns, weights, asset, groups,
            include_brinson=name in decision_stems)
    wf_labels = {
        "tsmom": "TSMOM (walk-forward out-of-sample)",
        "meanrev": "Mean-reversion (walk-forward out-of-sample)",
        "combined": "Combined (walk-forward out-of-sample)",
    }
    for family, rets in walk.items():
        weights, asset, groups = _prep_holdings(out_dir, f"wf_{family}", rets.index, factor_returns, holdings_panel)
        series[f"walkforward.{family}"] = attribute_series(
            f"walkforward.{family}", wf_labels.get(family, family), rets, factor_returns, weights, asset, groups)

    paper_block = {"available": False, "reason": "Paper account not requested."}
    if include_paper:
        paper, err = load_paper_returns()
        if err:
            paper_block = {"available": False, "reason": err}
        elif not paper or not paper.get("available"):
            paper_block = paper or {"available": False, "reason": "No paper equity history yet."}
        elif len(paper["returns"]) < MIN_OBS:
            paper_block = {
                "available": False,
                "reason": f"Paper history has only {len(paper['returns'])} sessions (need {MIN_OBS}).",
                "sessions": len(paper["returns"]), "start": paper.get("start"), "end": paper.get("end"),
            }
        else:
            series["paper.default"] = attribute_series(
                "paper.default", "Paper account (default)", paper["returns"], factor_returns,
                paper.get("weights"), None, None)
            paper_block = {"available": True, "account": paper.get("account"),
                           "start": paper.get("start"), "end": paper.get("end"),
                           "sessions": paper.get("sessions")}

    daytrade_block = {"available": False, "reason": "Day-trading research not requested."}
    if include_daytrade:
        day, err = load_daytrade_returns(out_dir)
        if err:
            daytrade_block = {"available": False, "reason": err}
        else:
            series["daytrade.full_defaults"] = attribute_series(
                "daytrade.full_defaults", "Day trading ORB (full-period defaults)", day["returns"],
                factor_returns, include_brinson=False)
            if day.get("note"):
                series["daytrade.full_defaults"]["reconstructionNote"] = day["note"]
            daytrade_block = {"available": True, "sessions": day["sessions"],
                              "daysWithTrades": day["daysWithTrades"],
                              "reconstructedTotalReturn": day["reconstructedTotalReturn"],
                              "savedTotalReturn": day["savedTotalReturn"],
                              "note": day.get("note")}

    headlines = {k: (v.get("verdict") or {}).get("summary") for k, v in series.items() if v.get("available")}
    payload = {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data": {
            "factorStart": str(factor_returns.index[0].date()),
            "factorEnd": str(factor_returns.index[-1].date()),
            "factorSessions": int(len(factor_returns)),
            "factors": [c for c in ALL_FACTORS if c in factor_returns.columns],
            "sampleNote": SAMPLE_NOTE,
        },
        "methods": {
            "returns": "Close-to-close total returns including dividends; strategy returns from equity curves.",
            "hacLags": "Newey-West 4*(n/100)^(2/9), same rule as engine/quant_tools.py.",
            "rollingWindows": list(ROLLING_WINDOWS),
            "sourcesModel": "SPY + IEF/TLT raw + sector/other ETFs residualized against SPY, so sector betas are incremental tilts.",
            "brinson": "Brinson-Fachler on forward-filled target weights from fill dates; stocks mapped to the sector ETF with the highest full-sample correlation (label only; a mild look-ahead).",
            "betaMatchedBlend": "w in SPY equal to clipped CAPM beta, remainder in IEF; plus a clipped SPY+IEF+TLT replicating mix.",
            "riskFreeRate": 0,
        },
        "series": series,
        "headlines": headlines,
        "paper": paper_block,
        "daytrade": daytrade_block,
        "caveats": _data_caveats([
            SAMPLE_NOTE,
            "Full-period backtests include the in-sample window used to set defaults; walk-forward OOS is the more honest test of residual skill.",
            "Sector assignment for individual stocks uses full-sample correlation with sector ETFs (look-ahead on the label, not on the returns).",
            "Brinson effects are arithmetic daily sums and will not exactly equal the compounded excess return; the identity residual is reported.",
            "The day-trading series is rebuilt from saved trades and a 0-return fill on sessions with no trade; holdings-based attribution is skipped because the book is flat overnight.",
            "No risk-free rate is subtracted (consistent with the rest of the research book).",
        ], panel),
        "runSeconds": round(time.time() - started, 1),
    }
    payload["strategies"] = {PANEL_NAMES.get(k, k): _to_panel_report(v) for k, v in series.items()}
    payload = _strip_models(payload)
    write_json(out_dir / "attribution.json", payload)
    print(f"wrote {out_dir / 'attribution.json'} in {payload['runSeconds']}s", file=sys.stderr, flush=True)
    return payload


def _strip_models(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            # Drop statsmodels leftovers, but keep numeric keys such as varianceShare["residual"].
            if k in ("residual", "fitted") and not isinstance(v, (int, float, type(None))):
                continue
            out[k] = _strip_models(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_strip_models(v) for v in value]
    if isinstance(value, pd.Series):
        return None
    return value


PREFERRED_ORDER = (
    "backtest.core_tsmom", "backtest.overlay_meanrev", "backtest.combined",
    "walkforward.tsmom", "walkforward.meanrev", "walkforward.combined",
    "daytrade.full_defaults", "paper.default",
    "backtest.benchmark_spy", "backtest.benchmark_equal_weight_etfs",
)


def markdown_section(attr):
    """Markdown block appended to research_report.md.

    Accepts the hybrid payload (series + strategies) or a panel-only {strategies: ...} object.
    """
    if not attr:
        return ""
    if not attr.get("series") and attr.get("strategies"):
        # Reconstruct a minimal series-like view from the panel schema so the tables still render.
        attr = dict(attr)
        series = {}
        for name, report in attr["strategies"].items():
            if report.get("skipped"):
                series[name] = {"available": False, "label": name, "reason": report["skipped"]}
                continue
            series[name] = {
                "available": True,
                "label": name,
                "period": report.get("period") or {"sessions": (report.get("market") or {}).get("n")},
                "capm": {
                    "beta": (report.get("market") or {}).get("spyBeta"),
                    "r2": (report.get("market") or {}).get("r2"),
                    "alphaAnnual": (report.get("market") or {}).get("alphaAnnual"),
                    "alphaT": (report.get("market") or {}).get("alphaT"),
                    "trackingError": (report.get("market") or {}).get("trackingError"),
                    "informationRatio": (report.get("market") or {}).get("informationRatio"),
                },
                "performance": report.get("performance") or ((report.get("betaMatched") or {}).get("strategy")),
                "betaMatchedBlend": {"simple": {
                    "blend": (report.get("betaMatched") or {}).get("cashMix") or (report.get("betaMatched") or {}).get("bondMix"),
                    "strategy": (report.get("betaMatched") or {}).get("strategy"),
                    "weights": ((report.get("betaMatched") or {}).get("cashMix") or {}).get("weights"),
                    "informationRatio": ((report.get("betaMatched") or {}).get("bondMix") or {}).get("informationRatio"),
                    "alphaVsBlendAnnual": ((report.get("betaMatched") or {}).get("bondMix") or {}).get("alphaAnnual"),
                    "alphaVsBlendT": ((report.get("betaMatched") or {}).get("bondMix") or {}).get("alphaT"),
                }},
                "sources": {"contributions": report.get("contributions"),
                            "full": report.get("sector")},
                "brinson": {"vsSpy": {**(report.get("brinson") or {}),
                                      "byGroup": (report.get("brinson") or {}).get("byGroup")}},
                "verdict": {"label": (report.get("verdict") or {}).get("label"),
                            "summary": (report.get("verdict") or {}).get("text"),
                            "reasons": (report.get("verdict") or {}).get("reasons") or []},
            }
        attr["series"] = series
    if not attr.get("series"):
        return ""
    lines = ["## Return attribution (market, sectors, residual skill)", "",
             "Full-sample OLS of daily strategy returns on factor returns, with HAC/Newey-West t-stats. "
             "A beta-matched SPY+IEF blend is the 'cheaper index fund' comparison. Holdings-based (Brinson) "
             "effects use the backtest's target weights.", ""]
    if attr.get("data", {}).get("sampleNote"):
        lines += [attr["data"]["sampleNote"], ""]
    lines += [
        "| Series | Sessions | Beta vs SPY | R² | Alpha %/yr | Alpha t (HAC) | TE vs SPY | IR vs SPY | Blend Sharpe | Strategy Sharpe | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = [k for k in PREFERRED_ORDER if k in attr["series"]]
    order += [k for k in attr["series"] if k not in PREFERRED_ORDER]
    for key in order:
        row = attr["series"][key]
        if not row.get("available"):
            lines.append(f"| {row.get('label', key)} | – | – | – | – | – | – | – | – | – | {row.get('reason', 'n/a')} |")
            continue
        capm = row.get("capm") or {}
        blend = ((row.get("betaMatchedBlend") or {}).get("simple") or {})
        lines.append(
            f"| {row.get('label', key)} | {row['period']['sessions']} | {_num(capm.get('beta'))} | {_num(capm.get('r2'))} | "
            f"{_pct(capm.get('alphaAnnual'))} | {_num(capm.get('alphaT'))} | {_pct(capm.get('trackingError'))} | "
            f"{_num(capm.get('informationRatio'))} | {_num((blend.get('blend') or {}).get('sharpe'))} | "
            f"{_num((row.get('performance') or {}).get('sharpe'))} | {(row.get('verdict') or {}).get('label', '')} |")
    lines.append("")
    for key in ("backtest.core_tsmom", "walkforward.tsmom", "backtest.combined", "walkforward.combined",
                "backtest.overlay_meanrev", "daytrade.full_defaults"):
        row = attr["series"].get(key)
        if not row or not row.get("available"):
            continue
        verdict_row = row.get("verdict") or {}
        lines += [f"### {row.get('label', key)}", "", verdict_row.get("summary", ""), ""]
        if verdict_row.get("reasons"):
            lines += [f"- {r}" for r in verdict_row["reasons"]] + [""]
        contrib = (row.get("sources") or {}).get("contributions") or {}
        if contrib:
            lines.append(
                f"Annualized return contributions (additive in arithmetic daily means): "
                f"alpha {_pct(contrib.get('alphaAnnual'))}, market {_pct(contrib.get('marketAnnual'))}, "
                f"sector tilts {_pct(contrib.get('sectorTiltsAnnual'))}, bonds {_pct(contrib.get('bondsAnnual'))}, "
                f"other {_pct(contrib.get('otherAnnual'))}.")
            lines.append("")
        br = (row.get("brinson") or {}).get("vsSpy")
        if br and br.get("byGroup"):
            lines += [f"Brinson vs SPY (arithmetic daily sums over {br['sessions']} sessions; "
                      f"compounded portfolio {_pct(br.get('portfolioReturn'))} vs SPY {_pct(br.get('benchmarkReturn'))}):",
                      "",
                      "| Group | Avg w (port) | Avg w (SPY) | Allocation | Selection | Interaction | Total |",
                      "|---|---|---|---|---|---|---|"]
            for g in br["byGroup"][:12]:
                lines.append(
                    f"| {g['group']} | {_pct(g['avgPortfolioWeight'], 1)} | {_pct(g['avgBenchmarkWeight'], 1)} | "
                    f"{_num(g['allocation'], 3)} | {_num(g['selection'], 3)} | {_num(g['interaction'], 3)} | "
                    f"{_num(g['total'], 3)} |")
            lines += ["",
                      f"Sums: allocation {_num(br.get('allocationSum'), 3)}, selection {_num(br.get('selectionSum'), 3)}, "
                      f"interaction {_num(br.get('interactionSum'), 3)}; identity residual "
                      f"{_num(br.get('identityResidualMeanAbs'), 6)} (mean absolute daily).", ""]
        blend = (row.get("betaMatchedBlend") or {}).get("simple")
        if blend:
            sw, bw = blend.get("strategy") or {}, blend.get("blend") or {}
            w = blend.get("weights") or {}
            wtxt = ", ".join(f"{k} {_pct(v, 0)}" for k, v in w.items() if v is not None and abs(v) > 0.005)
            lines.append(
                f"Beta-matched blend ({wtxt}): blend CAGR {_pct(bw.get('cagr'))}, Sharpe {_num(bw.get('sharpe'))}, "
                f"max DD {_pct(bw.get('maxDrawdown'))} vs strategy CAGR {_pct(sw.get('cagr'))}, "
                f"Sharpe {_num(sw.get('sharpe'))}, max DD {_pct(sw.get('maxDrawdown'))}. "
                f"IR vs blend {_num(blend.get('informationRatio'))}, alpha vs blend "
                f"{_pct(blend.get('alphaVsBlendAnnual'))} (t={_num(blend.get('alphaVsBlendT'))}).")
            lines.append("")
    paper = attr.get("paper") or {}
    if not paper.get("available"):
        lines += [f"Paper account: {paper.get('reason', 'no history yet.')}", ""]
    if attr.get("caveats"):
        lines += ["Attribution caveats:", ""] + [f"- {c}" for c in attr["caveats"]] + [""]
    return "\n".join(lines)


def _pct(x, digits=2):
    try:
        if x is None or not math.isfinite(float(x)):
            return "–"
    except (TypeError, ValueError):
        return "–"
    return f"{float(x) * 100:.{digits}f}%"


def _num(x, digits=2):
    try:
        if x is None or not math.isfinite(float(x)):
            return "–"
    except (TypeError, ValueError):
        return "–"
    return f"{float(x):.{digits}f}"


def tool_attribute_returns(args=None):
    """AI analyst entry point. See ANALYST_TOOL for the argument schema."""
    args = args or {}
    out_dir = Path(args.get("out_dir") or OUT_DIR)
    recompute = bool(args.get("recompute"))
    path = out_dir / "attribution.json"
    if recompute or not path.is_file():
        payload = run(out_dir)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    strategy = (args.get("strategy") or "all").strip().lower()
    source = (args.get("source") or "all").strip().lower()
    if strategy in ("", "all") and source in ("", "all"):
        return _finite(payload)
    aliases = {
        "tsmom": ["backtest.core_tsmom", "walkforward.tsmom"],
        "core_tsmom": ["backtest.core_tsmom"],
        "meanrev": ["backtest.overlay_meanrev", "walkforward.meanrev"],
        "overlay": ["backtest.overlay_meanrev"],
        "overlay_meanrev": ["backtest.overlay_meanrev"],
        "combined": ["backtest.combined", "walkforward.combined"],
        "daytrade": ["daytrade.full_defaults"],
        "day_trading": ["daytrade.full_defaults"],
        "paper": ["paper.default"],
    }
    if strategy not in ("", "all"):
        wanted = aliases.get(strategy, [strategy])
        if source == "backtest":
            wanted = [k for k in wanted if k.startswith("backtest.")] or wanted
        elif source == "walkforward":
            wanted = [k for k in wanted if k.startswith("walkforward.")] or wanted
    else:
        wanted = [k for k in (payload.get("series") or {}) if source == "all" or k.startswith(source + ".")]
    series = {k: v for k, v in (payload.get("series") or {}).items() if k in wanted}
    return _finite({
        "generatedAt": payload.get("generatedAt"),
        "methods": payload.get("methods"),
        "data": payload.get("data"),
        "series": series or {
            k: {"available": False, "reason": f"No series matched strategy={strategy!r} source={source!r}."}
            for k in (wanted or ["?"])
        },
        "headlines": {k: v for k, v in (payload.get("headlines") or {}).items() if k in series},
        "paper": payload.get("paper") if strategy in ("all", "paper", "") else None,
        "caveats": payload.get("caveats"),
    })


def update_research_artifacts(attr, out_dir=OUT_DIR):
    """Merge attribution into research_summary.json and refresh research_report.md when present."""
    out_dir = Path(out_dir)
    summary_path = out_dir / "research_summary.json"
    slim = json.loads(json.dumps(_finite(attr), allow_nan=False))
    for row in (slim.get("series") or {}).values():
        rolling = row.get("rollingCapm") or {}
        for window in rolling.values():
            if isinstance(window, dict):
                window.pop("sampledPath", None)
    if not summary_path.is_file():
        write_atomically(out_dir / "research_report_attribution.md", markdown_section(attr).encode())
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["attribution"] = slim
    write_atomically(summary_path, json.dumps(_finite(summary), indent=2, allow_nan=False).encode())
    try:
        from research import markdown
        write_atomically(out_dir / "research_report.md", markdown(summary).encode())
    except Exception:
        existing_path = out_dir / "research_report.md"
        existing = existing_path.read_text(encoding="utf-8") if existing_path.is_file() else ""
        section = markdown_section(attr)
        if "## Return attribution" in existing:
            pre = existing.split("## Return attribution")[0].rstrip()
            rest = existing.split("## Return attribution", 1)[1]
            caveats = ""
            if "\n## Caveats" in rest:
                caveats = "\n## Caveats" + rest.split("\n## Caveats", 1)[1]
            write_atomically(existing_path, (pre + "\n\n" + section + caveats).encode())
        elif "## Caveats" in existing:
            pre, caveats = existing.split("## Caveats", 1)
            write_atomically(existing_path, (pre + section + "\n## Caveats" + caveats).encode())
        else:
            write_atomically(existing_path, (existing.rstrip() + "\n\n" + section).encode())



# ------------------------------------------------------------------ hook adapters (research.py, paper_trader, daytrader, UI panel)

PANEL_NAMES = {
    "backtest.core_tsmom": "core_tsmom",
    "backtest.overlay_meanrev": "overlay_meanrev",
    "backtest.combined": "combined",
    "backtest.benchmark_spy": "benchmark_spy",
    "backtest.benchmark_equal_weight_etfs": "benchmark_equal_weight_etfs",
    "walkforward.tsmom": "walkforward_tsmom_oos",
    "walkforward.meanrev": "walkforward_meanrev_oos",
    "walkforward.combined": "walkforward_combined_oos",
    "paper.default": "paper_account",
    "daytrade.full_defaults": "daytrader",
}

PANEL_VERDICT = {
    "significant_alpha": "has_skill",
    "significant_negative_alpha": "underperforms_beta_mix",
    "mostly_market": "cheap_index_fund",
    "inconclusive": "inconclusive",
    "marginal": "marginal",
    "unavailable": "unavailable",
}


def factor_returns_from_panel(panel):
    """Slice factor ETF total returns out of a loaded price panel (used by research.py)."""
    rets = panel.total_return()
    cols = [c for c in ALL_FACTORS if c in rets.columns]
    if MARKET_FACTOR not in cols and MARKET_FACTOR in rets.columns:
        cols = [MARKET_FACTOR] + cols
    return rets[cols] if cols else rets.iloc[:, 0:0]


def _to_panel_report(row):
    """Convert attribute_series() output into the schema public/attribution-panel.js expects."""
    if not row:
        return {"skipped": "No attribution result."}
    if not row.get("available"):
        return {"name": row.get("name"), "skipped": row.get("reason") or "unavailable"}
    capm = row.get("capm") or {}
    full = (row.get("sources") or {}).get("full") or row.get("spyBonds") or {}
    contrib = (row.get("sources") or {}).get("contributions") or {}
    loadings = []
    for ticker, beta in (full.get("betas") or {}).items():
        if ticker == MARKET_FACTOR:
            continue
        loadings.append({"ticker": ticker, "loading": beta, "tStat": (full.get("betaT") or {}).get(ticker)})
    loadings.sort(key=lambda x: abs(x["loading"] or 0.0), reverse=True)
    blend = row.get("betaMatchedBlend") or {}
    simple = blend.get("simple") or {}
    # Frictionless beta*SPY + cash (rest uninvested) for the panel's cashMix card.
    cash_mix = None
    if capm.get("beta") is not None and simple.get("strategy"):
        w = min(max(float(capm["beta"]), 0.0), 1.0)
        cash_mix = {"weights": {"SPY": w, "cash": 1.0 - w},
                    "sharpe": (simple.get("blend") or {}).get("sharpe"),
                    "cagr": (simple.get("blend") or {}).get("cagr"),
                    "maxDrawdown": (simple.get("blend") or {}).get("maxDrawdown"),
                    "note": "Panel card uses the SPY+IEF blend Sharpe/CAGR when IEF is available (see bondMix)."}
    bond_mix = None
    if simple:
        bond_mix = {**simple.get("blend", {}), "weights": simple.get("weights"),
                    "trackingError": simple.get("trackingError"),
                    "informationRatio": simple.get("informationRatio"),
                    "alphaAnnual": simple.get("alphaVsBlendAnnual"),
                    "alphaT": simple.get("alphaVsBlendT")}
        # Prefer reporting the actual SPY+IEF blend as cashMix numbers so the UI is not empty;
        # the weights document that the remainder is IEF, not cash, when IEF is held.
        if simple.get("blend"):
            cash_mix = {**simple["blend"], "weights": simple.get("weights")}
    br = (row.get("brinson") or {}).get("vsSpy") or {}
    n = br.get("sessions") or (row.get("period") or {}).get("sessions") or 0
    scale = (TRADING_DAYS / n) if n else None
    brinson_ann = None
    if scale and br:
        brinson_ann = {
            "allocation": None if br.get("allocationSum") is None else br["allocationSum"] * scale,
            "selection": None if br.get("selectionSum") is None else br["selectionSum"] * scale,
            "interaction": None if br.get("interactionSum") is None else br["interactionSum"] * scale,
        }
    verdict_row = row.get("verdict") or {}
    return {
        "name": PANEL_NAMES.get(row.get("name"), row.get("name")),
        "period": row.get("period"),
        "performance": row.get("performance"),
        "market": {
            "spyBeta": capm.get("beta"),
            "betaT": capm.get("betaT"),
            "r2": capm.get("r2"),
            "alphaAnnual": capm.get("alphaAnnual"),
            "alphaT": capm.get("alphaT"),
            "alphaP": capm.get("alphaP"),
            "trackingError": capm.get("trackingError"),
            "informationRatio": capm.get("informationRatio"),
            "appraisalRatio": capm.get("appraisalRatio"),
            "n": capm.get("n"),
            "hacLags": capm.get("hacLags"),
        },
        "sector": {
            "alphaAnnual": full.get("alphaAnnual"),
            "alphaT": full.get("alphaT"),
            "alphaP": full.get("alphaP"),
            "r2": full.get("r2"),
            "n": full.get("n"),
        },
        "sectorLoadings": loadings[:12],
        "contributions": contrib,
        "betaMatched": {
            "strategy": simple.get("strategy") or row.get("performance"),
            "cashMix": cash_mix,
            "bondMix": bond_mix,
            "replicating": (blend.get("replicating") or {}).get("blend") if blend.get("replicating") else None,
        },
        "brinson": {"annualized": brinson_ann, "vsSpy": {k: v for k, v in br.items() if k != "byGroup"},
                    "byGroup": br.get("byGroup")},
        "rollingCapm": row.get("rollingCapm"),
        "verdict": {
            "label": PANEL_VERDICT.get(verdict_row.get("label"), verdict_row.get("label")),
            "text": verdict_row.get("summary"),
            "reasons": verdict_row.get("reasons") or [],
            "isCheaperIndexFund": verdict_row.get("isCheaperIndexFund"),
        },
    }


def _lag_holdings(holdings):
    """Beginning-of-period weights: yesterday's close weights applied to today's return."""
    if holdings is None or getattr(holdings, "empty", True):
        return None
    return holdings.shift(1).fillna(0.0)


def _attribute_sim(name, label, result, factor_returns, asset_returns=None):
    rets = result.returns.dropna() if result is not None and result.returns is not None else None
    if rets is None or len(rets) < MIN_OBS:
        return {"name": name, "label": label, "available": False,
                "reason": f"Only {0 if rets is None else len(rets)} sessions (need {MIN_OBS})."}
    weights = _lag_holdings(getattr(result, "holdings", None))
    held = asset_returns
    groups = assign_groups(list(weights.columns), factor_returns, held) if weights is not None else None
    return attribute_series(name, label, rets, factor_returns, weights, held, groups,
                            include_brinson=weights is not None and not weights.empty)


def attach_research_attribution(summary, factor_returns, out_dir, sim_results=None, walkforward_oos=None,
                                asset_returns=None):
    """Fill summary['attribution'] during research.py all/backtest/walkforward. Writes attribution.json."""
    out_dir = Path(out_dir or OUT_DIR)
    labels = {
        "core_tsmom": "TSMOM core (full-period backtest)",
        "overlay_meanrev": "Mean-reversion overlay (full-period backtest)",
        "combined": "Combined TSMOM + overlay (full-period backtest)",
        "benchmark_spy": "Buy-and-hold SPY (sanity check)",
        "benchmark_equal_weight_etfs": "Equal-weight core ETFs (benchmark)",
    }
    series = {}
    sim_results = sim_results or {}
    for name, result in sim_results.items():
        key = f"backtest.{name}"
        series[key] = _attribute_sim(key, labels.get(name, name), result, factor_returns, asset_returns)
    wf_labels = {
        "tsmom": "TSMOM (walk-forward out-of-sample)",
        "meanrev": "Mean-reversion (walk-forward out-of-sample)",
        "combined": "Combined (walk-forward out-of-sample)",
    }
    for family, result in (walkforward_oos or {}).items():
        key = f"walkforward.{family}"
        series[key] = _attribute_sim(key, wf_labels.get(family, family), result, factor_returns, asset_returns)

    # Fill any gaps from saved CSVs (e.g. walk-forward was not re-run this invocation).
    if not any(k.startswith("backtest.") for k in series):
        for name, rets in load_equity_curves(out_dir).items():
            key = f"backtest.{name}"
            weights = asset = groups = None
            if name in ("core_tsmom", "overlay_meanrev", "combined"):
                weights, asset, groups = _prep_holdings(out_dir, name, rets.index, factor_returns, asset_returns)
            series[key] = attribute_series(key, labels.get(name, name), rets, factor_returns, weights, asset, groups,
                                           include_brinson=name in ("core_tsmom", "overlay_meanrev", "combined"))
    if not any(k.startswith("walkforward.") for k in series):
        for family, rets in load_walkforward_returns(out_dir).items():
            key = f"walkforward.{family}"
            weights, asset, groups = _prep_holdings(out_dir, f"wf_{family}", rets.index, factor_returns, asset_returns)
            series[key] = attribute_series(key, wf_labels.get(family, family), rets, factor_returns,
                                           weights, asset, groups)

    paper_block, daytrade_block = _optional_live_series(series, factor_returns, out_dir)
    payload = _package(series, factor_returns, paper_block, daytrade_block, started=None)
    write_json(out_dir / "attribution.json", payload)
    summary["attribution"] = _finite(payload)
    return payload


def _optional_live_series(series, factor_returns, out_dir):
    paper, err = load_paper_returns()
    if err:
        paper_block = {"available": False, "reason": err}
    elif not paper or not paper.get("available"):
        paper_block = paper or {"available": False, "reason": "No paper equity history yet."}
    elif len(paper["returns"]) < MIN_OBS:
        paper_block = {"available": False,
                       "reason": f"Paper history has only {len(paper['returns'])} sessions (need {MIN_OBS}).",
                       "sessions": len(paper["returns"])}
    else:
        series["paper.default"] = attribute_series(
            "paper.default", "Paper account (default)", paper["returns"], factor_returns, paper.get("weights"))
        paper_block = {"available": True, "account": paper.get("account"),
                       "start": paper.get("start"), "end": paper.get("end"), "sessions": paper.get("sessions")}
    day, err = load_daytrade_returns(out_dir)
    if err:
        daytrade_block = {"available": False, "reason": err}
    else:
        series["daytrade.full_defaults"] = attribute_series(
            "daytrade.full_defaults", "Day trading ORB (full-period defaults)", day["returns"],
            factor_returns, include_brinson=False)
        daytrade_block = {"available": True, "sessions": day["sessions"],
                          "daysWithTrades": day["daysWithTrades"],
                          "reconstructedTotalReturn": day["reconstructedTotalReturn"],
                          "savedTotalReturn": day["savedTotalReturn"], "note": day.get("note")}
    return paper_block, daytrade_block


def _package(series, factor_returns, paper_block, daytrade_block, started):
    headlines = {k: (v.get("verdict") or {}).get("summary") for k, v in series.items() if v.get("available")}
    strategies = {}
    for key, row in series.items():
        strategies[PANEL_NAMES.get(key, key)] = _to_panel_report(row)
    payload = {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data": {
            "factorStart": str(factor_returns.index[0].date()) if len(factor_returns) else None,
            "factorEnd": str(factor_returns.index[-1].date()) if len(factor_returns) else None,
            "factorSessions": int(len(factor_returns)),
            "factors": [c for c in ALL_FACTORS if c in factor_returns.columns],
            "sampleNote": SAMPLE_NOTE,
        },
        "methods": {
            "returns": "Close-to-close total returns including dividends; strategy returns from equity curves or SimResult.returns.",
            "hacLags": "Newey-West 4*(n/100)^(2/9), same rule as engine/quant_tools.py.",
            "rollingWindows": list(ROLLING_WINDOWS),
            "sourcesModel": "SPY + IEF/TLT raw + sector/other ETFs residualized against SPY, so sector betas are incremental tilts.",
            "brinson": "Brinson-Fachler on lagged realized holdings when a SimResult is available, otherwise forward-filled target weights from fill dates.",
            "betaMatchedBlend": "w in SPY equal to clipped CAPM beta, remainder in IEF; plus a clipped SPY+IEF+TLT replicating mix.",
            "riskFreeRate": 0,
        },
        "series": series,
        "strategies": strategies,
        "headlines": headlines,
        "paper": paper_block,
        "daytrade": daytrade_block,
        "caveats": _data_caveats([
            SAMPLE_NOTE,
            "Full-period backtests include the in-sample window used to set defaults; walk-forward OOS is the more honest test of residual skill.",
            "Sector assignment for individual stocks uses full-sample correlation with sector ETFs (look-ahead on the label, not on the returns).",
            "Brinson effects are arithmetic daily sums and will not exactly equal the compounded excess return; the identity residual is reported.",
            "The day-trading series is rebuilt from saved trades and a 0-return fill on sessions with no trade unless a daily frame is provided; holdings-based attribution is skipped because the book is flat overnight.",
            "No risk-free rate is subtracted (consistent with the rest of the research book).",
        ]),
        "runSeconds": None if started is None else round(time.time() - started, 1),
    }
    return _strip_models(payload)


def attribute_paper_account(connection=None, account="default"):
    """Single-strategy report for paper_trader.py (panel schema). connection may be None on a dry run."""
    if connection is not None:
        try:
            history = pd.read_sql_query(
                "SELECT date, equity, daily_return FROM equity_history WHERE account = ? ORDER BY date",
                connection, params=(account,))
        except Exception as error:
            return {"name": "paper_account", "skipped": f"Could not read equity_history ({error})."}
        if history is None or history.empty:
            return {"name": "paper_account", "skipped": "No paper equity history yet. Run the swing paper trader and save a session."}
        history["date"] = pd.to_datetime(history["date"])
        history = history.set_index("date").sort_index()
        returns = history["daily_return"].astype(float) if history["daily_return"].notna().sum() >= MIN_OBS else returns_from_equity(history["equity"])
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) < MIN_OBS:
            return {"name": "paper_account", "skipped": f"Paper history has only {len(returns)} sessions (need {MIN_OBS})."}
        factors = load_factor_returns()
        row = attribute_series("paper.default", "Paper account (default)", returns, factors, include_brinson=False)
        return _to_panel_report(row)
    paper, err = load_paper_returns(account=account)
    if err:
        return {"name": "paper_account", "skipped": err}
    if not paper or not paper.get("available"):
        return {"name": "paper_account", "skipped": (paper or {}).get("reason") or "No paper equity history yet."}
    if len(paper["returns"]) < MIN_OBS:
        return {"name": "paper_account",
                "skipped": f"Paper history has only {len(paper['returns'])} sessions (need {MIN_OBS})."}
    factors = load_factor_returns()
    row = attribute_series("paper.default", "Paper account (default)", paper["returns"], factors,
                           paper.get("weights"), include_brinson=paper.get("weights") is not None)
    return _to_panel_report(row)


def attribute_daytrade_result(daily, name="daytrader"):
    """Single-strategy report for daytrader.py research/status hooks. `daily` has session_date and return."""
    if daily is None:
        return {"name": name, "skipped": "No day-trading daily frame."}
    frame = daily if isinstance(daily, pd.DataFrame) else pd.DataFrame(daily)
    if frame.empty:
        return {"name": name, "skipped": "No day-trading daily frame."}
    if "session_date" in frame.columns:
        frame = frame.copy()
        frame["session_date"] = pd.to_datetime(frame["session_date"])
        frame = frame.set_index("session_date")
    frame.index = pd.to_datetime(frame.index)
    if "return" in frame.columns:
        returns = frame["return"].astype(float)
    elif "equity" in frame.columns:
        returns = returns_from_equity(frame["equity"])
    else:
        return {"name": name, "skipped": "Day-trading daily frame has neither return nor equity."}
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < MIN_OBS:
        return {"name": name, "skipped": f"Day-trading history has only {len(returns)} sessions (need {MIN_OBS})."}
    factors = load_factor_returns()
    label = "Day trading ORB (walk-forward OOS)" if "oos" in str(name) else (
        "Day trading ORB (paper)" if "paper" in str(name) else "Day trading ORB (full-period defaults)")
    row = attribute_series(f"daytrade.{name}", label, returns, factors, include_brinson=False)
    report = _to_panel_report(row)
    report["name"] = name
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--holdings", action="store_true",
                        help="Reconstruct or use holdings for Brinson (always on when holdings can be built; accepted for the research.py hook).")
    parser.add_argument("--skip-paper", action="store_true")
    parser.add_argument("--skip-daytrade", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args.out, include_paper=not args.skip_paper, include_daytrade=not args.skip_daytrade)
    # Ensure the panel schema is present even when run() built the payload itself.
    if "strategies" not in payload:
        payload["strategies"] = {PANEL_NAMES.get(k, k): _to_panel_report(v) for k, v in (payload.get("series") or {}).items()}
        write_json(Path(args.out) / "attribution.json", _strip_models(payload))
    update_research_artifacts(payload, args.out)
    for key, line in (payload.get("headlines") or {}).items():
        print(f"{key}: {line}")
    return payload


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("attribution"):
        main()
