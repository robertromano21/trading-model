"""Duration- and credit-aware bond-ETF sleeve. Config-gated; uses only causal information.

Allocates across the liquid unlevered bond ETFs in the local price database (SHY, IEI, IEF, TLT,
TIP, LQD, HYG, JNK, MUB, BND, AGG, EMB, BIL, SGOV). Duration stance comes from the Treasury curve
(level change and slope) when macro data is available, otherwise from SHY vs TLT relative trend.
Credit stance comes from ICE BofA OAS proxies (and HYG/IEF) when available, otherwise from HYG vs IEF.

Real CDS/CDX and swap quotes are not used; see macro_data.PROXY_DISCLAIMER. This sleeve never places
broker orders — it only returns target weights for the shared backtest / paper trader.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from risk import apply_caps, blended_vol, hold_between_rebalances, keep_top, portfolio_vol_target

from .base import Strategy

# Approximate average duration (years) and credit bucket. Static metadata, not estimated in-sample.
BOND_ETFS = {
    "BIL":  {"duration": 0.20, "credit": "gov",  "bucket": "ultra_short"},
    "SGOV": {"duration": 0.15, "credit": "gov",  "bucket": "ultra_short"},
    "SHY":  {"duration": 1.90, "credit": "gov",  "bucket": "short"},
    "IEI":  {"duration": 4.50, "credit": "gov",  "bucket": "intermediate"},
    "IEF":  {"duration": 7.40, "credit": "gov",  "bucket": "intermediate"},
    "TLT":  {"duration": 16.5, "credit": "gov",  "bucket": "long"},
    "TIP":  {"duration": 6.80, "credit": "tips", "bucket": "tips"},
    "LQD":  {"duration": 8.20, "credit": "ig",   "bucket": "ig"},
    "HYG":  {"duration": 3.70, "credit": "hy",   "bucket": "hy"},
    "JNK":  {"duration": 3.60, "credit": "hy",   "bucket": "hy"},
    "MUB":  {"duration": 6.00, "credit": "muni", "bucket": "muni"},
    "BND":  {"duration": 6.20, "credit": "agg",  "bucket": "agg"},
    "AGG":  {"duration": 6.00, "credit": "agg",  "bucket": "agg"},
    "EMB":  {"duration": 7.00, "credit": "em",   "bucket": "em"},
}

# Duration / credit policy mix: remaining weight is cash.
# Keys are (duration_stance, credit_stance) -> bucket weights (not already normalized to 1).
POLICY = {
    ("short", "risk_off"): {"ultra_short": 0.70, "short": 0.20, "agg": 0.00, "ig": 0.00, "hy": 0.00, "tips": 0.00, "em": 0.00, "muni": 0.00, "intermediate": 0.00, "long": 0.00},
    ("short", "neutral"):  {"ultra_short": 0.40, "short": 0.35, "intermediate": 0.10, "agg": 0.05, "ig": 0.00, "hy": 0.00, "tips": 0.05, "muni": 0.00, "long": 0.00, "em": 0.00},
    ("short", "risk_on"):  {"ultra_short": 0.25, "short": 0.30, "intermediate": 0.10, "ig": 0.15, "hy": 0.10, "agg": 0.05, "tips": 0.05, "muni": 0.00, "long": 0.00, "em": 0.00},
    ("mid", "risk_off"):   {"ultra_short": 0.20, "short": 0.25, "intermediate": 0.35, "agg": 0.10, "tips": 0.05, "ig": 0.00, "hy": 0.00, "long": 0.00, "muni": 0.00, "em": 0.00},
    ("mid", "neutral"):    {"short": 0.15, "intermediate": 0.35, "long": 0.10, "agg": 0.15, "ig": 0.10, "hy": 0.00, "tips": 0.10, "muni": 0.05, "ultra_short": 0.00, "em": 0.00},
    ("mid", "risk_on"):    {"short": 0.10, "intermediate": 0.25, "long": 0.10, "ig": 0.20, "hy": 0.15, "agg": 0.10, "tips": 0.05, "em": 0.05, "muni": 0.00, "ultra_short": 0.00},
    ("long", "risk_off"):  {"intermediate": 0.30, "long": 0.35, "short": 0.15, "tips": 0.10, "agg": 0.10, "ig": 0.00, "hy": 0.00, "ultra_short": 0.00, "muni": 0.00, "em": 0.00},
    ("long", "neutral"):   {"intermediate": 0.25, "long": 0.30, "ig": 0.10, "agg": 0.10, "tips": 0.15, "short": 0.10, "hy": 0.00, "muni": 0.00, "ultra_short": 0.00, "em": 0.00},
    ("long", "risk_on"):   {"intermediate": 0.15, "long": 0.25, "ig": 0.20, "hy": 0.15, "tips": 0.10, "em": 0.10, "agg": 0.05, "short": 0.00, "muni": 0.00, "ultra_short": 0.00},
}

PREFERRED = {
    "ultra_short": ("SGOV", "BIL"),
    "short": ("SHY",),
    "intermediate": ("IEF", "IEI"),
    "long": ("TLT",),
    "tips": ("TIP",),
    "ig": ("LQD",),
    "hy": ("HYG", "JNK"),
    "muni": ("MUB",),
    "agg": ("AGG", "BND"),
    "em": ("EMB",),
}


class FixedIncomeStrategy(Strategy):
    name = "bonds"
    defaults = {
        "target_vol": 0.06, "max_positions": 6, "per_name_cap": 0.35, "gross_cap": 1.0,
        "max_leverage": 1.0, "vol_window": 60, "rebalance": "W-FRI",
        "slope_steep": 0.25,          # 10Y-2Y (percentage points) to take a long-duration stance
        "slope_flat": 0.00,           # at or below this (incl. inverted) -> short duration
        "rates_up_bp": 0.25,          # 20-session 10Y rise (pp) that forces short duration
        "hy_oas_risk_off_pct": 0.70,  # trailing percentile of HY OAS to cut credit
        "hy_oas_risk_on_pct": 0.40,
        "spread_widen_bp": 0.20,      # 20-session HY OAS widening (pp) that forces risk-off credit
        "allow_hy": True,
        "allow_em": True,
        "min_history": 40,
    }

    def universe(self, panel):
        names = [t for t in BOND_ETFS if t in panel.close.columns]
        mask = pd.DataFrame(False, index=panel.dates, columns=panel.close.columns)
        if names:
            mask[names] = panel.close[names].notna()
        return mask

    def _macro_features(self, panel):
        """Causal duration and credit features, lagged when they come from FRED."""
        features = pd.DataFrame(index=panel.dates)
        # ETF fallbacks (same-session close is ok: used for *next* day's target via the usual lag in simulate).
        close = panel.close
        if "TLT" in close.columns and "SHY" in close.columns:
            rel = np.log(close["TLT"] / close["SHY"])
            features["tlt_shy_ret20"] = rel.diff(20)
        if "HYG" in close.columns and "IEF" in close.columns:
            features["hyg_ief_ret20"] = np.log(close["HYG"] / close["IEF"]).diff(20)
        if "LQD" in close.columns and "IEF" in close.columns:
            features["lqd_ief_ret20"] = np.log(close["LQD"] / close["IEF"]).diff(20)
        try:
            from macro_data import load_macro
            macro = load_macro(lag=1)
        except Exception:
            macro = pd.DataFrame()
        if macro is not None and not macro.empty:
            macro = macro.reindex(panel.dates).ffill()
            for col in ("curve_slope", "curve_level_chg20", "curve_slope_chg20", "hy_oas", "hy_oas_chg20",
                        "ig_oas", "hy_ig_oas", "sofr_tbill_spread"):
                if col in macro.columns:
                    features[col] = macro[col]
            if "hy_oas" in features.columns:
                features["hy_oas_pct"] = features["hy_oas"].rolling(252, min_periods=60).apply(
                    lambda w: float((w <= w[-1]).mean()), raw=True)
        return features

    def _stances(self, features):
        p = self.params
        n = len(features)
        duration = pd.Series("mid", index=features.index, dtype=object)
        credit = pd.Series("neutral", index=features.index, dtype=object)

        slope = features["curve_slope"] if "curve_slope" in features.columns else pd.Series(np.nan, index=features.index)
        rates_up = features["curve_level_chg20"] if "curve_level_chg20" in features.columns else pd.Series(np.nan, index=features.index)
        # ETF fallback for duration: TLT outperforming SHY over 20d ~ falling rates / long-duration bid.
        if rates_up.isna().all() and "tlt_shy_ret20" in features.columns:
            # A 20d TLT/SHY log-return > 0 is a long-duration environment; very negative is short.
            proxy = features["tlt_shy_ret20"]
            duration = pd.Series(np.where(proxy > 0.02, "long", np.where(proxy < -0.02, "short", "mid")),
                                 index=features.index, dtype=object)
        else:
            short = (slope <= p["slope_flat"]) | (rates_up >= p["rates_up_bp"])
            long = (slope >= p["slope_steep"]) & (rates_up.fillna(0.0) < p["rates_up_bp"])
            duration = pd.Series(np.where(short.fillna(False), "short",
                                          np.where(long.fillna(False), "long", "mid")),
                                 index=features.index, dtype=object)

        oas_pct = features["hy_oas_pct"] if "hy_oas_pct" in features.columns else pd.Series(np.nan, index=features.index)
        widen = features["hy_oas_chg20"] if "hy_oas_chg20" in features.columns else pd.Series(np.nan, index=features.index)
        if oas_pct.isna().all() and "hyg_ief_ret20" in features.columns:
            proxy = features["hyg_ief_ret20"]
            credit = pd.Series(np.where(proxy < -0.02, "risk_off",
                                        np.where(proxy > 0.02, "risk_on", "neutral")),
                               index=features.index, dtype=object)
        else:
            off = (oas_pct >= p["hy_oas_risk_off_pct"]) | (widen >= p["spread_widen_bp"])
            on = (oas_pct <= p["hy_oas_risk_on_pct"]) & (widen.fillna(0.0) < p["spread_widen_bp"])
            credit = pd.Series(np.where(off.fillna(False), "risk_off",
                                        np.where(on.fillna(False), "risk_on", "neutral")),
                               index=features.index, dtype=object)
        if n:
            # Need a minimum history before taking an active stance.
            min_h = int(p["min_history"])
            if min_h > 0:
                duration.iloc[:min_h] = "mid"
                credit.iloc[:min_h] = "neutral"
        return duration, credit

    def signals(self, panel):
        mask = self.universe(panel)
        features = self._macro_features(panel)
        duration, credit = self._stances(features)
        raw = pd.DataFrame(0.0, index=panel.dates, columns=panel.close.columns)
        for day in panel.dates:
            mix = POLICY.get((duration.at[day], credit.at[day]), POLICY[("mid", "neutral")])
            for bucket, weight in mix.items():
                if weight <= 0:
                    continue
                if bucket == "hy" and not self.params["allow_hy"]:
                    continue
                if bucket == "em" and not self.params["allow_em"]:
                    continue
                names = [t for t in PREFERRED.get(bucket, ()) if t in mask.columns and bool(mask.at[day, t])]
                if not names:
                    continue
                each = weight / len(names)
                for ticker in names:
                    raw.at[day, ticker] = each
        self._duration_stance = duration
        self._credit_stance = credit
        return raw

    def target_weights(self, panel):
        p = self.params
        raw = self.signals(panel)
        cols = [c for c in raw.columns if raw[c].abs().sum() > 0]
        if not cols:
            return pd.DataFrame(0.0, index=panel.dates, columns=panel.close.columns)
        returns = panel.total_return()[cols]
        vol = blended_vol(returns, p["vol_window"])
        # Keep the policy mix (already a portfolio), scale to a vol target, cap names.
        scaled = portfolio_vol_target(raw[cols], returns, p["target_vol"], p["max_leverage"], window=p["vol_window"])
        scaled = keep_top(scaled, scaled.abs(), p["max_positions"])
        weights = apply_caps(scaled, p["per_name_cap"], min(p["gross_cap"], p["max_leverage"]))
        weights = hold_between_rebalances(weights, p["rebalance"])
        last_d = getattr(self, "_duration_stance", pd.Series(dtype=object))
        last_c = getattr(self, "_credit_stance", pd.Series(dtype=object))
        self.last_diagnostics = {
            "durationStance": None if last_d.empty else str(last_d.iloc[-1]),
            "creditStance": None if last_c.empty else str(last_c.iloc[-1]),
            "activeNames": int((weights.iloc[-1] > 0).sum()) if len(weights) else 0,
            "gross": float(weights.iloc[-1].abs().sum()) if len(weights) else 0.0,
        }
        return weights.reindex(columns=panel.close.columns, fill_value=0.0)
