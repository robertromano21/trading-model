"""MACRO tool operations for the AI analyst (fixed menu, no user-written code)."""

from __future__ import annotations

from macro_data import PROXY_DISCLAIMER, load_dxy, load_macro, snapshot, uup_crosscheck
from macro_research import HORIZONS, hac_ols, lead_lag, walk_forward_predict, forward_return


def op_macro(args):
    """Snapshot of dollar/rates/credit proxies, or a HAC/lead-lag/predictive test.

    args.method: snapshot (default) | betas | leadlag | predict | outlook
    args.tickers: optional, used by betas/leadlag/predict (default SPY, GLD, XLE, EEM, EFA)
    args.feature: dxy_ret (default) | dxy_ret20 | curve_slope | hy_oas_chg20 | sofr_tbill_spread
    args.horizon: 1, 5 or 20 for predict
    """
    args = args or {}
    method = str(args.get("method") or "snapshot").lower()
    if method in ("default", "status"):
        method = "snapshot"
    if method == "snapshot":
        snap = snapshot()
        snap["proxyDisclaimer"] = PROXY_DISCLAIMER
        snap["uup"] = snap.get("uup") or uup_crosscheck()
        try:
            from dxy_forecast import load_outlook
            outlook = load_outlook()
            if outlook:
                snap["dxyOutlook"] = outlook
        except Exception:
            pass
        return snap

    if method in ("outlook", "dxy_outlook"):
        from dxy_forecast import load_outlook
        data = load_outlook() or {"available": False, "reason": "No dollar outlook yet. Run python dxy_forecast.py research."}
        data["proxyDisclaimer"] = PROXY_DISCLAIMER
        data["tradingDefault"] = "off"
        return data
    tickers = args.get("tickers") or ["SPY", "GLD", "XLE", "DBC", "EEM", "EFA"]
    if isinstance(tickers, str):
        tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        tickers = [str(t).upper() for t in tickers]
    feature = str(args.get("feature") or ("dxy_ret20" if method == "predict" else "dxy_ret"))
    horizon = int(args.get("horizon") or 5)
    if horizon not in HORIZONS:
        horizon = 5

    from market_data import load_panel
    panel = load_panel(tickers=list(set(tickers) | {"SPY"}))
    rets = panel.total_return()
    macro = load_macro(lag=1)
    if feature == "dxy_ret":
        x = load_dxy(lag=0).pct_change().shift(1)
    elif macro is not None and feature in macro.columns:
        x = macro[feature]
    else:
        return {"error": f"unknown or unavailable feature {feature!r}", "proxyDisclaimer": PROXY_DISCLAIMER}

    out = {"method": method, "feature": feature, "proxyDisclaimer": PROXY_DISCLAIMER, "results": []}
    if method == "betas":
        for ticker in tickers:
            if ticker not in rets.columns:
                continue
            out["results"].append({"ticker": ticker, **(hac_ols(rets[ticker], x) or {"error": "not enough overlap"})})
        return out
    if method == "leadlag":
        for ticker in tickers:
            if ticker not in rets.columns:
                continue
            out["results"].append({"ticker": ticker, "leadLag": lead_lag(rets[ticker], x)})
        return out
    if method in ("predict", "forecast"):
        out["horizon"] = horizon
        for ticker in tickers:
            if ticker not in rets.columns:
                continue
            fwd = forward_return(rets[ticker], horizon)
            wf = walk_forward_predict(fwd, x, label_gap=horizon)
            out["results"].append({"ticker": ticker, "predict": wf})
        return out
    raise ValueError(f"unknown MACRO method {method!r}; one of snapshot, betas, leadlag, predict, outlook")
