"""Market-top / adverse-regime detector, scored honestly out of sample.

Signals (all causal: each date uses only rows dated <= that date)
  hmm_risk      HMM filtered P(risk_off) + P(stress) + 0.5 * P(chop) >= 0.5   (regime_hmm.regime_policy, walk-forward fits)
  breadth_div   SPY within 3% of its 252-day high while % of names above their 50-day SMA fell >= 15 points in 20 days
  trend_break   SPY closed below its 50-day SMA within the last 5 sessions after >= 20 sessions above it, or 50-day trend t < 0
  credit_widen  HYG minus IEF 20-day total return <= -1% (credit spreads widening proxy)
  dollar_strong dollar 20-day return >= +2%: GB-12 macro dollar index if engine/macro_features.py provides
                ``dollar_index(panel)`` (a date-indexed level series), else the UUP ETF as a proxy
  overextended  SPY (close - SMA50) / ATR14 >= 3 or RSI14 >= 75
  composite     count of the above >= k (k chosen on training data only in the walk-forward)

Targets (forward-looking, used only for scoring): fwd20_down (SPY 20-day forward return < 0), dd20_5 (max drawdown
from today's close within 20 sessions <= -5%), dd60_8 (within 60 sessions <= -8%).

Scoring per signal x target: hit rate when ON vs the unconditional base rate, lift, mean forward return ON vs OFF,
Newey-West (HAC, lag = horizon) t-stat of the forward return on the indicator, a circular-shift permutation p-value
(preserves the autocorrelation of both series; 2000 shifts), number of independent episodes (ON spells separated by
>= horizon sessions), and Holm + Benjamini-Hochberg corrections across all signal x target tests. The composite is
also walk-forward: k is picked on an expanding training window by balanced accuracy and applied to the next 63
sessions; out-of-sample Brier score of the implied probability is compared with climatology (the training base rate).

Honesty: the database holds ~2 years (one mostly-bull regime). Overlapping 20/60-day windows mean only a handful of
independent adverse episodes, so p-values are weak by construction. Report results, not hopes.

  python top_detector.py            # writes Massive/Research/top_detector.json and prints the summary
"""

import json
import math
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically
from risk import atr

OUT_PATH = DATA_DIR / "Research" / "top_detector.json"
SIGNALS = ("hmm_risk", "breadth_div", "trend_break", "credit_widen", "dollar_strong", "overextended")
TARGETS = {"fwd20_down": 20, "dd20_5": 20, "dd60_8": 60}
WARMUP = 120
MIN_EPISODES = 8          # independent ON episodes needed before a surviving test is called robust


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _dollar_series(panel):
    try:
        import macro_features  # GB-12 (optional): dollar_index(panel) -> pd.Series of index levels
        s = macro_features.dollar_index(panel)
        if s is not None and len(s.dropna()) > 60:
            return s.reindex(panel.dates).ffill(), "macro_features.dollar_index (GB-12)"
    except Exception:  # noqa: BLE001 - optional module
        pass
    if "UUP" in panel.close.columns:
        return panel.close["UUP"], "UUP ETF proxy"
    return None, None


def build_signals(panel, policy=None, hmm_overrides=None):
    """Date-indexed DataFrame of 0/1 signals plus the raw features behind them. policy: optional precomputed
    regime_hmm.regime_policy(panel, 'hmm') (expensive; pass it in to reuse)."""
    from regime_hmm import build_features, regime_policy
    spy = panel.close["SPY"]
    out = pd.DataFrame(index=panel.dates)
    if policy is None:
        policy = regime_policy(panel, "hmm", hmm_overrides)
    p = lambda name: policy.get(f"p_{name}", pd.Series(0.0, index=panel.dates)).fillna(0.0)
    out["hmm_prob"] = p("risk_off") + p("stress") + 0.5 * p("chop")
    has_fit = policy.get("source", pd.Series("", index=panel.dates)).eq("hmm")
    out["hmm_risk"] = ((out["hmm_prob"] >= 0.5) & has_fit).astype(float)
    feats = build_features(panel)
    breadth = feats.get("pct_above_50")
    high252 = spy.rolling(252, min_periods=120).max()
    near_high = spy >= 0.97 * high252
    if breadth is not None:
        out["breadth_drop20"] = breadth - breadth.shift(20)
        out["breadth_div"] = (near_high & (out["breadth_drop20"] <= -0.15)).astype(float)
    else:
        out["breadth_div"] = 0.0
    sma50 = spy.rolling(50, min_periods=50).mean()
    above = spy > sma50
    was_above = above.shift(1).rolling(20, min_periods=20).sum() >= 20
    cross_down = (~above) & was_above
    out["trend_break"] = ((cross_down.astype(float).rolling(5, min_periods=1).max() > 0)
                          | (feats.get("spy_trend_t", pd.Series(np.nan, index=panel.dates)) < 0)).astype(float)
    credit = feats.get("credit_excess20")
    out["credit_excess20"] = credit
    out["credit_widen"] = (credit <= -0.01).astype(float) if credit is not None else 0.0
    dollar, source = _dollar_series(panel)
    if dollar is not None:
        out["dollar_ret20"] = dollar / dollar.shift(20) - 1
        out["dollar_strong"] = (out["dollar_ret20"] >= 0.02).astype(float)
    else:
        out["dollar_strong"] = 0.0
    a = atr(panel.high["SPY"], panel.low["SPY"], spy, 14)
    out["ext_atr"] = (spy - sma50) / a
    out["rsi14"] = _rsi(spy)
    out["overextended"] = ((out["ext_atr"] >= 3) | (out["rsi14"] >= 75)).astype(float)
    out["count"] = out[list(SIGNALS)].sum(axis=1)
    out.attrs["dollarSource"] = source
    out.attrs["creditProxy"] = feats.attrs.get("creditProxy")
    return out


def targets(panel):
    spy = panel.close["SPY"]
    out = pd.DataFrame(index=panel.dates)
    out["fwd20_ret"] = spy.shift(-20) / spy - 1
    out["fwd20_down"] = (out["fwd20_ret"] < 0).astype(float).where(out["fwd20_ret"].notna())
    for name, h, thr in (("dd20_5", 20, -0.05), ("dd60_8", 60, -0.08)):
        fwd_min = pd.concat([spy.shift(-k) for k in range(1, h + 1)], axis=1).min(axis=1)
        valid = spy.shift(-h).notna()
        out[name] = ((fwd_min / spy - 1) <= thr).astype(float).where(valid)
    return out


def _episodes(on, gap):
    idx = np.flatnonzero(np.asarray(on) > 0)
    if not len(idx):
        return 0
    return int(1 + np.sum(np.diff(idx) > gap))


def _hac_t(y, x, lag):
    import statsmodels.api as sm
    X = sm.add_constant(np.asarray(x, dtype=float))
    res = sm.OLS(np.asarray(y, dtype=float), X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return float(res.params[1]), float(res.tvalues[1]), float(res.pvalues[1])


def _circular_p(sig, tgt, n=2000, seed=11):
    """One-sided p: P(hit rate under circular shifts of the signal >= observed). Shifts keep the signal's
    clustering, which a plain shuffle would destroy (a shuffle would overstate significance)."""
    s, y = np.asarray(sig, dtype=float), np.asarray(tgt, dtype=float)
    if s.sum() == 0:
        return None
    obs = (s * y).sum() / s.sum()
    rng = np.random.default_rng(seed)
    T = len(s)
    lo = max(1, T // 10)
    shifts = rng.integers(lo, T - lo, size=n) if T > 2 * lo + 1 else np.arange(1, T)
    null = np.array([(np.roll(s, k) * y).sum() / s.sum() for k in shifts])
    return float((1 + np.sum(null >= obs - 1e-12)) / (1 + len(null)))


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        adj[i] = prev
    return adj


def score(signals, tg, start=None):
    rows = []
    mask = pd.Series(True, index=signals.index)
    if start is not None:
        mask &= signals.index >= pd.Timestamp(start)
    for tname, h in TARGETS.items():
        y_all = tg[tname]
        for sname in SIGNALS + ("composite2",):
            s = (signals["count"] >= 2).astype(float) if sname == "composite2" else signals[sname]
            ok = mask & y_all.notna() & s.notna()
            y, x = y_all[ok].to_numpy(), s[ok].to_numpy()
            fwd = tg["fwd20_ret"][ok].to_numpy()
            n_on = int(x.sum())
            base = float(y.mean()) if len(y) else None
            row = {"signal": sname, "target": tname, "horizon": h, "n": int(len(y)), "daysOn": n_on,
                   "episodesOn": _episodes(x, h), "baseRate": base}
            if n_on >= 5 and n_on < len(x) - 5:
                hit = float(y[x > 0].mean())
                row.update({"hitRateOn": hit, "hitRateOff": float(y[x == 0].mean()), "lift": hit / base if base else None,
                            "fwd20On": float(np.nanmean(fwd[x > 0])), "fwd20Off": float(np.nanmean(fwd[x == 0]))})
                try:
                    b, t, p = _hac_t(y, x, h)
                    row.update({"hacCoef": b, "hacT": t, "hacP": p})
                except Exception as error:  # noqa: BLE001
                    row["hacError"] = str(error)[:120]
                row["permP"] = _circular_p(x, y)
            else:
                row["note"] = "signal almost never/always on in the scored window; not testable"
            rows.append(row)
    ps = [r.get("permP") for r in rows]
    idx = [i for i, v in enumerate(ps) if v is not None]
    if idx:
        h_adj, b_adj = holm([ps[i] for i in idx]), bh([ps[i] for i in idx])
        for j, i in enumerate(idx):
            rows[i]["holmP"] = float(h_adj[j])
            rows[i]["bhQ"] = float(b_adj[j])
    return rows


def walk_forward_composite(signals, tg, target="dd20_5", train_min=150, test_len=63):
    """Pick the count threshold k on the expanding training window (balanced accuracy; only training rows whose
    forward window has fully closed are used), apply it to the next test window. Returns OOS stats + Brier."""
    y_all = tg[target]
    h = TARGETS[target]
    dates = signals.index
    first = signals["count"].first_valid_index()
    start_i = max(WARMUP, int(np.searchsorted(dates, first)) if first is not None else WARMUP)
    preds, ys, picks = [], [], []
    i = start_i + train_min
    while i < len(dates):
        train_end = i - h                     # rows before this have a known label at date i (no lookahead)
        tr = slice(start_i, max(start_i, train_end))
        ytr, ctr = y_all.iloc[tr], signals["count"].iloc[tr]
        ok = ytr.notna()
        ytr, ctr = ytr[ok], ctr[ok]
        if len(ytr) < 60:
            i += test_len
            continue
        best_k, best_ba, rates = None, -1, {}
        for k in range(1, 5):
            pred = (ctr >= k).astype(float)
            tp = ((pred == 1) & (ytr == 1)).sum(); fn = ((pred == 0) & (ytr == 1)).sum()
            tn = ((pred == 0) & (ytr == 0)).sum(); fp = ((pred == 1) & (ytr == 0)).sum()
            tpr = tp / (tp + fn) if tp + fn else 0.0
            tnr = tn / (tn + fp) if tn + fp else 0.0
            ba = 0.5 * (tpr + tnr)
            on_rate = ytr[pred == 1].mean() if (pred == 1).any() else np.nan
            off_rate = ytr[pred == 0].mean() if (pred == 0).any() else np.nan
            rates[k] = (on_rate, off_rate)
            if ba > best_ba:
                best_k, best_ba = k, ba
        base = float(ytr.mean())
        te = slice(i, min(i + test_len, len(dates)))
        c_te, y_te = signals["count"].iloc[te], y_all.iloc[te]
        on, off = rates[best_k]
        prob = np.where(c_te >= best_k, on if np.isfinite(on) else base, off if np.isfinite(off) else base)
        for pr, yv, d in zip(prob, y_te, c_te.index):
            if np.isfinite(yv):
                preds.append((float(pr), base)); ys.append(float(yv))
        picks.append({"testStart": str(dates[i].date()), "k": best_k, "trainBalancedAccuracy": round(float(best_ba), 3),
                      "trainBaseRate": round(base, 3)})
        i += test_len
    if not ys:
        return {"target": target, "note": "not enough labeled out-of-sample rows", "folds": picks}
    p = np.array([a for a, _ in preds]); clim = np.array([b for _, b in preds]); y = np.array(ys)
    brier, brier_clim = float(np.mean((p - y) ** 2)), float(np.mean((clim - y) ** 2))
    # block-bootstrap the Brier difference (block = horizon) for a CI
    rng = np.random.default_rng(5)
    d = (clim - y) ** 2 - (p - y) ** 2
    T, B = len(d), max(1, h)
    boots = []
    for _ in range(2000):
        starts = rng.integers(0, T, size=math.ceil(T / B))
        idx = np.concatenate([(np.arange(s, s + B) % T) for s in starts])[:T]
        boots.append(d[idx].mean())
    boots = np.array(boots)
    return {"target": target, "oosRows": int(T), "brier": brier, "brierClimatology": brier_clim,
            "brierSkill": 1 - brier / brier_clim if brier_clim else None,
            "improvementCI95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "pImprovement": float((boots <= 0).mean()), "independentBlocks": int(T // B), "folds": picks}


def run(panel=None, policy=None, write=True):
    from quant_tools import fast_panel
    if panel is None:
        from market_data import load_panel
        panel = load_panel()
    sig = build_signals(panel, policy)
    tg = targets(panel)
    start = sig.index[min(WARMUP, len(sig) - 1)]
    rows = score(sig, tg, start)
    wf = {t: walk_forward_composite(sig, tg, t) for t in TARGETS}
    survivors = [r for r in rows if (r.get("holmP") or 1) < 0.05]
    robust = [r for r in survivors if r["episodesOn"] >= MIN_EPISODES]
    latest = sig.iloc[-1]
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"), "sessions": int(len(sig)), "scoredFrom": str(start.date()),
        "dollarSource": sig.attrs.get("dollarSource"), "creditProxy": sig.attrs.get("creditProxy"),
        "baseRates": {t: float(tg[t][tg.index >= start].mean()) for t in TARGETS},
        "tests": rows, "walkForwardComposite": wf, "testsRun": len([r for r in rows if r.get("permP") is not None]),
        "survivingHolm05": [f"{r['signal']}->{r['target']} ({r['episodesOn']} episodes)" for r in survivors],
        "robustSurvivors": [f"{r['signal']}->{r['target']}" for r in robust],
        "verdict": (("Signals that survive Holm with >= %d independent episodes: %s." % (MIN_EPISODES, ", ".join(
                        f"{r['signal']}->{r['target']}" for r in robust))) if robust else
                    ("Nominal only: %s survive Holm correction, but each rests on fewer than %d independent episodes, "
                     "so a few clustered events drive the result; the walk-forward composite must also beat "
                     "climatology out of sample before this counts as an edge." % (", ".join(
                        f"{r['signal']}->{r['target']}" for r in survivors), MIN_EPISODES)) if survivors else
                    "No signal beats chance after multiple-testing correction on this data (2 years, few independent "
                    "adverse episodes). Treat the detector as a risk flag, not a proven timing edge."),
        "latest": {"date": str(sig.index[-1].date()), **{k: (None if pd.isna(latest.get(k)) else float(latest.get(k)))
                                                         for k in list(SIGNALS) + ["count", "hmm_prob", "ext_atr", "rsi14",
                                                                                     "credit_excess20", "dollar_ret20", "breadth_drop20"]
                                                         if k in latest.index}},
    }
    summary = _clean(summary)
    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(OUT_PATH, json.dumps(summary, indent=1, default=str, allow_nan=False).encode())
    return summary, sig


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


if __name__ == "__main__":
    from structured_log import log_run
    with log_run("top_detector"):
        s, _ = run(write=True)
        brief = {k: s[k] for k in ("sessions", "scoredFrom", "baseRates", "testsRun", "survivingHolm05", "verdict", "latest")}
        brief["walkForward"] = {t: {k: v for k, v in w.items() if k != "folds"} for t, w in s["walkForwardComposite"].items()}
        print(json.dumps(brief, indent=1, default=str))
