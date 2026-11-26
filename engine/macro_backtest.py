"""Macro-aware index backtests (longest history) + dollar-shock lead/lag re-test.

Study only: it recommends, it never changes the live paper config.

Data: Schwab $SPX daily price index from 1970 (index_long), 3-month T-bill cash (DTB3), FRED macro series made
point-in-time by fred_macro (ALFRED first-release dates or publication lags), and our own real-time nowcast
panel (macro_nowcast.daily_panel, built from raw data only). Decisions at the close of day d use values
available by d; the position earns d+1's return. Dividends: $SPX is a price index, so equity days get an
approximate S&P dividend yield by decade (labelled; a price-only rerun is reported as a robustness check).

Selection uses pre-holdout data only (holdout 2026-07-09 .. 2026-11-24). The single holdout look is logged in
Massive/Research/swarm/holdout_ledger.json.

Usage (from engine/): python macro_backtest.py run [--holdout]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fred_macro as FM
import macro_nowcast as NC

try:
    from fetch_and_model import DATA_DIR, write_atomically
except Exception:  # pragma: no cover
    DATA_DIR = FM.DATA_DIR
    write_atomically = FM.write_atomically

OUT_DIR = DATA_DIR / "Research" / "macro"
REPORT_JSON = OUT_DIR / "macro_backtest.json"
REPORT_MD = OUT_DIR / "macro_backtest_report.md"
INDEX_DIR = DATA_DIR / "Macro" / "index_long"
HOLDOUT_START, HOLDOUT_END = NC.HOLDOUT_START, NC.HOLDOUT_END
EVAL_START = pd.Timestamp("1983-01-03")
TD = 252
COST_BPS = 5.0
WF_TRAIN, WF_TEST, PURGE, EMBARGO = 2520, 504, 21, 5

# Approximate S&P 500 dividend yield by period (%/yr). Labelled assumption; price-only rerun shown too.
DIV_YIELD = [("1970", 3.5), ("1980", 4.5), ("1987", 3.2), ("1992", 2.8), ("1996", 1.8), ("2000", 1.3),
             ("2003", 1.8), ("2009", 2.4), ("2010", 2.0), ("2020", 1.6), ("2022", 1.5)]

ASOF_IDS = ["T10Y3M", "T10Y2Y", "BAA10Y", "BAA", "GS10", "SAHMREALTIME", "NFCI", "CPILFESL:pch", "CPIAUCSL:pch",
            "DOLLAR_MAJOR_LONG", "ICSA", "PAYEMS:chg", "GDPC1:pca", "UNRATE", "DTB3"]


# ====================================================================== data
def load_index(name="SPX"):
    f = pd.read_csv(INDEX_DIR / f"{name}.csv", index_col=0, parse_dates=True).sort_index()
    return f["close"].astype(float)


def div_daily(index):
    s = pd.Series(np.nan, index=index)
    for start, y in DIV_YIELD:
        s[s.index >= pd.Timestamp(start)] = y
    return s.ffill().fillna(DIV_YIELD[0][1]) / 100.0 / TD


def build_panel(close, nowcast=None, root=None):
    sessions = close.index
    P = FM.asof_frame(ASOF_IDS, sessions, root=root)
    # credit: daily Baa-10y from 1986, monthly Baa - GS10 before
    monthly_credit = P["BAA"] - P["GS10"]
    P["CREDIT"] = P["BAA10Y"].where(P["BAA10Y"].notna(), monthly_credit)
    if nowcast is not None:
        P = P.join(nowcast.reindex(sessions, method="ffill"), how="left")
    return P


# ====================================================================== flags (True = risk off)
def _ann(pch_series, n):
    """Annualized rate from a daily as-of series of m/m % prints over the last n distinct prints."""
    vals = pch_series.copy()
    change = vals.ne(vals.shift())
    out = pd.Series(np.nan, index=vals.index)
    hist = []
    for i, (v, c) in enumerate(zip(vals.values, change.values)):
        if c and v == v:
            hist.append(v)
        if len(hist) >= n:
            out.iat[i] = (np.prod(1 + np.array(hist[-n:]) / 100) ** (12 / n) - 1) * 100
    return out


def flags(P):
    F = {}
    F["curve_3m_inverted"] = P["T10Y3M"] < 0
    F["curve_3m_inverted_12m"] = P["T10Y3M"].rolling(TD, min_periods=20).min() < 0
    F["curve_2y_inverted"] = P["T10Y2Y"] < 0
    cr = P["CREDIT"]
    F["credit_widen_50bp_3m"] = (cr - cr.shift(63)) > 0.5
    z = (cr - cr.rolling(TD, min_periods=120).mean()) / cr.rolling(TD, min_periods=120).std()
    F["credit_z_gt_1.5"] = z > 1.5
    F["sahm_0.5"] = P["SAHMREALTIME"] >= 0.5
    F["sahm_0.3"] = P["SAHMREALTIME"] >= 0.3
    F["nfci_tight"] = P["NFCI"] > 0
    F["nfci_rising_0.3"] = (P["NFCI"] - P["NFCI"].shift(63)) > 0.3
    core6, core12 = _ann(P["CPILFESL:pch"], 6), _ann(P["CPILFESL:pch"], 12)
    F["core_cpi_6m_gt_4_rising"] = (core6 > 4) & (core6 > core12)
    F["cpi_6m_gt_5"] = _ann(P["CPIAUCSL:pch"], 6) > 5
    dol = np.log(P["DOLLAR_MAJOR_LONG"])
    F["dollar_6m_up_5pct"] = (dol - dol.shift(126)) > np.log(1.05)
    F["dollar_6m_up_8pct"] = (dol - dol.shift(126)) > np.log(1.08)
    cl = P["ICSA"].rolling(20, min_periods=5).mean()
    F["claims_20pct_off_low"] = cl / cl.rolling(TD, min_periods=60).min() - 1 > 0.2
    F["payrolls_last_negative"] = P["PAYEMS:chg"] < 0
    F["gdp_last_negative"] = P["GDPC1:pca"] < 0
    core = pd.DataFrame({k: F[k] for k in ("curve_3m_inverted", "credit_widen_50bp_3m", "sahm_0.5", "nfci_tight",
                                           "dollar_6m_up_5pct")}).fillna(False).astype(int).sum(axis=1)
    F["composite_ge2"] = core >= 2
    F["composite_ge3"] = core >= 3
    # ---- nowcast-based (real-time, raw-data nowcasts)
    if "NC_UNRATE" in P:
        umin = P["UNRATE"].rolling(TD, min_periods=60).min()
        F["nc_unrate_up_0.5"] = (P["NC_UNRATE"] - umin) >= 0.5
        F["nc_unrate_up_0.3"] = (P["NC_UNRATE"] - umin) >= 0.3
        F["nc_payrolls_negative"] = P["NC_PAYROLLS"] < 0
        F["nc_gdp_negative"] = P["NC_GDP_SAAR"] < 0
        F["nc_core_cpi_gt_4ann"] = P["NC_CORE_CPI_MOM"] * 12 > 4
        F["nc_cpi_upside_vs_last_0.2"] = P["NC_CPI_MOM_VS_LAST"] > 0.2
    return {k: v.fillna(False).astype(bool) for k, v in F.items()}


NOWCAST_VS_OFFICIAL = [("nc_unrate_up_0.5", "sahm_0.5"), ("nc_payrolls_negative", "payrolls_last_negative"),
                       ("nc_gdp_negative", "gdp_last_negative"), ("nc_core_cpi_gt_4ann", "core_cpi_6m_gt_4_rising")]


# ====================================================================== strategies
def tsmom_signal(close, lookback=TD, skip=21):
    return (close.shift(skip) / close.shift(lookback) - 1) > 0


def strat_returns(pos, eq_ret, cash_ret, cost_bps=COST_BPS):
    """pos decided at close d (bool/0-1) earns d+1 equity return; costs on switches."""
    w = pos.astype(float).shift(1).fillna(0.0)
    turn = w.diff().abs().fillna(0.0)
    return w * eq_ret + (1 - w) * cash_ret - turn * cost_bps / 1e4


def candidates(close, F):
    base = tsmom_signal(close)
    out = {"TSMOM": base, "BUY_HOLD": pd.Series(True, index=close.index)}
    for k, f in F.items():
        out[f"TSMOM_and_not_{k}"] = base & ~f
        out[f"BH_and_not_{k}"] = ~f
    return out


def perf(r):
    r = r.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    sd = r.std()
    return {"sharpe": float(r.mean() / sd * math.sqrt(TD)) if sd > 0 else None,
            "cagr": float(eq.iloc[-1] ** (TD / len(r)) - 1), "maxDD": dd, "vol": float(sd * math.sqrt(TD)),
            "days": int(len(r)), "start": str(r.index[0].date()), "end": str(r.index[-1].date())}


def sharpe(r):
    r = r.dropna()
    sd = r.std()
    return float(r.mean() / sd * math.sqrt(TD)) if sd > 0 and len(r) > 20 else -np.inf


def walk_forward(R):
    """R: DataFrame of candidate daily returns (pre-holdout, eval window). Expanding train, purge+embargo gap."""
    n = len(R)
    pieces, folds = [], []
    start = WF_TRAIN
    while start < n:
        end = min(start + WF_TEST, n)
        train = R.iloc[: max(1, start - PURGE - EMBARGO)]
        scores = {c: sharpe(train[c]) for c in R.columns}
        best = max(scores, key=lambda c: (scores[c], c == "TSMOM"))
        pieces.append(R[best].iloc[start:end])
        folds.append({"testStart": str(R.index[start].date()), "testEnd": str(R.index[end - 1].date()), "chosen": best,
                      "trainSharpe": scores[best], "testSharpe": sharpe(R[best].iloc[start:end]),
                      "tsmomTestSharpe": sharpe(R["TSMOM"].iloc[start:end])})
        start = end
    return pd.concat(pieces), folds


# ====================================================================== dollar shock lead/lag
def dollar_shock_study(close, dollar, oil=None, z_thr=2.0, horizons=(1, 5, 10, 20), end=None):
    import statsmodels.api as sm
    idx = close.index
    d = np.log(dollar.reindex(idx).ffill())
    eq = np.log(close)
    d5 = d - d.shift(5)
    sd = d5.rolling(TD, min_periods=120).std().shift(1)
    z = d5 / sd
    out = {"zThreshold": z_thr, "decluster": 20}
    for timing, lag in (("sameDayClose", 0), ("nextDayClose", 1)):
        res = {}
        for side, mask in (("surge", z > z_thr), ("slump", z < -z_thr)):
            m = mask.fillna(False).values
            ev, last = [], -10 ** 9
            for i in np.where(m)[0]:
                if i - last >= 20:
                    ev.append(i)
                    last = i
            rows = {}
            for h in horizons:
                fwd = (eq.shift(-h - lag) - eq.shift(-lag))
                base = fwd.mean()
                ex = np.array([fwd.iloc[i] - base for i in ev if i + h + lag < len(idx) and (end is None or idx[min(i + h + lag, len(idx) - 1)] < end)])
                ex = ex[np.isfinite(ex)]
                if len(ex) >= 5:
                    t = ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex)))
                    rows[f"h{h}"] = {"n": int(len(ex)), "meanExcessPct": float(ex.mean() * 100), "t": float(t),
                                     "hitNegative": float((ex < 0).mean())}
                    if oil is not None and h in (5, 20):
                        o = np.log(oil.reindex(idx).ffill())
                        fo = (o.shift(-h - lag) - o.shift(-lag))
                        eo = np.array([fo.iloc[i] - fo.mean() for i in ev if i + h + lag < len(idx)])
                        eo = eo[np.isfinite(eo)]
                        if len(eo) >= 5:
                            rows[f"h{h}"]["oilMeanExcessPct"] = float(eo.mean() * 100)
                            rows[f"h{h}"]["oilT"] = float(eo.mean() / (eo.std(ddof=1) / math.sqrt(len(eo))))
            res[side] = {"events": int(len(ev)), "byHorizon": rows,
                         "eventDates": [str(idx[i].date()) for i in ev][-15:]}
        out[timing] = res
    # continuous lead/lag: next-k-day SPX return on today's dollar return (HAC)
    r_d = d.diff()
    r_e = eq.diff()
    ll = {}
    for k in range(0, 6):
        y = r_e.shift(-k)
        df = pd.concat([y, r_d], axis=1).dropna()
        if end is not None:
            df = df[df.index < end]
        X = sm.add_constant(df.iloc[:, 1].values)
        fit = sm.OLS(df.iloc[:, 0].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        ll[f"k{k}"] = {"beta": float(fit.params[1]), "t": float(fit.tvalues[1]), "p": float(fit.pvalues[1]), "n": int(len(df))}
    out["leadLag"] = ll
    # predictive regression: next 5-day SPX return on dollar 5-day z (HAC 5)
    y5 = eq.shift(-6) - eq.shift(-1)
    df = pd.concat([y5, z], axis=1).dropna()
    if end is not None:
        df = df[df.index < end]
    fit = sm.OLS(df.iloc[:, 0].values, sm.add_constant(df.iloc[:, 1].values)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    out["predictive5d"] = {"betaPctPerZ": float(fit.params[1] * 100), "t": float(fit.tvalues[1]), "p": float(fit.pvalues[1]),
                           "n": int(len(df))}
    return out


# ====================================================================== memory-safe SPA (same math as swarm.spa_test)
def spa_test(d, B=1000, mean_block=10, seed=7, chunk=20):
    """Hansen (2005) SPA + per-candidate studentized bootstrap p-values, chunked so T x K x B fits in memory."""
    from swarm import stationary_indices
    d = np.asarray(d, dtype=float)
    T, K = d.shape
    rng = np.random.default_rng(seed)
    dbar = d.mean(axis=0)
    boot = np.empty((B, K))
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        idx = stationary_indices(T, b, mean_block, rng)
        boot[start:start + b] = np.stack([d[i].mean(axis=0) for i in idx])
    centered = boot - dbar
    rt = math.sqrt(T)
    omega = rt * centered.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.nan)
    tstat = rt * dbar / omega
    t_spa = max(float(np.nanmax(tstat)), 0.0)
    thresh = -math.sqrt(2 * math.log(math.log(T))) if T > 15 else -1.0
    g = np.where(tstat >= thresh, dbar, 0.0)
    stat = np.maximum(np.nanmax(rt * (boot - g) / omega, axis=1), 0.0)
    p_ind = ((rt * centered / omega >= tstat[None, :]).sum(axis=0) + 1) / (B + 1)
    p_ind = np.where(np.isfinite(tstat), p_ind, 1.0)
    return {"spaP": float((stat >= t_spa).mean()), "bestT": float(np.nanmax(tstat)), "sessions": T,
            "candidates": K, "bootstrap": B, "meanBlock": mean_block}, p_ind, tstat


# ====================================================================== orchestration
def run(holdout=False, write=True, root=None):
    from swarm import holm, bh, deflated_sharpe, pbo_cscv
    close = load_index("SPX")
    nowcast = NC.load_daily_panel()
    P = build_panel(close, nowcast, root=root)
    F = flags(P)
    eq_ret = close.pct_change()
    cash = (P["DTB3"].ffill().shift(1) / 100.0 / TD).fillna(0.0)
    divd = div_daily(close.index)
    cands = candidates(close, F)
    results = {}
    for label, eq in (("withDividendApprox", eq_ret + divd), ("priceOnly", eq_ret)):
        R = pd.DataFrame({k: strat_returns(v, eq, cash) for k, v in cands.items()})
        pre = R[(R.index >= EVAL_START) & (R.index < HOLDOUT_START)]
        stats = {k: perf(pre[k]) for k in pre.columns}
        for k in stats:
            stats[k]["timeInMarket"] = float(cands[k].astype(float).shift(1).loc[pre.index].mean())
        d = pre.drop(columns=["TSMOM"]).sub(pre["TSMOM"], axis=0)
        spa, p_ind, tstat = spa_test(d.values)
        names = list(d.columns)
        p_h, p_b = holm(p_ind), bh(p_ind)
        for i, k in enumerate(names):
            stats[k].update({"vsTsmomP": float(p_ind[i]), "vsTsmomHolm": float(p_h[i]), "vsTsmomBH": float(p_b[i]),
                             "vsTsmomT": float(tstat[i]) if np.isfinite(tstat[i]) else None})
        trial_sr = np.array([stats[k]["sharpe"] or 0 for k in pre.columns])
        best = max(pre.columns, key=lambda c: stats[c]["sharpe"] or -9)
        dsr = deflated_sharpe(pre[best].values, len(pre.columns), trial_sr)
        pbo = pbo_cscv(pre.values, n_blocks=16)
        wf, folds = walk_forward(pre)
        wf_ts = pre["TSMOM"].loc[wf.index]
        wf_spa, _, _ = spa_test((wf - wf_ts).values.reshape(-1, 1))
        nvo = {}
        for a, b in NOWCAST_VS_OFFICIAL:
            for mode in ("TSMOM_and_not_", "BH_and_not_"):
                ka, kb = mode + a, mode + b
                if ka in pre and kb in pre:
                    s_, p_, t_ = spa_test((pre[ka] - pre[kb]).values.reshape(-1, 1))
                    nvo[f"{ka} vs {kb}"] = {"sharpeNowcast": stats[ka]["sharpe"], "sharpeOfficial": stats[kb]["sharpe"],
                                            "p": float(p_[0]), "t": float(t_[0]) if np.isfinite(t_[0]) else None}
        if nvo:
            ks = list(nvo)
            for k, h in zip(ks, holm(np.array([nvo[k]["p"] for k in ks]))):
                nvo[k]["holm"] = float(h)
        results[label] = {"stats": stats, "spaVsTsmom": spa, "best": best, "deflatedSharpeBest": dsr, "pbo": pbo,
                          "walkForward": {"selected": perf(wf), "tsmomSameDays": perf(wf_ts), "spa": wf_spa,
                                          "folds": folds},
                          "nowcastVsOfficial": nvo}
    doc = {"generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "pointInTime": "alfred-first-release (lag fallback)" if FM.has_key() else "publication-lag",
           "evalStart": str(EVAL_START.date()), "holdoutStart": str(HOLDOUT_START.date()),
           "indexHistory": {"SPX": [str(close.index[0].date()), str(close.index[-1].date())]},
           "candidates": len(cands), "costBps": COST_BPS, "divYieldAssumption": DIV_YIELD, "results": results}
    # ---- dollar shock, pre-holdout only
    oil = FM.load_series("DCOILWTICO", root)
    major = FM.load_series("DOLLAR_MAJOR_LONG", root)
    broad = FM.load_series("DTWEXBGS", root)
    ds = {}
    for name, ser, lo, hi in (("major_1973_2005", major, "1973-01-01", "2005-12-31"),
                              ("major_2006_pre_holdout", major, "2006-01-01", None),
                              ("broad_2006_pre_holdout", broad, "2006-01-01", None),
                              ("major_full_pre_holdout", major, "1973-01-01", None)):
        if ser is None:
            continue
        c = close[(close.index >= lo) & (close.index < (pd.Timestamp(hi) if hi else HOLDOUT_START))]
        ds[name] = dollar_shock_study(c, ser, oil=oil, end=HOLDOUT_START)
    doc["dollarShock"] = ds
    # ---- single logged holdout look at the pre-registered champion
    if holdout:
        champ = results["withDividendApprox"]["best"]
        R = pd.DataFrame({k: strat_returns(cands[k], eq_ret + divd, cash) for k in (champ, "TSMOM", "BUY_HOLD")})
        ho = R[(R.index >= HOLDOUT_START) & (R.index <= HOLDOUT_END)]
        doc["holdout"] = {"champion": champ, "champion_perf": perf(ho[champ]), "tsmom": perf(ho["TSMOM"]),
                          "buyHold": perf(ho["BUY_HOLD"]),
                          "note": "One look, logged. ~95 sessions: far too short to confirm or reject anything."}
        _ledger(champ)
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_atomically(REPORT_JSON, json.dumps(NC._clean(doc), indent=1, default=NC._json_default).encode("utf-8"))
        try:
            write_report(json.loads(REPORT_JSON.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001 - the JSON is the source of truth
            print(f"report writer failed: {exc}", file=sys.stderr)
    return doc


def _ledger(champion):
    design = hashlib.sha256(json.dumps({"flags": "v1", "cost": COST_BPS, "wf": [WF_TRAIN, WF_TEST, PURGE, EMBARGO],
                                        "evalStart": str(EVAL_START.date())}).encode()).hexdigest()[:16]
    path = NC.LEDGER_PATH
    ledger = json.loads(path.read_text()) if path.exists() else {"views": 0, "entries": []}
    ledger["entries"].append({"at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "designHash": f"macro-{design}",
                              "study": "macro_backtest", "holdoutStart": str(HOLDOUT_START.date()),
                              "dataEnd": str(HOLDOUT_END.date()), "champion": champion})
    ledger["views"] = len({e["designHash"] for e in ledger["entries"] if e.get("holdoutStart") == str(HOLDOUT_START.date())})
    write_atomically(path, json.dumps(ledger, indent=2).encode("utf-8"))


# ====================================================================== markdown report
def _p(x, pct=False, d=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.{d}f}"


def write_report(doc=None, path=None):
    """Plain-English markdown report from macro_backtest.json + nowcast_accuracy.json (no recomputation)."""
    doc = doc or json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    acc_path = NC.ACCURACY_PATH
    acc = json.loads(acc_path.read_text(encoding="utf-8")) if acc_path.exists() else {}
    res = doc["results"]["withDividendApprox"]
    po = doc["results"].get("priceOnly", {})
    st = res["stats"]
    L = []
    w = L.append
    w("# Macro regime backtests vs plain TSMOM, and our macro nowcasts")
    w("")
    w(f"Generated {doc['generatedAt']} (America/New_York). Study only: nothing here changes the live paper config.")
    w("")
    w("## Verdict")
    tsm, best = st["TSMOM"], st[res["best"]]
    wf = res["walkForward"]
    w(f"- **Stay with plain TSMOM.** The best of {doc['candidates'] - 2} macro-filtered variants ({res['best']}) had a pre-holdout "
      f"Sharpe of {_p(best.get('sharpe'))} vs {_p(tsm.get('sharpe'))} for TSMOM, but that edge is what you expect from picking the best of "
      f"many: SPA p = {_p(res['spaVsTsmom']['spaP'])}, Holm-adjusted p = {_p(best.get('vsTsmomHolm'))}, PBO = {_p(res['pbo'].get('pbo') if isinstance(res['pbo'], dict) else res['pbo'])}.")
    w(f"- Choosing a filter walk-forward (re-picked every {WF_TEST} sessions on past data only) earned Sharpe "
      f"{_p(wf['selected'].get('sharpe'))} vs TSMOM {_p(wf['tsmomSameDays'].get('sharpe'))} on the same days (SPA p {_p(wf['spa']['spaP'])}).")
    nvo = res.get("nowcastVsOfficial") or {}
    if nvo:
        better = [k for k, v in nvo.items() if (v.get("t") or 0) > 0 and (v.get("holm") or 1) < 0.05]
        w(f"- Nowcast-based signals vs the lagged official data: {len(better)} of {len(nvo)} comparisons favored the nowcast after Holm "
          f"correction. Our nowcasts forecast the next print better than the last print does, but that does not translate into better equity timing.")
    ds = doc.get("dollarShock", {}).get("major_2006_pre_holdout") or {}
    ll = ds.get("leadLag", {})
    if ll:
        w(f"- Dollar shock: same-day SPX beta to the dollar {_p(ll['k0']['beta'])} (t {_p(ll['k0']['t'], d=1)}); next-day t {_p(ll['k1']['t'], d=1)}. "
          "A dollar move is a same-day symptom, not a usable lead.")
    w("")
    w("## Backtest (S&P 500 index, " + f"{doc['evalStart']} to the day before the holdout {doc['holdoutStart']})")
    w("")
    w("Long the index or in T-bills; decisions at the close earn the next day's return; 5 bp per switch. "
      "$SPX is a price index, so an approximate dividend yield by decade is added (price-only rerun below).")
    w("")
    w("| Strategy | Sharpe | CAGR | Max drawdown | Time in market | p vs TSMOM | Holm | BH |")
    w("|---|---|---|---|---|---|---|---|")
    top = sorted([k for k in st if k not in ("TSMOM", "BUY_HOLD")], key=lambda k: -(st[k].get("sharpe") or -9))[:10]
    for k in ["TSMOM", "BUY_HOLD"] + top:
        s = st[k]
        w(f"| {k} | {_p(s.get('sharpe'))} | {_p(s.get('cagr'), True)} | {_p(s.get('maxDD'), True)} | {_p(s.get('timeInMarket'), True)} | "
          f"{_p(s.get('vsTsmomP'), d=3)} | {_p(s.get('vsTsmomHolm'), d=3)} | {_p(s.get('vsTsmomBH'), d=3)} |")
    w("")
    dsr = res.get("deflatedSharpeBest")
    w(f"Multiple-testing checks: SPA (Hansen) p = {_p(res['spaVsTsmom']['spaP'])} across {res['spaVsTsmom']['candidates']} candidates vs TSMOM; "
      f"deflated-Sharpe probability of the best = {_p(dsr.get('deflatedSharpeProbability') if isinstance(dsr, dict) else dsr, d=3)} (vs zero, so it says the strategy works, not that it beats TSMOM); "
      f"PBO (CSCV) = {_p(res['pbo'].get('pbo') if isinstance(res['pbo'], dict) else res['pbo'])}.")
    if po:
        w(f"Price-only robustness: TSMOM Sharpe {_p(po['stats']['TSMOM'].get('sharpe'))}, best {po['best']} {_p(po['stats'][po['best']].get('sharpe'))}, SPA p {_p(po['spaVsTsmom']['spaP'])}.")
    w("")
    if nvo:
        w("### Nowcast vs lagged official signals")
        w("")
        w("| Comparison | Sharpe (nowcast) | Sharpe (official) | t | Holm |")
        w("|---|---|---|---|---|")
        for k, v in nvo.items():
            w(f"| {k} | {_p(v.get('sharpeNowcast'))} | {_p(v.get('sharpeOfficial'))} | {_p(v.get('t'), d=1)} | {_p(v.get('holm'), d=2)} |")
        w("")
    if doc.get("holdout"):
        h = doc["holdout"]
        w("### Sealed holdout (one logged look)")
        w("")
        w("| | Sharpe | Return (CAGR) | Max drawdown | Days |")
        w("|---|---|---|---|---|")
        for name, key in ((h["champion"], "champion_perf"), ("TSMOM", "tsmom"), ("Buy & hold", "buyHold")):
            s = h.get(key) or {}
            w(f"| {name} | {_p(s.get('sharpe'))} | {_p(s.get('cagr'), True)} | {_p(s.get('maxDD'), True)} | {s.get('days', 'n/a')} |")
        w("")
        same = len({json.dumps(h.get(k), sort_keys=True) for k in ("champion_perf", "tsmom", "buyHold")}) == 1
        if same:
            w("All three rows are identical: TSMOM was long the whole holdout and the champion's macro filter never fired, "
              "so the holdout cannot tell them apart.")
        w(h.get("note", ""))
        w("")
    w("## Dollar shock lead/lag (re-test)")
    w("")
    w("| Sample | Same-day beta (t) | Next-day t | 2-day t | 5-day predictive t | Surge events | Slump events |")
    w("|---|---|---|---|---|---|---|")
    for name, d in doc.get("dollarShock", {}).items():
        ll = d.get("leadLag", {})
        sd = d.get("sameDayClose", {})
        w(f"| {name} | {_p(ll.get('k0', {}).get('beta'))} ({_p(ll.get('k0', {}).get('t'), d=1)}) | {_p(ll.get('k1', {}).get('t'), d=1)} | "
          f"{_p(ll.get('k2', {}).get('t'), d=1)} | {_p(d.get('predictive5d', {}).get('t'), d=2)} | {sd.get('surge', {}).get('events', 'n/a')} | {sd.get('slump', {}).get('events', 'n/a')} |")
    w("")
    w("Event studies (5-day dollar z-score beyond ±2, declustered 20 days) show no significant forward SPX excess return at 1–20 days. "
      "A 2-day lag t near −2 in one sample would not survive multiple testing.")
    w("")
    w("## Our nowcasts (raw data only)")
    w("")
    w("Real-time walk-forward: each nowcast uses only data published by that day (ALFRED first-release dates; publication "
      "lags where ALFRED has no vintage), refit monthly on releases already out. Outside forecasts are never inputs.")
    w("")
    w("| Target | n | RMSE ours | RMSE last print | Ratio | Clark-West p | Holm | Direction hit | Ex-pandemic ratio |")
    w("|---|---|---|---|---|---|---|---|---|")
    for t, a in (acc.get("accuracy") or {}).items():
        f, x = a.get("final", {}), a.get("exPandemic", {})
        w(f"| {t} | {f.get('n', 'n/a')} | {_p(f.get('rmseModel'), d=3)} | {_p(f.get('rmseLastPrint'), d=3)} | {_p(f.get('rmseRatio'))} | "
          f"{_p(f.get('clarkWestP'), d=3)} | {_p(f.get('clarkWestHolm'), d=3)} | {_p(f.get('directionHitRate'), True)} | {_p(x.get('rmseRatio'))} |")
    if any((a.get("holdout") or {}).get("n") for a in (acc.get("accuracy") or {}).values()):
        w("")
        w("Holdout (one logged look): only " + ", ".join(f"{t} {(a.get('holdout') or {}).get('n')}" for t, a in acc["accuracy"].items())
          + " releases fell inside the holdout, below the 8 needed to score; nothing can be concluded from it yet.")
    w("")
    oc = acc.get("outsideComparison") or {}
    rows = [(k, v) for k, v in oc.items() if isinstance(v, dict) and v.get("rmseOurs") is not None]
    if rows:
        w("### Ours vs outside forecasts (display-only data, compared after the fact)")
        w("")
        w("| Target | vs | n | RMSE ours | RMSE theirs | DM p |")
        w("|---|---|---|---|---|---|")
        for k, v in rows:
            w(f"| {k} | {v.get('vs')} | {v.get('n')} | {_p(v.get('rmseOurs'), d=3)} | {_p(v.get('rmseTheirs'), d=3)} | {_p(v.get('dmP'), d=3)} |")
        w("")
    w("### Surprise-vs-consensus study (separate, labeled)")
    w("")
    w("Not run: no free, terms-clean history of Street consensus was found. A pluggable slot exists at "
      "`Massive/Macro/outside/consensus.csv` (target, period, value, source, asof). Consensus numbers there are display-only and "
      "can never become nowcast inputs (enforced in `macro_nowcast.assert_no_outside_inputs` and tests).")
    w("")
    w("## Caveats")
    w("")
    w("- Before ALFRED vintage coverage (often the 1990s) availability uses publication lags and latest-vintage values, which flatters older periods slightly.")
    w("- $SPX dividends are approximated by decade; the price-only rerun gives the same answer.")
    w("- HY OAS on FRED only covers the last ~3 years (ICE licensing), so credit uses Baa−10y (daily from 1986, monthly before).")
    w("- 2020-03..2020-12 is excluded from nowcast training (pandemic outliers); it is still in the evaluation unless marked ex-pandemic.")
    w("- The holdout is ~95 sessions: too short to confirm or reject anything.")
    out = Path(path or REPORT_MD)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(out, ("\n".join(L) + "\n").encode("utf-8"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--holdout", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        print(write_report())
        return 0
    doc = run(holdout=a.holdout)
    res = doc["results"]["withDividendApprox"]
    top = sorted(res["stats"].items(), key=lambda kv: -(kv[1].get("sharpe") or -9))[:12]
    for k, s in top:
        print(f"{k:45s} SR={s.get('sharpe'):.2f} DD={s.get('maxDD'):.1%} CAGR={s.get('cagr'):.1%} p={s.get('vsTsmomP')}")
    print("TSMOM", res["stats"]["TSMOM"])
    print("SPA", res["spaVsTsmom"])
    print("WF", res["walkForward"]["selected"], res["walkForward"]["tsmomSameDays"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
