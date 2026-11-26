"""Our own macro nowcasts (GDPNow-style, constantly updated) built from RAW data only.

Hard rule: no outside projection (Street consensus, GDPNow, NY Fed Staff Nowcast, Cleveland Fed inflation
nowcast, SPF ...) is ever a model input. Every feature series must be in FEATURE_SERIES, which is checked
against fred_macro.OUTSIDE_FORECASTS and outside_forecasts.OUTSIDE_IDS at import time and by tests. Outside
estimates are attached to the published JSON *after* our numbers are computed, for display/comparison only.

Point-in-time: each observation carries an `available` date (ALFRED first-release date when a FRED key is
configured, else a conservative publication lag; see fred_macro). A nowcast made on day s only uses
observations with available <= s, and the walk-forward only trains on periods whose official print was out
by the refit date. Targets are first releases where ALFRED has them.

Targets: CPI m/m, core CPI m/m, core PCE m/m, unemployment rate, payroll change, real GDP q/q SAAR.

Usage (from engine/):
  python macro_nowcast.py publish        # refresh nowcast_latest.json + history (fast; daily / dashboard)
  python macro_nowcast.py refresh        # incremental FRED + outside-estimate update, then publish
                                         # (research panel rebuilt when older than 7 days); dashboard job
  python macro_nowcast.py research       # walk-forward accuracy + daily nowcast panel (slower)
  python macro_nowcast.py research --holdout   # also score the sealed holdout (logged in holdout_ledger.json)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import fred_macro as FM

try:
    from fetch_and_model import DATA_DIR, write_atomically
except Exception:  # pragma: no cover
    DATA_DIR = FM.DATA_DIR
    write_atomically = FM.write_atomically

OUT_DIR = DATA_DIR / "Research" / "macro"
LATEST_PATH = OUT_DIR / "nowcast_latest.json"
HISTORY_PATH = OUT_DIR / "nowcast_history.jsonl"
EVOLUTION_PATH = OUT_DIR / "nowcast_evolution.json"
PANEL_PATH = OUT_DIR / "nowcast_daily.csv"
ACCURACY_PATH = OUT_DIR / "nowcast_accuracy.json"
CALENDAR_PATH = DATA_DIR / "Macro" / "calendar.json"
HOLDOUT_START = pd.Timestamp("2026-07-09")
HOLDOUT_END = pd.Timestamp("2026-11-24")
LEDGER_PATH = DATA_DIR / "Research" / "swarm" / "holdout_ledger.json"

# Raw inputs only. Nothing here is anyone's forecast.
FEATURE_SERIES = frozenset({
    "CPIAUCSL", "CPILFESL", "PCEPILFE", "PPIACO", "DCOILWTICO", "GASREGW", "T5YIE", "DOLLAR_MAJOR_LONG",
    "UNRATE", "PAYEMS", "ICSA", "CCSA", "SAHMREALTIME", "NFCI", "WEI", "INDPRO", "GDPC1", "T10Y3M",
    # first-release growth rates computed inside each ALFRED vintage (fred_macro.VINTAGE_TRANSFORMS)
    "CPIAUCSL:pch", "CPILFESL:pch", "PCEPILFE:pch", "PPIACO:pch", "PAYEMS:chg", "GDPC1:pca", "INDPRO:pch",
})


def assert_no_outside_inputs(ids=FEATURE_SERIES):
    ids = {str(i).split(":")[0] for i in ids}
    bad = set(ids) & set(FM.OUTSIDE_FORECASTS)
    try:
        import outside_forecasts as OF  # only to read its id list; never its data
        bad |= set(ids) & set(OF.OUTSIDE_IDS)
    except Exception:  # noqa: BLE001
        pass
    if bad:
        raise RuntimeError(f"Outside forecasts may not be nowcast inputs: {sorted(bad)}")
    return True


assert_no_outside_inputs()

SNAPSHOTS_PER_PERIOD = 9      # weekly snapshots back from the day before the release
MIN_TRAIN_PERIODS = {"M": 60, "Q": 30}
RIDGE_ALPHA = 3.0
# Pandemic months (Mar-Dec 2020) produced payroll / GDP prints 20-50x normal size. They are excluded from
# model *training* (they would dominate the fit) but are still scored in the accuracy tables.
EXCLUDE_TRAIN = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-31"))
BAND_Z = 1.2816               # 80% band

TARGETS = {
    "CPI_MOM": {"sid": "CPIAUCSL", "src": "CPIAUCSL:pch", "freq": "M", "kind": "pct", "label": "CPI (headline) m/m %", "calendar": "cpi",
                "core": ["lastH", "core3", "ppi1", "dol3"], "full": ["oil", "gas"]},
    "CORE_CPI_MOM": {"sid": "CPILFESL", "src": "CPILFESL:pch", "freq": "M", "kind": "pct", "label": "Core CPI m/m %", "calendar": "core_cpi",
                     "core": ["core1", "core3", "core12", "ppi1", "dol6"], "full": ["be3", "oil3"]},
    "CORE_PCE_MOM": {"sid": "PCEPILFE", "src": "PCEPILFE:pch", "freq": "M", "kind": "pct", "label": "Core PCE m/m %", "calendar": "core_pce",
                     "core": ["pce1", "pce3", "cpiSame", "cpiSameKnown", "ppi1"], "full": ["be3"]},
    "UNRATE": {"sid": "UNRATE", "src": "UNRATE", "freq": "M", "kind": "level", "label": "Unemployment rate %", "calendar": "unrate",
               "core": ["gapm", "icsaU", "ccsaU", "dU1", "sahm", "nfci"], "full": []},
    "PAYROLLS": {"sid": "PAYEMS", "src": "PAYEMS:chg", "freq": "M", "kind": "diff", "label": "Nonfarm payrolls change (k)", "calendar": "nfp",
                 "core": ["pay1", "pay3", "icsa", "ccsa", "nfci"], "full": ["wei"]},
    "GDP_SAAR": {"sid": "GDPC1", "src": "GDPC1:pca", "freq": "Q", "kind": "saar", "label": "Real GDP q/q SAAR %", "calendar": "gdp",
                 "core": ["gdp1", "ip", "payq", "icsaq", "nfciq", "curve"], "full": ["wei"]},
}

FEATURE_LABELS = {
    "lastH": "last CPI m/m", "core3": "core CPI 3m avg", "core1": "last core CPI m/m", "core12": "core CPI 12m avg",
    "ppi1": "last PPI commodities m/m", "dol3": "dollar 3m change", "dol6": "dollar 6m change",
    "oil": "WTI oil month-to-date vs last month", "gas": "gasoline month-to-date vs last month",
    "be3": "5y breakeven 3m change", "oil3": "WTI 3m change", "pce1": "last core PCE m/m", "pce3": "core PCE 3m avg",
    "cpiSame": "core CPI for the same month", "cpiSameKnown": "same-month core CPI already out",
    "gapm": "months since last jobless-rate print", "icsaU": "initial claims vs last print month",
    "ccsaU": "continuing claims vs last print month", "dU1": "last jobless-rate change", "sahm": "Sahm (real-time)",
    "nfci": "financial conditions (NFCI)", "pay1": "last payroll change", "pay3": "payrolls 3m avg",
    "icsa": "initial claims month-to-date vs last month", "ccsa": "continuing claims vs last month",
    "wei": "NY Fed Weekly Economic Index (raw data index)", "gdp1": "last GDP print", "ip": "industrial production q-to-date",
    "payq": "payrolls q-to-date", "icsaq": "claims q-to-date", "nfciq": "NFCI q-to-date", "curve": "10y-3m curve",
}


# ====================================================================== data access
class Data:
    """Point-in-time view of the raw series. Observations hidden until their `available` date."""

    def __init__(self, root=None, prefer_vintage=True, frames=None):
        self.s = {}
        ids = set(FEATURE_SERIES) | {t["sid"] for t in TARGETS.values()} | {t["src"] for t in TARGETS.values()}
        assert_no_outside_inputs(ids)
        for sid in ids:
            obs = frames.get(sid) if frames is not None else FM.observations(sid, root, prefer_vintage=prefer_vintage)
            if obs is None or len(obs) == 0:
                continue
            obs = obs.sort_index()
            self.s[sid] = (obs.index.values.astype("datetime64[ns]"), obs["value"].values.astype(float),
                           pd.DatetimeIndex(obs["available"]).values.astype("datetime64[ns]"))

    def has(self, sid):
        return sid in self.s

    def known(self, sid, asof):
        if sid not in self.s:
            return np.array([], dtype="datetime64[ns]"), np.array([])
        d, v, a = self.s[sid]
        m = a <= np.datetime64(pd.Timestamp(asof), "ns")
        return d[m], v[m]

    def latest(self, sid, asof, k=0):
        d, v = self.known(sid, asof)
        if len(v) <= k:
            return np.nan, None
        return float(v[-1 - k]), pd.Timestamp(d[-1 - k])

    def value_at(self, sid, obs_date, asof):
        d, v = self.known(sid, asof)
        i = np.searchsorted(d, np.datetime64(pd.Timestamp(obs_date), "ns"))
        if i < len(d) and d[i] == np.datetime64(pd.Timestamp(obs_date), "ns"):
            return float(v[i])
        return np.nan

    def window_mean(self, sid, a, b, asof):
        d, v = self.known(sid, asof)
        if not len(d):
            return np.nan
        lo = np.searchsorted(d, np.datetime64(pd.Timestamp(a), "ns"), "left")
        hi = np.searchsorted(d, np.datetime64(pd.Timestamp(b), "ns"), "right")
        return float(np.nanmean(v[lo:hi])) if hi > lo else np.nan

    def pct_changes(self, sid, asof, n):
        """Last n m/m % changes known at asof (first-release growth rates when available)."""
        if f"{sid}:pch" in self.s:
            d, v = self.known(f"{sid}:pch", asof)
            return v[-n:] if len(v) >= n else np.array([])
        d, v = self.known(sid, asof)
        if len(v) < n + 1:
            return np.array([])
        return 100.0 * (v[-n:] / v[-n - 1:-1] - 1.0)


def _mtd_or_last(D, sid, a, b, s):
    m = D.window_mean(sid, a, b, s)
    if np.isnan(m):
        val, d = D.latest(sid, s)
        return val
    return m


def _logchg(cur, ref):
    if not (cur > 0 and ref > 0):
        return np.nan
    return 100.0 * math.log(cur / ref)


def _period_bounds(p, freq):
    p = pd.Timestamp(p)
    if freq == "Q":
        end = p + pd.offsets.QuarterEnd(0)
        prev = p - pd.offsets.QuarterBegin(1, startingMonth=1)
    else:
        end = p + pd.offsets.MonthEnd(0)
        prev = p - pd.offsets.MonthBegin(1)
    prev_end = p - pd.Timedelta(days=1)
    return p, end, prev, prev_end


# ====================================================================== features (raw data only)
def features(D, tname, p, s):
    t = TARGETS[tname]
    ps, pe, qs, qe = _period_bounds(p, t["freq"])
    s = pd.Timestamp(s)
    f = {}
    if tname == "CPI_MOM":
        ch = D.pct_changes("CPIAUCSL", s, 1)
        f["lastH"] = ch[-1] if len(ch) else np.nan
        c3 = D.pct_changes("CPILFESL", s, 3)
        f["core3"] = float(c3.mean()) if len(c3) == 3 else np.nan
        pp = D.pct_changes("PPIACO", s, 1)
        f["ppi1"] = pp[-1] if len(pp) else np.nan
        f["dol3"] = _logchg(D.window_mean("DOLLAR_MAJOR_LONG", s - pd.Timedelta(days=14), s, s),
                            D.window_mean("DOLLAR_MAJOR_LONG", s - pd.Timedelta(days=105), s - pd.Timedelta(days=91), s))
        f["oil"] = _logchg(_mtd_or_last(D, "DCOILWTICO", ps, pe, s), D.window_mean("DCOILWTICO", qs, qe, s))
        f["gas"] = _logchg(_mtd_or_last(D, "GASREGW", ps, pe, s), D.window_mean("GASREGW", qs, qe, s))
    elif tname == "CORE_CPI_MOM":
        c = D.pct_changes("CPILFESL", s, 12)
        f["core1"] = c[-1] if len(c) else np.nan
        f["core3"] = float(c[-3:].mean()) if len(c) >= 3 else np.nan
        f["core12"] = float(c.mean()) if len(c) == 12 else np.nan
        pp = D.pct_changes("PPIACO", s, 1)
        f["ppi1"] = pp[-1] if len(pp) else np.nan
        f["dol6"] = _logchg(D.window_mean("DOLLAR_MAJOR_LONG", s - pd.Timedelta(days=14), s, s),
                            D.window_mean("DOLLAR_MAJOR_LONG", s - pd.Timedelta(days=196), s - pd.Timedelta(days=182), s))
        a = D.window_mean("T5YIE", s - pd.Timedelta(days=14), s, s)
        b = D.window_mean("T5YIE", s - pd.Timedelta(days=105), s - pd.Timedelta(days=91), s)
        f["be3"] = a - b
        f["oil3"] = _logchg(D.window_mean("DCOILWTICO", s - pd.Timedelta(days=14), s, s),
                            D.window_mean("DCOILWTICO", s - pd.Timedelta(days=105), s - pd.Timedelta(days=91), s))
    elif tname == "CORE_PCE_MOM":
        c = D.pct_changes("PCEPILFE", s, 3)
        f["pce1"] = c[-1] if len(c) else np.nan
        f["pce3"] = float(c.mean()) if len(c) == 3 else np.nan
        same = D.value_at("CPILFESL:pch", ps, s)
        if same == same:
            f["cpiSame"], f["cpiSameKnown"] = same, 1.0
        else:
            cc = D.pct_changes("CPILFESL", s, 1)
            f["cpiSame"], f["cpiSameKnown"] = (cc[-1] if len(cc) else np.nan), 0.0
        pp = D.pct_changes("PPIACO", s, 1)
        f["ppi1"] = pp[-1] if len(pp) else np.nan
        a = D.window_mean("T5YIE", s - pd.Timedelta(days=14), s, s)
        b = D.window_mean("T5YIE", s - pd.Timedelta(days=105), s - pd.Timedelta(days=91), s)
        f["be3"] = a - b
    elif tname == "UNRATE":
        last, ld = D.latest("UNRATE", s)
        prev, _ = D.latest("UNRATE", s, 1)
        f["_last"] = last
        if ld is None:
            return f
        f["gapm"] = float((ps.year - ld.year) * 12 + ps.month - ld.month)
        l_s, l_e = ld, ld + pd.offsets.MonthEnd(0)
        f["icsaU"] = _logchg(_mtd_or_last(D, "ICSA", ps, pe, s), D.window_mean("ICSA", l_s, l_e, s))
        f["ccsaU"] = _logchg(_mtd_or_last(D, "CCSA", ps, pe, s), D.window_mean("CCSA", l_s, l_e, s))
        f["dU1"] = last - prev
        f["sahm"] = D.latest("SAHMREALTIME", s)[0]
        f["nfci"] = D.latest("NFCI", s)[0]
    elif tname == "PAYEMS" or tname == "PAYROLLS":
        d, v = D.known("PAYEMS:chg", s)
        f["pay1"] = float(v[-1]) if len(v) else np.nan
        f["pay3"] = float(v[-3:].mean()) if len(v) >= 3 else np.nan
        f["icsa"] = _logchg(_mtd_or_last(D, "ICSA", ps, pe, s), D.window_mean("ICSA", qs, qe, s))
        f["ccsa"] = _logchg(_mtd_or_last(D, "CCSA", ps, pe, s), D.window_mean("CCSA", qs, qe, s))
        f["nfci"] = D.latest("NFCI", s)[0]
        f["wei"] = _mtd_or_last(D, "WEI", ps, pe, s) if D.has("WEI") else np.nan
    elif tname == "GDP_SAAR":
        d, v = D.known("GDPC1:pca", s)
        f["gdp1"] = float(v[-1]) if len(v) else np.nan
        f["ip"] = _mtd_or_last(D, "INDPRO:pch", ps, pe, s) * 12      # annualized avg monthly IP growth in the quarter
        f["payq"] = _mtd_or_last(D, "PAYEMS:chg", ps, pe, s)         # avg monthly payroll change in the quarter
        f["icsaq"] = _logchg(_mtd_or_last(D, "ICSA", ps, pe, s), D.window_mean("ICSA", qs, qe, s))
        f["nfciq"] = _mtd_or_last(D, "NFCI", ps, pe, s)
        f["curve"] = D.latest("T10Y3M", s)[0]
        f["wei"] = _mtd_or_last(D, "WEI", ps, pe, s) if D.has("WEI") else np.nan
    return f


def target_value(D, tname, p, asof=None):
    """Official value for period p as first released (growth rates computed inside one vintage)."""
    t = TARGETS[tname]
    asof = asof or pd.Timestamp("2262-01-01")
    return D.value_at(t["src"], pd.Timestamp(p), asof)


def release_dates(D, tname):
    """Series of availability (release) dates per period for the target."""
    sid = TARGETS[tname]["src"]
    if sid not in D.s:
        return pd.Series(dtype="datetime64[ns]")
    d, v, a = D.s[sid]
    return pd.Series(pd.DatetimeIndex(a), index=pd.DatetimeIndex(d))


# ====================================================================== rows + walk-forward
def build_rows(D, tname, start="1968-01-01", snapshots=SNAPSHOTS_PER_PERIOD, extra_snapshot=None):
    t = TARGETS[tname]
    rel = release_dates(D, tname)
    rel = rel[rel.index >= pd.Timestamp(start)]
    rows = []
    for p, R in rel.items():
        ps, pe, qs, qe = _period_bounds(p, t["freq"])
        y = target_value(D, tname, p, asof=R)
        for k in range(snapshots):
            s = R - pd.Timedelta(days=1 + 7 * k)
            if s < ps:
                break
            f = features(D, tname, p, s)
            last = f.pop("_last", np.nan)
            rows.append({"period": p, "release": R, "snap": s, "k": k, "daysToRelease": int((R - s).days),
                         "y": y, "_last": last, **f})
    return pd.DataFrame(rows)


def _naive(tname, row):
    """Last-print benchmark in target units (random walk)."""
    t = TARGETS[tname]
    if t["kind"] == "level":
        return row.get("_last", np.nan)
    key = {"CPI_MOM": "lastH", "CORE_CPI_MOM": "core1", "CORE_PCE_MOM": "pce1", "PAYROLLS": "pay1", "GDP_SAAR": "gdp1"}[tname]
    return row.get(key, np.nan)


def _fit_ridge(X, y, alpha=RIDGE_ALPHA):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    ym = y.mean()
    n, k = Z.shape
    beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(k), Z.T @ (y - ym))
    resid = y - ym - Z @ beta
    return {"mu": mu, "sd": sd, "beta": beta, "intercept": ym, "rmseIn": float(np.sqrt(np.mean(resid ** 2))), "n": int(n)}


def _predict(model, x):
    z = (x - model["mu"]) / model["sd"]
    return float(model["intercept"] + z @ model["beta"]), z * model["beta"]


def _model_target(tname, rows):
    """Training target: change vs last print for the level series, else the print itself."""
    if TARGETS[tname]["kind"] == "level":
        return rows["y"] - rows["_last"]
    return rows["y"]


def fit_models(tname, rows, refit):
    t = TARGETS[tname]
    train = rows[(rows["release"] <= refit) & rows["y"].notna()
                 & ~rows["period"].between(EXCLUDE_TRAIN[0], EXCLUDE_TRAIN[1])]
    out = {}
    for name, cols in (("full", t["core"] + t["full"]), ("core", t["core"])):
        if name == "full" and not t["full"]:
            continue
        tr = train.dropna(subset=cols + (["_last"] if t["kind"] == "level" else []))
        if tr["period"].nunique() < MIN_TRAIN_PERIODS[t["freq"]]:
            continue
        out[name] = {**_fit_ridge(tr[cols].values.astype(float), _model_target(tname, tr).values.astype(float)),
                     "cols": cols, "periods": int(tr["period"].nunique())}
    return out


def _apply(tname, models, row):
    t = TARGETS[tname]
    for name in ("full", "core"):
        m = models.get(name)
        if not m:
            continue
        x = np.array([row.get(c, np.nan) for c in m["cols"]], dtype=float)
        if np.isnan(x).any():
            continue
        pred, contrib = _predict(m, x)
        if t["kind"] == "level":
            last = row.get("_last", np.nan)
            if last != last:
                continue
            pred = last + pred
        return pred, name, dict(zip(m["cols"], contrib.tolist()))
    return np.nan, None, {}


def walk_forward(tname, rows):
    """Monthly refits; each snapshot is predicted by the model fitted at the latest refit <= its date."""
    rows = rows.sort_values("snap").reset_index(drop=True)
    preds, which = np.full(len(rows), np.nan), [None] * len(rows)
    months = pd.date_range(rows["snap"].min().normalize().replace(day=1), rows["snap"].max() + pd.offsets.MonthBegin(1), freq="MS")
    snaps = rows["snap"].values
    for i, r in enumerate(months[:-1]):
        nxt = months[i + 1]
        idx = np.where((snaps >= np.datetime64(r)) & (snaps < np.datetime64(nxt)))[0]
        if not len(idx):
            continue
        models = fit_models(tname, rows, r)
        if not models:
            continue
        for j in idx:
            p, name, _ = _apply(tname, models, rows.iloc[j].to_dict())
            preds[j], which[j] = p, name
    rows["pred"] = preds
    rows["model"] = which
    rows["naive"] = [_naive(tname, r) for r in rows.to_dict("records")]
    return rows


def _bucket(days):
    return "late" if days <= 10 else ("mid" if days <= 30 else "early")


def band_rmse(rows, asof, bucket=None, years=10):
    done = rows[(rows["release"] <= asof) & rows["pred"].notna() & rows["y"].notna()
                & (rows["release"] > asof - pd.Timedelta(days=365 * years))]
    if bucket:
        done = done[done["daysToRelease"].map(_bucket) == bucket]
    if len(done) < 8:
        return np.nan
    # Robust scale (MAD x 1.4826) so one pandemic print does not blow the band up for a decade.
    err = (done["pred"] - done["y"]).values
    return float(1.4826 * np.median(np.abs(err - np.median(err))))


# ====================================================================== accuracy (significance discipline)
def _hac_t(x, lags=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return np.nan, np.nan
    m = x.mean()
    e = x - m
    s = (e @ e) / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / n
    se = math.sqrt(max(s, 1e-18) / n)
    t = m / se
    from math import erf
    p_one = 0.5 * (1 - erf(t / math.sqrt(2)))
    return float(t), float(p_one)


def accuracy(tname, rows, start=None, end=None, final_only=True):
    r = rows[rows["pred"].notna() & rows["y"].notna() & rows["naive"].notna()]
    if final_only:
        r = r[r["k"] == 0]
    if start is not None:
        r = r[r["release"] >= start]
    if end is not None:
        r = r[r["release"] < end]
    if len(r) < 8:
        return {"n": int(len(r))}
    e_m, e_n = r["y"] - r["pred"], r["y"] - r["naive"]
    rmse_m, rmse_n = float(np.sqrt((e_m ** 2).mean())), float(np.sqrt((e_n ** 2).mean()))
    # Clark-West (2007) adjusted MSPE: model nests the random walk in spirit (last print is a feature).
    cw = e_n ** 2 - (e_m ** 2 - (r["naive"] - r["pred"]) ** 2)
    cw_t, cw_p = _hac_t(cw.values, lags=1)
    dm_t, dm_p = _hac_t((e_n ** 2 - e_m ** 2).values, lags=1)
    hit = float((np.sign(r["pred"] - r["naive"]) == np.sign(r["y"] - r["naive"])).mean())
    return {"n": int(len(r)), "first": str(r["period"].min().date()), "last": str(r["period"].max().date()),
            "rmseModel": rmse_m, "rmseLastPrint": rmse_n, "rmseRatio": rmse_m / rmse_n if rmse_n else None,
            "clarkWestT": cw_t, "clarkWestP": cw_p, "dieboldMarianoT": dm_t, "dieboldMarianoP": dm_p,
            "directionHitRate": hit}


# ====================================================================== daily panel for backtests / other models
def daily_panel(all_rows, sessions):
    """Real-time nowcast at each session: latest OOS snapshot (snap <= d) for the next unreleased period."""
    sessions = pd.DatetimeIndex(sessions)
    cols = {}
    for tname, rows in all_rows.items():
        r = rows[rows["pred"].notna()].sort_values(["snap", "period"])
        if r.empty:
            continue
        snaps = r["snap"].values.astype("datetime64[ns]")
        vals, naive, rels, pers = r["pred"].values, r["naive"].values, r["release"].values.astype("datetime64[ns]"), r["period"].values
        out_v, out_s = np.full(len(sessions), np.nan), np.full(len(sessions), np.nan)
        j = -1
        latest_by_period = {}
        sv = sessions.values.astype("datetime64[ns]")
        for i, d in enumerate(sv):
            while j + 1 < len(snaps) and snaps[j + 1] <= d:
                j += 1
                latest_by_period[pers[j]] = (vals[j], naive[j], rels[j])
            # next unreleased period as of d
            cand = [(rel, v, nv) for per, (v, nv, rel) in latest_by_period.items() if rel > d]
            if cand:
                rel, v, nv = min(cand, key=lambda c: c[0])
                out_v[i], out_s[i] = v, v - nv
            if len(latest_by_period) > 40:
                for per in [p for p, (_, _, rel) in latest_by_period.items() if rel <= d]:
                    latest_by_period.pop(per, None)
        cols[f"NC_{tname}"] = out_v
        cols[f"NC_{tname}_VS_LAST"] = out_s
    return pd.DataFrame(cols, index=sessions)


# ====================================================================== publish (constantly updated)
def _calendar_next(cal_id, after):
    try:
        cal = json.loads(Path(CALENDAR_PATH).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    dates = sorted(e["date"] for e in cal.get("events", []) if e.get("id") == cal_id and e.get("date", "") > str(after.date()))
    return dates[0] if dates else None


def next_periods(D, tname, asof):
    """Unreleased periods that have started by asof (usually 1-2)."""
    t = TARGETS[tname]
    rel = release_dates(D, tname)
    known = rel[rel <= asof]
    last = known.index.max() if len(known) else None
    if last is None:
        return []
    step = pd.offsets.QuarterBegin(1, startingMonth=1) if t["freq"] == "Q" else pd.offsets.MonthBegin(1)
    out, p = [], last + step
    while p <= asof and len(out) < 2:
        out.append(p)
        p = p + step
    return out


def nowcast_one(D, tname, rows, asof, models=None):
    t = TARGETS[tname]
    models = models or fit_models(tname, rows, asof)
    out = []
    for p in next_periods(D, tname, asof):
        f = features(D, tname, p, asof)
        last = f.pop("_last", np.nan)
        row = {**f, "_last": last}
        pred, which, contrib = _apply(tname, models, row)
        naive = _naive(tname, row)
        est_release = FM.availability(pd.DatetimeIndex([p]), t["sid"])[0]
        p_end = FM.period_end(pd.DatetimeIndex([p]), t["freq"])[0]
        cal = _calendar_next(t["calendar"], max(asof, p_end))      # first scheduled release after the period ends
        next_release = pd.Timestamp(cal) if cal else est_release
        days = max(1, int((next_release - asof).days))
        rmse = band_rmse(rows, asof, _bucket(days))
        drivers = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:4]
        out.append({
            "target": tname, "label": t["label"], "period": str(p.date())[:7] if t["freq"] == "M" else f"{p.year}Q{(p.month - 1) // 3 + 1}",
            "nowcast": None if pred != pred else round(pred, 3), "model": which,
            "band80": None if pred != pred or rmse != rmse else [round(pred - BAND_Z * rmse, 3), round(pred + BAND_Z * rmse, 3)],
            "rmseHistory": None if rmse != rmse else round(rmse, 3),
            "lastPrint": None if naive != naive else round(float(naive), 3),
            "impliedSurpriseVsLastPrint": None if pred != pred or naive != naive else round(pred - naive, 3),
            "nextRelease": str(next_release.date()), "nextReleaseSource": "calendar" if cal else "estimated (publication lag)",
            "estimatedRelease": str(est_release.date()), "calendarNextRelease": cal,
            "drivers": [{"feature": k, "label": FEATURE_LABELS.get(k, k), "contribution": round(v, 3),
                         "value": None if row.get(k) != row.get(k) else round(float(row.get(k)), 3)} for k, v in drivers],
        })
    return out


def regime_summary(D, asof, nowcasts):
    s = pd.Timestamp(asof)
    flags = {}
    curve = D.latest("T10Y3M", s)[0]
    flags["curveInverted"] = bool(curve == curve and curve < 0)
    sahm = D.latest("SAHMREALTIME", s)[0]
    flags["sahmTriggered"] = bool(sahm == sahm and sahm >= 0.5)
    nfci = D.latest("NFCI", s)[0]
    flags["financialConditionsTight"] = bool(nfci == nfci and nfci > 0)
    c = D.pct_changes("CPILFESL", s, 3)
    core3 = float(((1 + c / 100).prod() ** 4 - 1) * 100) if len(c) == 3 else np.nan
    flags["coreInflationHot"] = bool(core3 == core3 and core3 > 3.5)
    nc = {n["target"]: n for n in nowcasts}
    pay = (nc.get("PAYROLLS") or {}).get("nowcast")
    gdp = (nc.get("GDP_SAAR") or {}).get("nowcast")
    flags["nowcastPayrollsNegative"] = bool(pay is not None and pay < 0)
    flags["nowcastGdpNegative"] = bool(gdp is not None and gdp < 0)
    growth_bad = sum(flags[k] for k in ("sahmTriggered", "nowcastPayrollsNegative", "nowcastGdpNegative"))
    stress = sum(flags[k] for k in ("curveInverted", "financialConditionsTight"))
    if growth_bad >= 2:
        label = "Contraction risk"
    elif growth_bad == 1 or stress == 2:
        label = "Slowdown"
    elif flags["coreInflationHot"]:
        label = "Expansion, inflation running hot"
    else:
        label = "Expansion"
    return {"label": label, "flags": flags, "values": {"curve10y3m": curve, "sahm": sahm, "nfci": nfci,
                                                       "coreCpi3mAnnualized": None if core3 != core3 else round(core3, 2)}}


def publish(root=None, asof=None, write=True, D=None):
    asof = pd.Timestamp(asof or date.today())
    D = D or Data(root)
    targets = []
    for tname in TARGETS:
        try:
            rows = walk_forward(tname, build_rows(D, tname, start=str((asof - pd.DateOffset(years=40)).date())))
            targets.extend(nowcast_one(D, tname, rows, asof))
        except Exception as exc:  # noqa: BLE001
            targets.append({"target": tname, "error": str(exc)[:200]})
    doc = {"generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "asof": str(asof.date()),
           "method": "Ridge nowcasts from raw data only, refit on releases known by the as-of date; 80% band from "
                     "the model's own past real-time errors at a similar distance to the release.",
           "inputsPolicy": "No outside projections (consensus, GDPNow, NY Fed, Cleveland Fed, SPF) are inputs.",
           "pointInTime": "alfred-first-release" if FM.has_key() else "publication-lag",
           "nowcasts": targets, "regime": regime_summary(D, asof, targets)}
    # ---- outside estimates: attached AFTER our numbers exist, display/comparison only
    try:
        import outside_forecasts as OF
        street = OF.latest_for_display()
        for n in doc["nowcasts"]:
            s = street.get(n.get("target"), {})
            if not s:
                continue
            per = n.get("period")
            # keep only outside numbers for the same period as our nowcast
            mine = {k.split(":")[0]: v for k, v in s.items() if isinstance(v, dict) and v.get("period") == per}
            if not mine:
                continue
            n["outside"] = mine
            ref_key = "consensus" if "consensus" in mine else next(iter(mine))
            ref = mine[ref_key].get("value")
            if ref is not None and n.get("nowcast") is not None:
                n["oursMinusOutside"] = round(n["nowcast"] - ref, 3)
                n["oursMinusOutsideRef"] = ref_key
        doc["outsideNote"] = street.get("_note")
    except Exception as exc:  # noqa: BLE001
        doc["outsideNote"] = f"Outside estimates unavailable ({type(exc).__name__})."
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_atomically(LATEST_PATH, json.dumps(doc, indent=2, default=_json_default).encode("utf-8"))
        with open(HISTORY_PATH, "a", encoding="utf-8") as fh:
            for n in doc["nowcasts"]:
                if n.get("nowcast") is None:
                    continue
                fh.write(json.dumps({"at": doc["generatedAt"], "asof": doc["asof"], "target": n["target"],
                                     "period": n["period"], "nowcast": n["nowcast"], "band80": n.get("band80"),
                                     "lastPrint": n.get("lastPrint"),
                                     "outside": {k: v.get("value") for k, v in (n.get("outside") or {}).items()
                                                 if isinstance(v, dict)}}, default=_json_default) + "\n")
    return doc


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return str(o)[:10]
    return str(o)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ====================================================================== research
def research(root=None, holdout=False, write=True):
    D = Data(root)
    all_rows, acc = {}, {}
    for tname in TARGETS:
        rows = walk_forward(tname, build_rows(D, tname))
        all_rows[tname] = rows
        acc[tname] = {"final": accuracy(tname, rows, end=HOLDOUT_START),
                      "early": accuracy(tname, rows[rows["daysToRelease"] > 21].assign(k=0), end=HOLDOUT_START),
                      "since2000": accuracy(tname, rows, start=pd.Timestamp("2000-01-01"), end=HOLDOUT_START),
                      "exPandemic": accuracy(tname, rows[~rows["period"].between(*EXCLUDE_TRAIN)], end=HOLDOUT_START)}
        if holdout:
            acc[tname]["holdout"] = accuracy(tname, rows, start=HOLDOUT_START, end=HOLDOUT_END + pd.Timedelta(days=1))
    # Holm / BH across targets on Clark-West p-values (pre-holdout, final nowcast)
    names = [n for n in acc if acc[n]["final"].get("clarkWestP") == acc[n]["final"].get("clarkWestP")
             and acc[n]["final"].get("clarkWestP") is not None]
    if names:
        from swarm import holm, bh
        ps = np.array([acc[n]["final"]["clarkWestP"] for n in names])
        for n, h, b in zip(names, holm(ps), bh(ps)):
            acc[n]["final"]["clarkWestHolm"], acc[n]["final"]["clarkWestBH"] = float(h), float(b)
    # Outside comparison (display-only data, compared AFTER the fact)
    outside_cmp = {}
    try:
        import outside_forecasts as OF
        outside_cmp = OF.compare_accuracy(all_rows, end=HOLDOUT_START)
    except Exception as exc:  # noqa: BLE001
        outside_cmp = {"error": str(exc)[:200]}
    sessions = pd.bdate_range("1970-01-01", date.today())
    panel = daily_panel(all_rows, sessions)
    evolution = {}
    for tname, rows in all_rows.items():
        r = rows[rows["pred"].notna()]
        recent = sorted(r["period"].unique())[-18:]
        evolution[tname] = [{"period": str(pd.Timestamp(p).date()), "release": str(r[r["period"] == p]["release"].iloc[0].date()),
                             "actual": _f(r[r["period"] == p]["y"].iloc[0]) if pd.Timestamp(r[r["period"] == p]["release"].iloc[0]) < HOLDOUT_START or holdout else None,
                             "path": [{"date": str(x.snap.date()), "nowcast": _f(x.pred)} for x in r[r["period"] == p].sort_values("snap").itertuples()]}
                            for p in recent]
    doc = {"generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "pointInTime": "alfred-first-release (lag fallback before ALFRED coverage)" if FM.has_key() else "publication-lag",
           "holdoutScored": bool(holdout), "accuracy": acc, "outsideComparison": outside_cmp}
    if holdout:
        _log_holdout({"study": "macro_nowcast", "targets": list(TARGETS)})
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_atomically(ACCURACY_PATH, json.dumps(_clean(doc), indent=2, default=_json_default).encode("utf-8"))
        write_atomically(EVOLUTION_PATH, json.dumps(_clean({"generatedAt": doc["generatedAt"],
                                                            "note": "Backfilled real-time walk-forward paths (each point used only data available that day).",
                                                            "targets": evolution}), default=_json_default).encode("utf-8"))
        write_atomically(PANEL_PATH, panel.to_csv(date_format="%Y-%m-%d", float_format="%.4f").encode("utf-8"))
    return doc, all_rows, panel


def _f(x):
    try:
        x = float(x)
        return None if not math.isfinite(x) else round(x, 4)
    except (TypeError, ValueError):
        return None


def _log_holdout(extra):
    design = hashlib.sha256(json.dumps({"targets": list(TARGETS), "alpha": RIDGE_ALPHA,
                                        "snaps": SNAPSHOTS_PER_PERIOD}, sort_keys=True).encode()).hexdigest()[:16]
    try:
        ledger = json.loads(LEDGER_PATH.read_text()) if LEDGER_PATH.exists() else {"views": 0, "entries": []}
        ledger["entries"].append({"at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "designHash": f"nowcast-{design}",
                                  "study": "macro_nowcast", "holdoutStart": str(HOLDOUT_START.date()),
                                  "dataEnd": str(HOLDOUT_END.date()), "champion": "ridge-nowcast", **(extra or {})})
        ledger["views"] = len({e["designHash"] for e in ledger["entries"] if e.get("holdoutStart") == str(HOLDOUT_START.date())})
        write_atomically(LEDGER_PATH, json.dumps(ledger, indent=2).encode("utf-8"))
    except Exception:  # noqa: BLE001
        pass


def load_daily_panel(path=None):
    """For other models (paper traders, crypto): real-time nowcast panel indexed by session date."""
    p = Path(path or PANEL_PATH)
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.is_file() else None


def load_point_in_time(path=None):
    """Daily real-time nowcast panel with an `asof` column (value known at that session's close).

    Consumed by crypto_research.load_nowcast and paper models. Columns: NC_<TARGET> (nowcast of the next
    unreleased print) and NC_<TARGET>_VS_LAST (nowcast minus last official print)."""
    f = load_daily_panel(path)
    if f is None:
        return None
    f = f.copy()
    f.index.name = "date"
    f = f.reset_index()
    f["asof"] = f["date"]
    return f


def refresh(research_max_age_days=7, log=print):
    """Incremental data update + publish. Used by run-paper-daily.ps1 and the dashboard (spawned detached)."""
    lock = OUT_DIR / "nowcast_refresh.lock"
    out = {"steps": {}}
    try:
        try:
            m = FM.update_all(vintages=FM.has_key(), log=None)
            out["steps"]["fred"] = {"mode": m.get("mode"), "errors": len(m.get("errors") or {})}
        except Exception as exc:  # noqa: BLE001 - publish from cached data anyway
            out["steps"]["fred"] = {"error": str(exc)[:200]}
        try:
            import outside_forecasts as OF
            OF.update_all()
            out["steps"]["outside"] = "ok"
        except Exception as exc:  # noqa: BLE001
            out["steps"]["outside"] = {"error": str(exc)[:200]}
        stale = (not PANEL_PATH.exists() or
                 (datetime.now().timestamp() - PANEL_PATH.stat().st_mtime) > research_max_age_days * 86400)
        if stale:
            try:
                research(holdout=False)
                out["steps"]["research"] = "rebuilt"
            except Exception as exc:  # noqa: BLE001
                out["steps"]["research"] = {"error": str(exc)[:200]}
        doc = publish()
        out["steps"]["publish"] = {"asof": doc["asof"], "n": len(doc["nowcasts"]), "regime": doc["regime"]["label"]}
    finally:
        try:
            lock.unlink()
        except OSError:
            pass
    if log:
        log(json.dumps(out, default=str))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Raw-data macro nowcasts (no outside projections as inputs).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("publish")
    sub.add_parser("refresh")
    r = sub.add_parser("research")
    r.add_argument("--holdout", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "refresh":
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] nowcast refresh")
        refresh()
        return 0
    if a.cmd == "publish":
        doc = publish()
        for n in doc["nowcasts"]:
            print(f"{n.get('target'):14s} {n.get('period')}  nowcast={n.get('nowcast')} band={n.get('band80')} last={n.get('lastPrint')} {n.get('error', '')}")
        print("regime:", doc["regime"]["label"])
        return 0
    doc, _, _ = research(holdout=a.holdout)
    print(json.dumps(_clean(doc["accuracy"]), indent=1, default=_json_default)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
