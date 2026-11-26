"""Short-BTC research with inverse bitcoin ETFs, modelled with US market hours (PAPER / research only).

Instruments (read-only Schwab market data for the real ETFs; no order or preview endpoint is ever called):
  SBIT  ProShares UltraShort Bitcoin ETF: -2x DAILY BTC (Bloomberg Bitcoin Index), launched 2024-04-02, ER 0.95%.
  BITI  ProShares Short Bitcoin ETF: -1x daily via CME futures, launched 2022-06-21, ER 0.95%.
  (Robert described SBIT as -1x; Schwab's reference data and the return regression both say SBIT is -2x and BITI
  is the -1x fund. Both are modelled; short exposure is expressed in "-1x units", so SBIT carries half the capital.)

Why market hours matter. Signals come from 24/7 BTC data at the UTC daily close (7-8 pm ET). An ETF order can only
fill in a US session: we fill at 10:00 ET (30 min after the open, conservative) on the next NYSE session, or at 08:00
ET pre-market for the extended-hours variant (wider cost). Weekends/holidays/overnight BTC moves become gaps on the
ETF. Protective stops (BTC +X% above the entry) are only checked on session hours (10:00-16:00 ET) and fill at that
hour, so an overnight squeeze is taken at the gap. The daily-reset NAV is simulated hour by hour from Coinbase BTC
hourly candles: NAV(t) = NAV(last 4 pm ET reset) * (1 + k * (BTC(t)/BTC(reset) - 1)), reset each session day at
16:00 ET, less the expense ratio and a tracking drag calibrated on the real ETF's history (flagged as synthetic before
the ETF existed). Comparators on the same signals (research only): a hypothetical 24/7 linear short (perpetual-style,
fills at the UTC close, zero funding assumed - optimistic) and a CME micro-futures (/MBT) style short (no weekend
fills: Fri 17:00 - Sun 18:00 ET closed, daily 17-18 ET break).

Discipline: same as crypto_research.py. Pre-holdout window only (data before 2026-07-09): the sealed holdout is NOT
scored and the holdout ledger is untouched (--no-holdout is the only mode). Walk-forward, deflated Sharpe, PBO
(CSCV), White RC / Hansen SPA vs BTC buy & hold and vs each variant's own long-only base, Holm and BH.

  python crypto_short.py data        # fetch BTC/ETH hourly history (Coinbase) and SBIT/BITI daily (Schwab)
  python crypto_short.py run --no-holdout
  python crypto_short.py paper       # forward shadow accounts 220-223 (also run by crypto_paper.py)
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_data as cd
import crypto_research as cr
from fetch_and_model import DATABASE

HOURLY_DIR = cd.CRYPTO_DIR / "hourly_full"
SHORT_DIR = cd.CRYPTO_DIR / "short"
REPORT_JSON = SHORT_DIR / "crypto_short_report.json"
REPORT_MD = SHORT_DIR / "crypto_short_report.md"
RETURNS_CSV = SHORT_DIR / "crypto_short_returns.csv"
PAPER_JSON = SHORT_DIR / "short_paper_status.json"
HOURLY_START = "2016-09-01"
ET = "America/New_York"
SPOT_COST = cr.FEE + cr.SLIP          # Schwab Crypto 0.75% + 5 bp
ETF_COST = 0.0010                     # $0 commission; ~5-10 bp spread + impact per side
ETF_EXT_COST = 0.0025                 # pre-market fills: wider spreads
CME_COST = 0.0003
PERP_COST = 0.0005
ER = 0.0095
ETFS = {"SBIT": {"k": -2.0, "launch": "2024-04-02"}, "BITI": {"k": -1.0, "launch": "2022-06-21"}}
FORWARD_START = "2026-11-25"
CODE_VERSION = "crypto-short-1"


def log(msg):
    print(f"[crypto_short] {msg}", file=sys.stderr, flush=True)


# ====================================================================== data
def update_hourly(coins=("BTC", "ETH")):
    """Coinbase hourly candles since HOURLY_START (incremental; ~300 requests for a first full pull per coin)."""
    import crypto_chart as cc
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for coin in coins:
        path = HOURLY_DIR / f"{coin}.csv"
        old = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        start = datetime.fromisoformat(HOURLY_START).replace(tzinfo=timezone.utc)
        if len(old):
            start = datetime.fromtimestamp(int(old["ts"].max()), timezone.utc) - timedelta(hours=3)
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        new = cc._fetch_hourly(coin, start, end)
        frame = pd.concat([old, new]).drop_duplicates("ts", keep="last").sort_values("ts")
        tmp = path.with_suffix(".tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(path)
        out[coin] = {"rows": int(len(frame)), "first": str(pd.to_datetime(frame["ts"].min(), unit="s")),
                     "last": str(pd.to_datetime(frame["ts"].max(), unit="s"))}
        log(f"{coin} hourly: {out[coin]}")
    return out


def update_etfs():
    import schwab_ondemand as od
    return {s: od.fetch(s, force=False, intraday_days=0) for s in ETFS}


def load_hourly(coins=("BTC", "ETH")):
    """Hourly close panel indexed by the candle CLOSE time (UTC, naive)."""
    cols = {}
    for coin in coins:
        f = pd.read_csv(HOURLY_DIR / f"{coin}.csv")
        s = pd.Series(f["close"].to_numpy(float), index=pd.to_datetime(f["ts"].astype("int64") + 3600, unit="s"))
        cols[coin] = s[~s.index.duplicated(keep="last")]
    P = pd.DataFrame(cols).sort_index()
    grid = pd.date_range(P.index[0].ceil("h"), P.index[-1], freq="h")
    # Coinbase has a few multi-hour outages; carry the last price (no trading inside a gap is assumed)
    return P.reindex(grid).ffill()


def session_dates():
    """NYSE session dates from SPY daily bars (Schwab daily_long since 2006 plus Massive daily_prices)."""
    con = sqlite3.connect(Path(DATABASE).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        d = {r[0] for r in con.execute("SELECT date FROM daily_prices WHERE ticker='SPY'")}
        try:
            d |= {r[0] for r in con.execute("SELECT date FROM daily_long WHERE ticker='SPY'")}
        except sqlite3.Error:
            pass
    finally:
        con.close()
    return pd.DatetimeIndex(sorted(d))


def etf_daily(symbol):
    con = sqlite3.connect(Path(DATABASE).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT date, close FROM daily_long WHERE ticker=? AND source='schwab' ORDER BY date", (symbol,)).fetchall()
    finally:
        con.close()
    return pd.Series({pd.Timestamp(d): float(c) for d, c in rows}, dtype=float)


def _et_to_utc_naive(dates, hour):
    ts = pd.DatetimeIndex(dates) + pd.Timedelta(hours=hour)
    return ts.tz_localize(ET, nonexistent="shift_forward", ambiguous=False).tz_convert("UTC").tz_localize(None)


def calendar_masks(grid, sessions):
    """Boolean masks over the hourly grid: ETF fill hours (regular / extended), 4 pm resets, CME open hours."""
    s = sessions[(sessions >= grid[0].normalize() - pd.Timedelta(days=3)) & (sessions <= grid[-1] + pd.Timedelta(days=3))]
    reg = set()
    for h in range(10, 17):
        reg |= set(_et_to_utc_naive(s, h))
    ext = set(reg)
    for h in (8, 9):
        ext |= set(_et_to_utc_naive(s, h))
    reset = set(_et_to_utc_naive(s, 16))
    m_reg = grid.isin(list(reg))
    m_ext = grid.isin(list(ext))
    m_reset = grid.isin(list(reset))
    et = grid.tz_localize("UTC").tz_convert(ET)
    wd, hr = et.weekday, et.hour
    closed = (wd == 5) | ((wd == 4) & (hr >= 17)) | ((wd == 6) & (hr < 18)) | (hr == 17)
    m_cme = ~closed
    return {"etf": np.asarray(m_reg), "etf_ext": np.asarray(m_ext), "reset": np.asarray(m_reset),
            "cme": np.asarray(m_cme), "perp": np.ones(len(grid), bool)}


def synthetic_nav(btc, reset_mask, k, annual_cost):
    """Hourly NAV of a daily-reset k-x fund on BTC, reset at each 4 pm ET session close, less annual_cost."""
    b = btc.to_numpy(float)
    nav = np.empty_like(b)
    ref_b, ref_nav = b[0], 1.0
    daily = annual_cost / 252.0
    for i in range(len(b)):
        x = ref_nav * (1.0 + k * (b[i] / ref_b - 1.0))
        x = max(x, 1e-9)
        nav[i] = x
        if reset_mask[i]:
            ref_nav, ref_b = x * (1.0 - daily), b[i]
    return pd.Series(nav, index=btc.index)


def calibrate(btc, masks, sessions):
    """Compare synthetic 4pm-to-4pm returns with the real ETF closes: beta, tracking error, extra drag per year."""
    out = {}
    for sym, meta in ETFS.items():
        real = etf_daily(sym)
        if len(real) < 60:
            out[sym] = {"available": False}
            continue
        nav = synthetic_nav(btc, masks["reset"], meta["k"], ER)
        at = nav[masks["reset"]]
        at.index = at.index.tz_localize("UTC").tz_convert(ET).tz_localize(None).normalize()
        rs, ss = real.pct_change(), at.pct_change()
        j = pd.concat([rs, ss], axis=1, keys=["real", "syn"]).dropna()
        j = j[(j["real"].abs() < 0.5) & (j.index >= pd.Timestamp(meta["launch"]) + pd.Timedelta(days=5))]
        if len(j) < 60:
            out[sym] = {"available": False}
            continue
        btc_d = btc[masks["reset"]]
        btc_d.index = btc_d.index.tz_localize("UTC").tz_convert(ET).tz_localize(None).normalize()
        bj = pd.concat([rs, btc_d.pct_change()], axis=1, keys=["real", "btc"]).dropna().loc[j.index]
        beta_btc = float(np.cov(bj["real"], bj["btc"])[0, 1] / np.var(bj["btc"], ddof=1))
        resid = j["real"] - j["syn"]
        # distributions (unadjusted Schwab closes) and data gaps show up as single-day outliers: trim |resid| > 5%
        trimmed = resid[resid.abs() <= 0.05]
        measured = float(trimmed.mean() * 252)
        out[sym] = {"available": True, "days": int(len(j)), "first": str(j.index[0].date()), "last": str(j.index[-1].date()),
                    "betaToBtc": beta_btc, "betaToSynthetic": float(np.cov(j["real"], j["syn"])[0, 1] / np.var(j["syn"], ddof=1)),
                    "trackingErrorAnnual": float(trimmed.std(ddof=1) * math.sqrt(252)), "outlierDays": int((resid.abs() > 0.05).sum()),
                    "realMinusSyntheticPerYear": measured,
                    # conservative: never credit the synthetic with the measured outperformance
                    "extraDragPerYear": max(0.0, -measured),
                    "corr": float(j.corr().iloc[0, 1])}
    return out


# ====================================================================== signals (daily, causal)
def daily_signals(close, drivers):
    """Daily features at each UTC close: BTC trend, TSMOM, RSI, z-score, exhaustion, dollar / real-yield, composite."""
    import indicators
    btc = close["BTC"]
    f = pd.DataFrame(index=close.index)
    for n in (100, 200):
        f[f"below{n}"] = btc.lt(btc.rolling(n).mean())
    moms = [(btc / btc.shift(L) - 1) for L in (20, 60, 120)]
    f["tsmom_btc"] = sum(m.gt(0).astype(float) for m in moms) / 3
    f["allneg"] = sum(m.lt(0).astype(int) for m in moms).eq(3)
    f["rsi14"] = indicators.rsi(btc, 14)
    f["z20"] = indicators.zscore(btc, 20)
    try:
        import exhaustion as ex
        daily = cd.read_csv(cd.SPOT_DIR / "BTC.csv")
        raw = ex.features_for_frame(daily.reindex(close.index))
        f["exh_up"] = raw["flag_up"].fillna(0).gt(0.5)
    except Exception:   # noqa: BLE001 - optional feature
        f["exh_up"] = False
    u = drivers.get("UUP")
    r = drivers.get("DFII10")
    f["dollar_up"] = (u / u.shift(90) - 1).gt(0) if u is not None else False
    f["real_up"] = (r - r.shift(30)).gt(0) if r is not None else False
    comp = cr.overlay_multiplier({"type": "composite", "floor": 0.25}, drivers, close.index)
    f["riskoff"] = comp.lt(0.99)
    return f


def long_weights(name, close, drivers):
    if name == "cash":
        return pd.DataFrame(0.0, index=close.index, columns=["BTC", "ETH"])
    spec = {"btc_tsmom": {"base": "tsmom", "universe": "btc", "lookbacks": [20, 60, 120], "weighting": "ew"},
            "btceth_tsmom": {"base": "tsmom", "universe": "btc_eth", "lookbacks": [20, 60, 120], "weighting": "invvol"}}[name]
    W = cr.target_weights(spec, close[["BTC", "ETH"]], drivers)
    return W.reindex(columns=["BTC", "ETH"], fill_value=0.0).fillna(0.0)


def short_signal(rule, f):
    t = rule["type"]
    if t == "trend":
        s = f[f"below{rule['sma']}"] & f["allneg"]
    elif t == "trend_macro":
        s = f["below200"] & f["allneg"] & (f["dollar_up"] | f["real_up"])
    elif t == "trend_riskoff":
        s = f["below200"] & f["allneg"] & f["riskoff"]
    elif t in ("ob", "ob_any", "ob_exh"):
        if t == "ob_exh":
            trig = f["exh_up"] & f["below200"]
        else:
            trig = f["rsi14"].gt(rule["rsi"]) | f["z20"].gt(2.0)
            if t == "ob":
                trig = trig & f["below200"]
        s = trig.astype(float).rolling(rule["hold"], min_periods=1).max().gt(0)
    else:
        raise ValueError(t)
    return s.fillna(False).astype(bool)


SHORT_RULES = [
    {"type": "trend", "sma": 200}, {"type": "trend", "sma": 100},
    {"type": "trend_macro"}, {"type": "trend_riskoff"},
    {"type": "ob", "rsi": 70, "hold": 5}, {"type": "ob", "rsi": 70, "hold": 10},
    {"type": "ob", "rsi": 80, "hold": 5}, {"type": "ob", "rsi": 80, "hold": 10},
    {"type": "ob_exh", "hold": 5}, {"type": "ob_any", "rsi": 85, "hold": 5},
]
LONGS = ["cash", "btc_tsmom", "btceth_tsmom"]


def rule_label(rule):
    t = rule["type"]
    return {"trend": f"downtrend (BTC<SMA{rule.get('sma')} & 20/60/120d mom<0)",
            "trend_macro": "downtrend + dollar or real yield rising",
            "trend_riskoff": "downtrend + macro composite risk-off",
            "ob": f"overbought in downtrend (RSI>{rule.get('rsi')} or z20>2, hold {rule.get('hold')}d)",
            "ob_exh": f"exhaustion flag in downtrend (hold {rule.get('hold')}d)",
            "ob_any": f"overbought fade (RSI>{rule.get('rsi')} or z20>2, hold {rule.get('hold')}d)"}[t]


def make_design():
    design = []
    for long in LONGS:
        for rule in SHORT_RULES:
            for w in (0.5, 1.0):
                for stop in (None, 0.10):
                    design.append({"long": long, "short": rule, "w": w, "stop": stop,
                                   "label": f"{long} | short {w:g}x when {rule_label(rule)}{' | stop +10%' if stop else ''}"})
    for i, s in enumerate(design):
        s["id"] = i + 1 + len(LONGS)      # 0 = BTC hold, 1..3 = long-only bases
    return design


# ====================================================================== simulator (hourly grid, vectorised over variants)
def simulate(P, decide_idx, WL, ES, stops, instrument, masks, nav=None, k=-1.0, start_i=0, end_i=None):
    """P: hourly prices (BTC, ETH). decide_idx[d]: grid index of day d's UTC close; WL (D x V x 2) long weights and
    ES (D x V) short exposure (in -1x units) decided at that close. Returns daily equity (D x V) at the decision hours.
    instrument: etf | etf_ext | cme | perp. ETF value follows `nav` (k-x daily reset); others are linear shorts."""
    end_i = len(P) if end_i is None else end_i
    B = P["BTC"].to_numpy(float)
    PX = P[["BTC", "ETH"]].to_numpy(float)
    N = nav.to_numpy(float) if nav is not None else None
    D, V = ES.shape
    cash = np.full(V, cr.INITIAL)
    qL = np.zeros((V, 2))
    u = np.zeros(V)            # ETF units (etf) or BTC units short (linear)
    entry = np.full(V, np.nan)
    pend = np.full(V, np.nan)  # pending short exposure target
    lock = np.zeros(V, bool)
    etf = instrument.startswith("etf")
    allowed = masks[{"etf": "etf", "etf_ext": "etf_ext", "etf247": "perp", "cme": "cme", "perp": "perp"}[instrument]]
    cost = {"etf": ETF_COST, "etf_ext": ETF_EXT_COST, "etf247": ETF_COST, "cme": CME_COST, "perp": PERP_COST}[instrument]
    stop_arr = np.array([np.nan if s is None else s for s in stops], float)
    eq_out = np.full((D, V), np.nan)
    day_at = {int(i): d for d, i in enumerate(decide_idx)}
    traded_short = np.zeros(V)

    def equity(i):
        ev = cash + qL @ PX[i]
        return ev + (u * N[i] if etf else -u * B[i])

    for i in range(start_i, end_i):
        if not np.isfinite(B[i]):
            continue
        d = day_at.get(i)
        if d is not None:
            eq = equity(i)
            eq_out[d] = eq
            # long leg: spot BTC/ETH trades 24/7 at the UTC close (Schwab Crypto cost), 5% band
            tgt = WL[d] * (ES[d] <= 0)[:, None]
            cur = (qL * PX[i]) / np.where(eq > 0, eq, np.nan)[:, None]
            diff = np.nan_to_num(tgt - cur)
            mask = (np.abs(diff) > 0.05) | ((tgt <= 1e-9) & (qL > 0))
            dv = np.where(mask, diff * eq[:, None], 0.0)
            dv = np.where((tgt <= 1e-9) & (qL > 0), -qL * PX[i], dv)
            dv[np.abs(dv) < cr.MIN_TRADE] = 0.0
            qL += dv / PX[i]
            cash -= dv.sum(axis=1) + np.abs(dv).sum(axis=1) * SPOT_COST
            # short leg: new target becomes a pending order for the next allowed fill
            lock &= ES[d] > 0
            pend = np.where(lock, 0.0, ES[d])
        # stops: BTC +X% over the short entry, only seen on allowed hours
        if allowed[i]:
            hold = (u > 0) & np.isfinite(stop_arr)
            if hold.any():
                hit = hold & (B[i] >= entry * (1 + stop_arr))
                if hit.any():
                    pend = np.where(hit, 0.0, pend)
                    lock |= hit
            todo = np.isfinite(pend)
            if todo.any():
                eq = equity(i)
                price = N[i] if etf else B[i]
                cur_x = (u * N[i]) * abs(k) / np.where(eq > 0, eq, np.nan) if etf else (u * B[i]) / np.where(eq > 0, eq, np.nan)
                cur_x = np.nan_to_num(cur_x)
                change = todo & ((np.abs(pend - cur_x) > 0.05) | ((pend <= 1e-9) & (u > 0)) | ((pend > 0) & (u <= 0)))
                if change.any():
                    tgt_units = np.where(pend > 0, pend * eq / (abs(k) if etf else 1.0) / price, 0.0)
                    du = np.where(change, tgt_units - u, 0.0)
                    notional = np.abs(du) * price
                    if etf:
                        cash -= du * price + notional * cost
                    else:
                        cash += du * price - notional * cost      # selling short adds proceeds
                    opened = change & (u <= 0) & (tgt_units > 0)
                    entry = np.where(opened, B[i], np.where(tgt_units <= 0, np.nan, entry))
                    u = np.where(change, tgt_units, u)
                    traded_short += notional / np.maximum(eq, 1e-9)
                pend = np.where(todo, np.nan, pend)
    return eq_out, traded_short


def daily_returns(eq, dates):
    E = pd.DataFrame(eq, index=dates)
    R = E.pct_change()
    R.iloc[0] = E.iloc[0] / cr.INITIAL - 1
    return R


# ====================================================================== run
def prepare(end=None):
    close = cd.load_spot(["BTC", "ETH"])
    P = load_hourly()
    drivers = cd.load_drivers(close.index)
    sessions = session_dates()
    masks = calendar_masks(P.index, sessions)
    # decision hour for UTC day d = 00:00 UTC of d+1 (close of the daily candle)
    dates = close.index[(close.index + pd.Timedelta(days=1) >= P.index[0]) & (close.index + pd.Timedelta(days=1) <= P.index[-1])]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    decide = P.index.get_indexer(dates + pd.Timedelta(days=1))
    ok = decide >= 0
    return close, P, drivers, masks, dates[ok], decide[ok]


def build_inputs(design, close, drivers, dates):
    f = daily_signals(close, drivers).reindex(dates)
    longs = {n: long_weights(n, close, drivers).reindex(dates).fillna(0.0) for n in LONGS}
    V = len(design)
    WL = np.zeros((len(dates), V, 2))
    ES = np.zeros((len(dates), V))
    for j, s in enumerate(design):
        WL[:, j, :] = longs[s["long"]].to_numpy()
        if s.get("short"):
            ES[:, j] = short_signal(s["short"], f).to_numpy(float) * s["w"]
    return WL, ES, f


def run(look_holdout=False):
    if look_holdout:
        raise SystemExit("crypto_short never scores the sealed holdout (use --no-holdout).")
    t0 = time.time()
    SHORT_DIR.mkdir(parents=True, exist_ok=True)
    pre_end = pd.Timestamp(cr.HOLDOUT_START) - pd.Timedelta(days=1)
    close, P, drivers, masks, dates, decide = prepare(end=pre_end)
    start = pd.Timestamp(cr.EVAL_START)
    keep = dates >= start - pd.Timedelta(days=1)
    dates, decide = dates[keep], decide[keep]
    design = make_design()
    bases = [{"id": 1 + i, "long": n, "short": None, "w": 0.0, "stop": None, "label": f"{n} long-only base (no short)"}
             for i, n in enumerate(LONGS)]
    hold = {"id": 0, "long": "hold", "label": "BTC buy & hold"}
    all_specs = bases + design
    WL, ES, feats = build_inputs(all_specs, close, drivers, dates)
    calib = calibrate(P["BTC"], masks, session_dates())
    drag = {s: (calib.get(s, {}).get("extraDragPerYear") or 0.0) if calib.get(s, {}).get("available") else 0.0 for s in ETFS}
    drag = {s: float(np.clip(v, -0.05, 0.10)) for s, v in drag.items()}
    navs = {s: synthetic_nav(P["BTC"], masks["reset"], ETFS[s]["k"], ER + drag[s]) for s in ETFS}
    stops = [s.get("stop") for s in all_specs]
    runs = {}
    for name, inst, nav, k in (("SBIT_hours", "etf", navs["SBIT"], -2.0), ("BITI_hours", "etf", navs["BITI"], -1.0),
                               ("SBIT_ext", "etf_ext", navs["SBIT"], -2.0), ("CME_hours", "cme", None, -1.0),
                               ("SBIT_if_24x7", "etf247", navs["SBIT"], -2.0), ("BITI_if_24x7", "etf247", navs["BITI"], -1.0),
                               ("perp_24x7", "perp", None, -1.0)):
        eq, turn = simulate(P, decide, WL, ES, stops, inst, masks, nav=nav, k=k, start_i=int(decide[0]))
        runs[name] = daily_returns(eq, dates)
        log(f"{name}: simulated {len(all_specs)} variants")
    btc = close["BTC"].reindex(dates)
    hold_r = btc.pct_change().fillna(0.0)
    hold_r.iloc[0] = -SPOT_COST
    main = runs["SBIT_hours"].copy()
    main.columns = [s["id"] for s in all_specs]
    main.insert(0, 0, hold_r.to_numpy())
    main = main.fillna(0.0)
    pre = main.loc[start:pre_end]
    pre.to_csv(RETURNS_CSV, index_label="date")
    ids = list(pre.columns)
    var_ids = [s["id"] for s in design]
    ranked, scores = cr.select(pre, ids)
    champion = ranked[0]
    M = pre.to_numpy()
    trial_sr = np.array([scores[i] for i in ids])
    eff = cr.effective_trials(M)
    dsr = cr.deflated(pre[champion].values, len(ids), trial_sr)
    dsr_eff = cr.deflated(pre[champion].values, eff, trial_sr)
    pbo_res = cr.pbo(M, 12)
    d_btc = pre[var_ids].to_numpy() - pre[[0]].to_numpy()
    spa_btc, p_btc = cr.spa_test(d_btc, seed=21)
    base_of = {s["id"]: 1 + LONGS.index(s["long"]) for s in design}
    d_base = np.column_stack([pre[v].values - pre[base_of[v]].values for v in var_ids])
    spa_base, p_base = cr.spa_test(d_base, seed=23)
    h_btc, b_btc = cr.holm(p_btc), cr.bh(p_btc)
    h_base, b_base = cr.holm(p_base), cr.bh(p_base)
    folds, stitched = cr.walk_forward(pre, ids)
    # market-hours cost: same signals, 24/7 linear short vs ETF hours (all -1x units)
    hours = {}
    for name, R in runs.items():
        Rp = R.loc[start:pre_end]
        sh = [cr.sharpe(Rp[j]) for j in range(len(bases), len(all_specs))]
        cg = [cr.perf(Rp[j])["cagr"] for j in range(len(bases), len(all_specs))]
        hours[name] = {"medianSharpe": float(np.nanmedian(sh)), "medianCagr": float(np.nanmedian(cg))}
    per_variant = []
    by_id = {s["id"]: s for s in all_specs}
    col = {s["id"]: j for j, s in enumerate(all_specs)}
    for n, v in enumerate(var_ids):
        pv = cr.perf(pre[v])
        alt = {name: cr.perf(R.loc[start:pre_end][col[v]]) for name, R in runs.items() if name != "SBIT_hours"}
        frac_short = float((ES[:, col[v]] > 0).mean())
        per_variant.append({"id": v, "label": by_id[v]["label"], "sharpe": pv.get("sharpe"), "cagr": pv.get("cagr"),
                            "maxDrawdown": pv.get("maxDrawdown"), "shortDaysFrac": frac_short,
                            "baseSharpe": cr.sharpe(pre[base_of[v]]), "pVsBtc": float(p_btc[n]), "pHolmVsBtc": float(h_btc[n]),
                            "pBHVsBtc": float(b_btc[n]), "pVsBase": float(p_base[n]), "pHolmVsBase": float(h_base[n]),
                            "pBHVsBase": float(b_base[n]),
                            "perp24x7Sharpe": alt["perp_24x7"].get("sharpe"), "cmeSharpe": alt["CME_hours"].get("sharpe"),
                            "bitiSharpe": alt["BITI_hours"].get("sharpe"), "sbitExtSharpe": alt["SBIT_ext"].get("sharpe"),
                            "perp24x7Cagr": alt["perp_24x7"].get("cagr"),
                            "sbitIf24x7Sharpe": alt["SBIT_if_24x7"].get("sharpe"), "sbitIf24x7Cagr": alt["SBIT_if_24x7"].get("cagr"),
                            "spec": {k: by_id[v][k] for k in ("long", "short", "w", "stop")}})
    per_variant.sort(key=lambda r: -(r["sharpe"] if r["sharpe"] == r["sharpe"] else -9))
    ok = lambda *x: all(v is not None and v == v for v in x)   # noqa: E731
    gap = [r["perp24x7Sharpe"] - r["sharpe"] for r in per_variant if ok(r["sharpe"], r["perp24x7Sharpe"])]
    gap_cagr = [r["perp24x7Cagr"] - r["cagr"] for r in per_variant if ok(r["cagr"], r["perp24x7Cagr"])]
    # pure hours effect: the same daily-reset SBIT, but fillable at any hour (hypothetical) vs real US hours
    hgap = [r["sbitIf24x7Sharpe"] - r["sharpe"] for r in per_variant if ok(r["sharpe"], r["sbitIf24x7Sharpe"])]
    hgap_cagr = [r["sbitIf24x7Cagr"] - r["cagr"] for r in per_variant if ok(r["cagr"], r["sbitIf24x7Cagr"])]
    short_only = [r for r in per_variant if r["spec"]["long"] == "cash"]
    so = lambda key: float(np.nanmedian([r[key] for r in short_only if r[key] is not None]))   # noqa: E731
    rep = {"generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"), "codeVersion": CODE_VERSION,
           "window": {"start": str(pre.index[0].date()), "end": str(pre.index[-1].date()), "days": int(len(pre))},
           "holdout": {"start": cr.HOLDOUT_START, "end": cr.HOLDOUT_END, "scored": False,
                       "note": "Sealed holdout not scored; holdout_ledger.json untouched."},
           "instruments": {"SBIT": "ProShares UltraShort Bitcoin ETF, -2x daily, launched 2024-04-02, ER 0.95%",
                           "BITI": "ProShares Short Bitcoin ETF, -1x daily (futures-based), launched 2022-06-21, ER 0.95%"},
           "modelling": {"etfFill": "next NYSE session, 10:00 ET hourly price (30 min after the open)",
                         "etfExtFill": "next session pre-market 08:00 ET, 25 bp per side",
                         "etfCostPerSide": ETF_COST, "spotCostPerSide": SPOT_COST, "cmeCostPerSide": CME_COST,
                         "perpCostPerSide": PERP_COST, "perpFunding": "0 (optimistic)",
                         "stops": "BTC +10% over the short entry, checked on allowed hours only (ETF: 10:00-16:00 ET)",
                         "syntheticBefore": {s: ETFS[s]["launch"] for s in ETFS}, "dragAddedPerYear": drag},
           "calibration": calib, "variants": len(design), "bases": len(bases),
           "champion": {"id": int(champion), "label": (by_id.get(champion) or hold)["label"], "sharpe": scores[champion]},
           "deflatedSharpe": dsr, "deflatedSharpeEffTrials": {**dsr_eff, "effectiveTrials": eff}, "pbo": pbo_res,
           "spaVsBtc": spa_btc, "spaVsOwnBase": spa_base,
           "passHolmVsBtc": int(sum(1 for x in h_btc if x < 0.05)), "passBHVsBtc": int(sum(1 for x in b_btc if x < 0.05)),
           "passHolmVsBase": int(sum(1 for x in h_base if x < 0.05)), "passBHVsBase": int(sum(1 for x in b_base if x < 0.05)),
           "walkForward": {"folds": folds, "stitched": cr.perf(stitched),
                           "btcSameDays": cr.perf(pre[0].reindex(stitched.index)) if len(stitched) else None},
           "btcHold": cr.perf(pre[0]), "bases": {s["label"]: cr.perf(pre[s["id"]]) for s in bases},
           "hoursCost": {"byInstrument": hours, "medianSharpeGap24x7MinusSbitHours": float(np.median(gap)) if gap else None,
                         "medianCagrGap24x7MinusSbitHours": float(np.median(gap_cagr)) if gap_cagr else None,
                         "medianSharpeGapSbitAnyHourMinusSbitHours": float(np.median(hgap)) if hgap else None,
                         "medianCagrGapSbitAnyHourMinusSbitHours": float(np.median(hgap_cagr)) if hgap_cagr else None,
                         "shortOnlyMedianSharpe": {"SBIT_hours": so("sharpe"), "SBIT_if_24x7": so("sbitIf24x7Sharpe"),
                                                   "CME_hours": so("cmeSharpe"), "perp_24x7": so("perp24x7Sharpe")}},
           "top": per_variant[:25], "all": per_variant, "seconds": round(time.time() - t0, 1)}
    cd.write_json(REPORT_JSON, rep)
    REPORT_MD.write_text(markdown(rep), encoding="utf-8")
    log(f"done in {rep['seconds']} s; champion {champion}")
    return rep


def _p(x, d=1):
    return "–" if x is None or x != x else f"{x * 100:.{d}f}%"


def _f(x, d=2):
    return "–" if x is None or x != x else f"{x:.{d}f}"


def markdown(rep):
    L = ["# Short-BTC with inverse ETFs (SBIT / BITI): market-hours backtest", "",
         f"Generated {rep['generatedAt']}. Pre-holdout window {rep['window']['start']} to {rep['window']['end']} "
         f"({rep['window']['days']} days). **The sealed holdout ({rep['holdout']['start']} to {rep['holdout']['end']}) "
         "was not scored.** Research and paper only; no orders or previews.", "",
         "**Instrument check:** SBIT is ProShares *UltraShort* Bitcoin (-2x daily); BITI is the -1x fund. Short exposure "
         "below is in -1x units (SBIT carries half the capital). History before each launch is synthetic "
         "(daily-reset NAV from Coinbase BTC hourly, 0.95% ER plus calibrated drag) and flagged as such.", "",
         "## Synthetic vs real ETF", "", "| ETF | days | beta to BTC | corr synth | tracking error (trimmed) | real minus synthetic/yr | outlier days |", "|---|---|---|---|---|---|---|"]
    for s, c in rep["calibration"].items():
        if c.get("available"):
            L.append(f"| {s} | {c['days']} | {_f(c['betaToBtc'])} | {_f(c['corr'], 3)} | {_p(c['trackingErrorAnnual'])} | {_p(c['realMinusSyntheticPerYear'])} | {c['outlierDays']} |")
    L += ["", "Outlier days (|real - synthetic| > 5%) are distributions in unadjusted closes or data gaps; the synthetic is never credited with "
          "the real fund's measured outperformance (conservative)."]
    hc = rep["hoursCost"]
    L += ["", "## Cost of US market hours (same signals, all variants)", "",
          "| instrument | median Sharpe | median CAGR |", "|---|---|---|"]
    for n, v in hc["byInstrument"].items():
        L.append(f"| {n} | {_f(v['medianSharpe'])} | {_p(v['medianCagr'])} |")
    so = hc.get("shortOnlyMedianSharpe", {})
    L += ["", f"- Pure market-hours effect (same daily-reset SBIT, fillable at any hour minus real US hours): median Sharpe gap "
          f"**{_f(hc.get('medianSharpeGapSbitAnyHourMinusSbitHours'))}**, CAGR gap **{_p(hc.get('medianCagrGapSbitAnyHourMinusSbitHours'))}**/yr.",
          f"- Linear 24/7 short (perpetual-style, zero funding: optimistic) minus SBIT market hours: Sharpe gap "
          f"**{_f(hc['medianSharpeGap24x7MinusSbitHours'])}**, CAGR gap **{_p(hc['medianCagrGap24x7MinusSbitHours'])}**/yr. "
          "A negative gap means the ETF did better: a daily-reset inverse fund shrinks its dollar short as BTC squeezes higher, "
          "while a linear short grows, and the delayed fill often entered after the first leg of an overbought spike.",
          f"- Short-only variants (cash otherwise), median Sharpe: SBIT hours {_f(so.get('SBIT_hours'))}, SBIT any-hour {_f(so.get('SBIT_if_24x7'))}, "
          f"CME hours {_f(so.get('CME_hours'))}, perp 24/7 {_f(so.get('perp_24x7'))}.", "",
          "## Significance (SBIT market-hours family)", "",
          f"- Variants: {rep['variants']} plus {len(rep['bases'])} long-only bases and BTC buy & hold "
          f"(Sharpe {_f(rep['btcHold'].get('sharpe'))}, maxDD {_p(rep['btcHold'].get('maxDrawdown'))}).",
          f"- In-sample champion #{rep['champion']['id']}: {rep['champion']['label']} (Sharpe {_f(rep['champion']['sharpe'])}).",
          f"- Deflated Sharpe probability {_f(rep['deflatedSharpe'].get('dsrProbability'), 3)} "
          f"({_f(rep['deflatedSharpeEffTrials'].get('dsrProbability'), 3)} with {rep['deflatedSharpeEffTrials'].get('effectiveTrials')} effective trials).",
          f"- PBO (CSCV) {_f(rep['pbo'].get('pbo'), 2)}.",
          f"- SPA vs BTC hold p = {_f(rep['spaVsBtc']['spaP'], 3)} (RC {_f(rep['spaVsBtc']['realityCheckP'], 3)}); "
          f"Holm passes {rep['passHolmVsBtc']}, BH passes {rep['passBHVsBtc']}.",
          f"- SPA vs each variant's own long-only base (does the short sleeve add value?) p = {_f(rep['spaVsOwnBase']['spaP'], 3)}; "
          f"Holm passes {rep['passHolmVsBase']}, BH passes {rep['passBHVsBase']}.",
          f"- Walk-forward stitched Sharpe {_f(rep['walkForward']['stitched'].get('sharpe'))} vs BTC same days "
          f"{_f((rep['walkForward']['btcSameDays'] or {}).get('sharpe'))}.", "",
          "## Top variants (pre-holdout, SBIT market hours)", "",
          "| id | variant | Sharpe | CAGR | maxDD | short days | base Sharpe | 24/7 Sharpe | CME Sharpe | BITI Sharpe | p vs base (BH) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["top"][:15]:
        L.append(f"| {r['id']} | {r['label']} | {_f(r['sharpe'])} | {_p(r['cagr'])} | {_p(r['maxDrawdown'])} | {_p(r['shortDaysFrac'], 0)} | "
                 f"{_f(r['baseSharpe'])} | {_f(r['perp24x7Sharpe'])} | {_f(r['cmeSharpe'])} | {_f(r['bitiSharpe'])} | {_f(r['pBHVsBase'], 3)} |")
    L += ["", "Bases: " + "; ".join(f"{k}: Sharpe {_f(v.get('sharpe'))}, maxDD {_p(v.get('maxDrawdown'))}" for k, v in rep["bases"].items()), ""]
    return "\n".join(L)


# ====================================================================== paper shadow accounts 220-223
PAPER_ACCOUNTS = [
    (220, "SHADOW: cash or SBIT 0.5x when BTC downtrend (SMA200 + 20/60/120d mom < 0), stop +10%",
     {"long": "cash", "short": {"type": "trend", "sma": 200}, "w": 0.5, "stop": 0.10}),
    (221, "SHADOW: BTC TSMOM long, SBIT 0.5x on overbought rallies in a downtrend (RSI>70/z>2, 5d), stop +10%",
     {"long": "btc_tsmom", "short": {"type": "ob", "rsi": 70, "hold": 5}, "w": 0.5, "stop": 0.10}),
    (222, "SHADOW: BTC/ETH TSMOM long, SBIT 0.5x in downtrend with dollar or real yields rising",
     {"long": "btceth_tsmom", "short": {"type": "trend_macro"}, "w": 0.5, "stop": None}),
    (223, "SHADOW: BTC/ETH TSMOM long, SBIT 1x in downtrend + macro composite risk-off, stop +10%",
     {"long": "btceth_tsmom", "short": {"type": "trend_riskoff"}, "w": 1.0, "stop": 0.10}),
]


def paper(write=True):
    """Forward shadow accounts from FORWARD_START, SBIT at market hours. Deterministic replay; paper only."""
    try:
        update_hourly()          # incremental: one Coinbase request per coin
    except Exception as error:   # noqa: BLE001 - use the cached candles
        log(f"hourly update skipped: {type(error).__name__}")
    close, P, drivers, masks, dates, decide = prepare()
    keep = dates >= pd.Timestamp(FORWARD_START)
    specs = [dict(s, id=i, label=l) for i, l, s in PAPER_ACCOUNTS]
    out = {"updatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"), "forwardStart": FORWARD_START, "accounts": [],
           "note": "SHADOW EXPERIMENTS, paper only, chosen regardless of significance. SBIT fills only in US market hours "
                   "(next session 10:00 ET); stops checked 10:00-16:00 ET. Synthetic SBIT NAV from BTC hourly."}
    if keep.sum() == 0:
        out["status"] = "awaiting"
        out["accounts"] = [{"id": s["id"], "label": s["label"], "equity": cr.INITIAL, "totalReturn": 0.0, "shadow": True,
                            "instrument": "SBIT"} for s in specs]
    else:
        dates_f, decide_f = dates[keep], decide[keep]
        WL, ES, f = build_inputs(specs, close, drivers, dates_f)
        nav = synthetic_nav(P["BTC"], masks["reset"], -2.0, ER)
        eq, _ = simulate(P, decide_f, WL, ES, [s["stop"] for s in specs], "etf", masks, nav=nav, k=-2.0, start_i=int(decide_f[0]))
        last = eq[-1]
        for j, s in enumerate(specs):
            short_on = ES[-1, j] > 0
            tw = {"SBIT": round(s["w"] / 2, 4)} if short_on else {c: round(float(x), 4) for c, x in zip(("BTC", "ETH"), WL[-1, j]) if x > 1e-4}
            out["accounts"].append({"id": s["id"], "label": s["label"], "shadow": True, "instrument": "SBIT",
                                    "equity": float(last[j]), "totalReturn": float(last[j] / cr.INITIAL - 1),
                                    "maxDrawdown": float(np.nanmin(eq[:, j] / np.fmax.accumulate(eq[:, j]) - 1)),
                                    "targetWeights": tw, "cash": round(1 - sum(tw.values()), 4),
                                    "shortSignalNow": bool(short_on), "schwabTradableNow": True, "dollarAware": "trend_macro" in str(s["short"])})
        out["status"] = "tracking"
        out["days"] = int(len(dates_f))
    if write:
        cd.write_json(PAPER_JSON, out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["data", "run", "paper"])
    ap.add_argument("--no-holdout", action="store_true", help="required for run: never score the sealed holdout")
    a = ap.parse_args(argv)
    if a.command == "data":
        print(json.dumps({"hourly": update_hourly(), "etfs": {k: v.get("status") for k, v in update_etfs().items()}}, default=str))
    elif a.command == "run":
        if not a.no_holdout:
            raise SystemExit("Pass --no-holdout: crypto_short.py never scores the sealed holdout.")
        rep = run(False)
        print(json.dumps({k: rep[k] for k in ("champion", "spaVsBtc", "spaVsOwnBase", "passHolmVsBtc", "passBHVsBase", "hoursCost", "seconds")}, default=str))
    else:
        print(json.dumps(paper(), default=str))


if __name__ == "__main__":
    main()
