"""Growth-adjusted mean reversion: buy dips below an expected GROWTH PATH, not below a flat rolling mean.

Fair path (all inputs dated <= t-1, so the deviation on day t uses no future data):
    g_t      = weighted blend of daily log drifts (renormalized over the components available for that name/day)
                 trend   : exponentially weighted regression slope of the name's log price (halflife drift_halflife)
                 fund    : fundamental growth, quarterly revenue YoY from cached SEC filings, usable only from
                           period end + fundamental_lag_days (point in time; analyst estimates if a frame is supplied)
                 sector  : EW slope of the sector ETF the name tracks best (causal trailing correlation, refreshed
                           every sector_refresh sessions)
                 market  : EW slope of SPY
    anchor_t = EWMA of log price through t-1 (halflife anchor_halflife) projected forward by g over the weighted
               mean age of that average:  fair_t = EWMA_{t-1} + g_{t-1} * (age_{t-1} + 1)
    z_t      = (log P_t - fair_t) / std of past deviations (resid_window sessions, through t-1)
With every weight on nothing (g = 0) this is the old flat-mean deviation, so the growth adjustment is the only change.

Weights: from config (w_trend, w_fund, w_sector, w_market) or weights='learned': every learn_every sessions the blend
with the most negative pooled correlation between z (z < 0 only) and the next learn_horizon-session return is chosen
from WEIGHT_GRID, using only observations whose forward return was complete before the refit date (purged).

Entry: z < entry_z, trend intact (g > 0 and close above its trend_filter-day SMA), liquid stock, regime gate open
(regime_filter 'fixed' = the original calm-and-ranging filter, 'hmm', or 'off'). Exit: z back to exit_z (the path),
a stop_atr x ATR stop, or max_hold sessions. Sizing, caps and stops reuse RegimeMeanReversion.
"""

import json
import math
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd

from risk import atr

from .meanrev import RegimeMeanReversion

SECTOR_ETFS = ("XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC")
MARKET = "SPY"
TRADING_DAYS = 252
# (trend, fund, sector, market) candidates for weights='learned'
WEIGHT_GRID = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.6, 0.0, 0.3, 0.1), (0.5, 0.2, 0.2, 0.1),
               (0.3, 0.1, 0.3, 0.3), (0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 0.0, 0.0), (0.25, 0.25, 0.25, 0.25))


def ew_slope(log_price, halflife, min_periods=20):
    """Exponentially weighted least-squares slope of log price on time (per session), causal (rows <= t)."""
    t = pd.DataFrame(np.arange(len(log_price), dtype=float)[:, None].repeat(log_price.shape[1], axis=1),
                     index=log_price.index, columns=log_price.columns).where(log_price.notna())
    ew = lambda f: f.ewm(halflife=halflife, min_periods=min_periods).mean()
    mt, mx = ew(t), ew(log_price)
    cov = ew(t * log_price) - mt * mx
    var = ew(t * t) - mt * mt
    return (cov / var.where(var > 1e-9)).where(log_price.notna())


def ew_anchor(log_price, halflife):
    """EWMA of log price and the weighted mean age (sessions) of that average, both through row t."""
    t = pd.DataFrame(np.arange(len(log_price), dtype=float)[:, None].repeat(log_price.shape[1], axis=1),
                     index=log_price.index, columns=log_price.columns)
    mean = log_price.ewm(halflife=halflife, min_periods=max(3, int(halflife))).mean()
    mean_t = t.where(log_price.notna()).ewm(halflife=halflife, min_periods=max(3, int(halflife))).mean()
    return mean, t - mean_t


def causal_sector_map(returns, sector_returns, window=126, every=21, min_obs=60):
    """dates x names frame of the best-correlated sector ETF (column label), using returns strictly before each
    refresh date; forward-filled between refreshes. NaN until enough history."""
    out = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns, dtype=object)
    if sector_returns is None or sector_returns.empty:
        return out
    labels = list(sector_returns.columns)
    rows = list(range(min_obs, len(returns), every))
    picks = {}
    R, S = returns.to_numpy(dtype=float), sector_returns.to_numpy(dtype=float)
    for r in rows:
        lo = max(0, r - window)
        a, b = R[lo:r], S[lo:r]                        # rows strictly before r
        ok_b = np.isfinite(b).all(axis=1)
        a, b = a[ok_b], b[ok_b]
        if len(b) < min_obs:
            continue
        a0 = np.where(np.isfinite(a), a, np.nan)
        am = np.nanmean(a0, axis=0)
        ac = np.where(np.isfinite(a0), a0 - am, 0.0)
        bc = b - b.mean(axis=0)
        n_ok = np.isfinite(a0).sum(axis=0)
        cov = ac.T @ bc
        sa = np.sqrt((ac ** 2).sum(axis=0))[:, None]
        sb = np.sqrt((bc ** 2).sum(axis=0))[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / (sa * sb)
        best = np.nanargmax(np.where(np.isfinite(corr), corr, -2), axis=1)
        choice = np.array([labels[k] for k in best], dtype=object)
        choice[n_ok < min_obs] = None
        picks[returns.index[r]] = choice
    if picks:
        frame = pd.DataFrame.from_dict(picks, orient="index", columns=returns.columns)
        out.loc[frame.index] = frame.to_numpy()
        out = out.ffill()
    return out


_FUND_MEMO = {}


def load_fundamental_growth(dates, columns, lag_days=60, path=None):
    """Point-in-time daily log drift from cached SEC quarterly revenue (Massive/ai_research.sqlite3).

    Growth for the quarter ending e = revenue_q / revenue_{q-4} - 1 (quarters about a year apart), usable from
    e + lag_days (10-Q filing deadline is 40-45 days; the cache has no filing date, so the lag is conservative).
    Returns a dates x columns frame (NaN where nothing was known yet)."""
    from fetch_and_model import DATA_DIR
    path = path or (DATA_DIR / "ai_research.sqlite3")
    key = (str(path), lag_days)
    if key not in _FUND_MEMO:
        series = {}
        try:
            with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
                for ticker, payload in db.execute("SELECT ticker, payload FROM source_cache WHERE source = 'SEC_FINANCIALS'"):
                    try:
                        q = json.loads(payload)["quarterly"]["revenue"]
                    except (ValueError, KeyError, TypeError):
                        continue
                    pts = sorted((pd.Timestamp(x["end"]), float(x["val"])) for x in q if x.get("val"))
                    rows = []
                    for i, (end, val) in enumerate(pts):
                        prior = [v for e, v in pts[:i] if 330 <= (end - e).days <= 400]
                        if prior and prior[-1] > 0 and val > 0:
                            rows.append((end + pd.Timedelta(days=lag_days), math.log(val / prior[-1]) / TRADING_DAYS))
                    if rows:
                        series[str(ticker).upper()] = pd.Series(dict(rows)).sort_index()
        except sqlite3.Error:
            pass
        _FUND_MEMO[key] = series
    frame = pd.DataFrame(np.nan, index=dates, columns=columns)
    for ticker, s in _FUND_MEMO[key].items():
        if ticker in frame.columns:
            frame[ticker] = s.reindex(s.index.union(dates)).ffill().reindex(dates)
    return frame


class GrowthMeanReversion(RegimeMeanReversion):
    name = "growth_mr"
    defaults = {**RegimeMeanReversion.defaults,
                "drift_halflife": 60, "anchor_halflife": 10, "resid_window": 60,
                "w_trend": 0.5, "w_fund": 0.2, "w_sector": 0.2, "w_market": 0.1, "weights": "config",
                "learn_every": 63, "learn_window": 252, "learn_horizon": 5, "learn_min_obs": 126,
                "drift_clip_annual": [-0.5, 1.0], "fundamental_lag_days": 60, "use_fundamentals": True,
                "sector_refresh": 21, "trend_filter": 100}

    def __init__(self, **params):
        super().__init__(**params)
        self._fund_override = None
        self.weight_log = []

    def set_fundamentals(self, frame):
        """Inject a point-in-time daily drift frame (e.g. analyst estimates); replaces the SEC cache."""
        self._fund_override = frame

    def regime(self, panel):
        base = super().regime(panel)
        if self.params.get("regime_filter") == "off" and self._external_gate is None:
            base["on"] = True
            base["source"] = "off"
        return base

    # -------------------------------------------------------------- drift components
    def components(self, panel, cols):
        p = self.params
        logp = np.log(panel.close[cols].where(panel.close[cols] > 0))
        lo, hi = (x / TRADING_DAYS for x in p["drift_clip_annual"])
        trend = ew_slope(logp, p["drift_halflife"]).clip(lo, hi)
        comps = {"trend": trend}
        fund = None
        if p["use_fundamentals"]:
            fund = (self._fund_override.reindex(index=panel.dates, columns=cols) if self._fund_override is not None
                    else load_fundamental_growth(panel.dates, cols, p["fundamental_lag_days"]))
            fund = fund.clip(lo, hi)
        comps["fund"] = fund if fund is not None else pd.DataFrame(np.nan, index=panel.dates, columns=cols)
        etfs = [e for e in SECTOR_ETFS if e in panel.close.columns]
        if MARKET in panel.close.columns:
            mkt_log = np.log(panel.close[[MARKET]])
            mkt = ew_slope(mkt_log, p["drift_halflife"])[MARKET].clip(lo, hi)
            comps["market"] = pd.DataFrame(np.repeat(mkt.to_numpy()[:, None], len(cols), axis=1),
                                           index=panel.dates, columns=cols).where(logp.notna())
        else:
            comps["market"] = pd.DataFrame(np.nan, index=panel.dates, columns=cols)
        if etfs:
            sec_slope = ew_slope(np.log(panel.close[etfs]), p["drift_halflife"]).clip(lo, hi)
            rets = panel.total_return()
            mapping = causal_sector_map(rets[cols], rets[etfs], every=int(p["sector_refresh"]))
            S = sec_slope.to_numpy()
            index_of = {e: k for k, e in enumerate(etfs)}
            M = mapping.to_numpy()
            idx = np.vectorize(lambda v: index_of.get(v, -1) if isinstance(v, str) else -1)(M) if M.size else M
            rows = np.arange(len(panel.dates))[:, None]
            vals = np.where(idx >= 0, S[rows, np.clip(idx, 0, None)], np.nan)
            comps["sector"] = pd.DataFrame(vals, index=panel.dates, columns=cols).where(logp.notna())
        else:
            comps["sector"] = pd.DataFrame(np.nan, index=panel.dates, columns=cols)
        return logp, comps

    @staticmethod
    def blend(comps, w):
        num = den = 0.0
        for name, weight in zip(("trend", "fund", "sector", "market"), w):
            if weight <= 0:
                continue
            f = comps[name]
            ok = f.notna()
            num = num + f.fillna(0.0) * weight
            den = den + ok.astype(float) * weight
        if isinstance(den, float):          # all weights zero: flat path
            return comps["trend"] * 0.0
        return (num / den.where(den > 0)).fillna(0.0).where(comps["trend"].notna() | comps["market"].notna(), 0.0)

    def zscore(self, logp, g):
        p = self.params
        mean, age = ew_anchor(logp, p["anchor_halflife"])
        fair = mean.shift(1) + g.shift(1) * (age.shift(1) + 1.0)
        dev = logp - fair
        sd = dev.rolling(p["resid_window"], min_periods=max(20, p["resid_window"] // 2)).std(ddof=0).shift(1)
        return dev / sd.where(sd > 0), fair

    def _learned_z(self, logp, comps):
        """Walk-forward choice of the blend: purged, using only rows whose forward return ended before the refit."""
        p = self.params
        h = int(p["learn_horizon"])
        fwd = (logp.shift(-h) - logp)              # forward return from t to t+h; known only at t+h
        zs, gs = [], []
        for w in WEIGHT_GRID:
            g = self.blend(comps, w)
            z, _ = self.zscore(logp, g)
            zs.append(z)
            gs.append(g)
        default = (p["w_trend"], p["w_fund"], p["w_sector"], p["w_market"])
        dz, _ = self.zscore(logp, self.blend(comps, default))
        dg = self.blend(comps, default)
        n = len(logp)
        out_z, out_g = dz.copy(), dg.copy()
        self.weight_log = []
        Z = [z.to_numpy() for z in zs]
        F = fwd.to_numpy()
        for start in range(int(p["learn_min_obs"]), n, int(p["learn_every"])):
            end = min(start + int(p["learn_every"]), n)
            lo = max(0, start - int(p["learn_window"]))
            hi = start - h                          # labels complete by row start-1  (row + h <= start - 1 + ... purge)
            if hi - lo < 40:
                continue
            best, best_ic = None, 0.0
            for k, zarr in enumerate(Z):
                a, b = zarr[lo:hi], F[lo:hi]
                m = np.isfinite(a) & np.isfinite(b) & (a < 0)
                if m.sum() < 200:
                    continue
                bb = np.clip(b[m], -0.3, 0.3)
                ic = float(np.corrcoef(a[m], bb)[0, 1])
                if np.isfinite(ic) and ic < best_ic:
                    best, best_ic = k, ic
            if best is None:
                continue
            rows = logp.index[start:end]
            out_z.loc[rows] = zs[best].loc[rows].to_numpy()
            out_g.loc[rows] = gs[best].loc[rows].to_numpy()
            self.weight_log.append({"from": str(rows[0].date()), "weights": list(WEIGHT_GRID[best]), "ic": best_ic})
        return out_z, out_g

    # -------------------------------------------------------------- state machine (same rules as the old overlay)
    def _states(self, panel):
        p = self.params
        mask = self.universe(panel)
        cols = list(mask.columns[mask.any()])
        regime = self.regime(panel)
        close = panel.close[cols]
        logp, comps = self.components(panel, cols)
        if p["weights"] == "learned":
            z, g = self._learned_z(logp, comps)
        else:
            g = self.blend(comps, (p["w_trend"], p["w_fund"], p["w_sector"], p["w_market"]))
            z, _ = self.zscore(logp, g)
        self.last_drift = g
        trend_ok = (g > 0) & (close > close.rolling(p["trend_filter"], min_periods=p["trend_filter"]).mean())
        entry_ok = (mask[cols] & trend_ok & (z < p["entry_z"])).to_numpy()
        zv = z.to_numpy()
        closes = close.to_numpy()
        atr_values = atr(panel.high[cols], panel.low[cols], close).to_numpy()
        regime_on = regime["on"].to_numpy()
        active = np.zeros_like(zv)
        stops = np.full_like(zv, np.nan)
        open_positions = {}
        for row in range(len(zv)):
            for col in list(open_positions):
                entry_row, stop = open_positions[col]
                age = row - entry_row
                price = closes[row, col]
                if np.isnan(price):
                    active[row, col], stops[row, col] = 1.0, stop
                    continue
                if (np.isfinite(zv[row, col]) and zv[row, col] >= p["exit_z"]) or age >= p["max_hold"] or price <= stop:
                    del open_positions[col]
                else:
                    active[row, col], stops[row, col] = 1.0, stop
            free = p["max_positions"] - len(open_positions)
            if regime_on[row] and free > 0:
                candidates = np.flatnonzero(entry_ok[row])
                candidates = [c for c in candidates if c not in open_positions and np.isfinite(atr_values[row, c])]
                candidates.sort(key=lambda c: (zv[row, c], c))
                for col in candidates[:free]:
                    stop = closes[row, col] - p["stop_atr"] * atr_values[row, col]
                    open_positions[col] = (row, stop)
                    active[row, col], stops[row, col] = 1.0, stop
        frame = lambda values: pd.DataFrame(values, index=panel.dates, columns=cols)
        return frame(active), frame(stops), z, regime
