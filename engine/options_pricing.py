"""Options math for paper trading, backtests and the analyst: Black-Scholes-Merton / Black-76 prices and greeks,
implied volatility, a modeled implied-vol surface, a bid/ask spread model, Schwab option fees, OCC symbols and
multi-leg payoff analytics (max loss / max gain / breakevens / probability of profit / expected move).

HONESTY: there are no historical option quotes in the local database. Anything priced with ``ModeledIV`` is a MODEL
(Black-Scholes on a vol forecast plus a skew and spread model), not a historical quote. Results produced from it carry
``"pricing": "modeled"`` and must be labeled that way in reports. Real quotes come only from Schwab chain snapshots
(options_data.py), saved locally each day so a real history builds up going forward.

No broker is called here; this is pure math.
"""

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import numpy as np

TRADING_DAYS = 252
CONTRACT_MULTIPLIER = 100
SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------------------------- distributions
def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def _npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------------------------- pricing
def bsm_price(spot, strike, t, vol, rate=0.04, div_yield=0.0, kind="call"):
    """Black-Scholes-Merton European price per share. t in years. kind 'call' | 'put'."""
    kind = _kind(kind)
    if t <= 0 or vol <= 0:
        return max(0.0, spot - strike) if kind == "call" else max(0.0, strike - spot)
    if spot <= 0:
        return 0.0 if kind == "call" else strike * math.exp(-rate * t)
    sq = vol * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * vol * vol) * t) / sq
    d2 = d1 - sq
    df_q, df_r = math.exp(-div_yield * t), math.exp(-rate * t)
    if kind == "call":
        return spot * df_q * _ncdf(d1) - strike * df_r * _ncdf(d2)
    return strike * df_r * _ncdf(-d2) - spot * df_q * _ncdf(-d1)


def black76_price(forward, strike, t, vol, rate=0.04, kind="call"):
    """Black-76 price on a forward / futures price (European)."""
    kind = _kind(kind)
    df = math.exp(-rate * max(t, 0.0))
    if t <= 0 or vol <= 0:
        return df * (max(0.0, forward - strike) if kind == "call" else max(0.0, strike - forward))
    sq = vol * math.sqrt(t)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * t) / sq
    d2 = d1 - sq
    if kind == "call":
        return df * (forward * _ncdf(d1) - strike * _ncdf(d2))
    return df * (strike * _ncdf(-d2) - forward * _ncdf(-d1))


def bsm_greeks(spot, strike, t, vol, rate=0.04, div_yield=0.0, kind="call"):
    """Per-share greeks: delta, gamma, vega (per 1 vol point = 0.01), theta (per calendar day), rho (per 1%)."""
    kind = _kind(kind)
    price = bsm_price(spot, strike, t, vol, rate, div_yield, kind)
    if t <= 0 or vol <= 0 or spot <= 0:
        itm = (spot > strike) if kind == "call" else (spot < strike)
        delta = (1.0 if itm else 0.0) * (1 if kind == "call" else -1)
        return {"price": price, "delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    sq = vol * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * vol * vol) * t) / sq
    d2 = d1 - sq
    df_q, df_r = math.exp(-div_yield * t), math.exp(-rate * t)
    gamma = df_q * _npdf(d1) / (spot * sq)
    vega = spot * df_q * _npdf(d1) * math.sqrt(t) / 100.0
    if kind == "call":
        delta = df_q * _ncdf(d1)
        theta = (-spot * df_q * _npdf(d1) * vol / (2 * math.sqrt(t)) - rate * strike * df_r * _ncdf(d2)
                 + div_yield * spot * df_q * _ncdf(d1))
        rho = strike * t * df_r * _ncdf(d2) / 100.0
    else:
        delta = -df_q * _ncdf(-d1)
        theta = (-spot * df_q * _npdf(d1) * vol / (2 * math.sqrt(t)) + rate * strike * df_r * _ncdf(-d2)
                 - div_yield * spot * df_q * _ncdf(-d1))
        rho = -strike * t * df_r * _ncdf(-d2) / 100.0
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta / 365.0, "rho": rho}


def implied_vol(price, spot, strike, t, rate=0.04, div_yield=0.0, kind="call", lo=1e-4, hi=5.0, tol=1e-7):
    """Implied volatility by bisection (robust; returns None outside no-arbitrage bounds)."""
    kind = _kind(kind)
    if t <= 0 or price is None or price <= 0:
        return None
    intrinsic = max(0.0, spot * math.exp(-div_yield * t) - strike * math.exp(-rate * t)) if kind == "call" else \
        max(0.0, strike * math.exp(-rate * t) - spot * math.exp(-div_yield * t))
    upper = spot if kind == "call" else strike
    if price < intrinsic - 1e-9 or price > upper:
        return None
    f = lambda v: bsm_price(spot, strike, t, v, rate, div_yield, kind) - price
    flo, fhi = f(lo), f(hi)
    if flo > 0 or fhi < 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol:
            return mid
        if fm < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _kind(kind):
    k = str(kind).lower()
    if k in ("c", "call"):
        return "call"
    if k in ("p", "put"):
        return "put"
    raise ValueError(f"option kind must be call or put, got {kind!r}")


# ---------------------------------------------------------------------------------------------- IV model
@dataclass
class ModeledIV:
    """Modeled implied volatility (labeled MODELED everywhere it is used).

    atm = max(GARCH forecast or blended realized vol, vol_floor) * variance-risk-premium multiplier * VIX scaling
    smile: iv(K) = atm * clip(1 + skew * x + smile * x^2, 0.5, 3), x = ln(K / S) / sqrt(T)
    VIX scaling: if a VIX proxy (VIXY 20-day log return) is given, atm *= clip(1 + vix_beta * r20, 0.7, 1.8).
    Defaults are rough equity-index values: skew -0.85 / smile 0.30 for ETFs, flatter for single stocks."""
    vrp: float = 1.12
    vol_floor: float = 0.08
    vol_cap: float = 2.5
    skew_etf: float = -0.85
    skew_stock: float = -0.45
    smile: float = 0.30
    vix_beta: float = 0.35
    term_slope: float = 0.0       # optional term structure tilt per year of maturity

    def atm(self, base_vol, vix_r20=None, t=None):
        base = max(float(base_vol or 0.0), self.vol_floor)
        iv = base * self.vrp
        if vix_r20 is not None and math.isfinite(vix_r20):
            iv *= min(1.8, max(0.7, 1.0 + self.vix_beta * vix_r20))
        if t:
            iv *= 1.0 + self.term_slope * t
        return min(iv, self.vol_cap)

    def iv(self, atm_vol, spot, strike, t, is_etf=True):
        t = max(t, 1.0 / 365)
        x = math.log(strike / spot) / math.sqrt(t)
        skew = self.skew_etf if is_etf else self.skew_stock
        x = max(-2.0, min(2.0, x))
        return min(self.vol_cap, atm_vol * min(3.0, max(0.5, 1.0 + skew * x + self.smile * x * x)))


def base_vol_forecast(returns, method="blend", lam=0.94, window=60):
    """Annualized vol forecast from daily simple/log returns using only the history given (no lookahead).
    method: 'garch' (arch package; falls back to blend), 'ewma', 'realized', 'blend' (mean of EWMA and realized)."""
    x = np.asarray([r for r in returns if r is not None and math.isfinite(r)], dtype=float)
    if len(x) < 20:
        return None
    realized = float(np.std(x[-window:], ddof=1) * math.sqrt(TRADING_DAYS))
    var = float(np.var(x[:20]))
    for v in x:
        var = lam * var + (1 - lam) * v * v
    ewma = math.sqrt(var * TRADING_DAYS)
    if method == "garch" and len(x) >= 120:
        try:
            from arch import arch_model
            res = arch_model(x * 100, vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
            f = np.asarray(res.forecast(horizon=21, reindex=False).variance)[-1]
            return float(math.sqrt(f.mean() * TRADING_DAYS) / 100)
        except Exception:  # noqa: BLE001 - deterministic fallback
            pass
    if method == "ewma":
        return ewma
    if method == "realized":
        return realized
    return 0.5 * (ewma + realized)


# ---------------------------------------------------------------------------------------------- spreads and fees
@dataclass
class OptionSpreadModel:
    """Modeled full bid/ask spread in dollars per share: max(min_spread, abs + rel * mid) * wing widening.
    ETF options (SPY/QQQ/IWM) trade at a penny or two; single-stock options are wider."""
    min_spread: float = 0.01
    abs_etf: float = 0.01
    rel_etf: float = 0.015
    abs_stock: float = 0.04
    rel_stock: float = 0.04
    wing_widen: float = 0.6       # extra relative widening for far OTM / deep ITM (|delta - 0.5| large)
    short_dte_widen: float = 0.3  # extra widening for < 7 DTE

    def spread(self, mid, delta=0.5, dte=30, is_etf=True):
        base = (self.abs_etf + self.rel_etf * mid) if is_etf else (self.abs_stock + self.rel_stock * mid)
        widen = 1.0 + self.wing_widen * max(0.0, abs(abs(delta) - 0.5) - 0.2) + (self.short_dte_widen if dte < 7 else 0.0)
        return max(self.min_spread, base * widen)

    def quote(self, mid, delta=0.5, dte=30, is_etf=True):
        s = self.spread(mid, delta, dte, is_etf)
        bid = max(0.0, mid - s / 2)
        tick = 0.01 if mid < 3 else 0.05 if not is_etf else 0.01
        bid = math.floor(bid / tick) * tick
        ask = math.ceil((mid + s / 2) / tick) * tick
        return round(bid, 2), round(max(ask, bid + 0.01), 2)


@dataclass
class OptionFees:
    """Schwab options costs (configurable). Sources checked 2026-11-25 (schwab.com/pricing): $0 base + $0.65 per
    contract online, waived on buy-to-close at $0.05 or less, no commission on exercise/assignment; ORF blended rate
    $0.01100 per contract (from 2026-11-01, both sides); FINRA TAF $0.00329 per contract sold (FINRA paused TAF
    Oct 1 - Dec 31 2026; we keep charging it to stay conservative); SEC Section 31 on sale proceeds ($20.60 / $1M);
    OCC clearing is not passed through to Schwab retail clients separately (set occ_per_contract if that changes)."""
    per_contract: float = 0.65
    orf_per_contract: float = 0.011
    taf_per_contract_sold: float = 0.00329
    sec_fee_per_million: float = 20.60
    occ_per_contract: float = 0.0
    btc_waiver_price: float = 0.05

    def fee(self, contracts, price, side):
        """Dollar fees for one leg fill. side: 'buy' | 'sell'; price per share; closing=buy-to-close waiver applies
        when price <= btc_waiver_price and side == 'buy_to_close'."""
        n = abs(float(contracts))
        if n == 0:
            return 0.0
        waived = side == "buy_to_close" and price <= self.btc_waiver_price
        total = (0.0 if waived else self.per_contract * n) + (self.orf_per_contract + self.occ_per_contract) * n
        if side in ("sell", "sell_to_open", "sell_to_close"):
            total += self.taf_per_contract_sold * n + self.sec_fee_per_million * n * price * CONTRACT_MULTIPLIER / 1e6
        return round(total, 4)


# ---------------------------------------------------------------------------------------------- OCC symbols
def occ_symbol(underlying, expiration, kind, strike):
    """Schwab / OCC symbol: underlying left-justified to 6, YYMMDD, C|P, strike * 1000 zero-padded to 8.
    Example: occ_symbol('SPY', '2026-12-18', 'C', 600) -> 'SPY   261218C00600000' (matches schwab-py OptionSymbol)."""
    root = str(underlying).upper().strip()
    if not root or len(root) > 6 or not all(c.isalnum() or c == "." for c in root):
        raise ValueError("underlying must be 1-6 letters")
    exp = _as_date(expiration)
    k = _kind(kind)
    milli = int(round(float(strike) * 1000))
    if milli <= 0 or milli >= 100_000_000:
        raise ValueError("strike out of range")
    return f"{root:<6}{exp:%y%m%d}{'C' if k == 'call' else 'P'}{milli:08d}"


def parse_occ(symbol):
    s = str(symbol)
    if len(s) != 21:
        raise ValueError("OCC option symbols are 21 characters (root padded to 6)")
    root, ymd, cp, strike = s[:6].rstrip(), s[6:12], s[12], s[13:]
    if cp not in "CP" or not ymd.isdigit() or not strike.isdigit():
        raise ValueError("bad OCC symbol")
    exp = datetime.strptime(ymd, "%y%m%d").date()
    return {"underlying": root, "expiration": exp.isoformat(), "kind": "call" if cp == "C" else "put",
            "strike": int(strike) / 1000.0}


def _as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def year_fraction(asof, expiration):
    """Calendar-day year fraction to the 16:00 ET expiration (>= 0)."""
    days = (_as_date(expiration) - _as_date(asof)).days
    return max(days, 0) / 365.0


# ---------------------------------------------------------------------------------------------- legs and payoff
@dataclass
class Leg:
    """One leg. kind: 'call' | 'put' | 'stock'. qty: signed contracts (or signed shares for stock).
    price: per-share entry price (premium or stock price). For stock legs strike/expiration are ignored."""
    kind: str
    qty: float          # int in the account; fractional only in reference backtests
    strike: float = 0.0
    expiration: str = ""
    price: float = 0.0
    iv: float | None = None

    def as_dict(self):
        return asdict(self)

    @property
    def multiplier(self):
        return 1 if self.kind == "stock" else CONTRACT_MULTIPLIER


def leg_from(d):
    if isinstance(d, Leg):
        return d
    kind = str(d.get("kind") or d.get("type") or "").lower()
    kind = "stock" if kind in ("stock", "equity", "shares") else _kind(kind)
    q = float(d["qty"])
    return Leg(kind=kind, qty=int(q) if q.is_integer() else q, strike=float(d.get("strike") or 0.0), expiration=str(d.get("expiration") or ""),
               price=float(d.get("price") or 0.0), iv=None if d.get("iv") in (None, "") else float(d["iv"]))


def leg_value_at_expiry(leg, s):
    if leg.kind == "stock":
        return leg.qty * s
    intrinsic = max(0.0, s - leg.strike) if leg.kind == "call" else max(0.0, leg.strike - s)
    return leg.qty * CONTRACT_MULTIPLIER * intrinsic


def payoff_at_expiry(legs, s):
    """P&L in dollars at the (first) expiration for underlying price s, net of entry prices (fees excluded)."""
    legs = [leg_from(l) for l in legs]
    return sum(leg_value_at_expiry(l, s) - l.qty * l.multiplier * l.price for l in legs)


def _critical_prices(legs, spot):
    strikes = sorted({l.strike for l in legs if l.kind != "stock"})
    pts = [0.0] + strikes
    hi = max([spot] + strikes) * 3 + 1
    return pts, hi


def analyze_legs(legs, spot, asof=None, rate=0.04, div_yield=0.0, vol=None, fees=0.0):
    """Max loss, max gain, breakevens, net debit/credit and whether risk is defined, for legs that all expire
    together (verticals, condors, straddles, covered calls, collars...). Calendar/diagonal legs are valued at the
    front expiration with the back leg priced by Black-Scholes at ``vol`` (or its own iv), so their max loss / gain are
    approximate and flagged 'approximate': True."""
    legs = [leg_from(l) for l in legs]
    options = [l for l in legs if l.kind != "stock"]
    expiries = sorted({l.expiration for l in options})
    approximate = len(expiries) > 1
    front = expiries[0] if expiries else None

    def value(s):
        if not approximate:
            return payoff_at_expiry(legs, s) - fees
        total = -fees
        for l in legs:
            if l.kind == "stock" or l.expiration == front:
                total += leg_value_at_expiry(l, s) - l.qty * l.multiplier * l.price
            else:
                t = max(year_fraction(front, l.expiration), 1 / 365)
                v = l.iv or vol or 0.25
                total += l.qty * CONTRACT_MULTIPLIER * (bsm_price(s, l.strike, t, v, rate, div_yield, l.kind) - l.price)
        return total

    pts, hi = _critical_prices(legs, spot)
    grid = sorted(set(np.concatenate([np.linspace(0.0, hi, 2001), np.array(pts)]).tolist()))
    values = [value(s) for s in grid]
    # slope beyond the highest strike tells whether gain/loss is unbounded to the upside
    net_calls = sum(l.qty for l in options if l.kind == "call") * CONTRACT_MULTIPLIER + sum(l.qty for l in legs if l.kind == "stock")
    upside_unbounded_loss = net_calls < 0
    upside_unbounded_gain = net_calls > 0
    max_loss = min(values)
    max_gain = max(values)
    breakevens = []
    for (a, va), (b, vb) in zip(zip(grid, values), zip(grid[1:], values[1:])):
        if va == 0:
            breakevens.append(a)
        elif va * vb < 0:
            breakevens.append(a + (b - a) * (-va) / (vb - va))
    breakevens = sorted({round(x, 2) for x in breakevens})
    net = sum(l.qty * l.multiplier * l.price for l in legs)
    return {
        "maxLoss": None if upside_unbounded_loss else round(max_loss, 2),
        "maxGain": None if upside_unbounded_gain else round(max_gain, 2),
        "unboundedLoss": upside_unbounded_loss, "unboundedGain": upside_unbounded_gain,
        "definedRisk": not upside_unbounded_loss,
        "breakevens": breakevens, "netDebit": round(net, 2) if net > 0 else 0.0,
        "netCredit": round(-net, 2) if net < 0 else 0.0, "fees": round(fees, 2),
        "approximate": approximate, "frontExpiration": front,
        "curve": [[round(s, 2), round(v, 2)] for s, v in zip(grid[::40], values[::40])],
    }


def position_greeks(legs, spot, asof, rate=0.04, div_yield=0.0, vol=None):
    """Dollar greeks of a multi-leg position (delta in shares-equivalent, gamma per $1, vega per vol point,
    theta per day, all already multiplied by contracts * 100)."""
    total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0, "value": 0.0}
    for l in (leg_from(x) for x in legs):
        if l.kind == "stock":
            total["delta"] += l.qty
            total["value"] += l.qty * spot
            continue
        t = year_fraction(asof, l.expiration)
        g = bsm_greeks(spot, l.strike, t, l.iv or vol or 0.25, rate, div_yield, l.kind)
        m = l.qty * CONTRACT_MULTIPLIER
        total["delta"] += g["delta"] * m
        total["gamma"] += g["gamma"] * m
        total["vega"] += g["vega"] * m
        total["theta"] += g["theta"] * m
        total["rho"] += g["rho"] * m
        total["value"] += g["price"] * m
    return {k: round(v, 4) for k, v in total.items()}


def probability_of_profit(legs, spot, asof, vol, rate=0.04, div_yield=0.0, fees=0.0, n=4001):
    """Risk-neutral-ish lognormal probability that the position's expiry P&L > 0 (drift = rate - q).
    Uses the front expiration's horizon. This is a MODEL probability, not a forecast of edge."""
    legs = [leg_from(l) for l in legs]
    exps = sorted({l.expiration for l in legs if l.kind != "stock"})
    if not exps:
        return None
    t = max(year_fraction(asof, exps[0]), 1 / 365)
    sd = vol * math.sqrt(t)
    mu = math.log(spot) + (rate - div_yield - 0.5 * vol * vol) * t
    z = np.linspace(-6, 6, n)
    s = np.exp(mu + sd * z)
    w = np.exp(-0.5 * z * z)
    w /= w.sum()
    info = analyze_legs(legs, spot, asof, rate, div_yield, vol, fees)
    if info["approximate"]:
        pnl = np.array([np.interp(x, [c[0] for c in info["curve"]], [c[1] for c in info["curve"]]) for x in s])
    else:
        pnl = np.array([payoff_at_expiry(legs, x) - fees for x in s])
    return float((w * (pnl > 0)).sum())


def expected_move(spot, vol, days, straddle=None):
    """1-sigma expected move over ``days`` calendar days: spot * vol * sqrt(days/365); and, if an ATM straddle price is
    given, the market-implied move (~0.85 * straddle)."""
    sigma = spot * vol * math.sqrt(max(days, 0) / 365.0)
    out = {"oneSigma": round(sigma, 4), "oneSigmaPct": round(sigma / spot, 6), "low": round(spot - sigma, 2),
           "high": round(spot + sigma, 2), "days": days, "vol": vol}
    if straddle:
        out["straddleImplied"] = round(0.85 * straddle, 4)
        out["straddleImpliedPct"] = round(0.85 * straddle / spot, 6)
    return out


# ---------------------------------------------------------------------------------------------- strategy templates
STRATEGIES = ("long_call", "long_put", "bull_call_spread", "bear_put_spread", "bull_put_spread", "bear_call_spread",
              "covered_call", "cash_secured_put", "collar", "protective_put", "straddle", "strangle", "short_straddle",
              "short_strangle", "iron_condor", "iron_butterfly", "call_butterfly", "put_butterfly", "calendar",
              "diagonal", "naked_call", "naked_put")

# Schwab complexOrderStrategyType per template (schwab-py ComplexOrderStrategyType values).
COMPLEX_TYPE = {"long_call": "NONE", "long_put": "NONE", "naked_call": "NONE", "naked_put": "NONE",
                "cash_secured_put": "NONE", "bull_call_spread": "VERTICAL", "bear_put_spread": "VERTICAL",
                "bull_put_spread": "VERTICAL", "bear_call_spread": "VERTICAL", "covered_call": "COVERED",
                "protective_put": "CUSTOM", "collar": "COLLAR_WITH_STOCK", "straddle": "STRADDLE",
                "short_straddle": "STRADDLE", "strangle": "STRANGLE", "short_strangle": "STRANGLE",
                "iron_condor": "IRON_CONDOR", "iron_butterfly": "BUTTERFLY", "call_butterfly": "BUTTERFLY",
                "put_butterfly": "BUTTERFLY", "calendar": "CALENDAR", "diagonal": "DIAGONAL"}


def build_strategy(name, spot, expiration, strikes=None, qty=1, back_expiration=None, width=None, shares=None):
    """Leg list (without prices) for a template. strikes: dict with keys used by the template (k, k1..k4, put, call);
    missing strikes default to round numbers around spot using ``width`` (default ~5% of spot)."""
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; one of {STRATEGIES}")
    st = dict(strikes or {})
    w = width or _round_strike(spot * 0.05, spot)
    atm = _round_strike(spot, spot)
    k = float(st.get("k", atm))
    exp, back = expiration, back_expiration or expiration
    L = lambda kind, q, strike, e=exp: Leg(kind=kind, qty=q * qty, strike=float(strike), expiration=e)
    stock = lambda q: Leg(kind="stock", qty=q)
    sh = int(shares if shares is not None else 100 * qty)
    if name == "long_call":
        return [L("call", 1, k)]
    if name == "long_put":
        return [L("put", 1, k)]
    if name == "naked_call":
        return [L("call", -1, k)]
    if name in ("naked_put", "cash_secured_put"):
        return [L("put", -1, st.get("k", atm - w))]
    if name == "bull_call_spread":
        return [L("call", 1, st.get("k1", atm)), L("call", -1, st.get("k2", atm + w))]
    if name == "bear_call_spread":
        return [L("call", -1, st.get("k1", atm)), L("call", 1, st.get("k2", atm + w))]
    if name == "bear_put_spread":
        return [L("put", 1, st.get("k2", atm)), L("put", -1, st.get("k1", atm - w))]
    if name == "bull_put_spread":
        return [L("put", -1, st.get("k2", atm)), L("put", 1, st.get("k1", atm - w))]
    if name == "covered_call":
        return [stock(sh), L("call", -1, st.get("k", atm + w))]
    if name == "protective_put":
        return [stock(sh), L("put", 1, st.get("k", atm - w))]
    if name == "collar":
        return [stock(sh), L("put", 1, st.get("put", atm - w)), L("call", -1, st.get("call", atm + w))]
    if name in ("straddle", "short_straddle"):
        s = 1 if name == "straddle" else -1
        return [L("call", s, k), L("put", s, k)]
    if name in ("strangle", "short_strangle"):
        s = 1 if name == "strangle" else -1
        return [L("call", s, st.get("call", atm + w)), L("put", s, st.get("put", atm - w))]
    if name == "iron_condor":
        k1, k2 = st.get("k1", atm - 2 * w), st.get("k2", atm - w)
        k3, k4 = st.get("k3", atm + w), st.get("k4", atm + 2 * w)
        return [L("put", 1, k1), L("put", -1, k2), L("call", -1, k3), L("call", 1, k4)]
    if name == "iron_butterfly":
        return [L("put", 1, st.get("k1", atm - w)), L("put", -1, k), L("call", -1, k), L("call", 1, st.get("k3", atm + w))]
    if name in ("call_butterfly", "put_butterfly"):
        kind = "call" if name == "call_butterfly" else "put"
        return [L(kind, 1, st.get("k1", atm - w)), L(kind, -2, k), L(kind, 1, st.get("k3", atm + w))]
    if name == "calendar":
        kind = st.get("kind", "call")
        return [L(kind, -1, k, exp), L(kind, 1, k, back)]
    if name == "diagonal":
        kind = st.get("kind", "call")
        return [L(kind, -1, st.get("k_short", atm + w), exp), L(kind, 1, st.get("k_long", k), back)]
    raise ValueError(name)


def _round_strike(x, spot):
    step = 0.5 if spot < 25 else 1.0 if spot < 200 else 5.0
    return max(step, round(x / step) * step)


def price_legs(legs, spot, asof, atm_vol, model=None, spread=None, rate=0.04, div_yield=0.0, is_etf=True, side="open"):
    """Fill each leg at a modeled price: longs pay the modeled ask, shorts receive the modeled bid (opening), the
    reverse when closing. Returns (priced legs, mid legs). Stock legs fill at spot (spread handled by CostModel)."""
    model = model or ModeledIV()
    spread = spread or OptionSpreadModel()
    out = []
    for l in (leg_from(x) for x in legs):
        if l.kind == "stock":
            out.append(Leg(**{**l.as_dict(), "price": float(spot)}))
            continue
        t = max(year_fraction(asof, l.expiration), 1 / 365)
        iv = l.iv or model.iv(atm_vol, spot, l.strike, t, is_etf)
        g = bsm_greeks(spot, l.strike, t, iv, rate, div_yield, l.kind)
        bid, ask = spread.quote(g["price"], g["delta"], t * 365, is_etf)
        buying = (l.qty > 0) == (side == "open")
        out.append(Leg(kind=l.kind, qty=l.qty, strike=l.strike, expiration=l.expiration, price=ask if buying else bid, iv=iv))
    return out


__all__ = ["bsm_price", "black76_price", "bsm_greeks", "implied_vol", "ModeledIV", "base_vol_forecast",
           "OptionSpreadModel", "OptionFees", "occ_symbol", "parse_occ", "Leg", "leg_from", "payoff_at_expiry",
           "analyze_legs", "position_greeks", "probability_of_profit", "expected_move", "build_strategy", "STRATEGIES",
           "COMPLEX_TYPE", "price_legs", "year_fraction", "CONTRACT_MULTIPLIER"]
