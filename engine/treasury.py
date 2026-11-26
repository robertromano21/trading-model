"""US Treasury pricing for paper trading only, marked to the FRED constant-maturity curve.

This module never sends orders. Individual cash Treasuries stay in a local paper ledger.
Bond ETFs (SHY, IEF, TLT, ...) already trade as equities through the existing paper/live equity path.

Pricing
-------
Notes and bonds: semi-annual coupons, 30/360 is NOT used. Accrued interest uses the actual/actual
ICMA convention (days accrued / days in the coupon period times the semi-annual coupon). Dirty price
is the present value of remaining coupons plus principal at the bond-equivalent yield (semi-annual).
Clean price = dirty price - accrued. Bills are zero-coupon instruments discounted on an actual/360
discount basis, converted to a bond-equivalent price.

The mark uses the FRED Treasury constant-maturity par curve, linearly interpolated at the bond's
remaining maturity. That is a curve proxy, not a CUSIP-level quote. Bid/ask is a simple tenor-based
spread model in 32nds of a point (1 point = 1% of par). Coupons credit cash on coupon dates; at
maturity the position pays par plus the final coupon and closes.

Yield, Macaulay and modified duration, and convexity are computed from the same cash-flow schedule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Tenor (years) -> half-spread in 32nds of a point. Bills are tighter than long bonds.
BID_ASK_32NDS = ((0.25, 0.5), (2.0, 1.0), (5.0, 1.5), (10.0, 2.0), (30.0, 4.0))
PAR = 100.0
# Standard generic paper instruments (CUSIP-less, constant-maturity at issue, then age).
GENERIC_TENORS = {
    "TSY_3M": 0.25,
    "TSY_2Y": 2.0,
    "TSY_5Y": 5.0,
    "TSY_10Y": 10.0,
    "TSY_30Y": 30.0,
}


def as_date(value):
    if value is None:
        return None
    if isinstance(value, date) and not hasattr(value, "hour"):
        # datetime is a subclass of date; exclude it
        from datetime import datetime as _dt
        if isinstance(value, _dt):
            return value.date()
        return value
    return pd.Timestamp(value).date()


def add_years(start, years):
    """Approximate calendar advance used for generic issue/maturity dates."""
    start = as_date(start)
    whole, frac = divmod(float(years), 1.0)
    month = start.month + int(round(frac * 12))
    year = start.year + int(whole)
    while month > 12:
        month -= 12
        year += 1
    while month < 1:
        month += 12
        year -= 1
    day = min(start.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year, month):
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day


def is_bill(maturity_years=None, coupon=None, kind=None):
    if kind == "bill":
        return True
    if coupon is not None and float(coupon) == 0.0:
        return True
    if maturity_years is not None and float(maturity_years) <= 1.0:
        return True
    return False


def coupon_dates(issue, maturity, freq=2):
    """Semi-annual coupon dates from maturity backward, strictly after issue, up to and including maturity."""
    issue, maturity = as_date(issue), as_date(maturity)
    if maturity <= issue:
        return []
    dates = []
    year, month, day = maturity.year, maturity.month, maturity.day
    step = 12 // int(freq)
    # Walk backward from maturity.
    current = maturity
    while current > issue:
        dates.append(current)
        month -= step
        year_delta, month = divmod(month - 1, 12)
        year += year_delta
        month += 1
        day_use = min(day, _days_in_month(year, month))
        current = date(year, month, day_use)
        if len(dates) > 80:
            break
    dates = sorted({d for d in dates if issue < d <= maturity})
    return dates


def coupon_period(settlement, schedule):
    """(previous coupon or issue-adjacent start, next coupon, days in period, days accrued)."""
    settlement = as_date(settlement)
    upcoming = [d for d in schedule if d > settlement]
    if not upcoming:
        return None
    nxt = upcoming[0]
    prior = [d for d in schedule if d <= settlement]
    # Period start is the previous coupon, or nxt minus 6 months if none (new issue).
    if prior:
        prev = prior[-1]
    else:
        month = nxt.month - 6
        year = nxt.year
        if month <= 0:
            month += 12
            year -= 1
        prev = date(year, month, min(nxt.day, _days_in_month(year, month)))
    days_period = (nxt - prev).days
    days_accrued = (settlement - prev).days
    return prev, nxt, days_period, max(0, days_accrued)


def accrued_interest(coupon_annual, settlement, schedule, face=PAR):
    """Actual/actual accrued interest in points (per 100 of par). Zero on a coupon date."""
    if coupon_annual is None or float(coupon_annual) == 0.0:
        return 0.0
    period = coupon_period(settlement, schedule)
    if period is None:
        return 0.0
    _, _, days_period, days_accrued = period
    if days_period <= 0 or days_accrued <= 0:
        return 0.0
    return float(coupon_annual) / 2.0 * (days_accrued / days_period) * (face / PAR)


def remaining_cashflows(settlement, schedule, coupon_annual, face=PAR):
    """List of (date, time_in_years, cashflow in points per `face`)."""
    settlement = as_date(settlement)
    remaining = [d for d in schedule if d > settlement]
    if not remaining:
        return []
    # Cashflows are always in points per 100 of par.
    out = []
    semi = float(coupon_annual or 0.0) / 2.0
    for i, day in enumerate(remaining):
        amount = semi + (PAR if i == len(remaining) - 1 else 0.0)
        years = (day - settlement).days / 365.2425
        out.append((day, years, amount))
    return out


def dirty_price_from_ytm(ytm, settlement, schedule, coupon_annual, face=PAR):
    """Present value in points per 100 of par at a bond-equivalent (semi-annual) yield."""
    settlement = as_date(settlement)
    flows = remaining_cashflows(settlement, schedule, coupon_annual, face=PAR)
    if not flows:
        return float("nan")
    y = float(ytm)
    # Street convention: discount each flow at (1+y/2)^(2*t)
    total = 0.0
    for _, years, amount in flows:
        total += amount / (1.0 + y / 2.0) ** (2.0 * years)
    return float(total)


def ytm_from_dirty(dirty, settlement, schedule, coupon_annual, guess=None, face=PAR):
    """Newton (with bisection fallback) for the bond-equivalent yield given a dirty price."""
    flows = remaining_cashflows(settlement, schedule, coupon_annual, face=PAR)
    if not flows or dirty is None or not math.isfinite(float(dirty)) or float(dirty) <= 0:
        return float("nan")
    dirty = float(dirty)
    if coupon_annual is None or float(coupon_annual) == 0.0:
        # Zero: 100 / (1+y/2)^{2t} = dirty
        years = flows[-1][1]
        if years <= 0 or dirty <= 0:
            return float("nan")
        return 2.0 * ((PAR / dirty) ** (1.0 / (2.0 * years)) - 1.0)

    def price(y):
        return dirty_price_from_ytm(y, settlement, schedule, coupon_annual, face=PAR)

    def deriv(y):
        # dP/dy ≈ -sum t * cf / (1+y/2)^{2t+1}
        total = 0.0
        for _, years, amount in flows:
            base = 1.0 + y / 2.0
            total += years * amount / base ** (2.0 * years + 1.0)
        return -total

    y = 0.04 if guess is None or not math.isfinite(guess) else float(guess)
    for _ in range(25):
        p = price(y)
        if not math.isfinite(p):
            break
        gap = p - dirty
        if abs(gap) < 1e-8:
            return float(y)
        dp = deriv(y)
        if not math.isfinite(dp) or abs(dp) < 1e-12:
            break
        y = y - gap / dp
        y = min(max(y, -0.5), 2.0)
    # Bisection fallback
    lo, hi = -0.2, 1.5
    plo, phi = price(lo) - dirty, price(hi) - dirty
    if plo * phi > 0:
        return float(y)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = price(mid) - dirty
        if abs(val) < 1e-8:
            return float(mid)
        if plo * val <= 0:
            hi, phi = mid, val
        else:
            lo, plo = mid, val
    return float(0.5 * (lo + hi))


def bill_price(discount_yield, settlement, maturity):
    """Actual/360 discount-basis price per 100 of par."""
    settlement, maturity = as_date(settlement), as_date(maturity)
    days = max((maturity - settlement).days, 0)
    if days <= 0:
        return PAR
    return PAR * (1.0 - float(discount_yield) * days / 360.0)


def bill_ytm(price, settlement, maturity):
    """Bond-equivalent yield for a bill given its price (semi-annual compound, actual/365)."""
    settlement, maturity = as_date(settlement), as_date(maturity)
    days = max((maturity - settlement).days, 0)
    if days <= 0 or price is None or float(price) <= 0:
        return float("nan")
    years = days / 365.2425
    return 2.0 * ((PAR / float(price)) ** (1.0 / (2.0 * years)) - 1.0)


def duration_convexity(ytm, settlement, schedule, coupon_annual):
    """Macaulay duration (years), modified duration, and convexity (years^2)."""
    flows = remaining_cashflows(settlement, schedule, coupon_annual)
    dirty = dirty_price_from_ytm(ytm, settlement, schedule, coupon_annual)
    if not flows or not math.isfinite(dirty) or dirty <= 0:
        return float("nan"), float("nan"), float("nan")
    y = float(ytm)
    mac = 0.0
    conv = 0.0
    for _, years, amount in flows:
        pv = amount / (1.0 + y / 2.0) ** (2.0 * years)
        mac += years * pv
        # Convexity in years^2: sum t*(t+1/2) * pv / (dirty * (1+y/2)^2) using period time.
        # With t in years and semi-annual compounding: sum years*(years+0.5) * pv / (dirty*(1+y/2)^2)
        conv += years * (years + 0.5) * pv
    mac /= dirty
    modified = mac / (1.0 + y / 2.0)
    convexity = conv / (dirty * (1.0 + y / 2.0) ** 2)
    return float(mac), float(modified), float(convexity)


def half_spread_points(maturity_years):
    """Half bid/ask spread in points (percent of par) from the tenor model."""
    years = max(float(maturity_years), 0.0)
    thirty_seconds = BID_ASK_32NDS[-1][1]
    for tenor, width in BID_ASK_32NDS:
        if years <= tenor:
            thirty_seconds = width
            break
    return (thirty_seconds / 32.0) / 2.0


@dataclass
class BondSpec:
    ticker: str
    issue: date
    maturity: date
    coupon: float  # annual coupon rate, decimal (0.04 = 4%)
    kind: str = "note"  # 'bill' or 'note'
    face: float = 1000.0  # dollars of par per unit

    def remaining_years(self, settlement):
        settlement = as_date(settlement)
        return max((self.maturity - settlement).days / 365.2425, 0.0)

    def schedule(self):
        if self.kind == "bill" or self.coupon == 0.0:
            return [self.maturity]
        return coupon_dates(self.issue, self.maturity, freq=2)


def mark_bond(spec, settlement, par_yield, mid_price=None):
    """Mark one bond on `settlement` using a par yield (decimal) at remaining maturity.

    Returns a dict with clean/dirty mid, bid, ask, accrued, ytm, duration, convexity, and cash
    due if `settlement` is a coupon or maturity date (points per 100 and dollars per unit).
    """
    settlement = as_date(settlement)
    years = spec.remaining_years(settlement)
    matured = settlement >= spec.maturity
    schedule = spec.schedule()
    coupon_annual = 0.0 if spec.kind == "bill" else float(spec.coupon) * 100.0  # percent
    # Internal functions use coupon in percent (6.0 = 6%).
    coupon_pct = 0.0 if spec.kind == "bill" else float(spec.coupon) * 100.0
    ytm = float(par_yield) if par_yield is not None else float("nan")

    cash_points = 0.0
    if matured:
        # Final coupon + par (bills: par only; the discount was in the purchase price).
        cash_points = PAR + (0.0 if spec.kind == "bill" else coupon_pct / 2.0)
        dirty = PAR
        accrued = 0.0 if spec.kind == "bill" else 0.0
        clean = PAR
        mac = mod = conv = 0.0
        bid = ask = PAR
    else:
        if spec.kind == "bill" or spec.coupon == 0.0:
            dirty = bill_price(ytm, settlement, spec.maturity) if math.isfinite(ytm) else float("nan")
            accrued = 0.0
            clean = dirty
            ytm = bill_ytm(dirty, settlement, spec.maturity) if math.isfinite(dirty) else float("nan")
            mac = years
            mod = years / (1.0 + (ytm or 0.0) / 2.0) if math.isfinite(ytm) else float("nan")
            conv = years * (years + 0.5) / ((1.0 + (ytm or 0.0) / 2.0) ** 2) if math.isfinite(ytm) else float("nan")
        else:
            if mid_price is not None:
                dirty = float(mid_price)
                ytm = ytm_from_dirty(dirty, settlement, schedule, coupon_pct, guess=ytm)
            else:
                dirty = dirty_price_from_ytm(ytm, settlement, schedule, coupon_pct)
            accrued = accrued_interest(coupon_pct, settlement, schedule)
            clean = dirty - accrued if math.isfinite(dirty) else float("nan")
            mac, mod, conv = duration_convexity(ytm, settlement, schedule, coupon_pct)
            # Coupon cash if settlement is on a coupon date (accrued just reset to 0, prior date in schedule).
            if settlement in schedule and settlement != spec.maturity:
                cash_points = coupon_pct / 2.0
        half = half_spread_points(years)
        bid = (clean - half) if math.isfinite(clean) else float("nan")
        ask = (clean + half) if math.isfinite(clean) else float("nan")

    dollars_per_point = spec.face / PAR
    return {
        "ticker": spec.ticker,
        "settlement": settlement.isoformat(),
        "maturity": spec.maturity.isoformat(),
        "remainingYears": years if not matured else 0.0,
        "coupon": spec.coupon,
        "kind": spec.kind,
        "matured": matured,
        "clean": clean,
        "dirty": dirty,
        "accrued": accrued,
        "bid": bid,
        "ask": ask,
        "halfSpread": half_spread_points(years) if not matured else 0.0,
        "ytm": ytm,
        "macDuration": mac,
        "modDuration": mod,
        "convexity": conv,
        "couponCashPoints": cash_points,
        "couponCashDollars": cash_points * dollars_per_point,
        "face": spec.face,
    }


def issue_generic(ticker, settlement, par_yield, face=1000.0):
    """Create a generic on-the-run-like Treasury: coupon equals the par yield, maturity = settlement + tenor."""
    if ticker not in GENERIC_TENORS:
        raise ValueError(f"unknown generic Treasury {ticker}; one of {sorted(GENERIC_TENORS)}")
    settlement = as_date(settlement)
    tenor = GENERIC_TENORS[ticker]
    maturity = add_years(settlement, tenor)
    kind = "bill" if tenor <= 1.0 else "note"
    coupon = 0.0 if kind == "bill" else float(par_yield)
    return BondSpec(ticker=ticker, issue=settlement, maturity=maturity, coupon=coupon, kind=kind, face=face)


def mark_from_curve(spec, settlement, curve_row):
    from macro_data import interpolate_par_yield
    years = spec.remaining_years(settlement)
    if years <= 0:
        return mark_bond(spec, settlement, 0.0)
    ytm_pct = interpolate_par_yield(curve_row, years)
    ytm = float(ytm_pct) / 100.0 if math.isfinite(ytm_pct) else float("nan")
    return mark_bond(spec, settlement, ytm)


@dataclass
class PaperTreasuryBook:
    """A local paper ledger of generic Treasuries. No broker calls."""
    cash: float = 100_000.0
    positions: dict = field(default_factory=dict)  # ticker -> {spec, quantity, cost_dirty}
    history: list = field(default_factory=list)

    def _curve_row(self, settlement, curve):
        if curve is None or curve.empty:
            return None
        settlement = pd.Timestamp(as_date(settlement))
        available = curve.loc[:settlement]
        if available.empty:
            return None
        return available.iloc[-1]

    def mark(self, settlement, curve):
        row = self._curve_row(settlement, curve)
        marks = {}
        cash_in = 0.0
        drop = []
        for ticker, pos in self.positions.items():
            spec = pos["spec"]
            mark = mark_from_curve(spec, settlement, row)
            qty = pos["quantity"]
            cash_in += mark["couponCashDollars"] * qty
            marks[ticker] = {**mark, "quantity": qty,
                             "marketValue": (mark["dirty"] * (spec.face / PAR) * qty) if math.isfinite(mark["dirty"]) else float("nan")}
            if mark["matured"]:
                # couponCashDollars already includes par plus the final coupon (or par for bills).
                drop.append(ticker)
        self.cash += cash_in
        for ticker in drop:
            self.history.append({"date": as_date(settlement).isoformat(), "action": "mature", "ticker": ticker,
                                 "quantity": self.positions[ticker]["quantity"]})
            del self.positions[ticker]
        equity = self.cash + sum(m["marketValue"] for m in marks.values() if m["ticker"] not in drop and math.isfinite(m.get("marketValue", float("nan"))))
        return {"settlement": as_date(settlement).isoformat(), "cash": self.cash, "equity": equity,
                "positions": {k: v for k, v in marks.items() if k not in drop}}

    def buy(self, ticker, settlement, curve, quantity=1, face=1000.0):
        """Buy `quantity` units at the ask (clean+accrued, i.e. dirty ask)."""
        settlement = as_date(settlement)
        row = self._curve_row(settlement, curve)
        if row is None:
            raise ValueError("no FRED curve available on or before this settlement date")
        from macro_data import interpolate_par_yield
        tenor = GENERIC_TENORS[ticker]
        ytm = interpolate_par_yield(row, tenor) / 100.0
        if ticker in self.positions:
            spec = self.positions[ticker]["spec"]
        else:
            spec = issue_generic(ticker, settlement, ytm, face=face)
        mark = mark_from_curve(spec, settlement, row)
        dirty_ask = mark["ask"] + mark["accrued"] if math.isfinite(mark["ask"]) else float("nan")
        cost = dirty_ask * (spec.face / PAR) * quantity
        if not math.isfinite(cost) or cost > self.cash + 1e-8:
            raise ValueError(f"insufficient cash to buy {quantity} {ticker} (need {cost}, have {self.cash})")
        self.cash -= cost
        if ticker in self.positions:
            prev = self.positions[ticker]
            new_qty = prev["quantity"] + quantity
            prev["cost_dirty"] = (prev["cost_dirty"] * prev["quantity"] + cost) / new_qty
            prev["quantity"] = new_qty
        else:
            self.positions[ticker] = {"spec": spec, "quantity": quantity, "cost_dirty": cost / quantity}
        self.history.append({"date": settlement.isoformat(), "action": "buy", "ticker": ticker,
                             "quantity": quantity, "dirtyAsk": dirty_ask, "cost": cost})
        return self.positions[ticker]

    def sell(self, ticker, settlement, curve, quantity=None):
        settlement = as_date(settlement)
        if ticker not in self.positions:
            raise ValueError(f"no position in {ticker}")
        pos = self.positions[ticker]
        quantity = pos["quantity"] if quantity is None else quantity
        if quantity > pos["quantity"] + 1e-12:
            raise ValueError("quantity exceeds position")
        row = self._curve_row(settlement, curve)
        mark = mark_from_curve(pos["spec"], settlement, row)
        dirty_bid = mark["bid"] + mark["accrued"] if math.isfinite(mark["bid"]) else float("nan")
        proceeds = dirty_bid * (pos["spec"].face / PAR) * quantity
        self.cash += proceeds
        pos["quantity"] -= quantity
        if pos["quantity"] <= 1e-12:
            del self.positions[ticker]
        self.history.append({"date": settlement.isoformat(), "action": "sell", "ticker": ticker,
                             "quantity": quantity, "dirtyBid": dirty_bid, "proceeds": proceeds})
        return proceeds
