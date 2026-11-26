"""US taxable-account lot ledger, wash-sale tracking and after-tax metrics.

This is a planning model, not tax advice. Rates, lot method and policy flags are
config-gated so the pre-tax engine path is unchanged when tax_aware is off.

Default federal rates (32% short-term, 15% long-term) are placeholders for a
hypothetical taxable account, not a filing status. State and NIIT default to 0
and can be set on TaxConfig. Wash sales use a 30-day window and a small list of
substantially identical ETF groups; they are tracked inside this account only
(including the paper account). A real taxpayer must also consider other accounts.

Simplifications: taxes are settled from cash on the first session of the next
calendar year (no estimated-tax penalties); the $3,000 capital-loss deduction
and loss carryforwards are modelled; state tax is a flat add-on with no SALT
deduction; spin-off cash adjustments from the simulator are not treated as
dividends; no AMT, collectibles rate, QSBS or wash-sale matching across accounts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta

SCHEMA = """
CREATE TABLE IF NOT EXISTS tax_lots (
  lot_id INTEGER PRIMARY KEY AUTOINCREMENT,
  account TEXT NOT NULL,
  ticker TEXT NOT NULL,
  shares REAL NOT NULL,
  remaining REAL NOT NULL,
  cost_per_share REAL NOT NULL,
  acquired TEXT NOT NULL,
  original_acquired TEXT NOT NULL,
  wash_adjustment REAL NOT NULL DEFAULT 0,
  closed TEXT,
  status TEXT NOT NULL DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS tax_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  account TEXT NOT NULL,
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  kind TEXT NOT NULL,
  shares REAL, proceeds REAL, basis REAL,
  st_gain REAL, lt_gain REAL, tax REAL,
  disallowed_loss REAL, note TEXT, lot_id INTEGER
);
CREATE TABLE IF NOT EXISTS tax_years (
  account TEXT NOT NULL,
  year INTEGER NOT NULL,
  st_gain REAL NOT NULL DEFAULT 0,
  lt_gain REAL NOT NULL DEFAULT 0,
  qualified_div REAL NOT NULL DEFAULT 0,
  ordinary_div REAL NOT NULL DEFAULT 0,
  disallowed_wash REAL NOT NULL DEFAULT 0,
  tax_accrued REAL NOT NULL DEFAULT 0,
  tax_paid REAL NOT NULL DEFAULT 0,
  carry_st REAL NOT NULL DEFAULT 0,
  carry_lt REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (account, year)
);
CREATE TABLE IF NOT EXISTS tax_pending_wash (
  account TEXT NOT NULL,
  ticker TEXT NOT NULL,
  sale_date TEXT NOT NULL,
  shares REAL NOT NULL,
  remaining REAL NOT NULL,
  loss_per_share REAL NOT NULL,
  original_acquired TEXT NOT NULL,
  sold_ticker TEXT NOT NULL
);
"""

# Same-index (or close equivalent) ETFs treated as substantially identical for wash sales.
DEFAULT_IDENTICAL_GROUPS = (
    frozenset({"SPY", "IVV", "VOO", "SPLG"}),
    frozenset({"QQQ", "QQQM"}),
    frozenset({"IWM", "VTWO"}),
    frozenset({"VTI", "ITOT", "SPTM"}),
)

# Correlated but not substantially identical substitutes used for loss harvesting.
# Only pairs that stay inside the project's CORE_ETFS (or common companions) are listed.
DEFAULT_HARVEST_SUBSTITUTES = {
    "SPY": "DIA", "DIA": "SPY",
    "QQQ": "XLK", "XLK": "QQQ",
    "TLT": "IEF", "IEF": "TLT",
    "EFA": "EEM",
    "IVV": "DIA", "VOO": "DIA",
    "QQQM": "XLK",
    "VTI": "SPY", "ITOT": "SPY",
    "IWM": "IJR", "VTWO": "IJR", "IJR": "IWM",
}

ANNUAL_LOSS_OFFSET = 3_000.0
QD_HOLD_DAYS = 61  # more than 60 days
WASH_DAYS = 30
LT_HOLDING = 365  # "more than one year" is handled by anniversary, not this raw count


def _as_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if hasattr(value, "date") and callable(value.date):
        try:
            return value.date()
        except Exception:
            pass
    text = str(value)[:10]
    return date.fromisoformat(text)


def _iso(value):
    value = _as_date(value)
    return None if value is None else value.isoformat()


def is_long_term(acquired, sold):
    """Long-term if sold after the one-year anniversary of the acquisition date (IRS: more than one year)."""
    acquired, sold = _as_date(acquired), _as_date(sold)
    try:
        anniversary = acquired.replace(year=acquired.year + 1)
    except ValueError:
        anniversary = acquired.replace(year=acquired.year + 1, day=28)
    return sold > anniversary


def days_to_long_term(acquired, asof):
    acquired, asof = _as_date(acquired), _as_date(asof)
    try:
        anniversary = acquired.replace(year=acquired.year + 1)
    except ValueError:
        anniversary = acquired.replace(year=acquired.year + 1, day=28)
    return (anniversary - asof).days + 1


def qualifies_for_dividend(acquired, ex_date):
    """Shares must be held more than 60 days in the 121-day window around the ex-date."""
    acquired, ex_date = _as_date(acquired), _as_date(ex_date)
    window_start = ex_date - timedelta(days=60)
    window_end = ex_date + timedelta(days=60)
    held_from = max(acquired, window_start)
    # Held through the ex-date at least; we assume the lot is still open on the ex-date.
    # Inclusive day count: (ex - (ex-60)) + 1 = 61, which is "more than 60 days".
    # Days after the ex-date are ignored here (conservative: a lot that only qualifies
    # by being held after the ex-date is treated as ordinary).
    held_to = min(ex_date, window_end)
    return (held_to - held_from).days + 1 > 60


@dataclass
class TaxConfig:
    enabled: bool = False
    method: str = "fifo"  # fifo | hifo | specific
    federal_st_rate: float = 0.32
    federal_lt_rate: float = 0.15
    niit_rate: float = 0.0
    state_rate: float = 0.0
    qualified_dividend_rate: float | None = None
    wash_sale_days: int = WASH_DAYS
    defer_near_lt: bool = True
    defer_window_days: int = 30
    defer_max_signal: float = 0.15  # max weight trim treated as a "weak" signal
    harvest_losses: bool = True
    harvest_min_loss: float = 100.0
    harvest_min_loss_pct: float = 0.05
    avoid_wash_sales: bool = True
    stock_harvest_etf: str | None = None
    identical_groups: tuple = DEFAULT_IDENTICAL_GROUPS
    harvest_substitutes: dict = field(default_factory=lambda: dict(DEFAULT_HARVEST_SUBSTITUTES))
    annual_loss_offset: float = ANNUAL_LOSS_OFFSET

    def __post_init__(self):
        method = (self.method or "fifo").lower()
        if method not in {"fifo", "hifo", "specific"}:
            raise ValueError(f"Unknown lot method {self.method!r}; use fifo, hifo or specific.")
        self.method = method
        if self.identical_groups and not isinstance(self.identical_groups[0], frozenset):
            self.identical_groups = tuple(frozenset(g) for g in self.identical_groups)
        if not isinstance(self.harvest_substitutes, dict):
            self.harvest_substitutes = dict(self.harvest_substitutes)

    @classmethod
    def from_dict(cls, raw=None):
        raw = dict(raw or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in raw.items() if k in known}
        if "identical_groups" in data:
            data["identical_groups"] = tuple(frozenset(g) for g in data["identical_groups"])
        return cls(**data)

    @classmethod
    def accounting(cls, **overrides):
        """Track lots and pay tax, but do not change trade decisions."""
        return cls(enabled=True, defer_near_lt=False, harvest_losses=False, avoid_wash_sales=False, **overrides)

    @classmethod
    def aware(cls, **overrides):
        """Track lots, pay tax, and apply defer / harvest / wash-avoidance policy."""
        return cls(enabled=True, **overrides)

    def st_rate(self):
        return self.federal_st_rate + self.niit_rate + self.state_rate

    def lt_rate(self):
        return self.federal_lt_rate + self.niit_rate + self.state_rate

    def qd_rate(self):
        base = self.federal_lt_rate if self.qualified_dividend_rate is None else self.qualified_dividend_rate
        return base + self.niit_rate + self.state_rate

    def identical_set(self, ticker):
        ticker = str(ticker).upper()
        names = {ticker}
        for group in self.identical_groups:
            if ticker in group:
                names |= set(group)
        return names

    def is_substantially_identical(self, a, b):
        return str(a).upper() in self.identical_set(b)

    def substitute(self, ticker, available=None):
        ticker = str(ticker).upper()
        choice = self.harvest_substitutes.get(ticker)
        if choice and not self.is_substantially_identical(ticker, choice):
            if available is None or choice in available:
                return choice
        if self.stock_harvest_etf:
            alt = str(self.stock_harvest_etf).upper()
            if alt != ticker and not self.is_substantially_identical(ticker, alt):
                if available is None or alt in available:
                    return alt
        return None

    def as_dict(self):
        data = asdict(self)
        data["identical_groups"] = [sorted(g) for g in self.identical_groups]
        return data


@dataclass
class Lot:
    lot_id: int
    ticker: str
    shares: float  # remaining open shares
    cost_per_share: float
    acquired: date
    original_acquired: date
    wash_adjustment: float = 0.0
    closed: date | None = None

    @property
    def status(self):
        return "closed" if self.shares <= 1e-12 else "open"

    def holding_period_start(self):
        return self.original_acquired


@dataclass
class WashPending:
    ticker: str  # replacement identity (the sold ticker's identical group key)
    sold_ticker: str
    sale_date: date
    remaining: float
    loss_per_share: float
    original_acquired: date


@dataclass
class YearStats:
    year: int
    st_gain: float = 0.0
    lt_gain: float = 0.0
    qualified_div: float = 0.0
    ordinary_div: float = 0.0
    disallowed_wash: float = 0.0
    tax_accrued: float = 0.0
    tax_paid: float = 0.0
    carry_st: float = 0.0
    carry_lt: float = 0.0

    def as_dict(self):
        return asdict(self)


def compute_year_tax(config: TaxConfig, st_gain, lt_gain, qualified_div, ordinary_div, carry_st=0.0, carry_lt=0.0):
    """Net ST/LT, apply the $3,000 ordinary offset and return a dict of tax, carries and taxable dividends."""
    return _year_tax_clean(config, st_gain, lt_gain, qualified_div, ordinary_div, carry_st, carry_lt)


def _year_tax_clean(config, st_gain, lt_gain, qualified_div, ordinary_div, carry_st=0.0, carry_lt=0.0):
    st = st_gain + carry_st
    lt = lt_gain + carry_lt
    if st < 0 and lt > 0:
        if lt >= -st:
            lt, st = lt + st, 0.0
        else:
            st, lt = st + lt, 0.0
    elif lt < 0 and st > 0:
        if st >= -lt:
            st, lt = st + lt, 0.0
        else:
            lt, st = lt + st, 0.0
    ordinary = max(ordinary_div, 0.0)
    qualified = max(qualified_div, 0.0)
    offset_cap = config.annual_loss_offset
    capital_loss = max(0.0, -(min(st, 0.0) + min(lt, 0.0)))
    offset = min(capital_loss, offset_cap)
    # Apply offset: first to ST loss, then LT loss. The deducted amount reduces taxable ordinary/QD income.
    leftover_offset = offset
    if st < 0:
        take = min(-st, leftover_offset)
        st += take
        leftover_offset -= take
    if lt < 0 and leftover_offset:
        take = min(-lt, leftover_offset)
        lt += take
        leftover_offset -= take
    taxable_ordinary = ordinary
    taxable_qd = qualified
    income_offset = offset
    use = min(taxable_ordinary, income_offset)
    taxable_ordinary -= use
    income_offset -= use
    use = min(taxable_qd, income_offset)
    taxable_qd -= use
    tax = (max(st, 0.0) * config.st_rate() + max(lt, 0.0) * config.lt_rate()
           + taxable_qd * config.qd_rate() + taxable_ordinary * config.st_rate())
    return {"tax": tax, "carrySt": min(st, 0.0), "carryLt": min(lt, 0.0), "offsetUsed": offset,
            "netSt": st, "netLt": lt, "taxableOrdinaryDiv": taxable_ordinary, "taxableQualifiedDiv": taxable_qd}


class LotLedger:
    """In-memory lot ledger used by the backtest simulator and (via sqlite) the paper trader."""

    def __init__(self, config=None, account="default"):
        self.config = config or TaxConfig()
        self.account = account
        self.lots: list[Lot] = []
        self._next_id = 1
        self.events: list[dict] = []
        self.pending_wash: list[WashPending] = []
        self.years: dict[int, YearStats] = {}
        self.unpaid_tax = 0.0
        self.paid_tax = 0.0
        self.last_settled_year: int | None = None
        self._purchases: list[tuple[date, str, int, float]] = []  # date, ticker, lot_id, shares (for lookback)

    def year(self, when):
        y = _as_date(when).year
        if y not in self.years:
            prev = self.years.get(y - 1)
            carry_st = prev.carry_st if prev else 0.0
            carry_lt = prev.carry_lt if prev else 0.0
            # carry is finalized at settlement; for an open year start at 0 until settle copies it
            self.years[y] = YearStats(year=y)
        return self.years[y]

    def _record(self, when, ticker, kind, **fields):
        event = {"date": _iso(when), "ticker": ticker, "kind": kind, **fields}
        self.events.append(event)
        return event

    def open_lots(self, ticker=None):
        rows = [lot for lot in self.lots if lot.shares > 1e-12]
        if ticker is not None:
            rows = [lot for lot in rows if lot.ticker == ticker]
        return rows

    def shares_of(self, ticker):
        return sum(lot.shares for lot in self.open_lots(ticker))

    def _select_lots(self, ticker, shares, method=None, specific_ids=None):
        method = (method or self.config.method).lower()
        open_lots = self.open_lots(ticker)
        if specific_ids:
            order = {int(i): n for n, i in enumerate(specific_ids)}
            open_lots = sorted(open_lots, key=lambda lot: order.get(lot.lot_id, 1_000_000 + lot.lot_id))
        elif method == "hifo":
            open_lots = sorted(open_lots, key=lambda lot: (-lot.cost_per_share, lot.acquired, lot.lot_id))
        else:  # fifo and default for specific without ids
            open_lots = sorted(open_lots, key=lambda lot: (lot.acquired, lot.lot_id))
        remaining = shares
        picked = []
        for lot in open_lots:
            if remaining <= 1e-12:
                break
            take = min(lot.shares, remaining)
            picked.append((lot, take))
            remaining -= take
        if remaining > 1e-6:
            raise ValueError(f"Cannot sell {shares} {ticker}; only {shares - remaining} shares in open lots.")
        return picked

    def _reverse_recognized_loss(self, pending, amount, when):
        """Take a previously recognized loss back out of year stats once a replacement is bought."""
        sale_year = pending.sale_date.year
        if sale_year in self.years and (self.last_settled_year is None or sale_year > self.last_settled_year):
            stats = self.years[sale_year]
            accrue_on = pending.sale_date
        else:
            stats = self.year(when)
            accrue_on = when
        if is_long_term(pending.original_acquired, pending.sale_date):
            stats.lt_gain += amount
        else:
            stats.st_gain += amount
        stats.disallowed_wash += amount
        self._accrue_year(accrue_on)
        if accrue_on != _as_date(when):
            self._accrue_year(when)

    def _match_wash_on_buy(self, ticker, shares, when):
        """Apply pending (sale-then-buy) wash to a new purchase. Returns (shares, extra_basis, tacked_date)."""
        when = _as_date(when)
        window = self.config.wash_sale_days
        extra_basis = 0.0
        tacked = None
        left = shares
        for pending in list(self.pending_wash):
            if left <= 1e-12:
                break
            if pending.remaining <= 1e-12:
                continue
            if ticker not in self.config.identical_set(pending.sold_ticker):
                continue
            if when < pending.sale_date or (when - pending.sale_date).days > window:
                continue
            take = min(left, pending.remaining)
            amount = take * pending.loss_per_share
            extra_basis += amount
            pending.remaining -= take
            left -= take
            if tacked is None or pending.original_acquired < tacked:
                tacked = pending.original_acquired
            self._reverse_recognized_loss(pending, amount, when)
            self._record(when, ticker, "wash_disallow", shares=take, disallowed_loss=amount,
                         note=f"replacement for {pending.sold_ticker} sold {pending.sale_date.isoformat()}")
        self.pending_wash = [p for p in self.pending_wash if p.remaining > 1e-12]
        return shares, extra_basis, tacked

    def buy(self, ticker, shares, price, when, costs=0.0):
        if shares <= 0:
            return None
        when = _as_date(when)
        ticker = str(ticker).upper()
        _, extra_basis, tacked = self._match_wash_on_buy(ticker, shares, when)
        basis = shares * price + max(costs, 0.0) + extra_basis
        lot = Lot(lot_id=self._next_id, ticker=ticker, shares=shares,
                  cost_per_share=basis / shares, acquired=when,
                  original_acquired=tacked or when, wash_adjustment=extra_basis)
        self._next_id += 1
        self.lots.append(lot)
        self._purchases.append((when, ticker, lot.lot_id, shares))
        self._record(when, ticker, "buy", shares=shares, proceeds=-(shares * price), basis=basis,
                     lot_id=lot.lot_id, disallowed_loss=extra_basis or None,
                     note="wash replacement basis added" if extra_basis else None)
        return lot

    def _lookback_replacements(self, ticker, when, shares):
        """Purchases of substantially identical securities in the 30 days before (and including) the sale."""
        when = _as_date(when)
        window = self.config.wash_sale_days
        identical = self.config.identical_set(ticker)
        found = []
        for lot in self.lots:
            if lot.ticker not in identical:
                continue
            if lot.lot_id is None:
                continue
            acquired = lot.acquired
            delta = (when - acquired).days
            if 0 <= delta <= window and lot.shares > 1e-12:
                # remaining shares of a lot bought in the window can be replacements
                found.append(lot)
        found.sort(key=lambda lot: (lot.acquired, lot.lot_id))
        return found

    def sell(self, ticker, shares, price, when, costs=0.0, method=None, specific_ids=None):
        if shares <= 0:
            return {"st_gain": 0.0, "lt_gain": 0.0, "disallowed": 0.0, "tax": 0.0, "lots": []}
        when = _as_date(when)
        ticker = str(ticker).upper()
        picked = self._select_lots(ticker, shares, method, specific_ids)
        st_gain = lt_gain = disallowed = 0.0
        cost_alloc = costs
        total = shares
        details = []
        # Pre-compute replacement capacity from lookback purchases still open
        replacements = self._lookback_replacements(ticker, when, shares)
        replace_left = {lot.lot_id: lot.shares for lot in replacements}

        for lot, take in picked:
            fee = cost_alloc * (take / total) if total else 0.0
            proceeds = take * price - fee
            basis = take * lot.cost_per_share
            gain = proceeds - basis
            long_term = is_long_term(lot.holding_period_start(), when)
            washed = 0.0
            if gain < 0:
                # Match against lookback replacement lots first, then record leftover as pending.
                need = take
                for repl in replacements:
                    if need <= 1e-12:
                        break
                    if repl.lot_id == lot.lot_id:
                        continue  # a lot cannot replace itself
                    avail = replace_left.get(repl.lot_id, 0.0)
                    if avail <= 1e-12:
                        continue
                    match = min(need, avail)
                    part = match / take * (-gain)
                    washed += part
                    repl.cost_per_share += part / repl.shares if repl.shares else 0.0
                    repl.wash_adjustment += part
                    if lot.holding_period_start() < repl.original_acquired:
                        repl.original_acquired = lot.holding_period_start()
                    replace_left[repl.lot_id] = avail - match
                    need -= match
                if need > 1e-12 and gain < 0:
                    # pending forward wash (sale then later buy)
                    self.pending_wash.append(WashPending(
                        ticker=ticker, sold_ticker=ticker, sale_date=when,
                        remaining=need, loss_per_share=(-gain) / take,
                        original_acquired=lot.holding_period_start()))
                    # The loss stays tentatively realized until a replacement is bought; if a buy
                    # arrives within 30 days it will be reversed via basis on the new lot. For
                    # accounting during the window we still disallow the unmatched portion only once
                    # a replacement is actually bought (see _match_wash_on_buy). So lookback is
                    # disallowed now; forward is disallowed later.
            recognized = gain + washed  # washed is positive (the disallowed loss), so recognized is less negative
            # wait: gain is negative, washed is the disallowed amount (positive). recognized = gain + washed
            # e.g. gain -100, washed 40 -> recognized -60. Good.
            # If all washed, recognized 0.
            if washed:
                recognized = gain + washed
            else:
                recognized = gain
            if long_term:
                lt_gain += recognized
            else:
                st_gain += recognized
            disallowed += washed
            lot.shares -= take
            if lot.shares <= 1e-12:
                lot.shares = 0.0
                lot.closed = when
            details.append({"lot_id": lot.lot_id, "shares": take, "proceeds": proceeds, "basis": basis,
                            "gain": gain, "recognized": recognized, "disallowed": washed,
                            "longTerm": long_term, "acquired": _iso(lot.holding_period_start())})
            self._record(when, ticker, "sell", shares=take, proceeds=proceeds, basis=basis,
                         st_gain=0.0 if long_term else recognized, lt_gain=recognized if long_term else 0.0,
                         disallowed_loss=washed or None, lot_id=lot.lot_id,
                         note="long-term" if long_term else "short-term")

        stats = self.year(when)
        stats.st_gain += st_gain
        stats.lt_gain += lt_gain
        stats.disallowed_wash += disallowed
        # Incremental tax vs previous accrual for this year
        tax_now = self._accrue_year(when)
        return {"st_gain": st_gain, "lt_gain": lt_gain, "disallowed": disallowed, "tax": tax_now,
                "lots": details}

    def _accrue_year(self, when):
        """Recompute the year's tax and update unpaid_tax for the change in accrued amount."""
        when = _as_date(when)
        stats = self.year(when)
        carry_st = carry_lt = 0.0
        prev = self.years.get(when.year - 1)
        if prev is not None:
            carry_st, carry_lt = prev.carry_st, prev.carry_lt
        elif self.last_settled_year is not None and self.last_settled_year < when.year:
            settled = self.years.get(self.last_settled_year)
            if settled:
                carry_st, carry_lt = settled.carry_st, settled.carry_lt
        result = _year_tax_clean(self.config, stats.st_gain, stats.lt_gain, stats.qualified_div,
                                 stats.ordinary_div, carry_st, carry_lt)
        new_tax = result["tax"]
        delta = new_tax - stats.tax_accrued
        stats.tax_accrued = new_tax
        self.unpaid_tax += delta
        return new_tax

    def dividend(self, ticker, per_share, when, shares=None):
        ticker = str(ticker).upper()
        when = _as_date(when)
        lots = self.open_lots(ticker)
        if shares is None:
            shares = sum(lot.shares for lot in lots)
        if not shares or not per_share:
            return {"qualified": 0.0, "ordinary": 0.0}
        # Allocate the dividend across lots: qualified if that lot meets the holding test.
        remaining = shares
        qualified = ordinary = 0.0
        for lot in sorted(lots, key=lambda item: (item.acquired, item.lot_id)):
            if remaining <= 1e-12:
                break
            take = min(lot.shares, remaining)
            amount = take * per_share
            if qualifies_for_dividend(lot.holding_period_start(), when):
                qualified += amount
            else:
                ordinary += amount
            remaining -= take
        if remaining > 1e-8:
            # no lots (shouldn't happen if the book is consistent); treat as ordinary
            ordinary += remaining * per_share
        stats = self.year(when)
        stats.qualified_div += qualified
        stats.ordinary_div += ordinary
        self._accrue_year(when)
        if qualified or ordinary:
            self._record(when, ticker, "dividend", shares=shares, proceeds=shares * per_share,
                         note=f"qualified {qualified:.2f}, ordinary {ordinary:.2f}")
        return {"qualified": qualified, "ordinary": ordinary}

    def maybe_settle(self, when, cash=None):
        """Pay prior-year tax on the first session of a new calendar year. Returns cash paid."""
        when = _as_date(when)
        paid = 0.0
        for year, stats in sorted(self.years.items()):
            if year >= when.year:
                continue
            if stats.tax_paid > 0 and abs(stats.tax_accrued - stats.tax_paid) < 1e-9:
                continue
            # Finalize carry into this year (and the next open year).
            prev = self.years.get(year - 1)
            carry_st = prev.carry_st if prev else 0.0
            carry_lt = prev.carry_lt if prev else 0.0
            result = _year_tax_clean(self.config, stats.st_gain, stats.lt_gain, stats.qualified_div,
                                     stats.ordinary_div, carry_st, carry_lt)
            due = result["tax"] - stats.tax_paid
            stats.tax_accrued = result["tax"]
            stats.carry_st = result["carrySt"]
            stats.carry_lt = result["carryLt"]
            nxt = self.year(date(year + 1, 1, 1))
            nxt.carry_st, nxt.carry_lt = result["carrySt"], result["carryLt"]
            if due > 1e-9:
                stats.tax_paid += due
                self.paid_tax += due
                self.unpaid_tax = max(0.0, self.unpaid_tax - due)
                paid += due
                self._record(when, "CASH", "tax_payment", tax=due, note=f"settlement for {year}")
            self.last_settled_year = year
        # Drop expired unmatched pending washes (loss stays realized — already the case).
        self.pending_wash = [p for p in self.pending_wash
                             if (when - p.sale_date).days <= self.config.wash_sale_days]
        return paid

    def recently_sold_at_loss(self, ticker, when):
        """True if this name (or a substantially identical one) was sold at a recognized or pending loss in the wash window."""
        when = _as_date(when)
        identical = self.config.identical_set(ticker)
        window = self.config.wash_sale_days
        for pending in self.pending_wash:
            if pending.sold_ticker in identical or pending.ticker in identical:
                if 0 <= (when - pending.sale_date).days <= window:
                    return True
        for event in self.events:
            if event["kind"] != "sell":
                continue
            if event["ticker"] not in identical:
                continue
            sold = _as_date(event["date"])
            if not (0 <= (when - sold).days <= window):
                continue
            gain = (event.get("st_gain") or 0.0) + (event.get("lt_gain") or 0.0)
            if gain < -1e-9 or (event.get("disallowed_loss") or 0.0) > 1e-9:
                return True
        return False

    def loss_sale_in_window(self, ticker, when):
        return self.recently_sold_at_loss(ticker, when)

    def unrealized(self, prices, when):
        when = _as_date(when)
        rows = []
        for lot in self.open_lots():
            price = prices.get(lot.ticker)
            if price is None or not math.isfinite(price):
                continue
            market = lot.shares * price
            basis = lot.shares * lot.cost_per_share
            gain = market - basis
            long_term = is_long_term(lot.holding_period_start(), when)
            rows.append({
                "lotId": lot.lot_id, "ticker": lot.ticker, "shares": lot.shares,
                "acquired": _iso(lot.acquired), "holdingPeriodStart": _iso(lot.holding_period_start()),
                "costPerShare": lot.cost_per_share, "price": price, "marketValue": market,
                "costBasis": basis, "unrealized": gain,
                "character": "long-term" if long_term else "short-term",
                "daysToLongTerm": 0 if long_term else max(0, days_to_long_term(lot.holding_period_start(), when)),
                "washAdjustment": lot.wash_adjustment,
            })
        return rows

    def harvest_candidates(self, prices, when):
        when = _as_date(when)
        by_ticker = {}
        for row in self.unrealized(prices, when):
            if row["unrealized"] >= 0:
                continue
            item = by_ticker.setdefault(row["ticker"], {"ticker": row["ticker"], "shares": 0.0,
                                                        "unrealized": 0.0, "costBasis": 0.0,
                                                        "marketValue": 0.0, "lots": []})
            item["shares"] += row["shares"]
            item["unrealized"] += row["unrealized"]
            item["costBasis"] += row["costBasis"]
            item["marketValue"] += row["marketValue"]
            item["lots"].append(row)
        out = []
        for ticker, item in sorted(by_ticker.items()):
            loss = -item["unrealized"]
            pct = loss / item["costBasis"] if item["costBasis"] else 0.0
            if loss < self.config.harvest_min_loss and pct < self.config.harvest_min_loss_pct:
                continue
            item["loss"] = loss
            item["lossPct"] = pct
            item["substitute"] = self.config.substitute(ticker, set(prices))
            item["washWindow"] = self.recently_sold_at_loss(ticker, when)
            out.append(item)
        return out

    def estimated_tax_bill(self, prices=None, when=None, include_unrealized=False):
        when = _as_date(when) or date.today()
        # Current-year accrued + unpaid prior + optional liquidation tax on unrealized
        realized_unpaid = self.unpaid_tax
        unrealized_tax = 0.0
        st_u = lt_u = 0.0
        if include_unrealized and prices:
            for row in self.unrealized(prices, when):
                if row["character"] == "long-term":
                    lt_u += row["unrealized"]
                else:
                    st_u += row["unrealized"]
            # Liquidation: add unrealized to this year's realized and recompute
            stats = self.years.get(when.year) or YearStats(year=when.year)
            prev = self.years.get(when.year - 1)
            carry_st = prev.carry_st if prev else 0.0
            carry_lt = prev.carry_lt if prev else 0.0
            liq = _year_tax_clean(self.config, stats.st_gain + st_u, stats.lt_gain + lt_u,
                                  stats.qualified_div, stats.ordinary_div, carry_st, carry_lt)
            unrealized_tax = max(0.0, liq["tax"] - stats.tax_accrued)
        years = {str(y): s.as_dict() for y, s in sorted(self.years.items())}
        return {
            "asOf": _iso(when),
            "unpaidRealizedTax": realized_unpaid,
            "taxPaidToDate": self.paid_tax,
            "unrealizedSt": st_u, "unrealizedLt": lt_u,
            "liquidationTax": unrealized_tax,
            "estimatedBillIfLiquidated": realized_unpaid + unrealized_tax,
            "rates": {"shortTerm": self.config.st_rate(), "longTerm": self.config.lt_rate(),
                      "qualifiedDividend": self.config.qd_rate()},
            "years": years,
            "disclaimer": "Planning estimate only; not tax advice.",
        }

    def snapshot(self, prices=None, when=None):
        when = _as_date(when) or date.today()
        prices = prices or {}
        lots = self.unrealized(prices, when)
        st = sum(r["unrealized"] for r in lots if r["character"] == "short-term")
        lt = sum(r["unrealized"] for r in lots if r["character"] == "long-term")
        bill = self.estimated_tax_bill(prices, when, include_unrealized=True)
        return {
            "enabled": self.config.enabled,
            "config": self.config.as_dict(),
            "asOf": _iso(when),
            "openLots": lots,
            "unrealizedShortTerm": st,
            "unrealizedLongTerm": lt,
            "harvestCandidates": self.harvest_candidates(prices, when) if prices else [],
            "taxBill": bill,
            "pendingWash": [{"ticker": p.sold_ticker, "saleDate": _iso(p.sale_date),
                             "remaining": p.remaining, "lossPerShare": p.loss_per_share}
                            for p in self.pending_wash],
            "recentEvents": self.events[-50:],
            "disclaimer": "Planning estimate only; not tax advice. Wash sales are tracked only inside this account.",
        }

    def report(self, prices=None, when=None):
        return self.snapshot(prices, when)

    # ---------------------------------------------------------------- sqlite
    def save_sqlite(self, connection, account=None):
        account = account or self.account
        ensure_schema(connection)
        connection.execute("DELETE FROM tax_lots WHERE account=?", (account,))
        connection.execute("DELETE FROM tax_events WHERE account=?", (account,))
        connection.execute("DELETE FROM tax_years WHERE account=?", (account,))
        connection.execute("DELETE FROM tax_pending_wash WHERE account=?", (account,))
        for lot in self.lots:
            connection.execute(
                """INSERT INTO tax_lots (lot_id, account, ticker, shares, remaining, cost_per_share, acquired,
                   original_acquired, wash_adjustment, closed, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (lot.lot_id, account, lot.ticker, lot.shares + (0 if lot.closed is None else 0),
                 lot.shares, lot.cost_per_share, _iso(lot.acquired), _iso(lot.original_acquired),
                 lot.wash_adjustment, _iso(lot.closed), lot.status))
        for event in self.events:
            connection.execute(
                """INSERT INTO tax_events (account, date, ticker, kind, shares, proceeds, basis, st_gain, lt_gain,
                   tax, disallowed_loss, note, lot_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (account, event.get("date"), event.get("ticker"), event.get("kind"), event.get("shares"),
                 event.get("proceeds"), event.get("basis"), event.get("st_gain"), event.get("lt_gain"),
                 event.get("tax"), event.get("disallowed_loss"), event.get("note"), event.get("lot_id")))
        for stats in self.years.values():
            connection.execute(
                """INSERT INTO tax_years (account, year, st_gain, lt_gain, qualified_div, ordinary_div,
                   disallowed_wash, tax_accrued, tax_paid, carry_st, carry_lt)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (account, stats.year, stats.st_gain, stats.lt_gain, stats.qualified_div, stats.ordinary_div,
                 stats.disallowed_wash, stats.tax_accrued, stats.tax_paid, stats.carry_st, stats.carry_lt))
        for pending in self.pending_wash:
            connection.execute(
                """INSERT INTO tax_pending_wash (account, ticker, sale_date, shares, remaining, loss_per_share,
                   original_acquired, sold_ticker) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (account, pending.ticker, _iso(pending.sale_date), pending.remaining, pending.remaining,
                 pending.loss_per_share, _iso(pending.original_acquired), pending.sold_ticker))
        # Keep AUTOINCREMENT ahead of our ids
        if self.lots:
            connection.execute("UPDATE sqlite_sequence SET seq=? WHERE name='tax_lots'", (self._next_id - 1,))
            # sqlite_sequence may not exist yet
            exists = connection.execute("SELECT 1 FROM sqlite_master WHERE name='sqlite_sequence'").fetchone()
            if exists and connection.execute("SELECT seq FROM sqlite_sequence WHERE name='tax_lots'").fetchone() is None:
                try:
                    connection.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('tax_lots', ?)",
                                       (self._next_id - 1,))
                except Exception:
                    pass

    @classmethod
    def load_sqlite(cls, connection, account, config=None):
        ensure_schema(connection)
        ledger = cls(config or TaxConfig(), account)
        for row in connection.execute("SELECT * FROM tax_lots WHERE account=? ORDER BY lot_id", (account,)):
            row = dict(row)
            lot = Lot(lot_id=int(row["lot_id"]), ticker=row["ticker"], shares=float(row["remaining"]),
                      cost_per_share=float(row["cost_per_share"]), acquired=_as_date(row["acquired"]),
                      original_acquired=_as_date(row["original_acquired"]),
                      wash_adjustment=float(row["wash_adjustment"] or 0),
                      closed=_as_date(row["closed"]))
            ledger.lots.append(lot)
            ledger._next_id = max(ledger._next_id, lot.lot_id + 1)
        for row in connection.execute("SELECT * FROM tax_events WHERE account=? ORDER BY event_id", (account,)):
            row = dict(row)
            ledger.events.append({k: row[k] for k in row.keys() if k not in {"event_id", "account"}})
        for row in connection.execute("SELECT * FROM tax_years WHERE account=? ORDER BY year", (account,)):
            row = dict(row)
            ledger.years[int(row["year"])] = YearStats(
                year=int(row["year"]), st_gain=row["st_gain"], lt_gain=row["lt_gain"],
                qualified_div=row["qualified_div"], ordinary_div=row["ordinary_div"],
                disallowed_wash=row["disallowed_wash"], tax_accrued=row["tax_accrued"],
                tax_paid=row["tax_paid"], carry_st=row["carry_st"], carry_lt=row["carry_lt"])
            if row["tax_paid"]:
                ledger.last_settled_year = int(row["year"])
        for row in connection.execute("SELECT * FROM tax_pending_wash WHERE account=?", (account,)):
            row = dict(row)
            ledger.pending_wash.append(WashPending(
                ticker=row["ticker"], sold_ticker=row["sold_ticker"], sale_date=_as_date(row["sale_date"]),
                remaining=float(row["remaining"]), loss_per_share=float(row["loss_per_share"]),
                original_acquired=_as_date(row["original_acquired"])))
        ledger.paid_tax = sum(s.tax_paid for s in ledger.years.values())
        ledger.unpaid_tax = max(0.0, sum(s.tax_accrued - s.tax_paid for s in ledger.years.values()))
        return ledger

    def seed_from_positions(self, positions, when=None):
        """Create one lot per existing position (avg cost, entry date) when tax-aware is first turned on."""
        if self.open_lots():
            return 0
        added = 0
        for pos in positions:
            shares = float(pos.get("shares") or 0)
            if shares <= 1e-12:
                continue
            acquired = _as_date(pos.get("entry_date") or when)
            price = float(pos.get("avg_cost") or pos.get("last_price") or 0)
            ticker = str(pos["ticker"]).upper()
            lot = Lot(lot_id=self._next_id, ticker=ticker, shares=shares, cost_per_share=price,
                      acquired=acquired, original_acquired=acquired)
            self._next_id += 1
            self.lots.append(lot)
            self._record(acquired, ticker, "seed", shares=shares, basis=shares * price, lot_id=lot.lot_id,
                         note="seeded from existing position")
            added += 1
        return added


def ensure_schema(connection):
    # Execute statements one by one. sqlite3.executescript() issues a COMMIT first,
    # which would abort an open paper-trader transaction.
    for statement in SCHEMA.split(";"):
        statement = statement.strip()
        if statement:
            connection.execute(statement)


def tax_config_from_settings(settings):
    settings = settings or {}
    raw = dict(settings.get("tax") or {})
    if "tax_aware" in settings:
        raw["enabled"] = bool(settings["tax_aware"])
    return TaxConfig.from_dict(raw)


def after_tax_performance(equity, after_tax_equity, pre_tax_performance=None):
    """After-tax CAGR / Sharpe and tax drag given pre-tax and after-tax equity series."""
    import pandas as pd
    from metrics import performance

    after_tax_equity = after_tax_equity.dropna()
    if len(after_tax_equity) < 2:
        return {"sessions": int(len(after_tax_equity))}
    returns = after_tax_equity.pct_change().dropna()
    after = performance(returns, after_tax_equity)
    drag = None
    if pre_tax_performance and after.get("cagr") is not None and pre_tax_performance.get("cagr") is not None:
        drag = pre_tax_performance["cagr"] - after["cagr"]
    return {
        "afterTax": after,
        "taxDragCagr": drag,
        "preTaxCagr": None if not pre_tax_performance else pre_tax_performance.get("cagr"),
        "preTaxSharpe": None if not pre_tax_performance else pre_tax_performance.get("sharpe"),
        "disclaimer": "Planning estimate only; not tax advice.",
    }


def summarize_tax_result(result, start=None):
    """Combine pre-tax metrics.summarize with after-tax figures when the sim tracked a ledger."""
    from metrics import summarize
    out = summarize(result, start=start)
    after_eq = getattr(result, "after_tax_equity", None)
    if after_eq is not None and len(after_eq):
        if start is not None:
            after_eq = after_eq.loc[__import__("pandas").Timestamp(start):]
            equity = result.equity.loc[__import__("pandas").Timestamp(start):]
        else:
            equity = result.equity
        out["afterTax"] = after_tax_performance(equity, after_eq, out.get("performance"))
    if getattr(result, "tax_report", None):
        out["tax"] = result.tax_report
    return out
