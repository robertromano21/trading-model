"""Paper options (and config-gated paper short-stock) engine shared by the options backtests and the paper runner.

PAPER ONLY. Nothing here talks to a broker. Prices come from a pricing provider: Schwab chain snapshots when they
exist for the date, otherwise the MODELED surface from options_pricing (positions remember which one was used, and
summaries report ``pricing: modeled|quoted|mixed``).

What it models
- Multi-leg positions from options_pricing.build_strategy templates (long call/put, verticals, covered call,
  cash-secured put, collar, protective put, straddle/strangle, iron condor/butterfly, butterflies, calendar, diagonal).
- Fills: longs pay the ask, shorts receive the bid (modeled spread), Schwab fees via OptionFees.
- Buying power: Reg-T style. Long premium paid in full; defined-risk spreads reserve their max loss; cash-secured
  puts reserve strike*100 - premium; covered calls need the shares; naked shorts use the Reg-T formula
  (max(20% * S - OTM + premium, 10% * (S or K) + premium) * 100) and are refused unless allow_undefined_risk.
  Cash accounts may not hold naked shorts or short stock at all.
- Risk engine: max total premium/max-loss at risk, per-position max loss, portfolio delta and vega limits (fractions
  of equity), integer contracts (a $10k account cannot hedge or write against SPY in size; the engine says so).
- Expiration: exercise-by-exception (ITM by >= $0.01 is exercised/assigned), pin-risk flag within pin_pct of a short
  strike, optional close of ITM / pinned legs on expiration day at modeled prices instead of taking delivery.
- Early assignment: a short call that is ITM with extrinsic value below the next dividend before its ex-date is
  assigned the session before the ex-date (the classic dividend-capture exercise). Puts: deep ITM with near-zero
  extrinsic are flagged.
- Short stock (config short_stock_enabled, default False): borrow fee accrual (annual rate, per-ticker overrides),
  Reg-T 150% initial requirement, maintenance (30% default) with forced cover, gap-aware squeeze stop (fills at the
  open when it gaps through), dividends owed on ex-dates.

Book file: Massive/PaperOptions/book.json (gitignored).
"""

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime

from fetch_and_model import DATA_DIR, write_atomically
from options_pricing import (CONTRACT_MULTIPLIER, Leg, ModeledIV, OptionFees, OptionSpreadModel, analyze_legs,
                             bsm_greeks, build_strategy, leg_from, year_fraction)

BOOK_DIR = DATA_DIR / "PaperOptions"
BOOK_PATH = BOOK_DIR / "book.json"


@dataclass
class OptionsConfig:
    enabled: bool = False                 # options overlays OFF by default (config-gated)
    account_type: str = "margin"          # "cash" | "margin" (Reg-T rules differ)
    initial_equity: float = 10_000.0
    max_premium_at_risk: float = 0.10     # sum of max losses of all option positions / equity
    max_position_loss: float = 0.03       # one position's max loss / equity
    max_abs_delta: float = 0.60           # |portfolio option delta dollars| / equity
    max_abs_vega: float = 0.005           # |portfolio vega dollars per vol point| / equity
    max_csp_notional: float = 0.30        # cash-secured put assignment value (strike*100) / equity, like a stock position cap
    allow_undefined_risk: bool = False    # naked short calls/puts, short straddles: refused unless True
    allow_naked_calls: bool = False       # separate, stricter switch
    close_itm_on_expiry: bool = True      # sell ITM longs / buy back ITM shorts at the last close instead of delivery
    pin_pct: float = 0.005
    early_assignment: bool = True
    rate: float = 0.04
    # short stock (paper only)
    short_stock_enabled: bool = False
    borrow_rate: float = 0.005            # annual general-collateral borrow fee
    borrow_overrides: dict = field(default_factory=dict)
    short_initial_req: float = 1.50       # Reg-T 150%
    short_maintenance: float = 0.30
    short_stop_pct: float = 0.10          # squeeze / gap stop above entry
    max_short_notional: float = 0.30      # / equity

    @classmethod
    def from_dict(cls, d=None):
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class RiskRefusal(ValueError):
    pass


class ModelPricer:
    """Modeled pricing provider: atm vol per underlying per date comes from a callable (date, underlying) -> vol."""

    def __init__(self, atm_vol_fn, is_etf_fn=lambda u: True, iv_model=None, spread_model=None, rate=0.04, div_yield=0.0):
        self.atm_vol_fn, self.is_etf_fn = atm_vol_fn, is_etf_fn
        self.iv_model, self.spread_model = iv_model or ModeledIV(), spread_model or OptionSpreadModel()
        self.rate, self.div_yield = rate, div_yield

    def quote(self, underlying, leg, asof, spot):
        t = max(year_fraction(asof, leg.expiration), 0.0)
        atm = self.atm_vol_fn(asof, underlying)
        is_etf = self.is_etf_fn(underlying)
        iv = self.iv_model.iv(atm, spot, leg.strike, max(t, 1 / 365), is_etf)
        g = bsm_greeks(spot, leg.strike, t, iv, self.rate, self.div_yield, leg.kind)
        if t <= 0:
            mid = g["price"]
            return {"bid": max(0.0, mid - 0.01), "ask": mid + 0.01, "mid": mid, "iv": iv, "greeks": g, "pricing": "modeled"}
        bid, ask = self.spread_model.quote(g["price"], g["delta"], t * 365, is_etf)
        return {"bid": bid, "ask": ask, "mid": g["price"], "iv": iv, "greeks": g, "pricing": "modeled"}


class SnapshotPricer:
    """Quotes from a stored Schwab chain (options_data.stored_chain); falls back to a ModelPricer per leg."""

    def __init__(self, chains, fallback):
        self.chains, self.fallback = chains, fallback   # chains: {underlying: normalized chain}

    def quote(self, underlying, leg, asof, spot):
        chain = self.chains.get(underlying)
        if chain:
            for c in chain["contracts"]:
                if c["kind"] == leg.kind and c["expiration"] == leg.expiration and abs((c["strike"] or 0) - leg.strike) < 1e-6:
                    if c.get("bid") is not None and c.get("ask") is not None and c["ask"] > 0:
                        mid = 0.5 * (c["bid"] + c["ask"])
                        g = {k: c.get(k) for k in ("delta", "gamma", "theta", "vega", "rho")}
                        if g["delta"] is None:
                            g = self.fallback.quote(underlying, leg, asof, spot)["greeks"]
                        g["price"] = mid
                        return {"bid": c["bid"], "ask": c["ask"], "mid": mid, "iv": c.get("iv"), "greeks": g, "pricing": "quoted"}
        return self.fallback.quote(underlying, leg, asof, spot)


def reg_t_requirement(legs, spot, account_type="margin"):
    """Buying power a position consumes (dollars, >= 0) and whether it is allowed in the account type."""
    legs = [leg_from(l) for l in legs]
    info = analyze_legs(legs, spot)
    stock = sum(l.qty for l in legs if l.kind == "stock")
    shorts = [l for l in legs if l.kind != "stock" and l.qty < 0]
    longs = [l for l in legs if l.kind != "stock" and l.qty > 0]
    premium_paid = sum(l.qty * CONTRACT_MULTIPLIER * l.price for l in longs)
    credit = -sum(l.qty * CONTRACT_MULTIPLIER * l.price for l in shorts)
    stock_cost = max(stock, 0) * spot
    if not shorts:
        return {"requirement": premium_paid + stock_cost, "kind": "long premium" + (" + stock" if stock else ""), "allowed": True}
    naked_puts_only = not longs and not stock and all(l.kind == "put" for l in shorts)   # bounded loss, but Reg-T margins it as naked
    if info["definedRisk"] and info["maxLoss"] is not None and not naked_puts_only:
        # spreads / covered / collars: reserve the worst case (covered calls: the stock itself)
        need = max(0.0, -info["maxLoss"]) if not stock else stock_cost - credit + premium_paid
        return {"requirement": max(need, 0.0), "kind": "defined risk (max loss reserved)", "allowed": True}  # Schwab allows defined-risk spreads / covered positions in cash and IRA accounts with approval
    # undefined risk: naked shorts. Cash-secured puts are the exception for cash accounts.
    req, naked_call = 0.0, False
    for l in shorts:
        n = -l.qty
        if l.kind == "put":
            covered_by_stock_short = stock < 0
            csp = l.strike * CONTRACT_MULTIPLIER * n - l.price * CONTRACT_MULTIPLIER * n
            if account_type == "cash":
                req += csp
            else:
                otm = max(0.0, spot - l.strike)
                req += n * CONTRACT_MULTIPLIER * (max(0.2 * spot - otm, 0.1 * l.strike) + l.price)
        else:
            calls_covered = max(stock, 0) / CONTRACT_MULTIPLIER
            if calls_covered >= n:
                continue
            naked_call = True
            otm = max(0.0, l.strike - spot)
            req += n * CONTRACT_MULTIPLIER * (max(0.2 * spot - otm, 0.1 * spot) + l.price)
    allowed = not (account_type == "cash" and naked_call)
    return {"requirement": req + premium_paid + stock_cost, "kind": "naked short (Reg-T)" if account_type == "margin" else "cash-secured",
            "allowed": allowed, "nakedCall": naked_call}


class OptionsBook:
    def __init__(self, config=None, state=None):
        self.config = config if isinstance(config, OptionsConfig) else OptionsConfig.from_dict(config)
        s = state or {}
        self.cash = float(s.get("cash", self.config.initial_equity))
        self.positions = s.get("positions", [])
        self.shorts = s.get("shorts", [])
        self.closed = s.get("closed", [])
        self.events = s.get("events", [])
        self.fees_paid = float(s.get("feesPaid", 0.0))
        self.borrow_paid = float(s.get("borrowPaid", 0.0))
        self.next_id = int(s.get("nextId", 1))
        self.last_marks = s.get("lastMarks", {})
        self.fees = OptionFees()
        # Optional account context set by the caller each session (overlay on top of an equity book):
        self.account_equity = None    # total account equity used for % limits
        self.cash_budget = None       # idle cash the options sleeve may use (book cash is then sleeve cash flows)
        self.external_delta = 0.0     # equity book delta dollars (a hedge that reduces |total delta| is allowed)

    # ------------------------------------------------------------------------------------------ state
    def state(self):
        return {"config": asdict(self.config), "cash": self.cash, "positions": self.positions, "shorts": self.shorts,
                "closed": self.closed[-500:], "events": self.events[-500:], "feesPaid": self.fees_paid,
                "borrowPaid": self.borrow_paid, "nextId": self.next_id, "lastMarks": self.last_marks}

    def log(self, asof, kind, **data):
        self.events.append({"date": str(asof)[:10], "event": kind, **data})

    # ------------------------------------------------------------------------------------------ valuation
    def _leg_quote(self, pricer, p, leg, asof, spot):
        if leg.kind == "stock":
            return {"bid": spot, "ask": spot, "mid": spot, "greeks": {"delta": 1.0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}, "pricing": "spot"}
        return pricer.quote(p["underlying"], leg, asof, spot)

    def position_value(self, p, pricer, asof, spot, side="mid"):
        """Liquidation value (dollars, signed) of a position: longs at bid, shorts at ask when side='exit'."""
        total, greeks, pricing = 0.0, {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}, set()
        for raw in p["legs"]:
            leg = leg_from(raw)
            q = self._leg_quote(pricer, p, leg, asof, spot)
            pricing.add(q["pricing"])
            m = leg.multiplier
            px = q["mid"] if side == "mid" else (q["bid"] if leg.qty > 0 else q["ask"])
            total += leg.qty * m * px
            g = q["greeks"]
            for k in greeks:
                greeks[k] += (g.get(k) or 0.0) * leg.qty * m
        return total, greeks, pricing

    def equity(self, pricer, asof, spots):
        value = self.cash
        for p in self.positions:
            value += self.position_value(p, pricer, asof, spots[p["underlying"]])[0]
        for s in self.shorts:
            value -= s["shares"] * spots.get(s["ticker"], s["entry"])
        return value

    def greeks(self, pricer, asof, spots):
        tot = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "deltaDollars": 0.0}
        for p in self.positions:
            spot = spots[p["underlying"]]
            _, g, _ = self.position_value(p, pricer, asof, spot)
            for k in ("delta", "gamma", "vega", "theta"):
                tot[k] += g[k]
            tot["deltaDollars"] += g["delta"] * spot
        for s in self.shorts:
            px = spots.get(s["ticker"], s["entry"])
            tot["delta"] -= s["shares"]
            tot["deltaDollars"] -= s["shares"] * px
        return tot

    def reserved(self, spots):
        """Cash that must stay locked: per position max(0, requirement + entry cash flow) (0 for paid debits, the
        spread width for credit spreads, strike*100 for cash-secured puts) plus the Reg-T initial requirement on shorts."""
        need = sum(p.get("reserve", 0.0) for p in self.positions)
        for s in self.shorts:
            need += s["shares"] * spots.get(s["ticker"], s["entry"]) * self.config.short_initial_req
        return need

    def requirement(self, spots):
        return self.reserved(spots)

    def free_cash(self, spots):
        if self.cash_budget is not None:
            return self.cash_budget + self.cash - self.reserved(spots)
        return self.cash - self.reserved(spots)

    def _account_equity(self, pricer, asof, spots):
        return self.account_equity if self.account_equity is not None else self.equity(pricer, asof, spots)

    # ------------------------------------------------------------------------------------------ open / close
    def open(self, strategy, underlying, legs, pricer, asof, spot, spots=None, tag="", reason="", dry_run=False,
             covered_shares=0):
        """Open a priced position after risk checks. legs: template legs (no prices). Raises RiskRefusal.
        covered_shares: shares of the underlying held in the equity book that cover short calls (covered-call overlay);
        they count for coverage and max-loss analysis but are not bought here."""
        cfg = self.config
        spots = dict(spots or {})
        spots.setdefault(underlying, spot)
        priced, pricing = [], set()
        for raw in legs:
            leg = leg_from(raw)
            if leg.kind == "stock":
                priced.append(Leg(kind="stock", qty=leg.qty, price=spot))
                continue
            q = pricer.quote(underlying, leg, asof, spot)
            pricing.add(q["pricing"])
            priced.append(Leg(kind=leg.kind, qty=leg.qty, strike=leg.strike, expiration=leg.expiration,
                              price=q["ask"] if leg.qty > 0 else q["bid"], iv=q.get("iv")))
        fees = sum(self.fees.fee(abs(l.qty), l.price, "buy_to_open" if l.qty > 0 else "sell_to_open")
                   for l in priced if l.kind != "stock")
        virtual = [Leg(kind="stock", qty=int(covered_shares), price=spot)] if covered_shares else []
        info = analyze_legs(priced + virtual, spot, asof, cfg.rate, fees=fees)
        req = reg_t_requirement(priced + virtual, spot, cfg.account_type)
        if virtual:
            req = {**req, "requirement": max(0.0, req["requirement"] - covered_shares * spot)}
        eq = self._account_equity(pricer, asof, spots)
        self._check(strategy, priced + virtual, info, req, eq, pricer, asof, spots, underlying, spot, real_legs=priced)
        cash_flow = -sum(l.qty * l.multiplier * l.price for l in priced) - fees
        position = {"id": self.next_id, "strategy": strategy, "underlying": underlying, "opened": str(asof)[:10],
                    "entrySpot": spot, "legs": [l.as_dict() for l in priced], "entryCash": round(cash_flow, 2),
                    "fees": round(fees, 2), "maxLoss": info["maxLoss"], "maxGain": info["maxGain"],
                    "breakevens": info["breakevens"], "definedRisk": info["definedRisk"], "requirement": round(req["requirement"], 2),
                    "reserve": round(max(0.0, req["requirement"] + cash_flow + fees), 2),
                    "optionRisk": round(self._option_risk(priced, info, req), 2),
                    "pricing": "quoted" if pricing == {"quoted"} else "modeled" if pricing <= {"modeled"} else "mixed",
                    "tag": tag, "reason": reason, "coveredShares": int(covered_shares)}
        if dry_run:
            return position
        self.next_id += 1
        self.cash += cash_flow
        self.fees_paid += fees
        self.positions.append(position)
        self.log(asof, "open", id=position["id"], strategy=strategy, underlying=underlying, cash=round(cash_flow, 2),
                 maxLoss=info["maxLoss"], pricing=position["pricing"], tag=tag, reason=reason)
        return position

    def _check(self, strategy, priced, info, req, eq, pricer, asof, spots, underlying, spot, real_legs=None):
        cfg = self.config
        if not req["allowed"]:
            raise RiskRefusal(f"{strategy}: not allowed in a {cfg.account_type} account")
        if req.get("nakedCall") and not cfg.allow_naked_calls:
            raise RiskRefusal(f"{strategy}: naked short calls are blocked (unlimited loss)")
        if not info["definedRisk"] and not cfg.allow_undefined_risk:
            raise RiskRefusal(f"{strategy}: undefined-risk position blocked (allow_undefined_risk is off)")
        has_stock = any(l.kind == "stock" for l in priced)
        option_loss = self._option_risk(priced, info, req)
        if strategy == "cash_secured_put":
            # economically a limit order to buy the stock: cap it like a stock position, not like premium at risk
            notional = sum(-l.qty * CONTRACT_MULTIPLIER * l.strike for l in priced if l.kind == "put" and l.qty < 0)
            if notional > cfg.max_csp_notional * eq + 1e-9:
                raise RiskRefusal(f"{strategy}: assignment value ${notional:,.0f} exceeds {cfg.max_csp_notional:.0%} of equity")
            option_loss = 0.0
        if option_loss > cfg.max_position_loss * eq + 1e-9:
            raise RiskRefusal(f"{strategy}: max loss ${option_loss:,.0f} exceeds {cfg.max_position_loss:.0%} of equity (${eq:,.0f})")
        at_risk = option_loss + sum(p.get("optionRisk", 0.0) for p in self.positions)
        if at_risk > cfg.max_premium_at_risk * eq + 1e-9:
            raise RiskRefusal(f"{strategy}: total options risk ${at_risk:,.0f} would exceed {cfg.max_premium_at_risk:.0%} of equity")
        cash_flow = -sum(l.qty * l.multiplier * l.price for l in (real_legs if real_legs is not None else priced))
        reserve = max(0.0, req["requirement"] + cash_flow)
        if self.free_cash(spots) + cash_flow - reserve < -1e-6:
            raise RiskRefusal(f"{strategy}: needs ${max(0.0, reserve - cash_flow):,.0f} buying power; free cash is "
                              f"${self.free_cash(spots):,.0f} in the ${eq:,.0f} account" + (" (100 shares per contract)" if has_stock else ""))
        g_new = {"delta": 0.0, "vega": 0.0}
        for l in priced:
            if l.kind == "stock":
                continue  # stock delta belongs to the equity book
            t = max(year_fraction(asof, l.expiration), 1 / 365)
            g = bsm_greeks(spot, l.strike, t, l.iv or 0.25, cfg.rate, 0.0, l.kind)
            g_new["delta"] += g["delta"] * l.qty * CONTRACT_MULTIPLIER * spot
            g_new["vega"] += g["vega"] * l.qty * CONTRACT_MULTIPLIER
        cur = self.greeks(pricer, asof, spots)
        before = cur["deltaDollars"] + self.external_delta
        after = before + g_new["delta"]
        if abs(after) > cfg.max_abs_delta * eq and abs(after) > abs(before):
            raise RiskRefusal(f"{strategy}: portfolio delta ${after:,.0f} would exceed {cfg.max_abs_delta:.0%} of equity")
        if abs(cur["vega"] + g_new["vega"]) > cfg.max_abs_vega * eq and abs(cur["vega"] + g_new["vega"]) > abs(cur["vega"]):
            raise RiskRefusal(f"{strategy}: vega ${cur['vega'] + g_new['vega']:,.0f}/pt would exceed {cfg.max_abs_vega:.1%} of equity per vol point")

    @staticmethod
    def _option_risk(priced, info, req):
        """Risk charged to the options budget: max loss for pure option positions (requirement for undefined risk);
        for stock + option combos (covered call, collar, protective put) only the long premium paid, because the
        share risk belongs to the equity exposure."""
        if any(l.kind == "stock" for l in priced):
            return max(0.0, sum(l.qty * CONTRACT_MULTIPLIER * l.price for l in priced if l.kind != "stock" and l.qty > 0))
        return -(info["maxLoss"] or 0.0) if info["definedRisk"] else req["requirement"]

    def close(self, position_id, pricer, asof, spot, reason="close"):
        p = next((x for x in self.positions if x["id"] == position_id), None)
        if not p:
            raise KeyError(position_id)
        cash, fees = 0.0, 0.0
        for raw in p["legs"]:
            leg = leg_from(raw)
            if leg.kind == "stock":
                cash += leg.qty * spot
                continue
            q = pricer.quote(p["underlying"], leg, asof, spot)
            px = q["bid"] if leg.qty > 0 else q["ask"]
            cash += leg.qty * CONTRACT_MULTIPLIER * px
            fees += self.fees.fee(abs(leg.qty), px, "sell_to_close" if leg.qty > 0 else "buy_to_close")
        return self._finish(p, cash - fees, fees, asof, reason)

    def _finish(self, p, cash, fees, asof, reason):
        self.cash += cash
        self.fees_paid += fees
        self.positions = [x for x in self.positions if x["id"] != p["id"]]
        pnl = round(p["entryCash"] + cash, 2)
        rec = {**p, "closed": str(asof)[:10], "exitCash": round(cash, 2), "pnl": pnl, "exitReason": reason}
        self.closed.append(rec)
        self.log(asof, "close", id=p["id"], strategy=p["strategy"], pnl=pnl, reason=reason)
        return rec

    # ------------------------------------------------------------------------------------------ daily lifecycle
    def daily(self, pricer, asof, spots, opens=None, dividends=None, next_dividends=None):
        """Run once per session close: early assignment, expirations, short-stock borrow/dividends/stops/margin.
        dividends: {ticker: cash per share going ex on asof}; next_dividends: {ticker: (ex_date, amount)}."""
        asof = str(asof)[:10]
        out = []
        for p in list(self.positions):
            spot = spots.get(p["underlying"])
            if spot is None:
                continue
            nd = (next_dividends or {}).get(p["underlying"])
            if self.config.early_assignment and nd:
                out += self._early_assignment(p, pricer, asof, spot, nd)
        for p in list(self.positions):
            spot = spots.get(p["underlying"])
            if spot is None:
                continue
            front = min((l["expiration"] for l in p["legs"] if l["kind"] != "stock"), default=None)
            if front and front <= asof:
                out.append(self._expire(p, pricer, asof, spot))
        out += self._shorts_daily(asof, spots, opens or {}, dividends or {})
        return out

    def _early_assignment(self, p, pricer, asof, spot, next_div):
        ex_date, amount = next_div
        if not ex_date or not amount:
            return []
        days = (datetime.strptime(str(ex_date)[:10], "%Y-%m-%d") - datetime.strptime(asof, "%Y-%m-%d")).days
        if not (0 < days <= 3):
            return []
        events = []
        for raw in p["legs"]:
            leg = leg_from(raw)
            if leg.kind != "call" or leg.qty >= 0 or leg.expiration < str(ex_date)[:10] or spot <= leg.strike:
                continue
            q = pricer.quote(p["underlying"], leg, asof, spot)
            extrinsic = q["mid"] - (spot - leg.strike)
            if extrinsic < amount:
                self.log(asof, "early_assignment_risk", id=p["id"], strike=leg.strike, extrinsic=round(extrinsic, 3), dividend=amount)
                rec = self._settle(p, pricer, asof, spot, reason=f"early assignment before ex-dividend {ex_date} "
                                   f"(extrinsic {extrinsic:.2f} < dividend {amount:.2f})")
                events.append(rec)
                break
        return events

    def _expire(self, p, pricer, asof, spot):
        cfg = self.config
        pinned = [l for l in p["legs"] if l["kind"] != "stock" and l["qty"] < 0 and abs(spot - l["strike"]) / spot < cfg.pin_pct]
        if pinned:
            self.log(asof, "pin_risk", id=p["id"], strikes=[l["strike"] for l in pinned], spot=spot)
        return self._settle(p, pricer, asof, spot, reason="expiration" + (" (pin risk: closed)" if pinned else ""))

    def _settle(self, p, pricer, asof, spot, reason):
        """Expiration / assignment settlement: options at intrinsic (exercise-by-exception >= $0.01). If
        close_itm_on_expiry, ITM legs are closed at intrinsic minus a modeled half-spread instead of delivery, and
        stock legs are sold at spot (the equity book keeps its own shares; this keeps the sleeve self-contained)."""
        cash, fees = 0.0, 0.0
        for raw in p["legs"]:
            leg = leg_from(raw)
            if leg.kind == "stock":
                cash += leg.qty * spot
                continue
            intrinsic = max(0.0, spot - leg.strike) if leg.kind == "call" else max(0.0, leg.strike - spot)
            if intrinsic >= 0.01:
                slip = 0.01 if self.config.close_itm_on_expiry else 0.0
                px = max(0.0, intrinsic - slip) if leg.qty > 0 else intrinsic + slip
                cash += leg.qty * CONTRACT_MULTIPLIER * px
                if self.config.close_itm_on_expiry:
                    fees += self.fees.fee(abs(leg.qty), px, "sell_to_close" if leg.qty > 0 else "buy_to_close")
        return self._finish(p, cash - fees, fees, asof, reason)

    # ------------------------------------------------------------------------------------------ short stock (paper)
    def short(self, ticker, shares, price, asof, spots=None, pricer=None, reason=""):
        cfg = self.config
        if not cfg.short_stock_enabled:
            raise RiskRefusal("short stock is disabled (short_stock_enabled=False)")
        if cfg.account_type == "cash":
            raise RiskRefusal("short selling needs a margin account")
        shares = int(shares) if float(shares).is_integer() else float(shares)
        if shares <= 0:
            raise RiskRefusal("shares must be positive")
        spots = dict(spots or {})
        spots[ticker] = price
        eq = self._account_equity(pricer, asof, spots) if pricer else (self.account_equity or self.cash)
        notional = shares * price
        current = sum(s["shares"] * spots.get(s["ticker"], s["entry"]) for s in self.shorts)
        if current + notional > cfg.max_short_notional * eq:
            raise RiskRefusal(f"short notional ${current + notional:,.0f} exceeds {cfg.max_short_notional:.0%} of equity")
        if self.free_cash(spots) + notional - notional * cfg.short_initial_req < 0:
            raise RiskRefusal("not enough equity for the Reg-T 150% initial requirement")
        from costs import CostModel
        fee = CostModel.for_broker("schwab").regulatory_fee(notional, "sell", price)
        self.cash += notional - fee
        self.fees_paid += fee
        self.shorts.append({"ticker": ticker, "shares": shares, "entry": price, "opened": str(asof)[:10],
                            "stop": round(price * (1 + cfg.short_stop_pct), 4), "borrowRate": cfg.borrow_overrides.get(ticker, cfg.borrow_rate),
                            "reason": reason})
        self.log(asof, "short", ticker=ticker, shares=shares, price=price, reason=reason,
                 warning="short stock has unlimited loss; stop is not guaranteed through gaps")

    def cover(self, ticker, price, asof, reason="cover"):
        keep, done = [], []
        for s in self.shorts:
            if s["ticker"] != ticker:
                keep.append(s)
                continue
            self.cash -= s["shares"] * price
            pnl = round(s["shares"] * (s["entry"] - price) - s.get("borrowAccrued", 0.0) - s.get("dividendsPaid", 0.0), 2)
            rec = {**s, "closed": str(asof)[:10], "exit": price, "pnl": pnl, "exitReason": reason}
            self.closed.append({"strategy": "short_stock", "underlying": ticker, **rec})
            self.log(asof, "cover", ticker=ticker, shares=s["shares"], price=price, pnl=pnl, reason=reason)
            done.append(rec)
        self.shorts = keep
        return done

    def _shorts_daily(self, asof, spots, opens, dividends):
        cfg, out = self.config, []
        for s in list(self.shorts):
            t = s["ticker"]
            px = spots.get(t)
            if px is None:
                continue
            fee = s["shares"] * px * s["borrowRate"] / 360.0
            self.cash -= fee
            self.borrow_paid += fee
            s["borrowAccrued"] = s.get("borrowAccrued", 0.0) + fee
            div = dividends.get(t, 0.0)
            if div:
                owed = div * s["shares"]
                self.cash -= owed
                s["dividendsPaid"] = s.get("dividendsPaid", 0.0) + owed
                self.log(asof, "short_dividend_owed", ticker=t, amount=round(owed, 2))
            o = opens.get(t)
            if o is not None and o >= s["stop"]:
                out += self.cover(t, o, asof, reason=f"gap through squeeze stop {s['stop']} (filled at open {o})")
                continue
            if px >= s["stop"]:
                out += self.cover(t, s["stop"], asof, reason=f"squeeze stop {s['stop']}")
                continue
        # maintenance margin: ACCOUNT equity must stay >= maintenance * short market value (the long book and idle
        # cash support the short; standalone books use their own cash minus the short liability)
        smv = sum(s["shares"] * spots.get(s["ticker"], s["entry"]) for s in self.shorts)
        if self.account_equity is not None:
            # account_equity is the last close's total (short marked there); subtract today's move against the short
            margin_equity = self.account_equity - sum(s["shares"] * (spots.get(s["ticker"], s["entry"]) - s.get("lastMark", s["entry"]))
                                                      for s in self.shorts)
        else:
            margin_equity = self.cash - smv
        for s in self.shorts:
            s["lastMark"] = spots.get(s["ticker"], s.get("lastMark", s["entry"]))
        if smv and margin_equity < cfg.short_maintenance * smv:
            for s in list(self.shorts):
                out += self.cover(s["ticker"], spots.get(s["ticker"], s["entry"]), asof, reason="maintenance margin call")
        return out

    # ------------------------------------------------------------------------------------------ reporting
    def summary(self, pricer, asof, spots):
        rows = []
        for p in self.positions:
            spot = spots.get(p["underlying"], p["entrySpot"])
            value, g, pricing = self.position_value(p, pricer, asof, spot)
            rows.append({"id": p["id"], "strategy": p["strategy"], "underlying": p["underlying"], "opened": p["opened"],
                         "legs": p["legs"], "spot": spot, "value": round(value, 2), "unrealized": round(p["entryCash"] + value, 2),
                         "maxLoss": p["maxLoss"], "maxGain": p["maxGain"], "breakevens": p["breakevens"],
                         "greeks": {k: round(v, 3) for k, v in g.items()}, "pricing": p["pricing"], "tag": p.get("tag")})
        eq = self.equity(pricer, asof, spots)
        return {"asof": str(asof)[:10], "equity": round(eq, 2), "cash": round(self.cash, 2), "positions": rows,
                "shorts": self.shorts, "greeks": {k: round(v, 2) for k, v in self.greeks(pricer, asof, spots).items()},
                "requirement": round(self.requirement(spots), 2), "feesPaid": round(self.fees_paid, 2),
                "borrowPaid": round(self.borrow_paid, 2), "realized": round(sum(c.get("pnl", 0) for c in self.closed), 2),
                "closedCount": len(self.closed), "config": asdict(self.config), "events": self.events[-30:],
                "pricingNote": "Prices are MODELED (Black-Scholes + skew + spread model) unless a position says 'quoted'."}


def load_book(path=BOOK_PATH):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_book(book, path=BOOK_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps(book.state(), indent=1, default=str).encode())


__all__ = ["OptionsConfig", "OptionsBook", "RiskRefusal", "ModelPricer", "SnapshotPricer", "reg_t_requirement",
           "load_book", "save_book", "BOOK_PATH"]
