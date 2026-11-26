"""Transaction cost model shared by the backtest simulator and the paper trader.

cost per order (bps of notional) = broker commission + half-spread(dollar volume) + impact(volatility, size / ADV)
                                   + sell-side regulatory fees (SEC Section 31 + FINRA TAF)
  half-spread = clip(1 + 30 / sqrt(ADV$ in millions), min, max)      e.g. $20M -> 7.7 bps, $1B -> 2.0, SPY -> ~1.1
  impact      = impact_coef * daily_vol * sqrt(order$ / ADV$)         square-root market impact
  SEC fee     = sec_fee_per_million / 1e6 * sell notional             $20.60 per $1M from 2026-04-04 (SEC FY2026 advisory)
  FINRA TAF   = taf_per_share * shares sold, capped at taf_max        $0.000195/share, max $9.79 (2026 schedule;
                                                                      FINRA paused TAF Oct 1 - Dec 31 2026; we keep
                                                                      charging it to stay conservative)
Broker profiles set the commission. 'schwab' (default): $0 online commission for US exchange-listed stocks and
ETFs (NYSE/Nasdaq/NYSE Arca); OTC names are excluded from the universe, so their fees never apply.
No broker is called; this only models costs.
"""

import math
from dataclasses import asdict, dataclass

BROKER_PROFILES = {
    "schwab": {"commission_bps": 0.0, "fixed_fee": 0.0},
    "zero_commission": {"commission_bps": 0.0, "fixed_fee": 0.0},
    "per_trade_1usd": {"commission_bps": 0.0, "fixed_fee": 1.0},
    "bps_0_5": {"commission_bps": 0.5, "fixed_fee": 0.0},
}


@dataclass
class CostModel:
    broker: str = "schwab"
    commission_bps: float = 0.0
    fixed_fee: float = 0.0          # dollars per order (ticket charge); hurts small accounts most
    sec_fee_per_million: float = 20.60
    taf_per_share: float = 0.000195
    taf_max: float = 9.79
    spread_coef: float = 30.0
    min_half_spread_bps: float = 1.0
    max_half_spread_bps: float = 100.0
    impact_coef: float = 0.7
    default_adv: float = 20e6
    default_daily_vol: float = 0.02

    @classmethod
    def for_broker(cls, broker="schwab", **overrides):
        if broker not in BROKER_PROFILES:
            raise ValueError(f"Unknown broker profile {broker!r}; choose from {sorted(BROKER_PROFILES)}.")
        return cls(broker=broker, **{**BROKER_PROFILES[broker], **overrides})

    def half_spread_bps(self, adv):
        adv = adv if adv and math.isfinite(adv) and adv > 0 else self.default_adv
        return min(self.max_half_spread_bps, max(self.min_half_spread_bps, 1 + self.spread_coef / math.sqrt(adv / 1e6)))

    def impact_bps(self, order_value, adv, daily_vol):
        adv = adv if adv and math.isfinite(adv) and adv > 0 else self.default_adv
        daily_vol = daily_vol if daily_vol and math.isfinite(daily_vol) and daily_vol > 0 else self.default_daily_vol
        return 1e4 * self.impact_coef * daily_vol * math.sqrt(abs(order_value) / adv)

    def regulatory_fee(self, order_value, side="buy", price=None):
        """Dollar SEC + FINRA TAF fees; only sales pay them."""
        if side != "sell" or not order_value:
            return 0.0
        notional = abs(order_value)
        fee = self.sec_fee_per_million * notional / 1e6
        if price and math.isfinite(price) and price > 0:
            fee += min(self.taf_per_share * notional / price, self.taf_max)
        return fee

    def cost_bps(self, order_value, adv, daily_vol, side="buy", price=None):
        if not order_value:
            return 0.0
        notional = abs(order_value)
        dollars = self.fixed_fee + self.regulatory_fee(notional, side, price)
        return (self.commission_bps + 1e4 * dollars / notional + self.half_spread_bps(adv)
                + self.impact_bps(notional, adv, daily_vol))

    def as_dict(self):
        return asdict(self)
