"""Tests for the strategy framework, simulator, cost model, data checks and dividend grace (synthetic data only)."""

import math
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
import pandas as pd

from backtester import run as legacy_run
from costs import CostModel
from fetch_and_model import check_session_date
from market_data import Panel
from portfolio_sim import simulate
from price_history import open_history, save_session
from risk import apply_caps, rebalance_days, value_at_risk
from strategies import CombinedStrategy, RegimeMeanReversion, TimeSeriesMomentum
from universe import CORE_ETFS


def synthetic_panel(sessions=320, stocks=12, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=sessions)
    names = list(CORE_ETFS[:6]) + [f"S{chr(65 + i)}" for i in range(stocks)]
    drift = rng.normal(0.0003, 0.0006, len(names))
    vol = rng.uniform(0.008, 0.03, len(names))
    returns = rng.standard_t(4, (sessions, len(names))) * vol / math.sqrt(2) + drift
    close = pd.DataFrame(50 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=names)
    gap = pd.DataFrame(rng.normal(0, 0.003, close.shape), index=dates, columns=names)
    opening = close.shift().fillna(close) * (1 + gap)
    high = np.maximum(opening, close) * (1 + rng.uniform(0, 0.01, close.shape))
    low = np.minimum(opening, close) * (1 - rng.uniform(0, 0.01, close.shape))
    volume = pd.DataFrame(2e6, index=dates, columns=names)
    dividends = pd.DataFrame(0.0, index=dates, columns=names)
    classes = pd.Series({n: "etf" if n in CORE_ETFS else "stock" for n in names})
    return Panel(opening, high, low, close, volume, dividends, classes,
                 pd.DataFrame(False, index=dates, columns=names))


class StrategyFrameworkTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = synthetic_panel()

    def strategies(self):
        loose = {"min_dollar_volume": 1e6}
        return [TimeSeriesMomentum(**loose), RegimeMeanReversion(entry_z=-1.5, regime_vol_pct=0.9, trend_t_max=5,
                                                                trend_filter=50, **loose),
                CombinedStrategy(sleeves=[TimeSeriesMomentum(**loose),
                                          RegimeMeanReversion(entry_z=-1.5, regime_vol_pct=0.9, trend_t_max=5,
                                                              trend_filter=50, **loose)])]

    def test_weights_are_causal_and_match_paper_path(self):
        for strategy in self.strategies():
            full = strategy.target_weights(self.panel)
            self.assertGreater(full.abs().to_numpy().sum(), 0, strategy.name)
            for asof in self.panel.dates[[150, 221, 260, -1]]:
                live = strategy.target_weights_at(self.panel, asof)
                pd.testing.assert_series_equal(full.loc[asof], live.reindex(full.columns).fillna(0.0),
                                               check_names=False, atol=1e-12, obj=f"{strategy.name} {asof.date()}")

    def test_future_data_does_not_change_past_weights(self):
        strategy = TimeSeriesMomentum(min_dollar_volume=1e6)
        cut = self.panel.dates[240]
        before = strategy.target_weights(self.panel.upto(cut))
        shocked = synthetic_panel()
        shocked.close.loc[shocked.close.index > cut] *= 3
        after = strategy.target_weights(shocked).loc[:cut]
        pd.testing.assert_frame_equal(before, after, atol=1e-12)

    def test_caps_positions_and_gross(self):
        strategy = CombinedStrategy(max_positions=5, per_name_cap=0.2,
                                    sleeves=[TimeSeriesMomentum(min_dollar_volume=1e6),
                                             RegimeMeanReversion(min_dollar_volume=1e6)])
        weights = strategy.target_weights(self.panel)
        self.assertLessEqual(weights.abs().sum(axis=1).max(), 1.0 + 1e-9)
        self.assertLessEqual(weights.max().max(), 0.2 + 1e-9)
        self.assertLessEqual(weights.ne(0).sum(axis=1).max(), 5)
        self.assertGreaterEqual(weights.min().min(), 0.0)
        capped = apply_caps(pd.DataFrame([[0.9, 0.9, 0.9]]), per_name_cap=0.5, gross_cap=1.0)
        self.assertAlmostEqual(capped.sum(axis=1).iloc[0], 1.0)

    def test_weekly_rebalance_uses_exchange_calendar(self):
        days = pd.DatetimeIndex(["2026-11-23", "2026-11-24", "2026-11-25", "2026-11-27", "2026-11-30"])
        self.assertEqual(rebalance_days(days).tolist(), [False, False, False, True, False])
        self.assertEqual(rebalance_days(days[:2]).tolist(), [False, False])   # truncated panel agrees

    def test_cost_model_scales_with_liquidity_size_and_vol(self):
        model = CostModel()
        self.assertEqual(model.cost_bps(0, 1e8, 0.02), 0.0)
        self.assertGreater(model.half_spread_bps(20e6), model.half_spread_bps(1e9))
        self.assertGreater(model.cost_bps(1e6, 1e8, 0.02), model.cost_bps(1e4, 1e8, 0.02))
        self.assertGreater(model.cost_bps(1e5, 1e8, 0.04), model.cost_bps(1e5, 1e8, 0.01))
        self.assertAlmostEqual(model.half_spread_bps(1e8), 1 + 30 / 10)

    def test_schwab_profile_zero_commission_and_sell_side_fees(self):
        model = CostModel.for_broker("schwab")
        self.assertEqual(model.commission_bps, 0.0)
        self.assertEqual(model.regulatory_fee(10_000, "buy", 100), 0.0)
        fee = model.regulatory_fee(10_000, "sell", 100)               # 100 shares
        self.assertAlmostEqual(fee, 10_000 * 20.60 / 1e6 + 100 * 0.000195)
        self.assertAlmostEqual(model.regulatory_fee(1e9, "sell", 1.0) - 1e9 * 20.60 / 1e6, 9.79)   # TAF cap
        self.assertGreater(model.cost_bps(10_000, 1e8, 0.02, "sell", 100), model.cost_bps(10_000, 1e8, 0.02, "buy", 100))
        ticket = CostModel.for_broker("per_trade_1usd")
        self.assertAlmostEqual(ticket.cost_bps(500, 1e8, 0.02) - model.cost_bps(500, 1e8, 0.02), 20.0)  # $1 on $500
        with self.assertRaises(ValueError):
            CostModel.for_broker("unknown")

    def test_simulator_fills_next_open_and_gaps_through_stops(self):
        dates = pd.bdate_range("2026-01-05", periods=6)
        frame = lambda values: pd.DataFrame({"AAA": values}, index=dates, dtype=float)
        close = frame([100, 100, 100, 100, 100, 100])
        opening = frame([100, 101, 100, 100, 90, 95])
        low = frame([99, 99, 99, 99, 85, 94])
        dividends = frame([0, 0, 0, 1.0, 0, 0])
        panel = Panel(opening, np.maximum(opening, close), low, close, frame([1e7] * 6), dividends,
                      pd.Series({"AAA": "stock"}), frame([False] * 6).astype(bool))
        weights = frame([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        stops = frame([np.nan, np.nan, np.nan, 95, 95, 95])
        zero = CostModel(commission_bps=0, spread_coef=0, min_half_spread_bps=0, impact_coef=0)
        result = simulate(panel, weights, stops, zero, initial_equity=10_000, min_trade_weight=0.05)
        trade = result.trades.iloc[0]
        self.assertEqual(trade["entry_date"], dates[1])
        self.assertAlmostEqual(trade["entry_price"], 101.0, places=6)          # next open, not the signal close
        self.assertEqual(trade["exit_reason"], "stop_gap")
        self.assertAlmostEqual(trade["exit_price"], 90.0)            # gapped below the 95 stop -> open
        self.assertEqual(result.decisions.query("reason == 'entry'").shape[0], 1)   # blocked after the stop
        shares = 5000 / 100
        self.assertAlmostEqual(trade["pnl"], shares * (90 - 101) + shares * 1.0, delta=1.5)  # dividend credited; ~1 bp floor cost

    def test_masked_corporate_action_day_is_value_neutral(self):
        dates = pd.bdate_range("2026-09-28", periods=5)
        frame = lambda values: pd.DataFrame({"AAA": values}, index=dates, dtype=float)
        close = frame([100, 100, 100, 70, 70])                 # spin-off: price drops 30% on day 3
        mask = frame([0, 0, 0, 1, 0]).astype(bool)
        panel = Panel(close, close, close, close, frame([1e7] * 5), frame([0] * 5), pd.Series({"AAA": "stock"}), mask)
        result = simulate(panel, frame([0.5] * 5), initial_equity=10_000)
        self.assertAlmostEqual(result.returns.iloc[3], 0.0, places=9)
        self.assertEqual(result.decisions.query("reason == 'corporate_action'").shape[0], 1)

    def test_costs_reduce_returns(self):
        weights = TimeSeriesMomentum(min_dollar_volume=1e6, rebalance="D").target_weights(self.panel)
        free = CostModel(commission_bps=0, spread_coef=0, min_half_spread_bps=0, impact_coef=0)
        cheap = simulate(self.panel, weights, cost_model=free).equity.iloc[-1]
        costly = simulate(self.panel, weights, cost_model=CostModel(commission_bps=20)).equity.iloc[-1]
        self.assertLess(costly, cheap)

    def test_var_models(self):
        rng = np.random.default_rng(1)
        normal = value_at_risk(rng.normal(0, 0.01, 20000), 0.99)
        self.assertAlmostEqual(normal["normal"], 0.0233, delta=0.001)
        self.assertAlmostEqual(normal["historical"], 0.0233, delta=0.0015)
        fat = value_at_risk(rng.standard_t(3, 20000) * 0.01, 0.99)
        self.assertGreater(fat["historical"], fat["normal"])
        self.assertGreater(fat["studentT"], fat["normal"])


class DataCheckTests(TestCase):
    def test_session_date_check_rejects_misdated_files(self):
        stamp = int(pd.Timestamp("2025-11-04 21:00", tz="UTC").timestamp() * 1000)
        records = [{"T": "AAA", "t": stamp, "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]
        check_session_date(records, "2025-11-04")
        with self.assertRaises(ValueError):
            check_session_date(records, "2026-11-06")

    def test_dividend_grace_allows_a_short_uncovered_tail(self):
        first = date(2025, 1, 1)
        dates = [(first + timedelta(days=i)).isoformat() for i in range(12)]
        stat = SimpleNamespace(st_size=1, st_mtime_ns=1)
        config = {"mode": "single", "tickers": ["AAA"], "strategy": "sma_crossover",
                  "params": {"fast": 2, "slow": 3}, "startDate": dates[0], "endDate": dates[-1],
                  "includeDividends": True,
                  "costs": {"initialCapital": 1000, "commissionPct": 0, "fixedFee": 0, "slippageBps": 0}}
        with TemporaryDirectory() as directory, open_history(Path(directory) / "p.sqlite3", writable=True) as db:
            for i, day in enumerate(dates):
                save_session(db, day, [{"ticker": "AAA", "open": 10 + i % 3, "close": 10 + i % 4, "low": 9,
                                        "high": 14, "volume": 100}], stat)
            db.execute("CREATE TABLE backtest_dividends (ticker TEXT, market_date TEXT, cash REAL NOT NULL, "
                       "unsupported INTEGER NOT NULL, PRIMARY KEY (ticker, market_date)) WITHOUT ROWID")
            db.execute("CREATE TABLE backtest_dividend_coverage (id INTEGER PRIMARY KEY CHECK (id = 1), "
                       "first_date TEXT NOT NULL, last_date TEXT NOT NULL, events INTEGER NOT NULL)")
            db.execute("INSERT INTO backtest_dividend_coverage VALUES (1, ?, ?, 0)", (dates[0], dates[-4]))
            result = legacy_run(db, config)
            self.assertTrue(any("price-only" in w for w in result["warnings"]))
            db.execute("UPDATE backtest_dividend_coverage SET last_date=?", (dates[3],))
            with self.assertRaises(ValueError):
                legacy_run(db, config)
