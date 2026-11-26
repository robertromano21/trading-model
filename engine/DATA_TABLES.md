# Local data tables (all under `Massive/`, gitignored, never committed)

## `Massive/intraday.sqlite3` — Schwab market data cache (written by `engine/schwab_data.py`)
Read-only market data from Schwab's `/marketdata/v1` API (no account or order endpoints are ever called).
Other components (e.g. the AI analyst layer) may read these tables; open SQLite read-only
(`file:...intraday.sqlite3?mode=ro`, WAL mode) and do not write to them.

| Table | Columns | Notes |
|---|---|---|
| `intraday_bars` | `symbol`, `ts` (bar start, epoch ms UTC), `session_date` (YYYY-MM-DD, US/Eastern), `minute` (minutes after midnight ET, 570 = 09:30), `open`, `high`, `low`, `close`, `volume`, `frequency` (bar minutes: 1 or 5), `source` ('schwab') | PK (`symbol`, `frequency`, `ts`). Regular session only (09:30-16:00 ET). Schwab keeps ~48 days of 1-minute and ~9 months of 5-minute bars. |
| `intraday_quotes` | `symbol`, `fetched_at` (local ISO time), `last`, `bid`, `ask`, `bid_size`, `ask_size`, `volume`, `quote_time` (epoch ms) | One row per symbol per fetch. Latest snapshot = `MAX(fetched_at)`. |
| `intraday_fetches` | `symbol`, `frequency`, `start_ms`, `end_ms`, `bars`, `fetched_at` | Fetch audit log. |

Refresh: `python engine/schwab_data.py fetch [--frequency 5 --days 270 --chunk-days 100] [SYMBOLS]`,
`python engine/schwab_data.py quotes SYMBOLS`, or `python engine/daytrader.py fetch` (default day-trading targets
plus the swing paper trader's current targets and positions). Needs a valid `schwab_token.json`
(and `SCHWAB_APP_KEY`/`SCHWAB_APP_SECRET` to refresh it every 30 minutes).

## `Massive/paper_trading.sqlite3` — paper-trading state
Swing trader (`engine/paper_trader.py`): `accounts`, `runs`, `signal_state` (targets per decision date),
`orders`, `fills`, `positions`, `decision_log` (same columns as backtest decisions:
`portfolio_sim.DECISION_COLUMNS`), `equity_history`, `reconciliation`.
Day trader (`engine/daytrader.py`): `dt_accounts`, `dt_trades`, `dt_daily`, `dt_runs`, `dt_decision_log`.

### Optional `advisory_verdicts` (input from the AI layer; see `engine/advisory.py`)
`ticker`, `asof` (YYYY-MM-DD), `flag` (GREEN | YELLOW | RED), `conviction` (0-1), `entry_low`, `entry_high`,
`stop`, `overextension`, `source`, `created_at`; PK (`ticker`, `asof`, `source`).
Alternatively `Massive/Research/advisory_verdicts.json` (`{"verdicts": {"AAPL": {"flag": "GREEN", ...}}}`).
Verdicts can only shrink or defer new entries (RED blocks, YELLOW halves, out-of-band defers); the risk engine is
final; missing/stale (> 3 days)/malformed verdicts change nothing. Default mode is `log` (record only);
use `--advisory apply` to act on them.

## `Massive/prices.sqlite3` — daily prices (Massive)

`daily_prices`, `backtest_dividends`, `backtest_dividend_coverage`, plus `corporate_splits`,
`price_anomalies` (status open | repaired | persists | masked), `ticker_repairs` from `engine/corporate_actions.py`.

### Optional `daily_long` (Schwab 20-year daily history, `engine/schwab_daily.py`)
Written by `python schwab_daily.py backfill`. Never overwrites `daily_prices` or `market_sessions`.

| Table | Columns | Notes |
|---|---|---|
| `daily_long` | `ticker`, `date` (YYYY-MM-DD, US/Eastern session), `open`, `high`, `low`, `close`, `volume`, `source` ('schwab'), `adjustment` ('split'), `fetched_at` | PK (`ticker`, `date`, `source`). Schwab `/marketdata/v1/pricehistory` daily candles are split-adjusted, not dividend-adjusted. |
| `daily_long_progress` | `ticker`, `first_date`, `last_date`, `rows`, `status`, `message`, `requested_start`, `requested_end`, `period_mode`, `fetched_at` | Resumable per-ticker cursor. |

`engine/market_data.py` `load_panel()` prefers Massive `daily_prices` on overlapping dates, extends the panel with `daily_long` before the first Massive session, and attaches survivorship-bias and overlap-mismatch warnings. Schwab only has currently listed symbols. Summary: `Massive/Research/schwab_daily_backfill.json`. Manual run: Research & data page, `POST /api/schwab/daily-backfill`, or the CLI above. Autopilot continues the backfill weekdays at 22:15 ET (20-minute budget) once a token exists.

## `Massive/Research/` — research outputs and status JSON
`research_summary.json`, `research_report.md`, `walkforward_*.csv`, `trades_*.csv`, `decisions_*.csv`,
`equity_curves.csv`, `paper_status.json`, `daytrade_summary.json`, `daytrade_trials.csv`,
`daytrade_status.json`.
