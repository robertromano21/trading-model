// Closed list of things the AI analyst may do on the user's behalf, one entry per user-facing route plus the
// read-only math tools. Pure data (no imports) so enums.js, the adjudicator, the parity test and the docs share it.
//
//   ROUTE_ACTIONS  id -> { route: "METHOD /api/path", kind: "read" | "write", label, args, example }
//                  Executed by calling the SAME route over loopback (server/ai/actions.js), so there is no
//                  second copy of any handler's logic. "write" actions always wait for a confirm chip in the dock.
//   MATH_ACTIONS   id -> { tool, method?, family?, args, label, example }  read-only; run through tools.execute,
//                  the same executor the analyst's tool loop uses.
//   HUMAN_ONLY     route -> reason   never exposed to the model (live orders, live toggle, keys/secrets, OAuth).
//   INTERNAL       route -> reason   plumbing, not a user action (heartbeats, streams, log intake, voice audio).
//
// Arg spec: { type, in?, req?, values?, min?, max?, def?, help? }
//   type: ticker | tickers | int | num | date | enum | text | bool | coin | json
//   in:   query (GET default) | body (POST default) | path
//   The model sends args as [{ name, value }]; actions.js validates every one against this spec.

const t = (help, extra = {}) => ({ type: "ticker", help, ...extra });
const ts = (help, extra = {}) => ({ type: "tickers", help, ...extra });
const d = (help, extra = {}) => ({ type: "date", help, ...extra });
const e = (values, help, extra = {}) => ({ type: "enum", values, help, ...extra });
const n = (min, max, help, extra = {}) => ({ type: "num", min, max, help, ...extra });
const i = (min, max, help, extra = {}) => ({ type: "int", min, max, help, ...extra });
const x = (help, extra = {}) => ({ type: "text", help, ...extra });
const b = (help, extra = {}) => ({ type: "bool", help, ...extra });
const j = (help, extra = {}) => ({ type: "json", help, ...extra });

export const ROUTE_ACTIONS = Object.freeze({
  // ---- dashboard: market data, history, ADX, old papertrade, model, backtester
  HEALTH_CHECK: { route: "GET /api/health", kind: "read", label: "Server health", args: {}, example: "is the server healthy?" },
  SERVER_STATUS: { route: "GET /api/status", kind: "read", label: "Dashboard status (jobs, data, keys set/not set)", args: {}, example: "what's the dashboard status?" },
  MARKET_DATES: { route: "GET /api/market-data/dates", kind: "read", label: "Saved market-data dates", args: {}, example: "which market dates are saved?" },
  MARKET_DATA_DAY: { route: "GET /api/market-data/:date", kind: "read", label: "Prices saved for one date", args: { date: d("Market date", { in: "path", req: true }) }, example: "show the saved prices for 2026-11-25" },
  HISTORY_STATUS: { route: "GET /api/history/status", kind: "read", label: "Price-history coverage", args: {}, example: "how much price history do we have?" },
  HISTORY_TICKER: { route: "GET /api/history/ticker", kind: "read", label: "Daily history for one ticker", args: { ticker: t("Ticker", { req: true }), through: d("Last date") }, example: "show NVDA daily history" },
  ADX_DATES: { route: "GET /api/adx-summary/dates", kind: "read", label: "ADX summary dates", args: {}, example: "which ADX summary dates exist?" },
  ADX_SUMMARY: { route: "GET /api/adx-summary", kind: "read", label: "ADX summary for a date", args: { date: d("Saved market date", { req: true }) }, example: "ADX summary for 2026-11-25" },
  ADX_FORWARD_RETURNS: { route: "POST /api/adx-summary/forward-returns", kind: "read", label: "5-day forward returns for ADX-filtered tickers", args: { through: d("Through date", { req: true }), tickers: ts("Tickers", { req: true }), filters: j("Filter object {columns:[...]} as on the ADX page", { req: true }) }, example: "forward returns for the ADX filter through 2026-11-20" },
  STRATEGY_LIST: { route: "GET /api/strategies", kind: "read", label: "Backtester strategy list", args: {}, example: "which backtester strategies are there?" },
  UNIVERSE: { route: "GET /api/universe", kind: "read", label: "Backtester universe / SQLite coverage", args: {}, example: "what's in the backtest universe?" },
  STRATEGY_RESULTS: { route: "GET /api/strategy-results/:id", kind: "read", label: "Rows of the last model run", args: { id: x("Run id from Run model", { in: "path", req: true }) }, example: "show the strategy results for run <id>" },
  WORKBENCH_BACKTEST: { route: "POST /api/backtest", kind: "read", label: "Backtester run (single ticker or portfolio)", args: { strategy: x("Strategy id from STRATEGY_LIST", { req: true }), mode: e(["single", "portfolio"], "single or portfolio", { req: true }), tickers: ts("Tickers", { req: true }), startDate: d("Start", { req: true }), endDate: d("End", { req: true }), includeDividends: b("Include dividends"), params: j("Strategy parameters"), costs: j("Cost overrides") }, example: "backtest the sma strategy on AAPL from 2024-01-02 to 2026-07-08" },
  RUN_MODEL: { route: "POST /api/run-model", kind: "write", label: "Run the dashboard model for a date range", args: { startDate: d("Start", { req: true }), endDate: d("End"), source: e(["saved", "pending"], "Price source"), updateWorkbook: b("Also update the Excel workbook") }, example: "run the model for 2026-11-25" },
  RUN_STRATEGIES: { route: "POST /api/run-strategies", kind: "write", label: "Run the old strategy screen for a date", args: { date: d("Date", { req: true }), baseline: e(["sma", "ema"], "Baseline"), fastEma: e(["5", "10"], "Fast EMA"), trailing: e(["ma", "atr"], "Trailing stop"), breakoutStop: e(["atr", "swing"], "Breakout stop"), trendDays: i(2, 60, "Trend days") }, example: "run strategies for 2026-11-25 with ema baseline" },
  BACKFILL_HISTORY: { route: "POST /api/backfill-history", kind: "write", label: "Backfill three years of history", args: { date: d("End date", { req: true }), mode: e(["local", "massive"], "Source", { req: true }) }, example: "backfill history through 2026-11-25 from massive" },
  SYNC_PRICES: { route: "POST /api/sync-prices", kind: "write", label: "Sync latest prices", args: {}, example: "sync prices" },
  SYNC_DIVIDENDS: { route: "POST /api/backtest/dividends", kind: "write", label: "Sync dividend history (Massive)", args: {}, example: "sync dividends" },
  PAPERTRADE_MARKS: { route: "POST /api/papertrade/marks", kind: "read", label: "Old papertrade marks (prices for the papertrade table)", args: { options: j("Same options object the Papertrade tab sends", { req: true }) }, example: "mark the old papertrade table" },
  LOG_LIST: { route: "GET /api/logs", kind: "read", label: "Server / price-sync logs", args: { stream: e(["server", "browser-sync"], "Log stream", { req: true }), date: d("Day"), level: e(["", "info", "warn", "error"], "Level"), q: x("Search text") }, example: "show today's server errors" },
  LOG_VIEWER: { route: "GET /api/log-viewer", kind: "read", label: "Log viewer (all streams)", args: { stream: x("Stream"), level: x("Level"), q: x("Search text"), date: d("Day") }, example: "search the logs for 'timeout'" },
  // ---- charts / chart picker / quotes
  CHART: { route: "GET /api/chart/:ticker", kind: "read", label: "Chart data for a ticker", args: { ticker: t("Ticker", { in: "path", req: true }), tf: e(["1d", "5m", "1m"], "Timeframe"), sessions: i(1, 60, "Intraday sessions") }, example: "pull up the QQQ daily chart" },
  QUOTES: { route: "GET /api/market/quotes", kind: "read", label: "Live quotes (Schwab market data, read-only)", args: { symbols: ts("1-40 symbols", { req: true }) }, example: "quote SPY and QQQ" },
  SYMBOL_LOOKUP: { route: "GET /api/symbols/lookup", kind: "read", label: "Chart picker symbol search", args: { q: x("Name or symbol", { req: true }), remote: b("Also search remote sources") }, example: "find the symbol for Grayscale Bitcoin" },
  SYMBOL_FETCH_STATUS: { route: "GET /api/symbols/fetch/:ticker", kind: "read", label: "Status of a symbol history fetch", args: { ticker: t("Ticker", { in: "path", req: true }) }, example: "is the SBIT history fetch done?" },
  SYMBOL_ADD: { route: "POST /api/symbols/fetch/:ticker", kind: "write", label: "Add a symbol: fetch its history (Schwab fill)", args: { ticker: t("Ticker", { in: "path", req: true }) }, example: "add SBIT to the chart picker" },
  CRYPTO_UNIVERSE: { route: "GET /api/crypto/universe", kind: "read", label: "Crypto chart universe", args: {}, example: "which coins can I chart?" },
  CRYPTO_CHART: { route: "GET /api/crypto/chart/:coin", kind: "read", label: "Crypto chart data", args: { coin: { type: "coin", in: "path", req: true, help: "Coin symbol, e.g. BTC" }, tf: x("Timeframe"), refresh: b("Refresh") }, example: "chart BTC" },
  // ---- paper trading pages
  PAPER2_STATUS: { route: "GET /api/paper2/status", kind: "read", label: "Swing paper account status", args: { dry: e(["0", "1"], "1 = last dry run") }, example: "how is the swing paper account doing?" },
  PAPER2_TABLES: { route: "GET /api/paper2/tables", kind: "read", label: "Paper account tables (orders, fills, positions)", args: { account: x("Account id"), limit: i(1, 1000, "Rows") }, example: "show the paper fills" },
  PAPER2_DRIFT: { route: "GET /api/paper2/drift", kind: "read", label: "Backtest-vs-paper drift report", args: {}, example: "is the paper account drifting from the backtest?" },
  DAYTRADE_STATUS: { route: "GET /api/daytrade/status", kind: "read", label: "Day-trading paper status", args: { dry: e(["0", "1"], "1 = last dry run") }, example: "day-trading paper status" },
  DAYTRADE_RESEARCH: { route: "GET /api/daytrade/research", kind: "read", label: "Day-trading research summary", args: {}, example: "day-trading research summary" },
  SWARM_STATUS: { route: "GET /api/swarm/status", kind: "read", label: "Swarm shadow accounts leaderboard", args: {}, example: "show the swarm leaderboard" },
  CRYPTO_PAPER_STATUS: { route: "GET /api/crypto/status", kind: "read", label: "Crypto paper accounts (200-219)", args: {}, example: "how are the crypto paper accounts doing?" },
  REGIME_ALLOCATION: { route: "GET /api/regime-allocation", kind: "read", label: "Regime-allocation study", args: {}, example: "show the regime allocation study" },
  AUTOPILOT_STATUS: { route: "GET /api/autopilot", kind: "read", label: "Autopilot (scheduled PAPER runs) state", args: {}, example: "is autopilot on?" },
  AUTOPILOT_RUN: { route: "POST /api/autopilot/run", kind: "write", label: "Run an autopilot PAPER task now", args: { task: e(["swing", "daytrade", "crypto", "options"], "Task", { req: true }) }, example: "run the swing autopilot now" },
  AUTOPILOT_TOGGLE: { route: "POST /api/autopilot/toggle", kind: "write", label: "Turn autopilot (PAPER) on or off", args: { enabled: b("On or off", { req: true }) }, example: "turn autopilot off" },
  DECISION_CHAT: { route: "GET /api/chat", kind: "read", label: "Paper decision chat log", args: {}, example: "show the paper decision log" },
  DECISION_CHAT_ASK: { route: "POST /api/chat/ask", kind: "read", label: "Ask about one paper decision", args: { question: x("Question", { req: true }), messageId: x("Decision message id") }, example: "why did the paper trader buy IEF?" },
  OPTIONS_PAPER_STATUS: { route: "GET /api/options/paper", kind: "read", label: "Options PAPER book", args: {}, example: "show the options paper book" },
  OPTIONS_PAPER_RUN: { route: "POST /api/options/paper/run", kind: "write", label: "Advance the options PAPER book", args: {}, example: "run the options paper book" },
  // ---- experiment accounts (user / analyst sandbox; ids 1000-9999, never the live paper config or other ranges)
  EXPERIMENT_LIST: { route: "GET /api/experiments", kind: "read", label: "Experiment paper accounts (1000-9999)", args: {}, example: "list my experiment accounts" },
  EXPERIMENT_CREATE: { route: "POST /api/experiments", kind: "write", label: "Open experiment PAPER account(s); batches and parameter sweeps", args: { name: x("Name"), strategy: e(["manual", "tsmom", "meanrev", "combined"], "What it runs", { req: true }), equity: n(1000, 1000000, "Starting cash per account", { def: 10000 }), count: i(1, 50, "How many accounts in this batch", { def: 1 }), tickers: ts("Universe (optional)"), params: j("Strategy parameters, e.g. {\"lookback_days\":126}"), sweep: j("Sweep one parameter across the batch: {\"param\":\"lookback_days\",\"from\":63,\"to\":252}") }, example: "open 10 $10k tsmom accounts sweeping lookback 3-12 months" },
  EXPERIMENT_ORDER: { route: "POST /api/experiments/:id/order", kind: "write", label: "Paper buy/sell in a manual experiment account", args: { id: i(1000, 9999, "Account id (1000 = default manual account)", { in: "path", def: 1000 }), side: e(["buy", "sell"], "Side", { req: true }), ticker: t("Ticker", { req: true }), qty: i(1, 100000, "Shares", { req: true }) }, example: "paper buy 10 SPY" },
  EXPERIMENT_ADVANCE: { route: "POST /api/experiments/:id/advance", kind: "write", label: "Mark / advance an experiment account", args: { id: i(1000, 9999, "Account id", { in: "path", req: true }) }, example: "update experiment account 1001" },
  EXPERIMENT_RETIRE: { route: "POST /api/experiments/:id/retire", kind: "write", label: "Retire an experiment account (kept, no longer active)", args: { id: i(1000, 9999, "Account id", { in: "path", req: true }) }, example: "retire experiment account 1001" },
  EXPERIMENT_DELETE: { route: "POST /api/experiments/:id/delete", kind: "write", label: "Delete an experiment account (its trial stays counted)", args: { id: i(1000, 9999, "Account id", { in: "path", req: true }) }, example: "delete experiment account 1001" },
  HOLDOUT_LOOK: { route: "POST /api/experiments/holdout-look", kind: "write", label: "Backtest INTO the sealed holdout (logged in the holdout ledger)", args: { strategy: e(["tsmom", "meanrev", "combined"], "Family", { req: true }), tickers: ts("Universe (optional)"), params: j("Parameters"), reason: x("Why look now", { req: true }), confirm: b("Always true; set by the Allow chip", { def: true }) }, example: "score tsmom on the holdout because the design is final" },
  // ---- research & data (ops)
  OPS_JOBS: { route: "GET /api/ops/jobs", kind: "read", label: "Research jobs and their status", args: {}, example: "what research jobs are running?" },
  OPS_JOB: { route: "GET /api/ops/jobs/:id", kind: "read", label: "One research job with its log", args: { id: x("Job id", { in: "path", req: true }) }, example: "show job <id>" },
  OPS_RUN: { route: "POST /api/ops/run", kind: "write", label: "Start a research / data job", args: { job: e(["research_all", "research_backtest", "research_walkforward", "research_multihorizon", "research_normality", "research_attribution", "research_prob", "daytrade_research", "paper_dry", "daytrade_dry", "paper_run", "daytrade_run", "schwab_fetch", "swarm_run", "swarm_forward", "crypto_data", "crypto_research", "crypto_research_pre", "crypto_paper", "macro_nowcast_refresh", "macro_nowcast_research", "macro_backtest", "index_long_update", "data_rescan"], "Job name", { req: true }) }, example: "try a dollar filter on the crypto trend" },
  DATA_HEALTH: { route: "GET /api/data-health", kind: "read", label: "Data health report", args: { refresh: e(["0", "1"], "1 = re-run the check") }, example: "is the price data healthy?" },
  RESEARCH_REPORT: { route: "GET /api/research/report", kind: "read", label: "Research report (markdown)", args: {}, example: "show the research report" },
  RESEARCH_SUMMARY: { route: "GET /api/research/summary", kind: "read", label: "Research summary (backtests, walk-forward)", args: {}, example: "summarize the research results" },
  RESEARCH_ATTRIBUTION: { route: "GET /api/research/attribution", kind: "read", label: "Saved return attribution", args: {}, example: "show the return attribution" },
  RESEARCH_PROBABILITIES: { route: "GET /api/research/probabilities", kind: "read", label: "Latest calibrated probabilities", args: {}, example: "show the latest probabilities" },
  RESEARCH_PROB_WALKFORWARD: { route: "GET /api/research/prob-walkforward", kind: "read", label: "Probability walk-forward study", args: {}, example: "show the probability walk-forward" },
  SCHWAB_STATUS: { route: "GET /api/schwab/status", kind: "read", label: "Schwab token / market-data status", args: {}, example: "is Schwab connected?" },
  SCHWAB_REAUTH_STATUS: { route: "GET /api/schwab/reauth", kind: "read", label: "Schwab reauthorization status", args: {}, example: "did the Schwab reauthorization finish?" },
  SCHWAB_DAILY_STATUS: { route: "GET /api/schwab/daily-backfill", kind: "read", label: "Long daily backfill status", args: {}, example: "how far is the 20-year backfill?" },
  SCHWAB_DAILY_RUN: { route: "POST /api/schwab/daily-backfill", kind: "write", label: "Start the long daily backfill", args: {}, example: "start the 20-year backfill" },
  KEY_STATUS_FRED: { route: "GET /api/keys/fred", kind: "read", label: "FRED key set / not set (never the value)", args: {}, example: "is the FRED key set?" },
  KEY_STATUS_MASSIVE: { route: "GET /api/keys/massive", kind: "read", label: "Massive key set / not set (never the value)", args: {}, example: "is the Massive key set?" },
  // ---- macro / black swan / dollar
  MACRO_SNAPSHOT: { route: "GET /api/macro/snapshot", kind: "read", label: "Dollar / rates / credit snapshot", args: {}, example: "macro snapshot" },
  MACRO_DXY: { route: "GET /api/macro/dxy", kind: "read", label: "Dollar outlook", args: {}, example: "what's the dollar outlook?" },
  MACRO_STATUS: { route: "GET /api/macro/status", kind: "read", label: "Macro calendar status", args: {}, example: "macro calendar status" },
  MACRO_CALENDAR: { route: "GET /api/macro/calendar", kind: "read", label: "Economic calendar", args: { from: d("From"), to: d("To") }, example: "economic calendar this week" },
  MACRO_UPCOMING: { route: "GET /api/macro/upcoming", kind: "read", label: "Upcoming releases", args: { days: i(1, 60, "Days ahead") }, example: "what macro releases are coming up?" },
  MACRO_RELEASES: { route: "GET /api/macro/releases", kind: "read", label: "Recent releases", args: { days: i(1, 365, "Days back") }, example: "recent macro releases" },
  MACRO_SURPRISES: { route: "GET /api/macro/surprises", kind: "read", label: "Release surprises", args: {}, example: "latest macro surprises" },
  MACRO_EVENT_STUDY: { route: "GET /api/macro/event-study", kind: "read", label: "Macro event study", args: {}, example: "how does SPY react to CPI?" },
  MACRO_OVERNIGHT: { route: "GET /api/macro/overnight", kind: "read", label: "Overnight macro moves", args: {}, example: "what happened overnight?" },
  MACRO_FEATURES: { route: "GET /api/macro/features", kind: "read", label: "Macro feature panel", args: {}, example: "show the macro features" },
  MACRO_WATCHER: { route: "GET /api/macro/watcher", kind: "read", label: "Release watcher status", args: {}, example: "is the release watcher running?" },
  MACRO_CONTEXT: { route: "GET /api/macro/context", kind: "read", label: "Macro context block", args: {}, example: "give me the macro context" },
  MACRO_CONSENSUS: { route: "POST /api/macro/consensus", kind: "write", label: "Enter a consensus forecast for a release", args: { eventId: x("Event id", { req: true }), field: x("Field"), period: x("Period"), value: n(-1e9, 1e9, "Consensus value", { req: true }) }, example: "set the CPI consensus to 0.3" },
  MACRO_REFRESH: { route: "POST /api/macro/refresh", kind: "write", label: "Rebuild the macro calendar", args: {}, example: "refresh the macro calendar" },
  NOWCAST: { route: "GET /api/macro/nowcast", kind: "read", label: "Our macro nowcasts", args: {}, example: "what's our CPI nowcast?" },
  DOLLAR_STUDY: { route: "GET /api/research/dollar-study", kind: "read", label: "Dollar growth, commodity and stock relationships, and trade forecasts", args: {}, example: "does export growth predict the dollar?" },
  NOWCAST_HISTORY: { route: "GET /api/macro/nowcast/history", kind: "read", label: "Nowcast evolution for one target", args: { target: e(["CPI_MOM", "CORE_CPI_MOM", "CORE_PCE_MOM", "UNRATE", "PAYROLLS", "GDP_SAAR", "IMPORTS_SAAR", "EXPORTS_SAAR"], "Target", { req: true }) }, example: "how has the real exports nowcast moved?" },
  NOWCAST_REFRESH: { route: "POST /api/macro/nowcast/refresh", kind: "write", label: "Refresh FRED data and the nowcasts", args: { force: b("Force") }, example: "refresh the nowcasts" },
  BLACK_SWAN_STATE: { route: "GET /api/black-swan", kind: "read", label: "Black swan monitor", args: {}, example: "any black swan risk right now?" },
  // ---- options analytics
  OPTIONS_CHAIN: { route: "GET /api/options/chain", kind: "read", label: "Option chain (market data)", args: { symbol: t("Underlying", { req: true }), strikes: i(1, 60, "Strikes"), expirations: i(1, 12, "Expirations"), prefer: e(["schwab", "snapshot", "model"], "Source") }, example: "SPY option chain" },
  OPTIONS_GREEKS_ROUTE: { route: "GET /api/options/greeks", kind: "read", label: "Option greeks for an underlying", args: { symbol: t("Underlying", { req: true }) }, example: "greeks for the QQQ chain" },
  OPTIONS_ANALYZE: { route: "POST /api/options/analyze", kind: "read", label: "Analyze an options strategy (payoff, POP)", args: { symbol: t("Underlying", { req: true }), strategy: x("Strategy, e.g. bear_put_spread", { req: true }), expiration: d("Expiration"), strikes: j("Strikes object"), qty: i(1, 100, "Contracts") }, example: "analyze a SPY bear put spread" },
  OPTIONS_EXPECTED_MOVE: { route: "GET /api/options/expected-move", kind: "read", label: "Expected move from the straddle", args: { symbol: t("Underlying", { req: true }), days: i(1, 365, "Days") }, example: "expected move for NVDA this week" },
  OPTIONS_TOP_RISK: { route: "GET /api/options/top-risk", kind: "read", label: "Top / adverse-regime detector", args: {}, example: "is the top detector firing?" },
  OPTIONS_VOL_REGIME: { route: "GET /api/options/vol-regime", kind: "read", label: "VIX vol regime and sleeve", args: {}, example: "what's the vol regime?" },
  OPTIONS_RESEARCH: { route: "GET /api/options/research", kind: "read", label: "Options research summary", args: {}, example: "options research summary" },
  // ---- analyst scorecard, alerts / mute, AI settings
  SCORECARD: { route: "GET /api/analyst-scorecard", kind: "read", label: "Analyst track record", args: { ticker: t("Ticker"), limit: i(1, 100, "Rows") }, example: "how good have the analyst's flags been?" },
  SCORECARD_GATE: { route: "GET /api/analyst-scorecard/gate", kind: "read", label: "Analyst sizing gate", args: {}, example: "is the analyst sizing gate open?" },
  SCORECARD_REFRESH: { route: "POST /api/analyst-scorecard/refresh", kind: "write", label: "Re-score the analyst", args: {}, example: "refresh the analyst scorecard" },
  ALERTS_RECENT: { route: "GET /api/alerts/recent", kind: "read", label: "Recent alerts", args: { limit: i(1, 200, "How many") }, example: "what alerts fired today?" },
  ALERTS_STATUS: { route: "GET /api/alerts/status", kind: "read", label: "Alert / voice status", args: {}, example: "is the voice working?" },
  ALERTS_TYPES: { route: "GET /api/alerts/types", kind: "read", label: "Alert types and categories", args: {}, example: "what alert categories are there?" },
  ALERT_PREFS: { route: "GET /api/alerts/prefs", kind: "read", label: "Voice / mute settings", args: {}, example: "what's muted?" },
  ALERT_PREFS_SET: { route: "POST /api/alerts/prefs", kind: "write", label: "Change voice / mute settings", args: { prefs: j("Settings patch, e.g. {\"speakCats\":{\"trade_swarm\":false}}", { req: true }) }, example: "mute the swarm alerts" },
  ALERT_DEMO: { route: "POST /api/alerts/demo", kind: "write", label: "Play a demo alert", args: { gender: e(["female", "male"], "Voice") }, example: "play a test alert" },
  AI_STATUS: { route: "GET /api/ai/status", kind: "read", label: "AI analyst status", args: {}, example: "which AI model are you using?" },
  AI_MODELS: { route: "GET /api/ai/models", kind: "read", label: "AI model catalog", args: {}, example: "which AI models can I pick?" },
  AI_MODEL_SET: { route: "POST /api/ai/models", kind: "write", label: "Switch the AI model", args: { id: x("Model id from AI_MODELS", { req: true }) }, example: "switch to grok" },
  AI_VERDICTS: { route: "GET /api/ai/verdicts", kind: "read", label: "Latest AI verdicts", args: { tickers: ts("Tickers") }, example: "show the latest verdicts" },
  AI_FLAG: { route: "POST /api/ai/flag", kind: "write", label: "Run red/yellow/green flags (writes advisory_verdicts.json)", args: { tickers: ts("Tickers (default paper targets)") }, example: "flag NVDA and AMD" },
  AI_FLAG_STATUS: { route: "GET /api/ai/flag/:id", kind: "read", label: "Flag job status", args: { id: x("Job id", { in: "path", req: true }) }, example: "is the flag job done?" },
  AI_DECISION: { route: "GET /api/ai/decision", kind: "read", label: "What the advisory hook would do", args: { tickers: ts("Tickers") }, example: "what would the advisory hook do with NVDA?" },
  AI_DECISION_PROPOSE: { route: "POST /api/ai/decision", kind: "read", label: "Advisory hook on proposed trades", args: { tickers: ts("Tickers", { req: true }), proposals: j("Proposals {TICKER: {...}}") }, example: "check a proposed NVDA entry against the advisory hook" },
  AI_RESEARCH: { route: "GET /api/ai/research/:ticker", kind: "read", label: "Research bundle for a ticker", args: { ticker: t("Ticker", { in: "path", req: true }), refresh: e(["0", "1"], "Refresh") }, example: "pull the research bundle for AMD" },
  AI_TOOLS: { route: "GET /api/ai/tools", kind: "read", label: "Analyst tool catalog", args: {}, example: "what tools can you run?" },
  AI_PROMPT: { route: "GET /api/ai/prompt", kind: "read", label: "The analyst's master prompt", args: {}, example: "show me your system prompt" },
  AI_CARD_REFRESH: { route: "POST /api/ai/strategy-card/refresh", kind: "write", label: "Rebuild the strategy card", args: {}, example: "refresh the strategy card" },
  AI_MEMORY_STATUS: { route: "GET /api/ai/memory/status", kind: "read", label: "Analyst memory status", args: {}, example: "how much do you remember?" },
  AI_MEMORY_SEARCH: { route: "GET /api/ai/memory/search", kind: "read", label: "Search analyst memory", args: { q: x("Query", { req: true }), tickers: ts("Tickers") }, example: "what do you remember about NVDA?" },
  AI_MEMORY_ITEMS: { route: "GET /api/ai/memory/items", kind: "read", label: "List memory items", args: { ticker: t("Ticker"), kind: x("Kind"), limit: i(1, 200, "How many") }, example: "list your notes on AMD" },
  AI_MEMORY_INGEST: { route: "POST /api/ai/memory/ingest", kind: "write", label: "Save a note to memory", args: { kind: e(["NOTE"], "Kind", { def: "NOTE" }), ticker: t("Ticker"), text: x("Note", { req: true }) }, example: "remember that NVDA reports on the 20th" },
  AI_MEMORY_REINDEX: { route: "POST /api/ai/memory/reindex", kind: "write", label: "Re-embed pending memory", args: {}, example: "reindex your memory" },
  LIFECYCLE: { route: "GET /api/lifecycle", kind: "read", label: "Auto-shutdown state", args: {}, example: "when will the dashboard shut down?" },
});

// Read-only math: the analyst's own quant tools (engine/quant_tools.py and Node summaries), exposed by prompt.
const S = (help) => ({ type: "tickers", help, req: true });
const H = i(1, 260, "Horizon in trading days");
export const MATH_ACTIONS = Object.freeze({
  MATH_DESCRIBE: { tool: "STATS", method: "DESCRIBE", label: "Return stats: mean, vol, skew, kurtosis, drawdown", args: { tickers: S("Tickers"), start: d("Start"), end: d("End") }, example: "describe SPY returns since 2020-01-01" },
  MATH_NORMALITY: { tool: "STATS", method: "NORMALITY", label: "Normality tests (JB, Shapiro, AD, K2) by horizon", args: { tickers: S("Tickers"), horizons: j("List of horizons, e.g. [1,5,20]") }, example: "are NVDA returns normal at 1, 5 and 20 days?" },
  MATH_BOOTSTRAP: { tool: "STATS", method: "BOOTSTRAP", label: "Block-bootstrap confidence interval", args: { tickers: S("Tickers"), statistic: e(["mean", "median", "sharpe", "vol", "total_return"], "Statistic"), confidence: n(0.5, 0.999, "Confidence") }, example: "bootstrap a CI for QQQ's Sharpe" },
  MATH_T_TEST: { tool: "STATS", method: "T_TEST", label: "t-test (incl. HAC) on returns", args: { tickers: S("1-2 tickers"), test: e(["one_sample", "welch", "paired"], "Test"), mu0: n(-1, 1, "Null mean") }, example: "is SPY's mean daily return different from zero?" },
  MATH_CORRELATION: { tool: "STATS", method: "CORRELATION", label: "Correlation matrix / rolling correlation", args: { tickers: S("2+ tickers"), method: e(["pearson", "spearman", "kendall"], "Method"), window: i(10, 504, "Rolling window") }, example: "correlation of TLT and SPY" },
  MATH_SHARPE: { tool: "STATS", method: "SHARPE_INFERENCE", label: "Sharpe CI, probabilistic and deflated Sharpe", args: { tickers: S("Tickers"), trials: i(1, 100000, "Number of trials tried (for deflation)"), benchmark_sharpe: n(-5, 10, "Benchmark Sharpe") }, example: "deflated Sharpe of QQQ with 50 trials" },
  MATH_ACCOUNT_SHARPE: { node: "accountSharpe", label: "Sharpe / PSR / deflated Sharpe of one swarm account (pre-holdout only)", args: { account: i(0, 999, "Swarm account id", { req: true }), book: e(["swing", "daytrade"], "Which returns file", { def: "swing" }) }, example: "what's the deflated Sharpe of account 12?" },
  MATH_SWARM_OVERFIT: { node: "swarmOverfit", label: "Swarm overfitting stats: PBO (CSCV), deflated Sharpe, White RC / Hansen SPA, Holm / BH", args: {}, example: "what's the swarm's probability of backtest overfitting?" },
  MATH_BETA: { tool: "ECONOMETRICS", method: "OLS", label: "Beta / OLS with HAC errors (first ticker = dependent)", args: { tickers: S("Dependent then regressors, e.g. NVDA SPY") }, example: "what's NVDA's beta to SPY?" },
  MATH_ROLLING_BETA: { tool: "ECONOMETRICS", method: "ROLLING_OLS", label: "Rolling beta", args: { tickers: S("Dependent then regressors"), window: i(20, 504, "Window") }, example: "rolling 60-day beta of TSLA to QQQ" },
  MATH_FACTOR_BETAS: { tool: "ECONOMETRICS", method: "FACTOR_BETAS", label: "Factor betas", args: { tickers: S("Tickers"), factors: ts("Factor ETFs, e.g. SPY IWM TLT") }, example: "factor betas of ARKK" },
  MATH_STATIONARITY: { tool: "ECONOMETRICS", method: "ADF_KPSS", label: "ADF / KPSS stationarity", args: { tickers: S("Tickers"), target: e(["log_prices", "returns"], "Series") }, example: "is the GLD/GDX spread stationary?" },
  MATH_COINTEGRATION: { tool: "ECONOMETRICS", method: "ENGLE_GRANGER", label: "Engle-Granger cointegration", args: { tickers: S("Two tickers") }, example: "are KO and PEP cointegrated?" },
  MATH_JOHANSEN: { tool: "ECONOMETRICS", method: "JOHANSEN", label: "Johansen cointegration", args: { tickers: S("2+ tickers"), lags: i(1, 12, "Lags") }, example: "Johansen test on XLE XOM CVX" },
  MATH_ARIMA: { tool: "ECONOMETRICS", method: "ARIMA", label: "AR / ARIMA forecast", args: { tickers: S("Ticker"), steps: i(1, 60, "Steps ahead") }, example: "ARIMA forecast for SPY 5 days" },
  MATH_HALF_LIFE: { tool: "ECONOMETRICS", method: "HALF_LIFE_HURST", label: "Mean-reversion half-life, Hurst, variance ratios", args: { tickers: S("Ticker(s)") }, example: "half-life and Hurst of the XLU" },
  MATH_GARCH: { tool: "VOL_FORECAST", method: "GARCH", label: "GARCH(1,1) volatility forecast", args: { tickers: S("Ticker"), horizonDays: H, dist: e(["t", "normal", "skewt"], "Distribution") }, example: "GARCH vol forecast for TSLA" },
  MATH_EWMA_VOL: { tool: "VOL_FORECAST", method: "EWMA", label: "EWMA volatility", args: { tickers: S("Ticker"), horizonDays: H }, example: "EWMA vol of BTC-USD" },
  MATH_VAR: { tool: "RISK", label: "VaR / CVaR (normal, Cornish-Fisher, t, historical)", args: { tickers: S("Ticker(s)"), horizonDays: H, confidences: j("List, e.g. [0.95,0.99]"), weights: j("Portfolio weights {TICKER: w}"), equity: n(100, 1e9, "Account equity") }, example: "1-day 99% VaR of a $10k SPY position" },
  MATH_SIZING: { tool: "SIZING", label: "Position sizing: vol target, Kelly / half-Kelly, stop-distance shares", args: { tickers: S("Ticker"), price: n(0.01, 1e6, "Entry price"), stop: n(0.01, 1e6, "Stop price"), equity: n(100, 1e9, "Equity"), risk_per_trade: n(0.0005, 0.1, "Risk per trade (fraction)") }, example: "how many shares of AMD with a stop at 140 and $10k?" },
  MATH_MONTE_CARLO: { tool: "MONTE_CARLO", method: "BOOTSTRAP_PATHS", label: "Monte Carlo equity paths (drawdown odds)", args: { tickers: S("Ticker(s)"), horizonDays: H, sims: i(100, 20000, "Simulations"), weights: j("Weights") }, example: "simulate a year of a 60/40 SPY TLT portfolio" },
  MATH_PORTFOLIO_OPT: { tool: "PORTFOLIO_OPT", method: "RISK_PARITY", label: "Portfolio optimization (risk parity / min var / max Sharpe)", args: { tickers: S("Tickers"), method: e(["risk_parity", "min_variance", "max_sharpe", "mean_variance", "equal_weight"], "Method"), per_name_cap: n(0.05, 1, "Cap per name") }, example: "risk parity weights for SPY TLT GLD" },
  MATH_INDICATORS: { tool: "INDICATORS", label: "Indicators: SMAs, Donchian, Bollinger, RSI, ATR, momentum", args: { tickers: S("Tickers") }, example: "RSI and ATR for AAPL" },
  MATH_REGIME: { tool: "MACRO", method: "SNAPSHOT", label: "Market regime (HMM regime state + macro snapshot)", args: { tickers: ts("Ticker for context") }, example: "run the regime model on QQQ" },
  MATH_MACRO_BETAS: { tool: "MACRO", method: "BETAS", label: "Macro factor betas (dollar, curve, credit)", args: { tickers: S("Tickers") }, example: "how sensitive is EEM to the dollar?" },
  MATH_MACRO_LEADLAG: { tool: "MACRO", method: "LEADLAG", label: "Macro lead / lag correlations", args: { tickers: S("Tickers"), feature: e(["dxy_ret", "dxy_ret20", "curve_slope", "hy_oas_chg20", "sofr_tbill_spread"], "Feature") }, example: "does the dollar lead GLD?" },
  MATH_DOLLAR_OUTLOOK: { tool: "MACRO", method: "OUTLOOK", label: "Dollar outlook model", args: {}, example: "what does the dollar model say?" },
  MATH_PROB: { tool: "PROB_MODEL", label: "Calibrated probabilities P(beat SPY), P(up)", args: { tickers: ts("Tickers") }, example: "probability NVDA beats SPY over 20 days" },
  MATH_ATTRIBUTION: { tool: "ATTRIBUTION", label: "Return attribution (beta, sectors, alpha)", args: { strategy: e(["tsmom", "meanrev", "combined", "daytrade", "paper", "all"], "Strategy"), source: e(["backtest", "walkforward", "paper", "daytrade", "all"], "Source") }, example: "how much of the paper return is just beta?" },
  MATH_BACKTEST: { tool: "RUN_BACKTEST", label: "Engine backtest of TSMOM / mean reversion / combined (pre-holdout by default)", args: { family: e(["TSMOM", "MEANREV", "COMBINED"], "Family", { req: true }), tickers: ts("Universe (optional)"), start: d("Start"), end: d("End (clamped before the sealed holdout)"), lookback_days: i(5, 504, "TSMOM lookback in trading days (6 months = 126)"), params: j("Other parameters, e.g. {\"max_positions\":5}") }, example: "backtest TSMOM on QQQ with a 6-month lookback" },
  MATH_WALK_FORWARD: { tool: "WALK_FORWARD", label: "Walk-forward (out-of-sample only, pre-holdout)", args: { family: e(["TSMOM", "MEANREV", "COMBINED"], "Family", { req: true }), tickers: ts("Universe") }, example: "walk-forward the mean-reversion sleeve" },
  MATH_DAYTRADE_BACKTEST: { tool: "DAYTRADE_BACKTEST", label: "Opening-range-breakout backtest", args: { tickers: ts("Tickers"), frequency: e(["1", "5"], "Bar minutes") }, example: "ORB backtest on SPY 5-minute bars" },
  MATH_RULE_STUDY: { tool: "RULE_STUDY", method: "SPY_200D_FILTER", label: "Quick rule study (200-day filter, RSI2 dip, Donchian, 12-1 momentum, pairs)", args: { rule: e(["SPY_200D_FILTER", "RSI2_DIP", "DONCHIAN_BREAKOUT", "XS_MOMENTUM_12_1", "XS_MOMENTUM_6_1", "PAIRS_ZSCORE"], "Rule", { req: true }), tickers: ts("Tickers") }, example: "study an RSI2 dip rule on QQQ" },
  MATH_OPTION_PRICE: { tool: "OPTIONS", method: "GREEKS", label: "Black-Scholes price and greeks", args: { tickers: S("Underlying"), strike: n(0.01, 1e6, "Strike", { req: true }), dte: i(1, 1500, "Days to expiry", { req: true }), kind: e(["call", "put"], "Call or put", { req: true }), spot: n(0.01, 1e6, "Spot (default: last price)"), vol: n(0.01, 5, "Volatility (default: realized)") }, example: "price a 30-day SPY 600 call" },
  MATH_OPTION_PAYOFF: { tool: "OPTIONS", method: "PAYOFF", label: "Option strategy payoff / breakevens / max loss", args: { tickers: S("Underlying"), strategy: x("Strategy, e.g. iron_condor", { req: true }), strikes: j("Strikes {k1..k4}"), expiration: d("Expiration") }, example: "payoff of a QQQ iron condor" },
  MATH_OPTION_POP: { tool: "OPTIONS", method: "POP", label: "Probability of profit", args: { tickers: S("Underlying"), strategy: x("Strategy", { req: true }), strikes: j("Strikes") }, example: "probability of profit for a SPY bull put spread" },
  MATH_EXPECTED_MOVE: { tool: "OPTIONS", method: "EXPECTED_MOVE", label: "Expected move", args: { tickers: S("Underlying"), horizonDays: H }, example: "expected move of TSLA into earnings" },
  MATH_TOP_RISK: { tool: "OPTIONS", method: "TOP_RISK", label: "Top / adverse-regime detector", args: {}, example: "how close are we to a market top signal?" },
  MATH_VOL_REGIME: { tool: "OPTIONS", method: "VOL_REGIME", label: "VIX vol regime + VIXY/SVIX sleeve", args: {}, example: "which vol regime are we in?" },
  MATH_NOWCAST: { tool: "MACRO_NOWCAST", label: "Nowcast model output for a target", args: { target: e(["CPI_MOM", "CORE_CPI_MOM", "CORE_PCE_MOM", "UNRATE", "PAYROLLS", "GDP_SAAR"], "Target") }, example: "nowcast for core PCE" },
  MATH_DILUTION: { tool: "RESEARCH", label: "Dilution flag (share-count YoY, offerings) and research bundle", args: { tickers: S("Ticker") }, example: "is PLTR diluting shareholders?" },
  MATH_CRYPTO: { tool: "CRYPTO_STATUS", label: "Crypto math: trend, vol, drawdown, dollar / liquidity overlays, ETP decay context", args: {}, example: "what's BTC's 90-day vol and trend?" },
  MATH_BLACK_SWAN: { tool: "BLACK_SWAN", label: "Black swan state + event study", args: {}, example: "black swan event study summary" },
  MATH_PAPER_BOOK: { tool: "PAPER_STATUS", label: "Paper book math: equity, exposure, PDT and breaker state", args: {}, example: "what's the paper account's exposure?" },
});

// Routes the model can never call. The adjudicator says it's a manual step and points at the page.
// Human-only routes (real-money orders, the live switch, keys, Schwab sign-in) are listed in
// server/human-only-routes.js, outside server/ai on purpose: no AI module names the live order path
// (see the order-path guard test).

// Plumbing: not user actions.
export const INTERNAL = Object.freeze({
  "POST /api/ai/command": "the analyst's own entry point", "POST /api/ai/ask": "the analyst's own entry point",
  "GET /api/ai/tools/run": "", "POST /api/ai/tools/run": "direct tool runner (math actions use the same executor)",
  "GET /api/ai/actions": "this catalog", "POST /api/ai/actions/run": "action entry point", "POST /api/ai/actions/confirm": "confirm chip",
  "POST /api/ai/actions/cancel": "cancel chip", "GET /api/ai/actions/jobs": "analyst job list",
  "GET /api/alerts/stream": "SSE stream", "POST /api/alerts": "engine alert intake", "POST /api/alerts/emit": "engine alert intake",
  "POST /api/alerts/speak.wav": "voice audio", "POST /api/client-logs": "browser log intake", "POST /api/browser-logs": "browser log intake",
  "GET /api/lifecycle/presence": "tab presence", "POST /api/lifecycle/heartbeat": "tab heartbeat", "POST /api/lifecycle/leave": "tab closed",
  "GET /assets/chart.umd.js": "static asset", "GET /voice/sid-worker.js": "static asset", "GET /voice/status": "voice engine status",
  "GET /voice/chip": "voice audio", "POST /voice/chip": "voice audio", "POST /voice/cry.sidvoice": "voice audio", "POST /voice/line.sidvoice": "voice audio",
  "GET /voice/line.wav": "voice audio", "POST /voice/line.wav": "voice audio", "POST /voice/prefetch": "voice cache", "POST /voice/profiles": "voice profiles",
});

// Belt and braces: even if someone adds one of these to ROUTE_ACTIONS, actions.js refuses to call it.
export const FORBIDDEN = Object.freeze([
  /\/live\//,                                                          // anything under the live API
  /^POST \/api\/keys\//, /provider-key/, /app-keys/, /^POST .*reauth/,   // secrets and Schwab sign-in
  /preview/i,                                                          // any order preview
  /^(?!POST \/api\/experiments\/)\w+ .*\border/i,                      // order routes, except experiment PAPER orders
]);
export const ACTION_IDS = Object.freeze([...Object.keys(ROUTE_ACTIONS), ...Object.keys(MATH_ACTIONS)]);
