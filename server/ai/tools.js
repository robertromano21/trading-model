// Server-side tools for the analyst's tool loop. The model picks a tool from the Retort enums (E.TOOL / TOOL_METHOD /
// STRATEGY_FAMILY) through the strict analystToolStep schema; this module validates the request, maps it onto a
// FIXED operation (engine/quant_tools.py op, or a Node function for memory / research / paper status), runs it with a
// timeout, logs the full result to ai_log (untruncated) and returns a compact summary for the prompt.
// Nothing here can place orders: quant_tools.py only reads the price database, research files and Schwab bar cache.
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import * as E from "./enums.js";
import { paperBook } from "./features.js";
import { researchFallbackText } from "./context.js";
import { cryptoStatus as loadCryptoStatus } from "./crypto.js";

const TICKER = /^[A-Z][A-Z0-9.-]{0,11}$/;
const DATE = /^\d{4}-\d{2}-\d{2}$/;
export const MAX_CALLS_PER_STEP = 4;
export const PROMPT_RESULT_CHARS = 3500;
export const CACHE_TTL_MS = 6 * 3600 * 1000;

// tool -> { methods: {METHOD: op}, default, timeoutMs, horizon: arg name, options: documented option names, help }
export const TOOLS = Object.freeze({
  RUN_BACKTEST: { op: "backtest", timeoutMs: 300000, cache: true, family: "COMBINED", options: "strategy parameters (tsmom: horizons, target_vol, max_positions, per_name_cap, rebalance, vol_window; meanrev: lookback, entry_z, exit_z, max_hold, max_positions, stop_atr, trend_filter, regime_vol_pct, trend_t_max; combined: target_vol, max_positions or sleeve keys like tsmom.max_positions / meanrev.entry_z)", help: "Engine backtest of the swing strategies (same code as research.py, Schwab costs, warm-up 110). Tickers optional (restricts the universe)." },
  WALK_FORWARD: { op: "walk_forward", timeoutMs: 900000, cache: true, family: "TSMOM", options: "grid.<param> = JSON list of values (<= 16 combinations), train_min, test_len", help: "Expanding-window walk-forward, out-of-sample only. Slow (minutes) for meanrev/combined." },
  DAYTRADE_BACKTEST: { op: "daytrade_backtest", timeoutMs: 300000, cache: true, options: "ORB parameters (or_minutes, entry_cutoff, target_r, stop_frac, min_rel_volume, max_vol_pct, risk_per_trade, daily_loss_limit, pdt_max_day_trades, allow_short), frequency (1|5), initial_equity", help: "Opening-range-breakout backtest on cached Schwab intraday bars (PDT + loss breaker)." },
  RULE_STUDY: { op: "rule_study", timeoutMs: 120000, cache: true, methods: { DEFAULT: "spy_200d_filter", SPY_200D_FILTER: "spy_200d_filter", RSI2_DIP: "rsi2_dip", DONCHIAN_BREAKOUT: "donchian_breakout", XS_MOMENTUM_12_1: "xs_momentum_12_1", XS_MOMENTUM_6_1: "xs_momentum_6_1", PAIRS_ZSCORE: "pairs_zscore" }, methodArg: "rule", options: "top_n, cost_bps", help: "Quick study of a candidate rule NOT in the engine (default tickers CORE_ETFS; pairs need 2 tickers)." },
  PAPER_STATUS: { node: "paperStatus", timeoutMs: 10000, help: "Swing and day-trade paper accounts: equity, cash, positions, open orders, recent decisions, PDT and breaker state." },
  STATS: { timeoutMs: 120000, cache: true, methods: { DEFAULT: "describe", DESCRIBE: "describe", NORMALITY: "normality", BOOTSTRAP: "bootstrap", T_TEST: "t_test", CORRELATION: "correlation", SHARPE_INFERENCE: "sharpe_inference" }, horizon: "horizon", options: "horizons (JSON list, NORMALITY), statistic (mean|median|sharpe|vol|total_return), n_boot, block, confidence, test (one_sample|welch|paired), mu0, method (pearson|spearman|kendall), window, trials, benchmark_sharpe, trial_sharpe_std", help: "Descriptive stats, normality tests (JB, Shapiro, Anderson-Darling, K2) by horizon, block bootstrap CIs, t-tests (incl. HAC), correlation, Sharpe CI / PSR / deflated Sharpe." },
  ECONOMETRICS: { timeoutMs: 120000, cache: true, methods: { DEFAULT: "ols", OLS: "ols", ROLLING_OLS: "rolling_ols", FACTOR_BETAS: "factor_betas", ADF_KPSS: "adf_kpss", ENGLE_GRANGER: "engle_granger", JOHANSEN: "johansen", ARIMA: "arima", HALF_LIFE_HURST: "half_life_hurst" }, horizon: "horizon", options: "window (ROLLING_OLS), factors (JSON list, FACTOR_BETAS), target (log_prices|returns, ADF_KPSS), lags (JOHANSEN), order (JSON [p,d,q]) and steps (ARIMA)", help: "OLS/rolling OLS with HAC errors (first ticker = dependent), factor betas, ADF/KPSS, Engle-Granger/Johansen cointegration, AR/ARIMA, half-life + Hurst + variance ratios." },
  VOL_FORECAST: { timeoutMs: 120000, cache: true, methods: { DEFAULT: "garch", GARCH: "garch", EWMA: "ewma" }, horizon: "steps", options: "dist (t|normal|skewt)", help: "GARCH(1,1) volatility forecast (EWMA fallback)." },
  INDICATORS: { op: "indicators", timeoutMs: 60000, cache: true, help: "Engine indicators: SMA/EMA 20-200, Donchian 20/55, Bollinger, z-scores, RSI14 and RSI2, ATR, realized vol, 12-1 / 6-1 momentum, SPY 200-day state." },
  RISK: { op: "var", timeoutMs: 120000, cache: true, horizon: "horizon_days", options: "confidences (JSON list), weights (JSON {ticker: weight}) for a portfolio, equity", help: "VaR and CVaR: normal, Cornish-Fisher, fitted Student-t, historical; also in dollars for the $10k account." },
  SIZING: { op: "sizing", timeoutMs: 60000, cache: true, options: "price, stop, equity, target_vol, per_name_cap, risk_per_trade", help: "Vol-target weight under the 30% cap, Kelly / half-Kelly (reference only), stop-distance share count." },
  MONTE_CARLO: { op: "monte_carlo", timeoutMs: 120000, cache: true, methods: { DEFAULT: "bootstrap", BOOTSTRAP_PATHS: "bootstrap", NORMAL_PATHS: "normal", STUDENT_T_PATHS: "student_t" }, methodArg: "method", horizon: "days", options: "sims, weights (JSON), drawdown_threshold, equity, block", help: "Equity-path simulation: terminal return quantiles, drawdown distribution, probability of loss." },
  PORTFOLIO_OPT: { op: "portfolio_opt", timeoutMs: 120000, cache: true, methods: { DEFAULT: "risk_parity", RISK_PARITY: "risk_parity", MIN_VARIANCE: "min_variance", MAX_SHARPE: "max_sharpe", MEAN_VARIANCE: "mean_variance", EQUAL_WEIGHT: "equal_weight" }, methodArg: "method", options: "per_name_cap, risk_free, risk_aversion, target_vol", help: "Long-only optimisation (Ledoit-Wolf covariance, 30% cap): risk parity, min variance, max Sharpe, mean-variance." },
  MEMORY_SEARCH: { node: "memorySearch", timeoutMs: 15000, help: "Search the analyst memory (earlier verdicts, answers, paper decisions, notes, news) with MiniLM." },
  RESEARCH: { node: "research", timeoutMs: 90000, help: "Fresh research bundle for one ticker: price features, Schwab quote, SEC fundamentals, latest news, and the dilution flag (split-adjusted share-count YoY plus any recent offering). Pass option asof=YYYY-MM-DD for a point-in-time look; history comes from Massive/Research/dilution/. With no ticker, returns local study summaries (regime allocation / overextension, black swan, dollar forecast, swarm) matching the query." },
  MACRO: { op: "macro", timeoutMs: 120000, cache: true, methods: { DEFAULT: "snapshot", SNAPSHOT: "snapshot", BETAS: "betas", FACTOR_BETAS: "betas", LEADLAG: "leadlag", PREDICT: "predict", OUTLOOK: "outlook" }, methodArg: "method", horizon: "horizon", options: "feature (dxy_ret|dxy_ret20|curve_slope|hy_oas_chg20|sofr_tbill_spread)", help: "Dollar / rates / credit-proxy snapshot, HAC betas, lead/lag correlations, walk-forward predictability, or the recommend-only dollar outlook (off for trading unless gates pass). These are FRED/ETF proxies, not live CDS or swap quotes." },
  OPTIONS: { timeoutMs: 120000, cache: false, methods: { DEFAULT: "options_chain", CHAIN: "options_chain", GREEKS: "options_greeks", PAYOFF: "options_payoff", POP: "options_pop", EXPECTED_MOVE: "expected_move", TOP_RISK: "top_risk", VOL_REGIME: "vol_regime" }, horizon: "days", options: "strikes, expirations, prefer (schwab|snapshot|model) for CHAIN; spot, vol, strike, dte, kind (call|put) for GREEKS; strategy (long_put, bear_put_spread, iron_condor, covered_call, ...), expiration, strikes (JSON {k|k1..k4}), back_expiration, qty or legs (JSON [{kind, qty, strike, expiration, price}]) for PAYOFF / POP; straddle for EXPECTED_MOVE", help: "Options analytics: chain (live Schwab market data when authorized, else local snapshot, else MODELED), Black-Scholes greeks, payoff / max loss / breakevens / probability of profit, expected move, the top / adverse-regime detector reading and the VIX vol-regime + VIXY/SVIX sleeve state. Market data only: never previews or places orders." },
  MACRO_CALENDAR: { node: "macroCalendar", timeoutMs: 20000, help: "Economic calendar (times in ET), upcoming releases, latest surprises, watcher status and the [MACRO_EVENTS] context block. Never places orders." },
  ATTRIBUTION: { op: "attribute_returns", timeoutMs: 180000, cache: true, options: "strategy (tsmom|meanrev|combined|daytrade|paper|all), source (backtest|walkforward|paper|daytrade|all), recompute (bool)", help: "Break strategy or paper-account returns into market beta, sector/bond bets and residual alpha (CAPM / multi-factor HAC, rolling alpha, Brinson). Uses the saved attribution.json unless recompute is true." },
  PROB_MODEL: { op: "prob_forecast", timeoutMs: 60000, cache: true, options: "ticker or tickers (optional)", help: "Walk-forward calibrated probabilities: P(k-day excess vs SPY > 0) and P(return > 0) for k=5 and 20, plus SPY up/down. Historical values are purged out-of-sample; the frozen pack is used after that tape ends. Advisory only." },
  MACRO_NOWCAST: { node: "macroNowcast", timeoutMs: 20000, options: "target (CPI_MOM|CORE_CPI_MOM|CORE_PCE_MOM|UNRATE|PAYROLLS|GDP_SAAR, optional)", help: "Our own GDPNow-style nowcasts of the next CPI / core CPI / core PCE / unemployment / payrolls / GDP prints from raw point-in-time data: nowcast, 80% band, last print, implied surprise, top drivers, regime, track record. Outside forecasts (Cleveland Fed, GDPNow, SPF, consensus) are shown for comparison only, never inputs. Research only; never places orders." },
  BLACK_SWAN: { node: "blackSwan", timeoutMs: 20000, help: "Current geopolitical / black swan state, market confirmation, playbook permissions, recent events, and the event-study summary. Never places orders." },
  CRYPTO_STATUS: { node: "cryptoStatus", timeoutMs: 50000, help: "Crypto market snapshot (refreshed if older than 10 min): BTC/ETH/SOL/AVAX/LINK/XRP/DOGE/LTC spot price, 24h/7d/30d/90d/1y change, 30/90-day vol, trend vs 50/200-day, drawdown from 1y high; Schwab quotes for /BTC /ETH futures, $BTC/$ETH indexes and ETPs (IBIT, FBTC, ETHA ...); dollar (UUP, broad dollar), real yield and net-liquidity context with the overlay states; macro nowcast regime; the 20 crypto PAPER accounts (ids 200-219) standings and positions; the crypto backtest verdict, and the SBIT (-2x inverse BTC ETF, US market hours only) / BITI (-1x) short research with shadow accounts 220-223. Market data only; never places orders." },
});

export function toolCatalog() {
  return Object.entries(TOOLS).map(([name, t]) => {
    const methods = t.methods ? ` methods: ${Object.keys(t.methods).filter((m) => m !== "DEFAULT").join("|")} (DEFAULT=${Object.entries(t.methods).find(([m, o]) => m !== "DEFAULT" && o === t.methods.DEFAULT)?.[0]}).` : "";
    const family = t.family ? ` family: TSMOM|MEANREV|COMBINED (default ${t.family}).` : "";
    const horizon = t.horizon ? ` horizonDays -> ${t.horizon}.` : "";
    return `- ${name}: ${t.help}${methods}${family}${horizon}${t.options ? ` options: ${t.options}.` : ""}`;
  }).join("\n");
}

const parseValue = (v) => {
  const s = String(v ?? "").trim();
  if (s === "") return null;
  try { return JSON.parse(s); } catch { return s; }
};
const stableKey = (obj) => JSON.stringify(obj, Object.keys(flatKeys(obj)).sort());
function flatKeys(obj, out = {}) { if (obj && typeof obj === "object") for (const [k, v] of Object.entries(obj)) { out[k] = 1; flatKeys(v, out); } return out; }

/** Validate one analystToolStep call and turn it into { tool, method, kind, op, args, timeoutMs } (throws a typed error). */
export function planCall(call) {
  const tool = String(call?.tool || "");
  const spec = TOOLS[tool];
  if (!spec || !E.TOOL.includes(tool)) throw Object.assign(new Error(`Unknown tool ${tool}`), { code: "TOOL_UNKNOWN" });
  const method = E.TOOL_METHOD.includes(call.method) ? call.method : "DEFAULT";
  const tickers = [...new Set((call.tickers || []).map((t) => String(t).toUpperCase().trim()).filter((t) => TICKER.test(t)))].slice(0, 60);
  const options = {};
  for (const o of call.options || []) {
    const name = String(o?.name || "").trim();
    if (/^[A-Za-z_][A-Za-z0-9_.]{0,60}$/.test(name)) options[name] = parseValue(o.value);
  }
  const args = {};
  if (tickers.length) args.tickers = tickers;
  if (DATE.test(call.start || "")) args.start = call.start;
  if (DATE.test(call.end || "")) args.end = call.end;
  const horizon = Number(call.horizonDays);
  let op = spec.op;
  if (spec.methods) {
    const chosen = spec.methods[method] || spec.methods.DEFAULT;
    if (spec.methodArg) { args[spec.methodArg] = chosen; op = spec.op; } else op = chosen;
  }
  if (spec.horizon && Number.isFinite(horizon) && horizon > 0) args[spec.horizon] = Math.round(horizon);
  if (tool === "RUN_BACKTEST" || tool === "WALK_FORWARD") {
    const family = call.family && call.family !== "NONE" ? call.family : spec.family;
    args.family = family.toLowerCase();
    if (tool === "RUN_BACKTEST") args.params = options;
    else {
      const grid = {};
      for (const [k, v] of Object.entries(options)) {
        if (k.startsWith("grid.")) grid[k.slice(5)] = Array.isArray(v) ? v : [v];
        else if (k === "train_min" || k === "test_len") args[k] = v;
      }
      if (Object.keys(grid).length) args.grid = grid;
    }
  } else if (tool === "DAYTRADE_BACKTEST") {
    const params = {};
    for (const [k, v] of Object.entries(options)) {
      if (k === "frequency" || k === "initial_equity") args[k] = v; else params[k] = v;
    }
    args.params = params;
  } else {
    for (const [k, v] of Object.entries(options)) if (!(k in args) && k !== "family") args[k] = v;
  }
  if (tool === "ECONOMETRICS" && op === "arima" && args.horizon) { args.steps = args.horizon; delete args.horizon; }
  if (tool === "MEMORY_SEARCH") args.query = String(call.query || call.purpose || "");
  return { tool, method, family: args.family || null, kind: spec.node ? "node" : "python", op: spec.node || op, args, timeoutMs: spec.timeoutMs, cache: Boolean(spec.cache) };
}

// ---------------------------------------------------------------- compact summaries for the prompt
const DROP_FIRST = ["monthlyEquity", "sampledPath", "lastTrades", "trialsTable", "parameterValueSelections", "params",
  "dailyMoments", "valueAtRisk", "byHoldingPeriod", "exitReasons", "config", "dollars", "quantiles", "firstVector"];
function round(value, depth = 0) {
  if (typeof value === "number") return Number.isFinite(value) ? Number(value.toPrecision(4)) : null;
  if (Array.isArray(value)) {
    const head = value.slice(0, 12).map((v) => round(v, depth + 1));
    return value.length > 12 ? [...head, `(+${value.length - 12} more)`] : head;
  }
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, round(v, depth + 1)]));
  return typeof value === "string" && value.length > 600 ? value.slice(0, 600) + "…" : value;
}
function without(value, keys) {
  if (Array.isArray(value)) return value.map((v) => without(v, keys));
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).filter(([k]) => !keys.includes(k)).map(([k, v]) => [k, without(v, keys)]));
  return value;
}
export function summarize(result, maxChars = PROMPT_RESULT_CHARS) {
  let value = round(result);
  let text = JSON.stringify(value);
  for (let i = 1; text.length > maxChars && i <= DROP_FIRST.length; i++) {
    value = without(round(result), DROP_FIRST.slice(0, i));
    text = JSON.stringify(value);
  }
  return text.length > maxChars ? text.slice(0, maxChars) + ` …(truncated for the prompt; the full result is in the AI log)` : text;
}

// ---------------------------------------------------------------- executor
export function createTools({ root, dataDir, quant, store, memory = null, research = null, python = null, log = () => {} }) {
  const readJson = (names) => {
    for (const n of names) {
      const p = join(dataDir, "Research", n);
      if (existsSync(p)) { try { return { file: n, ...JSON.parse(readFileSync(p, "utf8")) }; } catch { /* next */ } }
    }
    return null;
  };
  const node = {
    async paperStatus() {
      const swing = readJson(["paper_status.json", "paper_status_dryrun.json"]);
      const day = readJson(["daytrade_status.json", "daytrade_status_dryrun.json"]);
      const book = paperBook(dataDir);
      return {
        swing: swing && { file: swing.file, generatedAt: swing.generatedAt, dryRun: swing.dryRun, cash: swing.cash, initialEquity: swing.initialEquity,
          lastProcessed: swing.lastProcessed, circuitBreakerRemaining: swing.circuitBreakerRemaining,
          equity: swing.equityHistory?.at(-1), positions: swing.positions, openOrders: swing.openOrders?.slice(0, 20),
          recentDecisions: swing.recentDecisions?.slice(-15), targets: swing.targets?.slice?.(0, 20) },
        daytrade: day && { file: day.file, generatedAt: day.generatedAt, dryRun: day.dryRun, equity: day.equity, initialEquity: day.initialEquity,
          lastSession: day.lastSession, pdt: day.pdt, trades: day.trades?.slice(-10), params: day.params },
        ledger: book,
      };
    },
    async memorySearch(args) {
      if (!memory) throw Object.assign(new Error("memory is not available"), { code: "NO_MEMORY" });
      const r = await memory.recall({ query: args.query || "", tickers: args.tickers || [], budgetChars: 4000, keepRecent: 0 });
      return { method: r.method, text: r.text, items: r.items.length };
    },
    async research(args) {
      const t = (args.tickers || [])[0];
      if (!t) {
        const query = String(args.query || "");
        const text = researchFallbackText(query || "overview", { dataDir, budgetChars: 4000 });
        if (!text) throw Object.assign(new Error("no local research summaries are available"), { code: "NO_RESEARCH" });
        return { studies: true, query, text };
      }
      if (!research) throw Object.assign(new Error("research is not available"), { code: "NO_RESEARCH" });
      const r = await research(t);
      const f = r.features || {};
      return { ticker: r.ticker, features: f && { price: f.price, priceAsOf: f.priceAsOf, trend: f.trend, overextension: f.overextension, rsi14: f.rsi14, zPrice20: f.zPrice20,
        atr14: f.atr14, ret20: f.ret20, ret100: f.ret100, distSma50: f.distSma50, distSma200: f.distSma200, donchian20: f.donchian20 },
        quote: r.quote?.value && { last: r.quote.value.last, peRatio: r.quote.value.peRatio, eps: r.quote.value.eps, lastEarningsDate: r.quote.value.lastEarningsDate },
        fundamentals: r.financials?.value?.derived || null, news: (r.news?.value || []).slice(0, 8).map((n) => ({ date: n.published?.slice(0, 10), publisher: n.publisher, title: n.title })),
        dilution: r.dilution || r.financials?.value?.derived && { flags: r.financials.value.derived.dilutionFlags, flag: r.financials.value.derived.dilutionFlag, shareChangeYoY: r.financials.value.derived.dilutedShareChangeYoY, offering: r.financials.value.derived.dilutionOffering, offeringSeverity: r.financials.value.derived.dilutionOfferingSeverity, offeringSizePct: r.financials.value.derived.dilutionOfferingSizePct },
        gaps: r.gaps };
    },
    async macroCalendar(args) {
      const days = Math.max(1, Math.min(30, Number(args?.days) || 7));
      const calPath = join(dataDir, "Macro", "calendar.json");
      const surPath = join(dataDir, "Macro", "surprises.json");
      const watchPath = join(dataDir, "Macro", "watcher_status.json");
      const read = (p) => { try { return JSON.parse(readFileSync(p, "utf8")); } catch { return null; } };
      const cal = read(calPath);
      const surprises = (read(surPath) || {}).items || [];
      const watcher = read(watchPath) || {};
      const now = Date.now();
      const upcoming = ((cal && cal.events) || []).filter((e) => {
        const t = e.datetimeEt ? Date.parse(e.datetimeEt) : NaN;
        return Number.isFinite(t) && t >= now - 2 * 3600 * 1000 && t <= now + days * 86400 * 1000;
      });
      const latest = surprises.slice(-12).reverse();
      const lines = ["[MACRO_EVENTS] Upcoming releases (times in ET):"];
      for (const e of upcoming.slice(0, 16)) {
        lines.push(`  ${e.datetimeEt} ${e.name}${e.consensus != null ? ", consensus " + e.consensus : ""}${e.entry === "manual" ? " (manual entry)" : ""}`);
      }
      if (latest.length) {
        lines.push("Latest prints:");
        for (const s of latest.slice(0, 8)) {
          lines.push(`  ${s.scoredAt || s.period || ""} ${s.name || s.eventId}: actual ${s.actual ?? s.actualMid ?? s.highYield}, consensus ${s.consensus ?? s.impliedRate ?? s.benchmark}, z=${s.surpriseZ ?? s.tailZ ?? "n/a"}`);
        }
      }
      lines.push("Retail systems usually cannot trade the first tick; spreads widen around high-impact releases.");
      return {
        timezone: "America/New_York",
        builtAt: cal?.builtAt || null,
        upcoming,
        latestSurprises: latest,
        watcher,
        context: lines.join("\n"),
        provenRules: (read(join(dataDir, "Macro", "event_study.json")) || {}).nProvenHolm || 0,
      };
    },
    async macroNowcast(args) {
      const dir = join(dataDir, "Research", "macro");
      const read = (p) => { try { return JSON.parse(readFileSync(p, "utf8")); } catch { return null; } };
      const latest = read(join(dir, "nowcast_latest.json"));
      if (!latest) return { available: false, context: "[MACRO_NOWCAST] No nowcast published yet (engine/macro_nowcast.py refresh)." };
      const want = String(args?.target || "").toUpperCase();
      const nowcasts = (latest.nowcasts || []).filter((n) => !want || n.target === want);
      const acc = read(join(dir, "nowcast_accuracy.json")) || {};
      const track = Object.fromEntries(Object.entries(acc.accuracy || {}).filter(([k]) => !want || k === want).map(([k, v]) => {
        const a = v.exPandemic || v.final || {};
        return [k, { n: a.n, rmseRatioVsLastPrint: a.rmseRatio, clarkWestP: a.clarkWestP, directionHitRate: a.directionHitRate }];
      }));
      const lines = nowcasts.map((n) => `${n.target} ${n.period}: ${n.nowcast} (80% ${JSON.stringify(n.band80)}), last ${n.lastPrint}, surprise vs last ${n.impliedSurpriseVsLastPrint}, next ${n.nextRelease || n.estimatedRelease}`
        + (n.oursMinusOutside != null ? `, ours-minus-${n.oursMinusOutsideRef} ${n.oursMinusOutside}` : ""));
      return {
        available: true, asof: latest.asof, generatedAt: latest.generatedAt, regime: latest.regime, inputsPolicy: latest.inputsPolicy,
        nowcasts: nowcasts.map((n) => ({ ...n, drivers: (n.drivers || []).slice(0, 4) })), trackRecord: track,
        outsideComparison: acc.outsideComparison || null, outsideNote: latest.outsideNote,
        context: [`[MACRO_NOWCAST] as of ${latest.asof}. Regime: ${latest.regime?.label || "n/a"}.`, ...lines].join("\n"),
      };
    },
    async cryptoStatus() {
      return loadCryptoStatus({ root, dataDir, python, log });
    },
    async blackSwan() {
      const dir = join(dataDir, "Research", "black_swan");
      const read = (p) => { try { return JSON.parse(readFileSync(p, "utf8")); } catch { return null; } };
      const state = read(join(dir, "state.json")) || {};
      const feeds = read(join(dir, "feeds_status.json")) || {};
      const study = read(join(dir, "study.json")) || {};
      const events = [];
      try {
        const raw = readFileSync(join(dir, "events.jsonl"), "utf8").trim().split(/\r?\n/).filter(Boolean).slice(-12);
        for (const line of raw) { try { events.push(JSON.parse(line)); } catch { /* skip */ } }
      } catch { /* none yet */ }
      const lines = [`[BLACK_SWAN] state ${state.state || "UNKNOWN"} (${state.direction || "n/a"}). ${state.headline || ""}`];
      if (state.policy) {
        lines.push(`Playbook: gross x${state.policy.scaleGross ?? 1}, freezeLongs=${!!state.policy.freezeNewLongs}, VIXY=${!!state.policy.allowVixy}, SVIX=${!!state.policy.allowSvix}.`);
      }
      return {
        disabled: String(process.env.TRADING_BLACKSWAN || "").toLowerCase() === "off",
        state, feeds, recentEvents: events.slice(-8),
        study: {
          generatedAt: study.generatedAt, nEvents: study.nEvents, honesty: study.honesty,
          walkForward: study.walkForward && {
            ok: study.walkForward.ok, nApplied: study.walkForward.nApplied,
            eventWindowExcessMean: study.walkForward.eventWindowExcessMean,
            eventWindowExcessSE: study.walkForward.eventWindowExcessSE, honesty: study.walkForward.honesty,
          },
        },
        context: lines.join("\n"),
      };
    },
  };

  const withTimeout = (promise, ms, label) => Promise.race([promise, new Promise((_, reject) =>
    setTimeout(() => reject(Object.assign(new Error(`${label} timed out after ${Math.round(ms / 1000)} s`), { code: "TIMEOUT" })), ms).unref?.())]);

  /** Execute one tool call. Never throws: returns { ok, tool, method, op, args, result | error, ms, cached, summary }. */
  async function execute(call, { deadline = Infinity, refresh = false, promptChars = PROMPT_RESULT_CHARS } = {}) {
    const started = Date.now();
    let plan;
    try { plan = planCall(call); } catch (error) {
      return { ok: false, tool: call?.tool, error: { code: error.code || "BAD_ARGS", message: error.message }, ms: 0, summary: `ERROR ${error.message}` };
    }
    const timeoutMs = Math.max(1000, Math.min(plan.timeoutMs, deadline - Date.now()));
    const key = "tool:" + createHash("sha256").update(plan.op + "|" + stableKey(plan.args)).digest("hex").slice(0, 32);
    let out, cached = false;
    if (plan.cache && !refresh && store) {
      const hit = store.cacheGet(key);
      if (hit) { out = hit.value; cached = true; }
    }
    if (!out) {
      if (timeoutMs <= 1000 && deadline !== Infinity) out = { ok: false, error: { code: "DEADLINE", message: "tool budget for this question is used up" } };
      else if (plan.kind === "node") {
        try { out = { ok: true, result: await withTimeout(node[plan.op](plan.args), timeoutMs, plan.tool) }; }
        catch (error) { out = { ok: false, error: { code: error.code || "TOOL_FAILED", message: error.message } }; }
      } else if (!quant) out = { ok: false, error: { code: "NO_PYTHON", message: "quant tools are not configured" } };
      else out = await quant.run(plan.op, plan.args, { timeoutMs });
      if (out.ok && plan.cache && store) { try { store.cachePut(key, "tool", plan.args.tickers?.[0] || null, out, CACHE_TTL_MS); } catch { /* cache only */ } }
    }
    const record = { ok: Boolean(out.ok), tool: plan.tool, method: plan.method, family: plan.family, op: plan.op, args: plan.args,
      ms: Date.now() - started, cached, ...(out.ok ? { result: out.result } : { error: out.error }) };
    try { store?.log({ kind: "tool", model: null, status: out.ok ? "ok" : "failed", input: { tool: plan.tool, method: plan.method, op: plan.op, args: plan.args, purpose: call.purpose || null },
      output: out.ok ? out.result : null, error: out.ok ? null : `${out.error?.code}: ${out.error?.message}` }); } catch { /* logging only */ }
    log(out.ok ? "info" : "warn", "AI tool run", { tool: plan.tool, op: plan.op, ms: record.ms, cached, error: out.ok ? undefined : out.error });
    record.summary = out.ok ? summarize(out.result, promptChars) : `ERROR ${out.error?.code}: ${out.error?.message}`;
    return record;
  }
  return { execute, catalog: toolCatalog, list: () => TOOLS };
}

/**
 * Parse a slash command into a tool call (no model involved):
 *   /tool STATS/NORMALITY NVDA SPY start=2025-01-01 horizons=[1,5,20]
 *   /tool RUN_BACKTEST/TSMOM max_positions=5          /tool RISK NVDA horizon=5 confidences=[0.95,0.99]
 * The token after the tool may be a METHOD or a FAMILY; key=value pairs become options (start/end/horizon/query are fields).
 */
export function parseToolCommand(text) {
  const words = String(text || "").trim().replace(/^\/(tool|run)\s*/i, "").split(/\s+/).filter(Boolean);
  const head = (words.shift() || "").toUpperCase();
  const [tool, sub] = head.split("/");
  if (!TOOLS[tool]) throw Object.assign(new Error(`Unknown tool. Tools: ${Object.keys(TOOLS).join(", ")}`), { code: "TOOL_UNKNOWN" });
  const call = { tool, method: "DEFAULT", family: "NONE", tickers: [], start: "", end: "", horizonDays: null, options: [], query: "", purpose: "slash command" };
  if (sub && E.TOOL_METHOD.includes(sub)) call.method = sub;
  else if (sub && E.STRATEGY_FAMILY.includes(sub)) call.family = sub;
  const free = [];
  for (const w of words) {
    const m = w.match(/^([A-Za-z_][A-Za-z0-9_.]*)=(.*)$/);
    if (!m) { if (TICKER.test(w.toUpperCase()) && tool !== "MEMORY_SEARCH") call.tickers.push(w.toUpperCase()); else free.push(w); continue; }
    const [, key, value] = m;
    if (key === "start" || key === "end") call[key] = value;
    else if (key === "horizon" || key === "horizonDays") call.horizonDays = Number(value);
    else if (key === "method" && E.TOOL_METHOD.includes(value.toUpperCase())) call.method = value.toUpperCase();
    else if (key === "family" && E.STRATEGY_FAMILY.includes(value.toUpperCase())) call.family = value.toUpperCase();
    else call.options.push({ name: key, value });
  }
  if (tool === "MEMORY_SEARCH") call.query = free.join(" ");
  return call;
}
