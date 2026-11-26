// AI analyst actions: the analyst can do what the user can do in the dashboard, through the SAME route handlers.
//
//   user prompt -> adjudicate(): one strict-JSON model call picks { action (closed enum E.APP_ACTION), args,
//   confidence, needsConfirm, manualTopic, clarifyQuestion } -> legalize(): enum + per-action arg schema
//   (actionRegistry.js) -> unknown / low confidence / missing args => a clarifying question in the dock
//   -> human-only topics => "that's a manual step" + where in the UI -> read actions run now -> state-changing
//   actions return an Allow / Cancel prompt (token, expires, never auto-allowed) -> confirm(token) calls the route.
//
// Routes are called over loopback (http://127.0.0.1:<port>), so every write goes through the exact handler the
// page uses and writes the same records. Requests carry x-tm-origin: analyst_user_request. Math actions run
// through tools.execute (the analyst's own executor). Live orders, the live toggle, keys/secrets and Schwab
// sign-in are not in the enum at all, and FORBIDDEN blocks those routes even if someone adds them.
import { randomBytes } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { ROUTE_ACTIONS, MATH_ACTIONS, FORBIDDEN } from "./actionRegistry.js";
import { APP_ACTION, ACTION_CONFIDENCE, MANUAL_TOPIC as MANUAL_TOPICS } from "./enums.js";

export const ORIGIN = "analyst_user_request";
export const CONFIRM_TTL_MS = 5 * 60 * 1000;
export const META_ACTIONS = Object.freeze(["ASK_ANALYST", "CLARIFY", "MANUAL_STEP"]);
export const APP_ACTIONS = APP_ACTION;
export const CONFIDENCE = ACTION_CONFIDENCE;
export const MANUAL_TOPIC = MANUAL_TOPICS;
export const MANUAL_TEXT = Object.freeze({
  LIVE_ORDER: "Live Schwab orders are a manual step: use Paper trading > Live trading (ticket -> Preview -> Confirm). I can only paper trade, e.g. \"paper buy 10 SPY\".",
  LIVE_TOGGLE: "Turning live trading on or off is a manual step: Paper trading > Live trading switch.",
  API_KEYS: "Keys are a manual step: enter them yourself in Research & data > Data keys (Massive, FRED), the AI analyst page (AI keys) or the Schwab setup dialog. I never read or write secrets.",
  SCHWAB_SIGNIN: "Schwab sign-in is a manual step: click \"Click here to reauthorize\" in the banner and finish the login in your browser.",
  OTHER: "That's a manual step in the dashboard; I can't do it from here.",
});
// Jobs whose engine run evaluates the sealed holdout (the engine appends the look to holdout_ledger.json).
export const HOLDOUT_JOBS = Object.freeze(["swarm_run", "crypto_research", "research_all", "research_backtest", "research_walkforward"]);
// Started-job watchers: action -> status route (polled until done, then announced in the dock + voice).
const WATCH = { OPS_RUN: "GET /api/ops/jobs/:id", SCHWAB_DAILY_RUN: "GET /api/ops/jobs/:id", AI_FLAG: "GET /api/ai/flag/:id", SYMBOL_ADD: "GET /api/symbols/fetch/:ticker" };
const RUNNING = new Set(["running", "queued", "pending", "started", "fetching"]);

export const isForbidden = (route) => FORBIDDEN.some((re) => re.test(route));
export const specOf = (id) => ROUTE_ACTIONS[id] || MATH_ACTIONS[id] || null;
export const isWrite = (id) => ROUTE_ACTIONS[id]?.kind === "write";

// ------------------------------------------------------------------ validation (legalizer)
const TICKER = /^[A-Z][A-Z0-9.-]{0,11}$/;
const DATE = /^\d{4}-\d{2}-\d{2}$/;
function parseLoose(v) { if (typeof v !== "string") return v; const s = v.trim(); if (/^[[{]/.test(s)) { try { return JSON.parse(s); } catch { return v; } } return s; }
function one(name, spec, raw) {
  const v = parseLoose(raw);
  switch (spec.type) {
    case "ticker": { const s = String(v).toUpperCase().replace(/^\$/, "").trim(); if (!TICKER.test(s)) throw new Error(`${name}: not a ticker`); return s; }
    case "coin": { const s = String(v).toUpperCase().trim(); if (!/^[A-Z]{2,10}$/.test(s)) throw new Error(`${name}: not a coin symbol`); return s; }
    case "tickers": {
      const list = (Array.isArray(v) ? v : String(v).split(/[\s,]+/)).map((t) => String(t).toUpperCase().replace(/^\$/, "").trim()).filter(Boolean);
      if (!list.length || list.length > 60 || !list.every((t) => TICKER.test(t))) throw new Error(`${name}: 1-60 tickers`);
      return [...new Set(list)];
    }
    case "int": case "num": {
      const x = Number(String(v).replace(/[$,_]/g, "").replace(/k$/i, "000"));
      if (!Number.isFinite(x) || (spec.type === "int" && !Number.isInteger(x))) throw new Error(`${name}: not a${spec.type === "int" ? "n integer" : " number"}`);
      if ((spec.min != null && x < spec.min) || (spec.max != null && x > spec.max)) throw new Error(`${name}: must be ${spec.min}-${spec.max}`);
      return x;
    }
    case "date": { const s = String(v).trim(); if (!DATE.test(s)) throw new Error(`${name}: use YYYY-MM-DD`); return s; }
    case "enum": { const hit = spec.values.find((o) => String(o).toLowerCase() === String(v).toLowerCase().trim()); if (hit === undefined) throw new Error(`${name}: one of ${spec.values.join(", ")}`); return hit; }
    case "bool": { const s = String(v).toLowerCase().trim(); if (["true", "yes", "on", "1"].includes(s)) return true; if (["false", "no", "off", "0"].includes(s)) return false; throw new Error(`${name}: yes or no`); }
    case "json": { if (v === null || typeof v !== "object") throw new Error(`${name}: JSON object or list`); if (JSON.stringify(v).length > 4000) throw new Error(`${name}: too large`); return v; }
    default: { const s = String(v ?? "").replace(/[\u0000-\u001f]/g, " ").trim(); if (s.length > 500) throw new Error(`${name}: too long`); return s; }
  }
}
/** Validate model/user args ([{name, value}] or an object) against the action's spec. Unknown names are rejected. */
export function legalize(id, rawArgs) {
  const spec = specOf(id);
  if (!spec) return { ok: false, errors: [`unknown action ${id}`] };
  const pairs = Array.isArray(rawArgs) ? rawArgs.map((a) => [String(a?.name ?? ""), a?.value]) : Object.entries(rawArgs || {});
  const args = {}, errors = [];
  for (const [name, value] of pairs) {
    if (value === "" || value == null) continue;
    const s = spec.args[name];
    if (!s) { errors.push(`unknown argument ${name}`); continue; }
    try { args[name] = one(name, s, value); } catch (e) { errors.push(e.message); }
  }
  for (const [name, s] of Object.entries(spec.args)) {
    if (!(name in args) && s.def !== undefined) args[name] = s.def;
    if (s.req && !(name in args)) errors.push(`missing ${name} (${s.help})`);
  }
  return { ok: !errors.length, args, errors };
}

// ------------------------------------------------------------------ adjudication schema + prompt
const pick = (values, description) => ({ type: String, enum: [...values], description });
export const ADJUDICATION = {
  action: pick(APP_ACTIONS, "The one dashboard action that does what the user asked; ASK_ANALYST for questions, opinions or analysis that need no specific action; CLARIFY when the request is ambiguous or key details are missing; MANUAL_STEP for real-money orders, the live switch, API keys / secrets or Schwab sign-in."),
  args: { type: [{ type: { name: { type: String, description: "Argument name exactly as listed for the action." },
    value: { type: String, description: "Value as text or JSON (lists like [\"SPY\",\"QQQ\"], objects like {\"lookback_days\":126})." } } }],
    description: "Arguments for the action; empty when none." },
  confidence: pick(CONFIDENCE, "HIGH = clearly this action with these args; MEDIUM = probably; LOW = guessing."),
  needsConfirm: pick(["YES", "NO"], "YES if the action changes anything (creates accounts, trades on paper, starts jobs, changes settings)."),
  manualTopic: pick(MANUAL_TOPIC, "For MANUAL_STEP: which manual step; NONE otherwise."),
  clarifyQuestion: { type: String, description: "For CLARIFY (or LOW confidence): one short question to the user; empty otherwise." },
};
export function catalogText() {
  const line = (id, s) => `${id}: ${s.label}${Object.keys(s.args).length ? ` | args: ${Object.entries(s.args).map(([k, a]) => `${k}${a.req ? "*" : ""}${a.type === "enum" ? "(" + a.values.join("|") + ")" : ":" + a.type}`).join(", ")}` : ""}${s.example ? ` | e.g. "${s.example}"` : ""}`;
  return ["Dashboard actions (read = runs now; write = the user must Allow it):",
    ...Object.entries(ROUTE_ACTIONS).map(([id, s]) => `${line(id, s)} [${s.kind}]`),
    "Math actions (read-only, run now):", ...Object.entries(MATH_ACTIONS).map(([id, s]) => line(id, s))].join("\n");
}
export const ADJUDICATOR_SYSTEM = `You route a trading-dashboard user's request to ONE action from a closed list.
Return ONLY the structured JSON object.
Rules:
- Pick the action that does exactly what the user asked, with args from the list (* = required). Never invent args or values the user did not give; leave optional ones out.
- Questions, opinions and explanations ("what do you think of NVDA?", "why is the market down?") are ASK_ANALYST.
- If two actions fit equally, or a required arg is missing and cannot be inferred, use CLARIFY with one short question.
- Live (real-money) orders, turning live trading on/off, entering/reading API keys or secrets, and Schwab sign-in are MANUAL_STEP with manualTopic set. Paper trading is fine ("paper buy" -> EXPERIMENT_ORDER).
- Backtests and studies never look into the sealed holdout unless the user explicitly asks to score the holdout (HOLDOUT_LOOK).
- Tickers are upper case; company names become tickers (Nvidia -> NVDA). Months of lookback = 21 trading days each.
Examples:
"backtest TSMOM on QQQ with a 6-month lookback" -> MATH_BACKTEST family=TSMOM tickers=["QQQ"] lookback_days=126
"open a paper account running tsmom with $10k" -> EXPERIMENT_CREATE strategy=tsmom equity=10000
"10 $10k accounts sweeping lookback 3-12m" -> EXPERIMENT_CREATE strategy=tsmom equity=10000 count=10 sweep={"param":"lookback_days","from":63,"to":252}
"paper buy 10 SPY" -> EXPERIMENT_ORDER side=buy ticker=SPY qty=10
"try a dollar filter on the crypto trend" -> OPS_RUN job=crypto_research_pre
"what's the deflated Sharpe of account 12" -> MATH_ACCOUNT_SHARPE account=12
"run the regime model on QQQ" -> MATH_REGIME tickers=["QQQ"]
"price a 30-day SPY 600 call" -> MATH_OPTION_PRICE tickers=["SPY"] strike=600 dte=30 kind=call
"mute the swarm alerts" -> ALERT_PREFS_SET prefs={"speakCats":{"trade_swarm":false}}
"buy 10 SPY in my Schwab account" -> MANUAL_STEP manualTopic=LIVE_ORDER
"what do you think of Nvidia today?" -> ASK_ANALYST
"run a backtest" (no strategy) -> CLARIFY "Which strategy (TSMOM, mean reversion or combined) and on which tickers?"`;

// ------------------------------------------------------------------ deterministic fallback (no model / model failed)
const MONTHS = (m) => Math.round(Number(m) * 21);
export function fallbackAdjudicate(text) {
  const s = String(text || "").trim(), l = s.toLowerCase();
  const out = (action, args = {}, extra = {}) => ({ action, args: Object.entries(args).map(([name, value]) => ({ name, value: typeof value === "string" ? value : JSON.stringify(value) })), confidence: "HIGH", needsConfirm: isWrite(action) ? "YES" : "NO", manualTopic: "NONE", clarifyQuestion: "", source: "fallback", ...extra });
  const tick = (re) => (s.match(re)?.[1] || "").toUpperCase();
  let m;
  if ((m = s.match(/^\/do\s+([A-Z_]+)\s*(.*)$/i))) {
    const args = {}; for (const kv of m[2].match(/(\w+)=("[^"]*"|\S+)/g) || []) { const [k, ...v] = kv.split("="); args[k] = v.join("=").replace(/^"|"$/g, ""); }
    return out(m[1].toUpperCase(), args);
  }
  if (/\b(schwab|live|real)\b.*\b(buy|sell|order)\b|\b(buy|sell)\b.*\b(in|on) (my )?(schwab|live)\b/.test(l) && !/\bpaper\b/.test(l)) return out("MANUAL_STEP", {}, { manualTopic: "LIVE_ORDER" });
  if (/\b(turn|switch) (on|off) live|live trading (on|off)\b/.test(l)) return out("MANUAL_STEP", {}, { manualTopic: "LIVE_TOGGLE" });
  if (/\b(api key|secret|password|token)\b/.test(l) && /\b(set|enter|save|show|read|what is|change|paste)\b/.test(l)) return out("MANUAL_STEP", {}, { manualTopic: "API_KEYS" });
  if ((m = l.match(/\bpaper (buy|sell)\s+(\d+)\s+(?:shares? of\s+)?\$?([a-z][a-z0-9.-]{0,11})\b(?:.*?\baccount\s+(\d{4}))?/))) return out("EXPERIMENT_ORDER", { side: m[1], qty: m[2], ticker: m[3].toUpperCase(), ...(m[4] ? { id: m[4] } : {}) });
  if ((m = l.match(/\b(\d+)\s+\$?(\d+(?:\.\d+)?)(k)?\s+(?:paper\s+)?accounts?\b.*?\bsweep\w*\s+(?:the\s+)?lookback\s+(\d+)\s*(?:-|to)\s*(\d+)\s*(m|months?|d|days?)\b/))) {
    const scale = /^m/.test(m[6]) ? 21 : 1;
    return out("EXPERIMENT_CREATE", { strategy: /meanrev|mean reversion/.test(l) ? "meanrev" : /combined/.test(l) ? "combined" : "tsmom", count: m[1], equity: String(Number(m[2]) * (m[3] ? 1000 : 1)), sweep: { param: "lookback_days", from: Number(m[4]) * scale, to: Number(m[5]) * scale } });
  }
  if ((m = l.match(/\bopen (?:a |an )?(?:new )?(?:experiment(?:al)? )?paper account\b(?:\s+(?:running|with|using)\s+(tsmom|momentum|trend|meanrev|mean reversion|combined|manual))?.*?(?:\$\s?(\d+(?:\.\d+)?)\s*(k)?)?\s*$/))) {
    const strat = !m[1] ? "manual" : /mean/.test(m[1]) ? "meanrev" : m[1] === "combined" ? "combined" : m[1] === "manual" ? "manual" : "tsmom";
    return out("EXPERIMENT_CREATE", { strategy: strat, ...(m[2] ? { equity: String(Number(m[2]) * (m[3] ? 1000 : 1)) } : {}) });
  }
  if ((m = l.match(/\b(retire|delete|close)\s+(?:experiment\s+)?account\s+(\d{4})\b/))) return out(m[1] === "delete" ? "EXPERIMENT_DELETE" : "EXPERIMENT_RETIRE", { id: m[2] });
  if (/\b(list|show)\b.*\bexperiment/.test(l)) return out("EXPERIMENT_LIST");
  if ((m = l.match(/\bbacktest\s+(tsmom|momentum|trend|meanrev|mean[- ]reversion|combined)\b/))) {
    const family = /mean/.test(m[1]) ? "MEANREV" : m[1] === "combined" ? "COMBINED" : "TSMOM";
    const args = { family };
    const on = s.match(/\bon\s+((?:\$?[A-Z]{1,5}(?:\.[A-Z])?(?:\s*,\s*|\s+and\s+|\s+)?)+)/);
    if (on) { const list = on[1].split(/[\s,]+|and/).map((x) => x.replace("$", "").trim()).filter((x) => /^[A-Z]{1,5}(\.[A-Z])?$/.test(x)); if (list.length) args.tickers = list; }
    const lb = l.match(/(\d+)[- ]?(month|mo|week|day|year)s?\b.*?lookback|lookback (?:of )?(\d+)\s*(month|mo|week|day|year)/);
    if (lb) { const nn = Number(lb[1] || lb[3]), unit = lb[2] || lb[4]; args.lookback_days = String(unit.startsWith("mo") ? MONTHS(nn) : unit === "week" ? nn * 5 : unit === "year" ? nn * 252 : nn); }
    return out("MATH_BACKTEST", args);
  }
  if (/\bbacktest\b/.test(l) && !/\b(sma|ema|strategy)\b/.test(l)) return out("CLARIFY", {}, { clarifyQuestion: "Which strategy should I backtest (TSMOM, mean reversion or combined), and on which tickers?" });
  if (/\bdollar\b.*\bcrypto\b|\bcrypto\b.*\bdollar\b/.test(l) && /\b(try|test|filter|study|backtest|run)\b/.test(l)) return out("OPS_RUN", { job: "crypto_research_pre" });
  if ((m = l.match(/\bdeflated sharpe\b.*\baccount\s+(\d{1,3})\b/))) return out("MATH_ACCOUNT_SHARPE", { account: m[1] });
  if (/\b(pbo|probability of (backtest )?overfitting|overfit)/.test(l) && /\bswarm\b/.test(l)) return out("MATH_SWARM_OVERFIT");
  if ((m = l.match(/\bregime (?:model )?on\s+\$?([a-z]{1,5})\b/))) return out("MATH_REGIME", { tickers: [m[1].toUpperCase()] });
  if ((m = l.match(/\bprice (?:a |an )?(\d+)[- ]?day\s+\$?([a-z]{1,5})\s+(\d+(?:\.\d+)?)\s+(call|put)\b/))) return out("MATH_OPTION_PRICE", { tickers: [m[2].toUpperCase()], dte: m[1], strike: m[3], kind: m[4] });
  if ((m = l.match(/\bbeta of\s+\$?([a-z]{1,5})\s+(?:to|vs\.?|against)\s+\$?([a-z]{1,5})\b/))) return out("MATH_BETA", { tickers: [m[1].toUpperCase(), m[2].toUpperCase()] });
  if ((m = l.match(/\bgarch\b.*?\b(?:for|of|on)\s+\$?([a-z]{1,5})\b/))) return out("MATH_GARCH", { tickers: [m[1].toUpperCase()] });
  if ((m = l.match(/^quote\s+(.+)$/))) return out("QUOTES", { symbols: m[1].toUpperCase().split(/[\s,]+|and/).filter(Boolean) });
  if ((m = l.match(/\bmute (?:the )?(analyst|swarm|crypto|options|main|signals?|black swan|dollar|regime|macro|system)\b/))) {
    const cat = { analyst: "analyst", swarm: "trade_swarm", crypto: "trade_crypto", options: "trade_options", main: "trade_main", signal: "signals", signals: "signals", "black swan": "black_swan", dollar: "dollar_shock", regime: "regime", macro: "macro", system: "system" }[m[1]];
    return out("ALERT_PREFS_SET", { prefs: { speakCats: { [cat]: false } } });
  }
  void tick;
  return out("ASK_ANALYST", {}, { confidence: "LOW" });
}

// ------------------------------------------------------------------ runner
export function createActions({ fetchImpl = globalThis.fetch, tools = null, llm = null, experiments = null, memory = null, store = null,
  dataDir = null, log = () => {}, now = () => Date.now(), maxJobs = Number(process.env.TM_MAX_ANALYST_JOBS) || 2, jobTimeoutMs = 90 * 60 * 1000, pollMs = 5000 } = {}) {
  let baseUrl = null, emit = () => {};
  const pending = new Map();     // token -> { id, args, expires, summary }
  const watched = new Map();     // job key -> { action, label, startedAt }

  function attach(opts = {}) { if (opts.baseUrl) baseUrl = opts.baseUrl; if (opts.emit) emit = opts.emit; if (opts.tools) tools = opts.tools; if (opts.experiments) experiments = opts.experiments; if (opts.llm !== undefined) llm = opts.llm; return api; }
  const learnLog = (entry) => { try { store?.log?.({ kind: "action", model: null, status: entry.ok === false ? "failed" : "ok", input: { action: entry.action, args: entry.args, origin: ORIGIN }, output: entry.output ?? null, error: entry.error ?? null }); } catch { /* log only */ } };
  const remember = (item) => { try { memory?.ingest?.({ ...item, meta: { ...(item.meta || {}), origin: ORIGIN } }); } catch { /* optional */ } };

  function buildRequest(route, args) {
    const [method, pattern] = route.split(" ");
    const spec = Object.values(ROUTE_ACTIONS).find((s) => s.route === route)?.args || {};
    const query = new URLSearchParams(), body = {};
    let path = pattern;
    for (const [k, v] of Object.entries(args)) {
      const where = spec[k]?.in || (pattern.includes(":" + k) ? "path" : method === "GET" ? "query" : "body");
      if (where === "path") path = path.replace(":" + k, encodeURIComponent(String(v)));
      else if (where === "query") query.set(k, Array.isArray(v) ? v.join(",") : typeof v === "object" ? JSON.stringify(v) : String(v));
      else body[k] = v;
    }
    if (/:\w+/.test(path)) throw Object.assign(new Error(`missing path argument in ${pattern}`), { code: "BAD_ARGS" });
    return { method, url: path + (query.toString() ? "?" + query : ""), body: method === "GET" ? undefined : body };
  }
  async function callRoute(route, args) {
    if (isForbidden(route)) throw Object.assign(new Error("That route is human-only."), { code: "HUMAN_ONLY" });
    if (!baseUrl) throw Object.assign(new Error("Actions are only available while the dashboard server runs."), { code: "NO_SERVER" });
    const { method, url, body } = buildRequest(route, args);
    const r = await fetchImpl(baseUrl + url, { method, headers: { "Content-Type": "application/json", "x-tm-origin": ORIGIN }, body: body === undefined ? undefined : JSON.stringify(body) });
    const text = await r.text();
    let value; try { value = JSON.parse(text); } catch { value = { text: text.slice(0, 20000) }; }
    if (!r.ok) throw Object.assign(new Error(value?.error || `HTTP ${r.status}`), { code: "ROUTE_FAILED", status: r.status });
    return value;
  }

  const holdoutStart = () => { try { const d = JSON.parse(readFileSync(join(dataDir, "Research", "swarm", "design.json"), "utf8")); if (/^\d{4}-\d{2}-\d{2}$/.test(d.holdoutStart)) return d.holdoutStart; } catch { /* default */ } return "2026-07-09"; };
  const before = (date) => { const t = new Date(date + "T12:00:00Z"); t.setUTCDate(t.getUTCDate() - 1); return t.toISOString().slice(0, 10); };
  const trialCount = () => { try { return experiments?.trialCount?.() || 0; } catch { return 0; } };

  // Node-side math (read-only): uses the engine's Sharpe inference on files the swarm already wrote.
  const nodeMath = {
    async accountSharpe(args) {
      const file = join(dataDir, "Research", "swarm", args.book === "daytrade" ? "returns_daytrade.csv" : "returns_swing.csv");
      if (!existsSync(file)) throw new Error("No swarm returns yet. Run the swarm job first.");
      const [head, ...rows] = readFileSync(file, "utf8").trim().split(/\r?\n/);
      const cols = head.split(","), col = cols.indexOf(String(args.account));
      if (col < 1) throw new Error(`No swarm account ${args.account} (accounts 0-${cols.length - 2}).`);
      const hs = holdoutStart();
      const series = rows.map((r) => r.split(",")).filter((c) => c[0] < hs).map((c) => Number(c[col])).filter(Number.isFinite);
      const swarmTrials = cols.length - 1, extra = trialCount();
      const run = await tools.execute({ tool: "STATS", method: "SHARPE_INFERENCE", family: "NONE", tickers: [], start: "", end: "", horizonDays: null, query: "",
        purpose: "account Sharpe", options: [{ name: "series", value: JSON.stringify(series) }, { name: "trials", value: String(swarmTrials + extra) }] });
      if (!run.ok) throw new Error(run.error?.message || "Sharpe inference failed");
      return { account: args.account, book: args.book || "swing", sessions: series.length, through: before(hs), preHoldoutOnly: true,
        trialsUsed: swarmTrials + extra, trialsFromSwarm: swarmTrials, trialsFromUserExperiments: extra, ...run.result };
    },
    async swarmOverfit() {
      const p = join(dataDir, "Research", "swarm", "swarm_status.json");
      if (!existsSync(p)) throw new Error("No swarm results yet.");
      const doc = JSON.parse(readFileSync(p, "utf8")), sw = doc.swing || {};
      const tests = Object.fromEntries(Object.entries(sw.tests || {}).filter(([k]) => /preHoldout/.test(k)).map(([k, t]) => [k, { whiteRealityCheckP: t.realityCheckP, hansenSpaP: t.spaP, holmSignificant: t.significantHolm?.length ?? 0, bhSignificant: t.significantBH?.length ?? 0 }]));
      return { pbo: sw.pboPre || null, deflatedSharpe: sw.deflatedSharpePre || null, effectiveTrials: sw.effectiveTrials ?? null, preHoldoutTests: tests,
        userExperimentTrials: trialCount(), note: "Pre-holdout statistics only. User/analyst experiments are counted separately in trial_registry.jsonl." };
    },
  };

  async function runMath(id, args) {
    const spec = MATH_ACTIONS[id];
    if (spec.node) return nodeMath[spec.node](args);
    if (!tools) throw new Error("Math tools are not configured.");
    const call = { tool: spec.tool, method: spec.method || "DEFAULT", family: "NONE", tickers: args.tickers || [], start: args.start || "", end: args.end || "",
      horizonDays: args.horizonDays ?? null, options: [], query: "", purpose: `user request (${id})` };
    const toolMethods = tools.list?.()[spec.tool]?.methods || {};
    let note = null;
    for (const [k, v] of Object.entries(args)) {
      if (["tickers", "start", "end", "horizonDays"].includes(k)) continue;
      if (k === "family") { call.family = String(v).toUpperCase(); continue; }
      if (k === "rule" || (k === "method" && toolMethods[String(v).toUpperCase()])) { call.method = String(v).toUpperCase(); continue; }
      if (k === "lookback_days") { call.options.push({ name: "horizons", value: JSON.stringify([v]) }); continue; }
      if (k === "params") { for (const [pk, pv] of Object.entries(v)) call.options.push({ name: pk, value: typeof pv === "string" ? pv : JSON.stringify(pv) }); continue; }
      call.options.push({ name: k, value: typeof v === "string" ? v : JSON.stringify(v) });
    }
    if (id === "MATH_SHARPE" && !("trials" in args)) call.options.push({ name: "trials", value: String(Math.max(1, trialCount())) });
    if (["MATH_BACKTEST", "MATH_WALK_FORWARD", "MATH_RULE_STUDY"].includes(id)) {
      const cut = before(holdoutStart());
      if (!call.end || call.end > cut) { note = `Ended ${cut}: the holdout from ${holdoutStart()} stays sealed (ask to "score it on the holdout" to look, which is logged).`; call.end = cut; }
    }
    const run = await tools.execute(call, { refresh: false });
    if (!run.ok) throw new Error(run.error?.message || "tool failed");
    if (["MATH_BACKTEST", "MATH_WALK_FORWARD", "MATH_RULE_STUDY"].includes(id)) {
      const perf = run.result?.performance || {};
      try { experiments?.recordTrial?.({ kind: "strategy", origin: ORIGIN, action: id, configHash: JSON.stringify([call.tool, call.method, call.family, [...call.tickers].sort(), call.options]), sharpe: perf.sharpe ?? null, end: call.end, holdoutLook: false }); } catch { /* registry optional */ }
    }
    return note ? { holdoutNote: note, ...run.result } : run.result;
  }

  function summaryFor(id, args) {
    const spec = specOf(id);
    const parts = Object.entries(args).map(([k, v]) => `${k} = ${typeof v === "object" ? JSON.stringify(v) : v}`);
    const affects = [];
    if (id === "EXPERIMENT_CREATE") {
      let ids = "";
      try { const l = experiments?.list?.(); if (l) { const used = new Set(l.accounts.map((a) => a.id)); const next = []; for (let k = 1001; k <= 9999 && next.length < (args.count || 1); k++) if (!used.has(k)) next.push(k); ids = next.length > 1 ? `${next[0]}-${next[next.length - 1]}` : String(next[0]); affects.push(`${l.active + (args.count || 1)} of ${l.cap} active experiment accounts after this`); } } catch { /* preview only */ }
      affects.unshift(`Creates ${args.count || 1} new PAPER account(s)${ids ? ` (ids ${ids})` : ""} with $${Number(args.equity || 10000).toLocaleString("en-US")} each`);
      if (args.sweep) affects.push(`Sweeps ${args.sweep.param} from ${args.sweep.from} to ${args.sweep.to}`);
    }
    if (id === "EXPERIMENT_ORDER") affects.push(`PAPER ${String(args.side).toUpperCase()} ${args.qty} ${args.ticker} in experiment account ${args.id ?? 1000} at the latest price (no broker)`);
    if (["EXPERIMENT_RETIRE", "EXPERIMENT_DELETE", "EXPERIMENT_ADVANCE"].includes(id)) affects.push(`Experiment account ${args.id}`);
    if (id === "EXPERIMENT_DELETE") affects.push("Deletes it for good (its trial stays counted)");
    if (id === "HOLDOUT_LOOK" || (id === "OPS_RUN" && HOLDOUT_JOBS.includes(args.job))) affects.push("Looks at the SEALED HOLDOUT. The look is written to the holdout ledger and the result is not fed to learning.");
    if (id === "OPS_RUN") affects.push(`Starts the ${args.job} job (background; the result is posted here and spoken)`);
    if (id === "AUTOPILOT_TOGGLE") affects.push(`Autopilot PAPER runs ${args.enabled ? "ON" : "OFF"} (live trading is not affected)`);
    return { action: id, title: spec.label, args: parts, affects };
  }

  /** Run a validated action: reads now, writes -> Allow/Cancel prompt. Never throws. */
  async function run(id, rawArgs) {
    const spec = specOf(id);
    if (!spec) return { ok: false, kind: "error", code: "UNKNOWN_ACTION", text: `Unknown action ${id}.` };
    if (spec.route && isForbidden(spec.route)) return { ok: false, kind: "manual", code: "HUMAN_ONLY", text: MANUAL_TEXT.OTHER };
    const v = legalize(id, rawArgs);
    if (!v.ok) return { ok: false, kind: "clarify", code: "BAD_ARGS", errors: v.errors, text: `I need a bit more for "${spec.label}": ${v.errors.join("; ")}.` };
    if (id === "HOLDOUT_LOOK") v.args.confirm = true;
    if (spec.route && spec.kind === "write") {
      for (const [t, p] of pending) if (p.expires < now()) pending.delete(t);
      const token = randomBytes(12).toString("hex");
      const summary = summaryFor(id, v.args);
      pending.set(token, { id, args: v.args, expires: now() + CONFIRM_TTL_MS, summary });
      return { ok: true, kind: "confirm", token, expiresAt: new Date(now() + CONFIRM_TTL_MS).toISOString(), summary };
    }
    try {
      const result = spec.route ? await callRoute(spec.route, v.args) : await runMath(id, v.args);
      learnLog({ action: id, args: v.args, output: { summary: brief(result).slice(0, 2000) } });
      if (MATH_ACTIONS[id] && !result?.holdoutLook) remember({ kind: "EVENT", text: `User-requested ${spec.label} (${JSON.stringify(v.args)}): ${brief(result).slice(0, 900)}` });
      return { ok: true, kind: "action", action: id, label: spec.label, args: v.args, ...present(result) };
    } catch (e) {
      learnLog({ action: id, args: v.args, ok: false, error: e.message });
      return { ok: false, kind: "error", action: id, code: e.code || "FAILED", text: `${spec.label} failed: ${e.message}` };
    }
  }

  async function confirm(token) {
    const p = pending.get(String(token || ""));
    if (!p) return { ok: false, kind: "error", code: "EXPIRED", text: "That request expired or was already handled. Ask again." };
    pending.delete(token);
    if (p.expires < now()) return { ok: false, kind: "error", code: "EXPIRED", text: "That request expired. Ask again." };
    const spec = ROUTE_ACTIONS[p.id];
    try {
      // The Allow click is the user's confirmation; routes that demand confirm:true (state-saving jobs) get it.
      const result = await callRoute(spec.route, spec.route.startsWith("POST") ? { ...p.args, confirm: true } : p.args);
      learnLog({ action: p.id, args: p.args, output: { summary: brief(result).slice(0, 2000) } });
      const job = WATCH[p.id] && jobRef(p.id, p.args, result);
      if (job) watch(p.id, p.args.job ? `${spec.label} (${p.args.job})` : spec.label, job, p.args);
      return { ok: true, kind: "action", action: p.id, label: spec.label, args: p.args, started: Boolean(job), ...present(result) };
    } catch (e) {
      learnLog({ action: p.id, args: p.args, ok: false, error: e.message });
      return { ok: false, kind: "error", action: p.id, code: e.code || "FAILED", text: `${spec.label} failed: ${e.message}` };
    }
  }
  const cancel = (token) => ({ ok: pending.delete(String(token || "")), kind: "cancelled", text: "Cancelled. Nothing was changed." });

  function jobRef(id, args, result) {
    const status = String(result?.status || result?.job?.status || "").toLowerCase();
    const jid = result?.id || result?.job?.id;
    if (id === "SYMBOL_ADD") return RUNNING.has(status) ? { route: WATCH[id], args: { ticker: args.ticker } } : null;
    return jid && (RUNNING.has(status) || !status) ? { route: WATCH[id], args: { id: jid } } : null;
  }
  function watch(id, label, job, actionArgs = {}) {
    const key = job.route + JSON.stringify(job.args);
    if (watched.size >= maxJobs) { emit({ category: "analyst", type: "analyst", severity: "info", title: "AI analyst", text: `${label} started; I'm already watching ${watched.size} jobs, so check Research & data for this one.` }); return; }
    const started = now();
    watched.set(key, { id, label, started });
    const tick = async () => {
      try {
        const v = await callRoute(job.route, job.args);
        const st = String(v?.status || "").toLowerCase();
        if (RUNNING.has(st) && now() - started < jobTimeoutMs) return void setTimeout(tick, pollMs).unref?.();
        watched.delete(key);
        const ok = !RUNNING.has(st) && !/fail|error/.test(st);
        const tail = typeof v?.log === "string" ? v.log.trim().split(/\r?\n/).slice(-3).join(" ").slice(0, 300) : brief(v?.result || v).slice(0, 300);
        const text = RUNNING.has(st) ? `${label} is still running after ${Math.round(jobTimeoutMs / 60000)} minutes; I stopped watching it.` : `${label} ${ok ? "finished" : "failed"}. ${tail}`;
        emit({ category: "analyst", type: "analyst", severity: ok ? "success" : "warning", title: "AI analyst: job " + (ok ? "done" : "problem"), text, meta: { origin: ORIGIN, action: id } });
        if (ok && id === "OPS_RUN" && !HOLDOUT_JOBS.includes(actionArgs.job)) remember({ kind: "EVENT", text: `Research job ${label} finished on the user's request: ${tail}` });
      } catch (e) { watched.delete(key); emit({ category: "analyst", type: "analyst", severity: "warning", title: "AI analyst", text: `${label}: could not read the job status (${e.message}).` }); }
    };
    setTimeout(tick, pollMs).unref?.();
  }

  /** Adjudicate a prompt. llm({system, user}) -> value matching ADJUDICATION (stubbed in tests). */
  async function adjudicate(text, conversation) {
    let value = null, source = "model";
    if (llm) { try { value = await llm({ system: ADJUDICATOR_SYSTEM + "\n\n" + catalogText(), user:
      (conversation ? conversation + "\n\n" : "") + "Latest user request: " + JSON.stringify(String(text).slice(0, 2000)) }); } catch { value = null; } }
    if (!value) { value = fallbackAdjudicate(text); source = "fallback"; }
    return { ...sanitize(value), source };
  }
  function sanitize(value) {
    const action = APP_ACTIONS.includes(value?.action) ? value.action : "CLARIFY";
    const rejected = action === "CLARIFY" && value?.action && !APP_ACTIONS.includes(value.action);
    return { action, args: Array.isArray(value?.args) ? value.args.slice(0, 20) : [], confidence: CONFIDENCE.includes(value?.confidence) ? value.confidence : "LOW",
      manualTopic: MANUAL_TOPIC.includes(value?.manualTopic) ? value.manualTopic : "NONE", clarifyQuestion: String(value?.clarifyQuestion || "").slice(0, 300), rejected };
  }

  /** Entry from the dock: returns a reply object, or null to let the normal analyst Q&A handle it. */
  async function handle(text, conversation) {
    const a = await adjudicate(text, conversation);
    if (a.action === "ASK_ANALYST") return null;
    if (a.action === "MANUAL_STEP") return { kind: "manual", text: MANUAL_TEXT[a.manualTopic] || MANUAL_TEXT.OTHER, adjudication: a };
    if (a.action === "CLARIFY" || a.confidence === "LOW") {
      return { kind: "clarify", text: a.clarifyQuestion || (a.rejected ? "I couldn't match that to a dashboard action. What should I do, e.g. \"backtest TSMOM on QQQ\" or \"paper buy 10 SPY\"?" : "Can you say a little more about what you want me to do?"), adjudication: a };
    }
    const r = await run(a.action, a.args);
    return { ...r, adjudication: { action: a.action, confidence: a.confidence, source: a.source } };
  }

  const api = { attach, run, confirm, cancel, adjudicate, handle, legalize, catalog: () => ({ routes: ROUTE_ACTIONS, math: MATH_ACTIONS, manualSteps: MANUAL_TEXT }), jobs: () => [...watched.values()], _pending: pending, _buildRequest: buildRequest };
  return api;
}

// ------------------------------------------------------------------ result presentation (numbers + a small table)
export function brief(value) {
  const out = [];
  const walk = (v, path, depth) => {
    if (out.length >= 16) return;
    if (v == null) return;
    if (typeof v === "number") out.push(`${path}: ${Number.isInteger(v) ? v : Number(v.toPrecision(4))}`);
    else if (typeof v === "string") { if (v.length <= 160 && path) out.push(`${path}: ${v}`); }
    else if (typeof v === "boolean") out.push(`${path}: ${v ? "yes" : "no"}`);
    else if (Array.isArray(v)) out.push(`${path}: ${v.length} item${v.length === 1 ? "" : "s"}`);
    else if (depth < 2) for (const [k, x] of Object.entries(v)) walk(x, path ? `${path}.${k}` : k, depth + 1);
  };
  // Headline numbers first (performance / result blocks), never the holdout note (shown separately).
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const first = ["performance", "summary", "result"].filter((k) => value[k] && typeof value[k] === "object");
    for (const k of first) walk(value[k], k, 1);
    for (const [k, x] of Object.entries(value)) if (!first.includes(k) && k !== "holdoutNote") walk(x, k, 1);
  } else walk(value, "", 0);
  return out.join("\n");
}
export function present(result) {
  const lines = brief(result).split("\n").filter(Boolean);
  let table = null;
  const firstTable = (v, depth = 0) => {
    if (!v || typeof v !== "object" || depth > 2) return null;
    for (const x of Object.values(v)) {
      if (Array.isArray(x) && x.length && x.every((r) => r && typeof r === "object" && !Array.isArray(r))) return x;
      const deeper = !Array.isArray(x) && firstTable(x, depth + 1); if (deeper) return deeper;
    }
    return null;
  };
  const rows = Array.isArray(result) ? result : firstTable(result);
  if (rows) {
    const columns = [...new Set(rows.slice(0, 12).flatMap((r) => Object.keys(r).filter((k) => r[k] == null || typeof r[k] !== "object")))].slice(0, 7);
    table = { columns, rows: rows.slice(0, 12).map((r) => columns.map((c) => (typeof r[c] === "number" ? Number(r[c].toPrecision(5)) : r[c] ?? ""))) };
  }
  const note = result?.holdoutNote ? `\n\n_${result.holdoutNote}_` : "";
  return { answer: (lines.length ? lines.map((l) => "- " + l).join("\n") : "Done.") + note, table };
}
