// The analyst: research -> structured verdict / answer -> validation & clamps -> local storage -> paper-trader contract.
// The LLM is advisory only. Every model call is schema-enforced (Retort enums), validated, retried and time-limited;
// any failure produces a deterministic FALLBACK verdict that the paper-trader export ignores (= no trade change).
import { renameSync, writeFileSync, mkdirSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import * as E from "./enums.js";
import { ANSWER, TOOL_STEP, VERDICT } from "./schemas.js";
import { ANSWER_TASK, VERDICT_TASK, systemPrompt, toolTask } from "./prompts.js";
import { renderMaster } from "./strategyCard.js";
import { MAX_CALLS_PER_STEP } from "./tools.js";
import { buildContext, researchFallbackText } from "./context.js";
import { computeFeatures, diversification, loadHmmRegime, marketRegime, paperBook } from "./features.js";
import { converse } from "./runtime.js";
import { annotateFinancials, applyDilutionToVerdict } from "./dilution.js";

const DEFAULT_TARGETS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO", "AMD"];
const r2 = (x) => (x == null || !Number.isFinite(x) ? null : Math.round(x * 100) / 100);
export const easternDate = (d = new Date()) => d.toLocaleDateString("en-CA", { timeZone: "America/New_York" });

// ---------------------------------------------------------------- deterministic levels (also the fallback)
export function deterministicLevels(f) {
  if (!f?.price) return null;
  const p = f.price, atr = f.atr14 || p * 0.02;
  const up = f.trend === "UP" || f.trend === "STRONG_UP";
  const down = f.trend === "DOWN" || f.trend === "STRONG_DOWN";
  let band = null, opportunity = "NONE";
  if (!down) {
    if (f.overextension === "OVEREXTENDED" || f.overextension === "EXTENDED") {
      const high = p - 1 * atr, low = Math.max(p - 3 * atr, f.sma20 && f.sma20 < high ? f.sma20 - 0.5 * atr : p - 3 * atr);
      band = [low, high];                                   // wait for a pullback
    } else if (up && (f.overextension === "OVERSOLD" || f.overextension === "DEEPLY_OVERSOLD")) {
      band = [p - 1 * atr, p + 0.5 * atr]; opportunity = "OVERSOLD_REVERSAL";
    } else {
      band = [p - 1.5 * atr, p + 0.5 * atr];
      if (up && f.distSma20 != null && f.distSma20 < 0 && f.distSma50 > 0) opportunity = "PULLBACK_IN_UPTREND";
      if (f.breakout20) opportunity = "BREAKOUT";
    }
  }
  const swingLow = f.donchian20?.low;
  let stop = p - 2.5 * atr;
  if (swingLow && swingLow < p && p - swingLow <= 3.5 * atr && p - swingLow >= 1.5 * atr) stop = swingLow - 0.25 * atr;
  return { price: p, atr, stop: r2(stop), entryLow: band ? r2(band[0]) : null, entryHigh: band ? r2(band[1]) : null,
    limitPrice: band ? r2(Math.min(p, band[1])) : null, opportunity };
}

function dilutionOf(research) {
  if (research?.dilution) return research.dilution;
  const fin = research?.financials?.value;
  if (!fin) return null;
  return annotateFinancials({ ...fin, ticker: research.ticker }, {
    splits: research.splits?.value, asof: easternDate(), flagsFile: research.dilutionFlags || null,
  }).dilution;
}

function fundamentalHealth(fin, dilution) {
  const d = fin?.derived;
  if (!d || d.revenueTTM == null) return "UNKNOWN";
  const change = dilution?.shareChangeYoY ?? d.dilutedShareChangeYoY;
  const bad = [d.fcfTTM != null && d.fcfTTM < 0, d.revenueGrowthTTM != null && d.revenueGrowthTTM < -0.05,
    d.interestCoverage != null && d.interestCoverage < 2, d.netDebtToOperatingIncome != null && d.netDebtToOperatingIncome > 4,
    change != null && change > 0.05].filter(Boolean).length;
  let health = "ADEQUATE";
  if (bad >= 3) health = "DISTRESSED";
  else if (bad === 2) health = "WEAK";
  else if (bad === 0 && (d.fcfMarginTTM ?? 0) > 0.1 && (d.revenueGrowthTTM ?? 0) > 0.05) health = "STRONG";
  if ((dilution?.red || (change != null && change >= 0.10)) && health !== "DISTRESSED") return "WEAK";
  return health;
}

export function fallbackVerdict(research, reason) {
  const f = research.features, lv = deterministicLevels(f);
  const raw = {
    ticker: research.ticker, source: "FALLBACK", model: null, reason,
    flag: "YELLOW", conviction: "LOW", fundamentalHealth: fundamentalHealth(research.financials?.value, dilutionOf(research)),
    trend: f?.trend || "SIDEWAYS", overextension: f?.overextension || "NEUTRAL", opportunity: lv?.opportunity || "NONE",
    action: "HOLD", horizon: "POSITION_20_60D",
    summary: `Model verdict unavailable (${reason}). Deterministic levels only; the paper trader is not changed by this verdict.`,
    entry: { orderType: "NONE", limitPrice: lv?.limitPrice ?? null, bandLow: lv?.entryLow ?? null, bandHigh: lv?.entryHigh ?? null, rationale: "ATR / moving-average band computed in code." },
    stop: { price: lv?.stop ?? null, atrMultiple: lv && f?.atr14 ? r2((lv.price - lv.stop) / f.atr14) : null, rationale: "2.5 ATR or just under the 20-day low." },
    takeProfit: { price: null, rationale: "" }, sizeHint: "NONE",
    diversification: { status: research.diversification?.status || "OK", note: "" },
    catalysts: [], risks: [], dataGaps: research.gaps, citations: [], clamps: [],
  };
  return applyDilutionToVerdict(raw, dilutionOf(research)).verdict;
}

// ---------------------------------------------------------------- validation / clamps (code beats model)
const rank = (list, v) => list.indexOf(v);
export function sanitizeVerdict(raw, research) {
  const v = structuredClone(raw), clamps = [];
  const f = research.features, lv = deterministicLevels(f);
  if (v.ticker !== research.ticker) { clamps.push(`ticker ${v.ticker} -> ${research.ticker}`); v.ticker = research.ticker; }
  if (!lv) { clamps.push("no price data: verdict forced to YELLOW/WATCH"); v.flag = v.flag === "RED" ? "RED" : "YELLOW"; v.action = "WATCH"; v.clamps = clamps; return applyDilutionToVerdict(v, dilutionOf(research)).verdict; }
  const p = lv.price, atr = lv.atr;
  // Stop: below price, 0.5-6 ATR away.
  const s = v.stop?.price;
  if (s == null || !(s < p) || p - s < 0.5 * atr || p - s > 6 * atr) {
    clamps.push(`stop ${s} -> ${lv.stop} (must be 0.5-6 ATR below ${r2(p)})`);
    v.stop = { ...(v.stop || {}), price: lv.stop, atrMultiple: r2((p - lv.stop) / atr) };
  } else v.stop.atrMultiple = r2((p - s) / atr);
  // Entry band: ordered, within [p-4ATR, p+1.5ATR]; none for RED.
  const e = v.entry || {};
  if (v.flag === "RED" || ["AVOID", "EXIT", "TRIM"].includes(v.action)) {
    if (e.bandLow != null || e.bandHigh != null) clamps.push("entry band removed (no new entry advised)");
    v.entry = { ...e, orderType: "NONE", bandLow: null, bandHigh: null, limitPrice: null };
  } else if (e.bandLow != null || e.bandHigh != null) {
    let lo = e.bandLow ?? lv.entryLow, hi = e.bandHigh ?? lv.entryHigh;
    if (lo == null || hi == null || !(lo < hi) || lo < p - 4 * atr || hi > p + 1.5 * atr) {
      clamps.push(`entry band ${e.bandLow}-${e.bandHigh} -> ${lv.entryLow}-${lv.entryHigh}`);
      lo = lv.entryLow; hi = lv.entryHigh;
    }
    let lim = e.limitPrice;
    if (lim != null && lo != null && (lim < lo || lim > hi)) { clamps.push(`limit ${lim} clamped into band`); lim = r2(Math.min(Math.max(lim, lo), hi)); }
    v.entry = { ...e, bandLow: lo, bandHigh: hi, limitPrice: lim };
  }
  if (v.takeProfit?.price != null && !(v.takeProfit.price > p)) { clamps.push("take-profit below price removed"); v.takeProfit.price = null; }
  // Overextension: keep the code's classification unless the model is within one step of it.
  if (Math.abs(rank(E.OVEREXTENSION, v.overextension) - rank(E.OVEREXTENSION, f.overextension)) >= 2) {
    clamps.push(`overextension ${v.overextension} -> ${f.overextension} (code)`); v.overextension = f.overextension;
  }
  // Consistency and deterministic diversification guard.
  if (v.flag === "RED" && ["BUY", "ADD"].includes(v.action)) { clamps.push("RED + BUY -> WATCH"); v.action = "WATCH"; }
  const div = research.diversification?.status;
  if (div === "WOULD_BREACH_CAP" && ["BUY", "ADD"].includes(v.action)) {
    clamps.push("diversification cap: BUY -> WATCH"); v.action = "WATCH";
    if (v.flag === "GREEN") v.flag = "YELLOW";
  }
  if (div && div !== "OK" && rank(E.SIZE_HINT, v.sizeHint) > rank(E.SIZE_HINT, "HALF")) { clamps.push(`size ${v.sizeHint} -> HALF (${div})`); v.sizeHint = "HALF"; }
  if (v.overextension === "OVEREXTENDED" && v.flag === "GREEN" && v.action === "BUY" && (v.entry?.bandHigh ?? Infinity) >= p) {
    clamps.push("overextended: entry band capped 1 ATR below price"); v.entry.bandHigh = r2(p - atr);
    if (v.entry.bandLow != null && v.entry.bandLow >= v.entry.bandHigh) v.entry.bandLow = r2(p - 3 * atr);
    if (v.entry.limitPrice != null && v.entry.limitPrice > v.entry.bandHigh) v.entry.limitPrice = v.entry.bandHigh;
  }
  const stale = f.lastBarDate && (Date.now() - Date.parse(f.lastBarDate)) / 86400000 > 7 && f.priceSource !== "schwab_live";
  if (stale && v.flag === "GREEN") { clamps.push("price data older than 7 days: GREEN -> YELLOW"); v.flag = "YELLOW"; }
  v.clamps = clamps;
  return applyDilutionToVerdict(v, dilutionOf(research)).verdict;
}

// engine/advisory.py adjust_entry, mirrored for the decision API (never increases size).
export function adjustEntry(verdict, amount, refPrice, yellowScale = 0.5) {
  if (!verdict || amount <= 0) return { amount, reason: null };
  if (verdict.flag === "RED") return { amount: 0, reason: "advisory_red" };
  if (verdict.entry_low != null && refPrice != null && refPrice < verdict.entry_low) return { amount: 0, reason: "advisory_below_band" };
  if (verdict.entry_high != null && refPrice != null && refPrice > verdict.entry_high) return { amount: 0, reason: "advisory_above_band" };
  if (verdict.flag === "YELLOW") return { amount: amount * Math.min(Math.max(yellowScale, 0), 1), reason: "advisory_yellow" };
  return { amount, reason: null };
}

// ---------------------------------------------------------------- advisor
export function createAdvisor({ dataDir, models, store, sources, prices, memory = null, tools = null, card = null, log = () => {},
  env = process.env, fetchImpl, sleep }) {
  const maxToolSteps = Math.max(0, Number(env.TRADING_AI_TOOL_STEPS ?? 4));
  const toolBudgetMs = Math.max(10000, Number(env.TRADING_AI_TOOL_BUDGET_MS ?? 20 * 60 * 1000));
  // ---- master system prompt: grounding rules + TRADING MODEL REFERENCE generated from engine code / research files.
  // Any failure falls back to the cached card, then to the rules alone (never blocks a call).
  async function system() {
    if (!card) return systemPrompt("");
    try { return systemPrompt(renderMaster(await card.get())); }
    catch (error) {
      log("ai-card", { status: "failed", message: error.message });
      try { return systemPrompt(renderMaster(card.cached())); } catch { return systemPrompt(""); }
    }
  }
  const canCallModel = (modelId) => {
    try { const profile = modelId ? models.profile(modelId) : models.current(); return Boolean(models.apiKey(profile.provider)); }
    catch { return false; }
  };
  // ---- analyst memory hooks (all optional and failure-proof: memory problems never block an answer or verdict)
  const remember = (item) => { if (!memory) return; try { memory.ingest(item); } catch (error) { log("ai-memory", { status: "failed", message: error.message }); } };
  function rememberResearch(r) {
    for (const n of (r.news?.value || []).slice(0, 25))
      remember({ kind: "NEWS", ticker: r.ticker, ts: n.published, text: `${n.pressRelease ? "Press release" : n.publisher}: ${n.title}. ${n.description || ""}${n.sentiment ? ` (sentiment ${n.sentiment})` : ""}`,
        meta: { dedupeKey: n.url || n.title, url: n.url, publisher: n.publisher, pressRelease: n.pressRelease } });
    const d = r.financials?.value?.derived;
    if (d?.lastQuarterEnd) {
      const p = (x) => (x == null ? "n/a" : (x * 100).toFixed(1) + "%"), b = (x) => (x == null ? "n/a" : (x / 1e9).toFixed(2) + "B");
      remember({ kind: "FUNDAMENTALS", ticker: r.ticker, ts: d.lastQuarterEnd + "T21:00:00Z", meta: { dedupeKey: "sec:" + d.lastQuarterEnd },
        text: `SEC snapshot for quarter ended ${d.lastQuarterEnd}: TTM revenue ${b(d.revenueTTM)} (growth ${p(d.revenueGrowthTTM)}, last quarter YoY ${p(d.revenueGrowthLastQuarterYoY)}), operating margin ${p(d.operatingMarginTTM)}, FCF ${b(d.fcfTTM)} (${p(d.fcfMarginTTM)}), net debt ${b(d.netDebt)}, interest coverage ${d.interestCoverage == null ? "n/a" : d.interestCoverage.toFixed(1) + "x"}, diluted shares YoY ${p(d.dilutedShareChangeYoY)}.` });
    }
  }
  async function recallFor(query, list, kinds = null) {
    if (!memory) return "";
    const prices = Object.fromEntries(list.filter((r) => r.features?.price).map((r) => [r.ticker, r.features.price]));
    const m = await memory.recall({ query, tickers: list.map((r) => r.ticker), kinds, prices });
    return m.text;
  }

  const researchDir = join(dataDir, "Research");
  const advisoryPath = () => env.ADVISORY_VERDICTS_PATH || join(researchDir, "advisory_verdicts.json");
  const maxAgeDays = Number(env.TRADING_AI_MAX_AGE_DAYS || 3);

  async function attempt(gaps, label, work) {
    try {
      const r = await work();
      if (r?.warning) gaps.push(`${label}: using cached copy from ${r.fetchedAt} (${r.warning})`);
      return r;
    } catch (error) { gaps.push(`${label}: ${String(error.message).split(/(?<=\.)\s/)[0]}`); return null; }
  }

  async function research(ticker, { refresh = false, book = null } = {}) {
    ticker = ticker.toUpperCase();
    const gaps = [];
    const [details, news, dividends, splits, quote] = await Promise.all([
      attempt(gaps, "ticker details", () => sources.details(ticker, { refresh })),
      attempt(gaps, "news", () => sources.news(ticker, { refresh })),
      attempt(gaps, "dividends", () => sources.dividends(ticker, { refresh })),
      attempt(gaps, "splits", () => sources.splits(ticker, { refresh })),
      attempt(gaps, "Schwab quote", async () => {
        const r = await sources.schwabQuotes([ticker], { refresh });
        return r.value?.[ticker] ? { ...r, value: r.value[ticker] } : Promise.reject(new Error("no quote returned"));
      }),
    ]);
    const financials = details?.value?.type === "ETF" ? null
      : await attempt(gaps, "SEC financials", () => sources.secFinancials(ticker, details?.value?.cik, { refresh }));
    if (details?.value?.type === "ETF") gaps.push("SEC financials: not applicable (ETF)");
    const bars = prices ? prices.bars(ticker, 300) : [];
    if (!bars.length) gaps.push("local price history: none for " + ticker);
    const spy = prices ? prices.bars("SPY", 300) : [];
    const features = computeFeatures(bars, { live: quote?.value, benchmark: spy });
    book = book || paperBook(dataDir);
    const industries = {};
    for (const p of book.positions) {
      const d = store.cacheGet(`details:${p.ticker}`, { allowStale: true }) ||
        await attempt([], "details", () => sources.details(p.ticker));
      if (d?.value?.industry) industries[p.ticker] = d.value.industry;
    }
    const div = diversification({ ticker, industry: details?.value?.industry, book, industries, prices });
    let flagsFile = null;
    const flagsPath = join(dataDir, "Research", "dilution_flags.json");
    try { if (existsSync(flagsPath)) flagsFile = JSON.parse(readFileSync(flagsPath, "utf8")); } catch { flagsFile = null; }
    let dilution = null;
    let financialsOut = financials;
    if (financials?.value) {
      const annotated = annotateFinancials({ ...financials.value, ticker }, {
        splits: splits?.value, flagsFile, asof: easternDate(),
      });
      financialsOut = { ...financials, value: { ...financials.value, derived: annotated.derived, history: financials.value.history || {} } };
      dilution = annotated.dilution;
    }
    const bundle = { ticker, details, news, dividends, splits, quote, financials: financialsOut, features, diversification: div, book, gaps,
      dilution, dilutionFlags: flagsFile, collectedAt: new Date().toISOString() };
    rememberResearch(bundle);
    return bundle;
  }

  const regime = () => {
    const spy = prices ? marketRegime(prices.bars("SPY", 300)) : null;
    const hmm = loadHmmRegime(dataDir);
    if (!spy && !hmm) return null;
    return { ...(spy || {}), hmm };
  };

  function toRecord(v, research, modelId) {
    const f = research.features || {};
    return { ...v, ticker: research.ticker, createdAt: new Date().toISOString(), asof: easternDate(), model: modelId || v.model || null,
      convictionScore: E.CONVICTION_SCORE[v.conviction] ?? null, overextensionScore: f.zPrice20 ?? null,
      refPrice: f.price ?? null, atr: f.atr14 ?? null, entryLow: v.entry?.bandLow ?? null, entryHigh: v.entry?.bandHigh ?? null,
      limitPrice: v.entry?.limitPrice ?? null, orderType: v.entry?.orderType ?? null, stop: v.stop?.price ?? null,
      takeProfit: v.takeProfit?.price ?? null, diversification: v.diversification?.status ?? null,
      dataAsOf: { lastBar: f.lastBarDate ?? null, price: f.priceAsOf ?? null, news: research.news?.fetchedAt ?? null,
        financials: research.financials?.value?.derived?.lastQuarterEnd ?? null, quote: research.quote?.fetchedAt ?? null } };
  }

  async function verdict(ticker, { refresh = false, modelId, book } = {}) {
    const res = await research(ticker, { refresh, book });
    const memoryText = await recallFor(`${res.ticker} outlook: prior flags, verdicts, decisions, catalysts, risks, earnings, guidance`, [res]);
    const context = buildContext([res], { regime: regime(), book: res.book }) + (memoryText ? "\n\n" + memoryText : "");
    let record;
    const sys = await system();
    try {
      const result = await converse(($) => {
        $.system`${sys}`;
        $.user`${context}`;
        $.user`${VERDICT_TASK} Ticker: ${res.ticker}. Today (US Eastern): ${easternDate()}.`;
      }, { models, modelId, name: "tradeVerdict", description: "Structured red/yellow/green verdict for one ticker.",
        parameters: VERDICT, temperature: 0.2, fetchImpl, sleep, log: (e) => log("ai-model", e) });
      const clean = sanitizeVerdict(result.value, res);
      record = toRecord({ ...clean, source: "LLM" }, res, result.modelId);
      store.log({ kind: "verdict", model: result.modelId, status: "ok", input: { ticker: res.ticker, contextChars: context.length },
        output: record, attempts: result.attempts, usage: result.usage });
      remember({ kind: "VERDICT", ticker: record.ticker, ts: record.createdAt,
        text: `Flagged ${record.flag} (${record.conviction}), action ${record.action}, ${record.horizon}; trend ${record.trend}, stretch ${record.overextension}; entry band ${record.entryLow ?? "none"}-${record.entryHigh ?? "none"}, stop ${record.stop ?? "n/a"}. ${record.summary}`,
        meta: { refPrice: record.refPrice, flag: record.flag, action: record.action, reason: String(record.summary || "").split(/(?<=[.!?])\s/)[0], entryLow: record.entryLow, entryHigh: record.entryHigh, stop: record.stop, model: record.model } });
    } catch (error) {
      record = toRecord(fallbackVerdict(res, `${error.code || "ERROR"}: ${error.message}`), res, null);
      store.log({ kind: "verdict", model: modelId || models.current().id, status: "fallback", input: { ticker: res.ticker },
        error: `${error.code || ""} ${error.message}`, attempts: error.attempts });
    }
    store.saveVerdict(record);
    return record;
  }

  // Contract file read by engine/advisory.py (only fresh LLM verdicts; FALLBACK never changes trading).
  function exportAdvisory() {
    const cutoff = Date.now() - maxAgeDays * 86400000;
    const verdicts = {};
    for (const v of store.latestVerdicts()) {
      if (v.source !== "LLM" || Date.parse(v.createdAt) < cutoff) continue;
      verdicts[v.ticker] = { flag: v.flag, conviction: v.convictionScore, entry_low: v.entryLow, entry_high: v.entryHigh,
        stop: v.stop, overextension: v.overextensionScore, asof: v.asof, source: "ai:" + (v.model || "model"),
        action: v.action, size_hint: v.sizeHint, size_scale: E.SIZE_SCALE[v.sizeHint] ?? null, limit_price: v.limitPrice,
        order_type: v.orderType, take_profit: v.takeProfit, horizon: v.horizon, trend: v.trend,
        overextension_class: v.overextension, opportunity: v.opportunity, fundamental_health: v.fundamentalHealth,
        diversification: v.diversification, ref_price: v.refPrice, atr: v.atr, created_at: v.createdAt };
    }
    const payload = { schema: "advisory_verdicts/v1", generatedAt: new Date().toISOString(), asof: easternDate(), verdicts };
    mkdirSync(researchDir, { recursive: true });
    const temporary = advisoryPath() + "." + randomUUID() + ".tmp";
    writeFileSync(temporary, JSON.stringify(payload, null, 2));
    renameSync(temporary, advisoryPath());
    return { path: advisoryPath(), count: Object.keys(verdicts).length };
  }

  function targets() {
    if (env.TRADING_AI_TARGETS) return env.TRADING_AI_TARGETS.split(/[\s,]+/).filter(Boolean).map((t) => t.toUpperCase());
    const book = paperBook(dataDir);
    const list = [...new Set([...book.positions.map((p) => p.ticker), ...book.targets.filter((t) => t.target_weight > 0).map((t) => t.ticker)])];
    return list.length ? list : DEFAULT_TARGETS;
  }

  async function flagTargets(list = null, { refresh = false, modelId, onProgress = () => {}, concurrency = 2 } = {}) {
    list = (list?.length ? list : targets()).map((t) => t.toUpperCase());
    const book = paperBook(dataDir), results = [];
    let index = 0;
    async function worker() {
      while (index < list.length) {
        const t = list[index++];
        try { results.push(await verdict(t, { refresh, modelId, book })); }
        catch (error) { results.push({ ticker: t, source: "ERROR", flag: null, summary: error.message }); }
        onProgress({ done: results.length, total: list.length, ticker: t });
      }
    }
    await Promise.all(Array.from({ length: Math.min(concurrency, list.length) }, worker));
    const exported = exportAdvisory();
    return { tickers: list, verdicts: results.sort((a, b) => a.ticker.localeCompare(b.ticker)), exported };
  }

  // Advisory decision API: latest verdict per ticker + the exact effect engine/advisory.py would apply.
  async function decision(tickers, { proposals = {}, yellowScale = 0.5, includeMemory = true } = {}) {
    const rows = decisionSync(tickers, { proposals, yellowScale });
    if (!memory || !includeMemory) return rows;
    for (const row of rows) {
      const m = await memory.recall({ query: `${row.ticker} entry stop size decision flag`, tickers: [row.ticker],
        kinds: ["VERDICT", "DECISION", "NOTE", "EVENT"], budgetChars: 4000, keepRecent: 0, prices: row.refPrice ? { [row.ticker]: row.refPrice } : {} });
      row.memory = m.text;
    }
    return rows;
  }
  function decisionSync(tickers, { proposals = {}, yellowScale = 0.5 } = {}) {
    const cutoff = Date.now() - maxAgeDays * 86400000;
    return tickers.map((t) => {
      t = t.toUpperCase();
      const v = store.latestVerdicts([t])[0];
      const usable = v && v.source === "LLM" && Date.parse(v.createdAt) >= cutoff;
      const contract = usable ? { flag: v.flag, entry_low: v.entryLow, entry_high: v.entryHigh } : null;
      const p = proposals[t];
      const effect = p ? adjustEntry(contract, Number(p.amount ?? p.weight ?? p.shares ?? 0), Number(p.refPrice ?? p.price ?? v?.refPrice), yellowScale) : null;
      return { ticker: t, usable, reason: usable ? null : !v ? "no verdict" : v.source !== "LLM" ? "fallback verdict (no change)" : "stale verdict (no change)",
        flag: v?.flag ?? null, action: v?.action ?? null, conviction: v?.conviction ?? null, allowNewEntry: usable ? v.flag !== "RED" : true,
        sizeScale: usable ? (v.flag === "RED" ? 0 : v.flag === "YELLOW" ? yellowScale : 1) : 1,
        entryBand: usable ? [v.entryLow, v.entryHigh] : null, limitPrice: usable ? v.limitPrice : null, orderType: usable ? v.orderType : null,
        stop: v?.stop ?? null, takeProfit: v?.takeProfit ?? null, overextension: v?.overextension ?? null,
        diversification: v?.diversification ?? null, createdAt: v?.createdAt ?? null, model: v?.model ?? null, refPrice: v?.refPrice ?? null, proposalEffect: effect,
        note: "Advisory only: the risk engine (caps, sector limits, stops, PDT, loss breaker) stays final and verdicts can only reduce or defer risk." };
    });
  }

  function fallbackAnswer(list, reason, question = "") {
    const noKey = /NO_KEY|needs an .* API key/i.test(reason);
    const lines = [noKey
      ? `**An AI key is needed for a full answer** (${reason}). Here is what the local research files and data say:`
      : `**Model unavailable** (${reason}). Here is the data summary the model would have used:`];
    for (const r of list) {
      const f = r.features, lv = deterministicLevels(f), q = r.quote?.value, d = r.financials?.value?.derived;
      lines.push(`\n### ${r.ticker}${r.details?.value?.name ? " — " + r.details.value.name : ""}`);
      if (f) lines.push(`- Price ${f.price} (${f.priceSource}, ${f.priceAsOf}); trend **${f.trend}**, stretch **${f.overextension}** (z20 ${f.zPrice20}, RSI14 ${f.rsi14}).`,
        `- Returns 20d ${(f.ret20 * 100).toFixed(1)}%, 100d ${f.ret100 == null ? "n/a" : (f.ret100 * 100).toFixed(1) + "%"}; vs SMA50 ${(f.distSma50 * 100).toFixed(1)}%, vs SMA200 ${f.distSma200 == null ? "n/a" : (f.distSma200 * 100).toFixed(1) + "%"}; ATR14 ${f.atr14}.`);
      if (lv) lines.push(`- Code levels: entry band ${lv.entryLow ?? "none"}–${lv.entryHigh ?? "none"}, stop ${lv.stop}.`);
      if (q) lines.push(`- Schwab: P/E ${q.peRatio}, EPS ${q.eps}, last earnings ${q.lastEarningsDate?.slice(0, 10) || "n/a"}.`);
      if (d) lines.push(`- SEC: TTM revenue growth ${d.revenueGrowthTTM == null ? "n/a" : (d.revenueGrowthTTM * 100).toFixed(1) + "%"}, FCF margin ${d.fcfMarginTTM == null ? "n/a" : (d.fcfMarginTTM * 100).toFixed(1) + "%"}, net debt ${d.netDebt}.`);
      for (const n of (r.news?.value || []).slice(0, 5)) lines.push(`- ${n.published.slice(0, 10)} ${n.publisher}: ${n.title}`);
      if (r.gaps.length) lines.push(`- Data gaps: ${r.gaps.join("; ")}`);
    }
    const studies = researchFallbackText(question, { dataDir, budgetChars: 3500 });
    if (studies) lines.push("\n" + studies);
    else if (!list.length) lines.push("\nNo matching local research summary was found. Add an AI key on the AI analyst page for a full answer, or name a ticker.");
    return lines.join("\n");
  }

  // ---- tool loop: the model picks tools via the Retort enums (analystToolStep), the server runs them, results are fed
  // back as [TOOL_RESULTS]; bounded by steps, calls per step and a wall-clock budget. Any model or tool failure ends
  // the loop and the answer is produced with whatever evidence exists (never an exception to the caller).
  async function toolLoop({ sys, context, question, modelId }) {
    const runs = [], seen = new Set(), steps = [];
    if (!tools || maxToolSteps === 0 || !canCallModel(modelId)) return { runs, steps };
    const deadline = Date.now() + toolBudgetMs;
    for (let step = 1; step <= maxToolSteps && Date.now() < deadline; step++) {
      let result;
      try {
        result = await converse(($) => {
          $.system`${sys}`;
          $.user`${context}`;
          if (runs.length) $.user`${toolResultsBlock(runs)}`;
          $.user`${toolTask({ catalog: tools.catalog(), stepsLeft: maxToolSteps - step + 1, maxCalls: MAX_CALLS_PER_STEP })}\nToday (US Eastern): ${easternDate()}.\nQuestion: ${question}`;
        }, { models, modelId, name: "analystToolStep", description: "Choose server-side tools to run before answering, or ANSWER.",
          parameters: TOOL_STEP, temperature: 0.1, fetchImpl, sleep, log: (e) => log("ai-model", e) });
      } catch (error) {
        steps.push({ step, status: "failed", error: `${error.code || ""} ${error.message}`.trim() });
        store.log({ kind: "tool-step", model: modelId || models.current().id, status: "failed", input: { question, step }, error: `${error.code || ""} ${error.message}` });
        break;
      }
      const value = result.value;
      store.log({ kind: "tool-step", model: result.modelId, status: "ok", input: { question, step }, output: value, attempts: result.attempts, usage: result.usage });
      steps.push({ step, decision: value.decision, reasoning: value.reasoning, calls: value.calls.length });
      if (value.decision !== "CALL_TOOLS" || !value.calls.length) break;
      const fresh = [];
      for (const call of value.calls.slice(0, MAX_CALLS_PER_STEP)) {
        const key = JSON.stringify([call.tool, call.method, call.family, [...call.tickers].sort(), call.start, call.end, call.horizonDays, call.options, call.query]);
        if (!seen.has(key)) { seen.add(key); fresh.push(call); }
      }
      if (!fresh.length) break;                       // only repeats: nothing new to learn
      const perResult = Math.max(1200, Math.floor(16000 / Math.max(1, runs.length + fresh.length)));
      for (const call of fresh) runs.push({ step, purpose: call.purpose, ...(await tools.execute(call, { deadline, promptChars: perResult })) });
    }
    return { runs, steps };
  }
  function toolResultsBlock(runs) {
    return "[TOOL_RESULTS] (server computations you requested; full outputs are logged)\n" + runs.map((r, i) =>
      `#${i + 1} ${r.tool}${r.method && r.method !== "DEFAULT" ? "/" + r.method : ""}${r.family ? " " + r.family : ""} ${JSON.stringify(r.args)}` +
      `${r.purpose ? ` — ${r.purpose}` : ""}\n${r.ok ? "OK" : "FAILED"} (${r.ms} ms${r.cached ? ", cached" : ""}): ${r.summary}`).join("\n\n");
  }

  async function ask(question, tickers, { modelId, refresh = false, useTools = true } = {}) {
    const names = (tickers || []).map((t) => String(t).toUpperCase()).filter(Boolean).slice(0, 4);
    const list = names.length ? await Promise.all(names.map((t) => research(t, { refresh }))) : [];
    let context = buildContext(list, { regime: regime(), book: list[0]?.book || paperBook(dataDir), dataDir, question });
    const prior = names.length ? store.latestVerdicts(names).filter((v) => v.source === "LLM")
      .map((v) => `[PRIOR_VERDICT] ${v.ticker} ${v.createdAt}: ${v.flag}/${v.conviction}, action ${v.action}, band ${v.entryLow}-${v.entryHigh}, stop ${v.stop}. ${v.summary}`) : [];
    if (prior.length) context += "\n\n" + prior.join("\n");
    const memoryText = await recallFor(question, list);
    if (memoryText) context += "\n\n" + memoryText;
    const sys = await system();
    const { runs, steps } = useTools ? await toolLoop({ sys, context, question, modelId }) : { runs: [], steps: [] };
    const toolSummary = runs.map((r) => ({ tool: r.tool, method: r.method, family: r.family, args: r.args, ok: r.ok, ms: r.ms, cached: r.cached,
      purpose: r.purpose, error: r.ok ? undefined : r.error, summary: r.summary }));
    const dataAsOf = list.map((r) => ({ ticker: r.ticker, lastBar: r.features?.lastBarDate, price: r.features?.priceAsOf, news: r.news?.fetchedAt, gaps: r.gaps }));
    try {
      const result = await converse(($) => {
        $.system`${sys}`;
        $.user`${context}`;
        if (runs.length) $.user`${toolResultsBlock(runs)}`;
        $.user`${ANSWER_TASK}\nToday (US Eastern): ${easternDate()}.\nQuestion: ${question}`;
      }, { models, modelId, name: "analystAnswer", description: "Answer the question using only the supplied data, research summaries and tool results.",
        parameters: ANSWER, temperature: 0.3, fetchImpl, sleep, log: (e) => log("ai-model", e) });
      store.log({ kind: "ask", model: result.modelId, status: "ok", input: { question, tickers: names, contextChars: context.length, tools: toolSummary.length },
        output: result.value, attempts: result.attempts, usage: result.usage });
      remember({ kind: "QA", ticker: names[0] || null, text: `Q: ${question}\nA (${result.value.stance}/${result.value.conviction}): ${result.value.answer}`,
        meta: { refPrice: list[0]?.features?.price ?? null, tickers: names, stance: result.value.stance, model: result.modelId } });
      return { source: "LLM", model: result.modelId, tickers: names, ...result.value, contextChars: context.length, memoryChars: memoryText.length,
        systemChars: sys.length, tools: toolSummary, toolSteps: steps, dataAsOf };
    } catch (error) {
      store.log({ kind: "ask", model: modelId || models.current().id, status: "fallback", input: { question, tickers: names }, error: `${error.code || ""} ${error.message}`, attempts: error.attempts });
      const toolText = runs.length ? "\n\n### Tool results\n" + runs.map((r) => `- ${r.tool}${r.method !== "DEFAULT" ? "/" + r.method : ""}: ${r.ok ? r.summary.slice(0, 600) : "failed — " + r.error?.message}`).join("\n") : "";
      return { source: "FALLBACK", model: null, tickers: names, answer: fallbackAnswer(list, `${error.code || "ERROR"}: ${error.message}`, question) + toolText,
        stance: "YELLOW", conviction: "LOW", keyPoints: [], levels: [], citations: [], followUps: [], error: error.message,
        tools: toolSummary, toolSteps: steps, dataAsOf };
    }
  }

  /** Run one tool directly (slash command / HTTP / CLI); no model involved. */
  async function runTool(call, options = {}) {
    if (!tools) return { ok: false, error: { code: "NO_TOOLS", message: "tools are not configured" } };
    return tools.execute({ tool: "STATS", method: "DEFAULT", family: "NONE", tickers: [], start: "", end: "", horizonDays: null, options: [], query: "", purpose: "direct", ...call }, options);
  }

  return { research, verdict, flagTargets, exportAdvisory, decision, decisionSync, ask, runTool, system, targets, regime, advisoryPath, memory };
}
