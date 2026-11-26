// Express routes for the AI analyst. Mounted by server.js with one call: mountAi(app, {...}).
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { createModelService } from "./models.js";
import { openStore } from "./store.js";
import { createSources } from "./sources.js";
import { openPrices } from "./features.js";
import { createAdvisor } from "./advisor.js";
import { classify, HELP, isHelpRequest } from "./command.js";
import { getMemory, KINDS, BUDGET_CHARS } from "./memory/memory.js";
import { createQuantRunner } from "./quant.js";
import { createStrategyCard, renderMaster } from "./strategyCard.js";
import { createTools, parseToolCommand, TOOLS } from "./tools.js";
import { startCryptoRefresher } from "./crypto.js";
import { mountCryptoChart } from "../crypto-chart.js";
import { complete } from "./transport.js";
import { createActions, ADJUDICATION } from "./actions.js";

const TICKER = /^[A-Z]{1,5}(\.[A-Z])?$/;
const cleanTickers = (list) => [...new Set((Array.isArray(list) ? list : String(list || "").split(/[\s,]+/))
  .map((t) => String(t).toUpperCase().trim()).filter((t) => TICKER.test(t)))].slice(0, 40);

export function createAi({ root, dataDir, python, log = () => {}, env = process.env, fetchImpl = globalThis.fetch }) {
  const models = createModelService({ root, env, settingsPath: join(dataDir, "ai_settings.json") });
  const store = openStore(dataDir);
  let prices = null;
  try { prices = openPrices(dataDir); } catch (error) { log("warn", "AI analyst: price database unavailable", { error: error.message }); }
  const sources = createSources({ store, env, fetchImpl, root, python });
  let memory = null;
  try {
    memory = getMemory({ dataDir, modelDir: env.TRADING_AI_MEMORY_MODEL_DIR || join(root, "models", "memory"),
      logger: (level, message, meta) => log(level, message, meta) });
  } catch (error) { log("warn", "AI memory unavailable", { error: error.message }); }
  // Quant tools (engine/quant_tools.py via a JSON runner), the code-generated strategy card and the tool executor.
  const quant = createQuantRunner({ root, python });
  const card = createStrategyCard({ root, dataDir, quant, log });
  let advisor = null;
  const tools = createTools({ root, dataDir, quant, store, memory, research: (t) => advisor.research(t), python, log });
  advisor = createAdvisor({ dataDir, models, store, sources, prices, memory, tools, card, env, fetchImpl,
    log: (kind, entry) => log(entry.status === "failed" ? "warn" : "info", "AI model call", { kind, ...entry }) });
  // Adjudicated actions (actions.js): the server attaches its loopback URL, the alert emitter and the
  // experiments module after it starts listening. The adjudicator model call is strict JSON (ADJUDICATION).
  const actions = createActions({ tools, memory, store, dataDir, log,
    llm: async ({ system, user }) => {
      if (!models.current().ready) return null;
      const r = await complete({ models, name: "adjudicateAction", description: "Pick the one dashboard action for a user request.",
        parameters: ADJUDICATION, temperature: 0, timeoutMs: 20000, messages: [{ role: "system", content: system }, { role: "user", content: user }] });
      return r.value;
    } });
  return { models, store, sources, prices, advisor, memory, quant, card, tools, actions };
}

/** Dispatch one command-prompt message. Help is only for empty input, /help, or a bare "help"/"?". */
export function createCommandHandler(ai, { startFlagJob } = {}) {
  const { models, advisor, prices, sources } = ai;
  return async function dispatch(text, modelId) {
    const raw = String(text ?? "");
    if (isHelpRequest(raw)) return { kind: "help", intent: { intent: "HELP", tickers: [], question: raw, source: "help" }, text: HELP };
    // Actions first: the adjudicator returns null for plain questions (ASK_ANALYST), which fall through to Q&A.
    // Existing slash commands (/ask, /flag, /tool, /note, ...) keep their own handling.
    if (ai.actions && (!raw.trim().startsWith("/") || /^\/do\b/i.test(raw.trim()))) {
      const acted = await ai.actions.handle(raw);
      if (acted) return acted;
    }
    const c = await classify(raw, { models, prices, sources });
    if (c.intent === "FLAG_TICKERS") return { kind: "job", intent: c, job: startFlagJob ? startFlagJob(c.tickers, modelId) : { error: "flag jobs are only available on the dashboard" } };
    if (c.intent === "DECISION" && c.tickers.length && /^\/decid/i.test(raw)) return { kind: "decision", intent: c, decisions: await advisor.decision(c.tickers) };
    if (c.intent === "TOOL") {
      let call;
      try { call = parseToolCommand(c.question); } catch (error) { return { kind: "help", intent: c, text: error.message }; }
      return { kind: "tool", intent: c, call, run: await advisor.runTool(call) };
    }
    if (c.intent === "NOTE") {
      const [ticker, ...rest] = c.question.replace(/^\/note\s*/i, "").split(/\s+/);
      const known = /^[A-Z]{1,5}(\.[A-Z])?$/.test(ticker.toUpperCase()) && prices?.has(ticker.toUpperCase());
      const r = ai.memory.ingest({ kind: "NOTE", ticker: known ? ticker.toUpperCase() : null, text: (known ? rest : [ticker, ...rest]).join(" ") });
      return { kind: "help", intent: c, text: r.duplicate ? "That note is already saved." : "Note saved to memory." };
    }
    if (c.intent === "MEMORY") {
      const q = c.question.replace(/^\/memory\s*/i, "");
      const m = await ai.memory.recall({ query: q, tickers: c.tickers.filter((t) => prices?.has(t)), budgetChars: BUDGET_CHARS });
      return { kind: "help", intent: c, text: m.text || "Nothing relevant in memory yet." };
    }
    if (c.intent === "DATA") return { kind: "data", intent: c, research: await Promise.all(c.tickers.slice(0, 3).map((t) => advisor.research(t))) };
    if (c.intent === "VERDICTS") return { kind: "verdicts", intent: c, verdicts: ai.store.latestVerdicts() };
    if (c.intent === "MARKET" && !c.tickers.length) return { kind: "answer", intent: c, ...(await advisor.ask(c.question, ["SPY", "QQQ", "IWM"], { modelId })) };
    if (c.intent === "PORTFOLIO" && !c.tickers.length) {
      const held = advisor.targets();
      return { kind: "answer", intent: c, ...(await advisor.ask(c.question, held.slice(0, 4), { modelId })) };
    }
    // General questions with no ticker go to the analyst (research context + tools), not the help text.
    return { kind: "answer", intent: c, ...(await advisor.ask(c.question || raw.trim(), c.tickers, { modelId })) };
  };
}

export function mountAi(app, options) {
  const ai = createAi(options);
  // Crypto snapshot every 5 min while the dashboard runs (engine/crypto_snapshot.py; market data only).
  // Chart picker lookup (/api/symbols/lookup, local only) and crypto charts (/api/crypto/chart/:coin).
  ai.cryptoChart = mountCryptoChart(app, { root: options.root, dataDir: options.dataDir, python: options.python || ai.quant?.python, log: options.log });
  ai.stopCryptoRefresher = startCryptoRefresher({ root: options.root, python: options.python || ai.quant?.python, env: options.env || process.env, log: options.log });
  const { models, advisor, prices, sources } = ai;
  const jobs = new Map();
  const fail = (response, error) => response.status(error.status || 500).json({ error: error.status ? error.message : "The AI analyst could not complete the request: " + error.message });

  // ----- models and provider keys
  app.get("/api/ai/models", (_q, r) => r.json(models.view()));
  app.post("/api/ai/models", (q, r) => { try { r.json({ current: models.select(String(q.body?.id || "")) }); } catch (e) { fail(r, e); } });
  app.post("/api/ai/provider-key", (q, r) => {
    try {
      const keys = models.saveKey(String(q.body?.provider || ""), q.body?.key, { persistOnly: q.body?.persistOnly === true });
      options.log?.("info", "AI provider key saved", { provider: q.body?.provider }); // provider name only, never the key
      r.json({ keys, storedKeys: models.storedKeys(), current: models.current() });
    } catch (e) { r.status(e.status || 500).json({ error: e.status ? e.message : "Could not save the key." }); }
  });

  app.get("/api/ai/status", (_q, r) => r.json({ model: models.current(), keys: models.keys(), massive: sources.hasMassive(),
    pricesDb: Boolean(prices), targets: advisor.targets(), advisoryPath: advisor.advisoryPath(), regime: advisor.regime() }));

  // ----- command prompt / Q&A
  const dispatch = createCommandHandler(ai, { startFlagJob });
  app.post("/api/ai/command", async (q, r) => {
    try { r.json(await dispatch(String(q.body?.text ?? q.body?.question ?? ""), q.body?.model || undefined)); } catch (e) { fail(r, e); }
  });
  // ----- adjudicated actions: catalog, direct run, Allow / Cancel
  app.get("/api/ai/actions", (_q, r) => r.json({ ...ai.actions.catalog(), jobs: ai.actions.jobs() }));
  app.get("/api/ai/actions/jobs", (_q, r) => r.json({ jobs: ai.actions.jobs() }));
  app.post("/api/ai/actions/run", async (q, r) => {
    try { r.json(await ai.actions.run(String(q.body?.action || ""), q.body?.args || {})); } catch (e) { fail(r, e); }
  });
  app.post("/api/ai/actions/confirm", async (q, r) => {
    try { r.json(await ai.actions.confirm(String(q.body?.token || ""))); } catch (e) { fail(r, e); }
  });
  app.post("/api/ai/actions/cancel", (q, r) => r.json(ai.actions.cancel(String(q.body?.token || ""))));
  app.post("/api/ai/ask", async (q, r) => {
    try {
      const tickers = cleanTickers(q.body?.tickers);
      if (!tickers.length) return r.json(await dispatch(String(q.body?.question || ""), q.body?.model || undefined));
      r.json({ kind: "answer", ...(await advisor.ask(String(q.body?.question || ""), tickers, { modelId: q.body?.model || undefined })) });
    } catch (e) { fail(r, e); }
  });
  app.get("/api/ai/research/:ticker", async (q, r) => {
    const [t] = cleanTickers([q.params.ticker]);
    if (!t) return r.status(400).json({ error: "Unknown ticker." });
    try { r.json(await advisor.research(t, { refresh: q.query.refresh === "1" })); } catch (e) { fail(r, e); }
  });

  // ----- flagging job, verdicts, paper-trader decision API
  function startFlagJob(tickers, modelId) {
    const id = randomUUID(), job = { id, status: "running", startedAt: new Date().toISOString(), progress: null, result: null, error: null };
    jobs.set(id, job);
    advisor.flagTargets(tickers, { modelId, onProgress: (p) => { job.progress = p; } })
      .then((result) => { job.status = "complete"; job.result = result; })
      .catch((error) => { job.status = "failed"; job.error = error.message; })
      .finally(() => { job.finishedAt = new Date().toISOString(); });
    return { id, status: job.status };
  }
  app.post("/api/ai/flag", (q, r) => r.json(startFlagJob(cleanTickers(q.body?.tickers), q.body?.model || undefined)));
  app.get("/api/ai/flag/:id", (q, r) => { const j = jobs.get(q.params.id); j ? r.json(j) : r.status(404).json({ error: "Unknown job." }); });
  app.get("/api/ai/verdicts", (q, r) => r.json({ verdicts: ai.store.latestVerdicts(q.query.tickers ? cleanTickers(q.query.tickers) : null) }));
  app.get("/api/ai/decision", async (q, r) => {
    try { r.json({ decisions: await advisor.decision(cleanTickers(q.query.tickers || advisor.targets()), { includeMemory: q.query.memory !== "0" }) }); } catch (e) { fail(r, e); }
  });
  app.post("/api/ai/decision", async (q, r) => {
    try {
      const proposals = Object.fromEntries(Object.entries(q.body?.proposals || {}).map(([k, v]) => [k.toUpperCase(), v]));
      const tickers = cleanTickers(q.body?.tickers?.length ? q.body.tickers : Object.keys(proposals));
      r.json({ decisions: await advisor.decision(tickers, { proposals, yellowScale: Number(q.body?.yellowScale ?? 0.5), includeMemory: q.body?.memory !== false }) });
    } catch (e) { fail(r, e); }
  });

  // ----- tools and the master prompt (inspection / direct runs; the model uses them through advisor.ask's tool loop)
  app.get("/api/ai/tools", (_q, r) => r.json({ tools: Object.fromEntries(Object.entries(TOOLS).map(([k, v]) => [k, { help: v.help, methods: v.methods ? Object.keys(v.methods) : [], options: v.options || null, timeoutMs: v.timeoutMs }])), python: ai.quant.python }));
  app.post("/api/ai/tools/run", async (q, r) => {
    try {
      const b = q.body || {};
      const options = Array.isArray(b.options) ? b.options : Object.entries(b.options || {}).map(([name, value]) => ({ name, value: typeof value === "string" ? value : JSON.stringify(value) }));
      r.json(await advisor.runTool({ tool: String(b.tool || "").toUpperCase(), method: String(b.method || "DEFAULT").toUpperCase(),
        family: String(b.family || "NONE").toUpperCase(), tickers: cleanTickers(b.tickers || []), start: b.start || "", end: b.end || "",
        horizonDays: b.horizonDays ?? null, options, query: String(b.query || ""), purpose: "api" }, { refresh: b.refresh === true }));
    } catch (e) { fail(r, e); }
  });
  app.get("/api/ai/prompt", async (q, r) => {
    try { if (q.query.refresh === "1") await ai.card.get({ refresh: true }); r.type("text/plain").send(await advisor.system()); } catch (e) { fail(r, e); }
  });

  app.post("/api/ai/strategy-card/refresh", async (_q, r) => {
    try { const card = await ai.card.get({ refresh: true }); r.json({ ok: Boolean(card), generatedAt: card?.generatedAt || null, promptChars: renderMaster(card).length }); } catch (e) { fail(r, e); }
  });

  // ----- analyst memory (MiniLM vector store). ingest is the hook for the paper-trader decision / chat-log writer.
  const needMemory = (r) => { if (!ai.memory) { r.status(503).json({ error: "AI memory is unavailable." }); return false; } return true; };
  app.get("/api/ai/memory/status", (_q, r) => { if (needMemory(r)) r.json({ ...ai.memory.stats(), kinds: KINDS }); });
  app.post("/api/ai/memory/ingest", (q, r) => {
    if (!needMemory(r)) return;
    try {
      const items = Array.isArray(q.body?.items) ? q.body.items.slice(0, 200) : [q.body];
      r.json({ results: items.map((item) => ai.memory.ingest(item)) });
    } catch (e) { r.status(e.status || 500).json({ error: e.message }); }
  });
  app.get("/api/ai/memory/search", async (q, r) => {
    if (!needMemory(r)) return;
    try {
      const m = await ai.memory.recall({ query: String(q.query.q || ""), tickers: cleanTickers(q.query.tickers || []),
        kinds: q.query.kinds ? String(q.query.kinds).toUpperCase().split(",").filter((k) => KINDS.includes(k)) : null,
        since: q.query.since || null, budgetChars: Math.min(20000, Number(q.query.budget) || BUDGET_CHARS) });
      r.json(m);
    } catch (e) { fail(r, e); }
  });
  app.get("/api/ai/memory/items", (q, r) => { if (needMemory(r)) r.json({ items: ai.memory.list({ ticker: q.query.ticker, kind: q.query.kind, limit: Number(q.query.limit) || 50 }) }); });
  app.post("/api/ai/memory/reindex", async (_q, r) => { if (!needMemory(r)) return; try { r.json(await ai.memory.embedPending(5000)); } catch (e) { fail(r, e); } });
  return ai;
}
