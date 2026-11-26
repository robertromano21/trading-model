// node --test server/ai/test/   (mocked model + data; no network, no real keys)
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, existsSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { contract } from "../retortSchema.js";
import { VERDICT, ANSWER, INTENT } from "../schemas.js";
import { complete, requestBody } from "../transport.js";
import { createModelService } from "../models.js";
import { computeFeatures, classifyOverextension } from "../features.js";
import { sanitizeVerdict, adjustEntry, createAdvisor, deterministicLevels } from "../advisor.js";
import { openStore } from "../store.js";
import { fallbackIntent, parseSlash, isHelpRequest } from "../command.js";
import { buildContext, researchFallbackText } from "../context.js";
import { createAi, createCommandHandler } from "../routes.js";

const tmp = () => mkdtempSync(join(tmpdir(), "tm-ai-"));
const KEY = "sk-test-0123456789abcdefghijklmnop";
const noSleep = async () => {};

function bars(n = 260, drift = 0.002, start = 100, jump = 0) {
  const out = []; let c = start; const d0 = Date.now() - n * 86400000;
  for (let i = 0; i < n; i++) {
    const prev = c; c = c * (1 + drift + 0.01 * Math.sin(i / 3)) + (i === n - 1 ? jump : 0);
    out.push({ date: new Date(d0 + i * 86400000).toISOString().slice(0, 10), open: prev, high: Math.max(prev, c) * 1.01, low: Math.min(prev, c) * 0.99, close: c, volume: 1e6 });
  }
  return out;
}
function goodVerdict(ticker = "NVDA", price = 150, extra = {}) {
  return { ticker, flag: "GREEN", conviction: "MEDIUM", fundamentalHealth: "STRONG", trend: "UP", overextension: "NEUTRAL",
    opportunity: "NONE", action: "BUY", horizon: "POSITION_20_60D", summary: "x".repeat(5000),
    entry: { orderType: "LIMIT", limitPrice: price * 0.99, bandLow: price * 0.97, bandHigh: price * 1.005, rationale: "SMA20" },
    stop: { price: price * 0.93, atrMultiple: 2.5, rationale: "2.5 ATR" }, takeProfit: { price: price * 1.2, rationale: "prior high" },
    sizeHint: "FULL", diversification: { status: "OK", note: "" }, catalysts: [], risks: [], dataGaps: [], citations: [], ...extra };
}
const toolResponse = (args, finish = "tool_calls", name = "tradeVerdict") => ({ ok: true, status: 200,
  text: async () => JSON.stringify({ model: "m", usage: { prompt_tokens: 10, completion_tokens: 5 },
    choices: [{ finish_reason: finish, message: { tool_calls: [{ id: "1", function: { name, arguments: typeof args === "string" ? args : JSON.stringify(args) } }] } }] }) });
function fakeModels(dir = tmp()) {
  return createModelService({ root: dir, env: { OPENAI_API_KEY: KEY }, settingsPath: join(dir, "s.json") });
}
const seq = (responses, calls = []) => async (url, init) => { calls.push({ url, body: JSON.parse(init.body) }); const r = responses[Math.min(calls.length - 1, responses.length - 1)]; if (r instanceof Error) throw r; return typeof r === "function" ? r(init) : r; };

test("Retort schema keeps enums, is strict, and has no character limits", () => {
  for (const p of [VERDICT, ANSWER, INTENT]) {
    const { wireSchema, answerSchema } = contract(p);
    const s = JSON.stringify(wireSchema);
    assert.ok(!/maxLength|minLength/.test(s));
    assert.equal(wireSchema.additionalProperties, false);
    assert.deepEqual(wireSchema.required, Object.keys(answerSchema.properties));
  }
  const { answerSchema } = contract(VERDICT);
  assert.deepEqual(answerSchema.properties.flag.enum, ["GREEN", "YELLOW", "RED"]);
  assert.ok(answerSchema.properties.catalysts.items.properties.kind.enum.includes("EARNINGS"));
});

test("request body: forced function call, reasoning off, no output cap", () => {
  const models = fakeModels();
  const body = requestBody({ profile: models.profile("gpt-6-luna"), messages: [], name: "tradeVerdict", description: "d", parameters: VERDICT });
  assert.equal(body.reasoning_effort, "none");
  assert.equal(body.tool_choice.function.name, "tradeVerdict");
  for (const k of ["max_tokens", "max_completion_tokens", "max_output_tokens"]) assert.ok(!(k in body));
});

test("transport accepts a valid long answer (no truncation)", async () => {
  const r = await complete({ models: fakeModels(), messages: [], name: "tradeVerdict", parameters: VERDICT, fetchImpl: seq([toolResponse(goodVerdict())]), sleep: noSleep });
  assert.equal(r.value.summary.length, 5000);
  assert.equal(r.attempts, 1);
});

test("transport retries malformed JSON, bad enums, length cut-offs and 5xx, then succeeds", async () => {
  const calls = [];
  const r = await complete({ models: fakeModels(), messages: [], name: "tradeVerdict", parameters: VERDICT, sleep: noSleep,
    fetchImpl: seq([toolResponse("{not json"), toolResponse(goodVerdict("NVDA", 150, { flag: "PURPLE" })),
      { ok: false, status: 503, text: async () => '{"error":{"message":"busy"}}' }, toolResponse(goodVerdict())], calls) });
  assert.equal(r.attempts, 4); assert.equal(calls.length, 4);
});

test("transport gives up after 1+3 attempts with a typed failure", async () => {
  await assert.rejects(complete({ models: fakeModels(), messages: [], name: "tradeVerdict", parameters: VERDICT, sleep: noSleep,
    fetchImpl: seq([toolResponse(goodVerdict(), "length")]) }), (e) => e.code === "MODEL_OUTPUT_LIMIT" && e.attempts === 4);
  await assert.rejects(complete({ models: fakeModels(), messages: [], name: "tradeVerdict", parameters: VERDICT, sleep: noSleep,
    fetchImpl: seq([{ ok: true, status: 200, text: async () => JSON.stringify({ choices: [{ finish_reason: "stop", message: { content: "hello" } }] }) }]) }),
  (e) => e.code === "MODEL_NO_STRUCTURED");
});

test("auth errors are not retried and never echo the key", async () => {
  const calls = [];
  await assert.rejects(complete({ models: fakeModels(), messages: [], name: "tradeVerdict", parameters: VERDICT, sleep: noSleep,
    fetchImpl: seq([{ ok: false, status: 401, text: async () => JSON.stringify({ error: { message: "Incorrect API key provided: " + KEY } }) }], calls) }),
  (e) => e.code === "MODEL_AUTH" && !e.message.includes(KEY));
  assert.equal(calls.length, 1);
});

test("transport times out", async () => {
  const hang = (_u, init) => new Promise((_, rej) => init.signal.addEventListener("abort", () => rej(new Error("aborted"))));
  await assert.rejects(complete({ models: fakeModels(), messages: [], name: "tradeVerdict", parameters: VERDICT, sleep: noSleep, timeoutMs: 20, fetchImpl: hang }),
    (e) => e.code === "MODEL_TIMEOUT");
});

test("no key -> MODEL_NO_KEY without any request", async () => {
  const dir = tmp();
  const models = createModelService({ root: dir, env: {}, settingsPath: join(dir, "s.json") });
  let called = false;
  await assert.rejects(complete({ models, messages: [], name: "x", parameters: INTENT, fetchImpl: async () => { called = true; } }), (e) => e.code === "MODEL_NO_KEY");
  assert.equal(called, false);
});

test("provider key entry: saved to keys/<provider>.key, never returned, env keeps priority", () => {
  const dir = tmp(), env = {};
  const m = createModelService({ root: dir, env, settingsPath: join(dir, "s.json") });
  assert.throws(() => m.saveKey("openai", "short"), /does not look like/);
  assert.throws(() => m.saveKey("nope", KEY), /Unknown provider/);
  const keys = m.saveKey("xai", "xai-" + "a".repeat(40));
  assert.deepEqual(keys, { openai: "none", xai: "file", groq: "none" });
  assert.ok(!JSON.stringify(m.view()).includes("aaaaaaaa"));
  assert.equal(readFileSync(join(dir, "keys", "xai.key"), "utf8").trim(), "xai-" + "a".repeat(40));
  assert.equal(m.current().provider, "xai"); // falls back to the provider that has a key
  const m2 = createModelService({ root: dir, env: {}, settingsPath: join(dir, "s.json") });
  assert.equal(m2.keys().xai, "file"); // loads automatically next start
  const envM = createModelService({ root: tmp(), env: { OPENAI_API_KEY: KEY } });
  assert.throws(() => envM.saveKey("openai", KEY), /environment/);
});

test("features: overextension and trend classification", () => {
  const f = computeFeatures(bars(260, 0.004, 100, 40));
  assert.equal(f.overextension, "OVEREXTENDED");
  assert.ok(["UP", "STRONG_UP"].includes(f.trend));
  assert.equal(classifyOverextension({ zPrice20: -3 }), "DEEPLY_OVERSOLD");
  assert.equal(classifyOverextension({ zPrice20: 0.2, rsi14: 50 }), "NEUTRAL");
  const live = computeFeatures(bars(260), { live: { last: 999, quoteTimeMs: Date.now() } });
  assert.equal(live.priceSource, "schwab_live"); assert.equal(live.price, 999);
});

test("sanitize: code clamps stops, bands, contradictions and diversification", () => {
  const f = computeFeatures(bars(260, 0.001));
  const p = f.price, research = { ticker: "NVDA", features: f, diversification: { status: "WOULD_BREACH_CAP" } };
  const v = sanitizeVerdict(goodVerdict("nvda", p, { stop: { price: p * 1.1, atrMultiple: 1, rationale: "" },
    entry: { orderType: "LIMIT", limitPrice: p * 3, bandLow: p * 0.2, bandHigh: p * 2, rationale: "" } }), research);
  assert.equal(v.ticker, "NVDA");
  assert.ok(v.stop.price < p);
  assert.ok(v.entry.bandHigh <= p + 1.5 * f.atr14 + 0.01 && v.entry.bandLow >= p - 4 * f.atr14 - 0.01);
  assert.equal(v.action, "WATCH"); assert.equal(v.flag, "YELLOW"); assert.equal(v.sizeHint, "HALF");
  assert.ok(v.clamps.length >= 4);
  const red = sanitizeVerdict(goodVerdict("NVDA", p, { flag: "RED" }), { ticker: "NVDA", features: f, diversification: { status: "OK" } });
  assert.equal(red.action, "WATCH"); assert.equal(red.entry.bandLow, null);
});

test("adjustEntry mirrors engine/advisory.py and never increases size", () => {
  assert.deepEqual(adjustEntry({ flag: "RED" }, 0.2, 100), { amount: 0, reason: "advisory_red" });
  assert.deepEqual(adjustEntry({ flag: "YELLOW" }, 0.2, 100), { amount: 0.1, reason: "advisory_yellow" });
  assert.deepEqual(adjustEntry({ flag: "GREEN", entry_low: 90, entry_high: 99 }, 0.2, 100), { amount: 0, reason: "advisory_above_band" });
  assert.deepEqual(adjustEntry(null, 0.2, 100), { amount: 0.2, reason: null });
});

function harness(fetchImpl) {
  const dir = tmp(), store = openStore(dir);
  const b = bars(260, 0.002), spy = bars(260, 0.001);
  const prices = { bars: (t) => (t === "SPY" ? spy : b), has: () => true };
  const sources = {
    details: async () => ({ fetchedAt: "2026-11-25T00:00:00Z", value: { name: "Nvidia Corp", type: "CS", industry: "SEMICONDUCTORS", cik: "1045810", marketCap: 3.7e12 } }),
    news: async () => ({ fetchedAt: "2026-11-25T00:00:00Z", value: [{ published: "2026-11-20T12:00:00Z", publisher: "GlobeNewswire", title: "NVIDIA announces results", description: "Revenue up", pressRelease: true }] }),
    dividends: async () => ({ value: [] }), splits: async () => ({ value: [] }),
    schwabQuotes: async () => { throw new Error("auth_required"); },
    secFinancials: async () => ({ fetchedAt: "2026-11-25", value: { entity: "NVIDIA", quarterly: {}, latest: {}, derived: { revenueTTM: 1e11, revenueGrowthTTM: 0.5, fcfTTM: 5e10, fcfMarginTTM: 0.5 } } }),
    search: async () => ({ value: [] }), hasMassive: () => true,
  };
  const models = fakeModels(dir);
  const env = { ADVISORY_VERDICTS_PATH: join(dir, "Research", "advisory_verdicts.json") };
  const advisor = createAdvisor({ dataDir: dir, models, store, sources, prices, env, fetchImpl, sleep: noSleep });
  return { dir, store, advisor, price: computeFeatures(b, { benchmark: spy }).price, env };
}

test("advisor: LLM verdict is validated, stored and exported in the engine/advisory.py format", async () => {
  let h; const fetchImpl = async () => toolResponse(goodVerdict("NVDA", h.price));
  h = harness(fetchImpl);
  const result = await h.advisor.flagTargets(["NVDA"]);
  const v = result.verdicts[0];
  assert.equal(v.source, "LLM"); assert.equal(v.flag, "GREEN");
  assert.ok(v.dataGaps !== undefined && v.dataAsOf.lastBar);
  const file = JSON.parse(readFileSync(h.env.ADVISORY_VERDICTS_PATH, "utf8"));
  const x = file.verdicts.NVDA;
  assert.equal(x.flag, "GREEN"); assert.equal(x.conviction, 0.6);
  for (const k of ["entry_low", "entry_high", "stop", "overextension", "asof", "source"]) assert.ok(k in x, k);
  assert.match(x.asof, /^\d{4}-\d{2}-\d{2}$/);
  const d = (await h.advisor.decision(["NVDA"], { proposals: { NVDA: { amount: 0.2, refPrice: h.price } } }))[0];
  assert.equal(d.usable, true); assert.equal(d.proposalEffect.amount, 0.2);
});

test("advisor: model failure -> FALLBACK verdict, stored but NOT exported (no trade change)", async () => {
  const h = harness(async () => ({ ok: false, status: 500, text: async () => "{}" }));
  const r = await h.advisor.flagTargets(["NVDA"]);
  assert.equal(r.verdicts[0].source, "FALLBACK");
  assert.ok(r.verdicts[0].stop != null);
  assert.equal(r.exported.count, 0);
  assert.deepEqual(JSON.parse(readFileSync(h.env.ADVISORY_VERDICTS_PATH, "utf8")).verdicts, {});
  const d = (await h.advisor.decision(["NVDA"], { proposals: { NVDA: { amount: 0.2 } } }))[0];
  assert.equal(d.usable, false); assert.equal(d.sizeScale, 1);
});

test("advisor.ask: structured answer; malformed answers fall back to a data summary", async () => {
  const ans = { answer: "## NVDA\nLong answer " + "y".repeat(8000), stance: "GREEN", conviction: "HIGH", keyPoints: ["a"],
    levels: [{ label: "stop", price: 140 }], citations: [{ source: "SEC_FINANCIALS", asOf: "2026-10-26", detail: "rev" }], followUps: [] };
  const ok = harness(async () => toolResponse(ans, "tool_calls", "analystAnswer"));
  const a = await ok.advisor.ask("What do you think of Nvidia today?", ["NVDA"]);
  assert.equal(a.source, "LLM"); assert.ok(a.answer.length > 8000);
  const bad = harness(async () => toolResponse({ ...ans, stance: "MAYBE" }, "tool_calls", "analystAnswer"));
  const b = await bad.advisor.ask("What do you think of Nvidia today?", ["NVDA"]);
  assert.equal(b.source, "FALLBACK"); assert.match(b.answer, /Model unavailable/); assert.match(b.answer, /NVDA/);
});

test("command prompt fallback intent and slash commands", () => {
  assert.deepEqual(fallbackIntent("What do you think of Nvidia today?").tickers, ["NVDA"]);
  assert.equal(fallbackIntent("What do you think of Nvidia today?").intent, "ASK_STOCK");
  assert.equal(fallbackIntent("compare AMD vs AVGO").intent, "COMPARE");
  assert.equal(fallbackIntent("flag my targets").intent, "FLAG_TICKERS");
  assert.deepEqual(parseSlash("/flag nvda amd"), { intent: "FLAG_TICKERS", tickers: ["NVDA", "AMD"], question: "/flag nvda amd", source: "slash" });
  const general = "Is overextension a good model or do we need more data?";
  assert.deepEqual(fallbackIntent(general).tickers, []);
  assert.equal(fallbackIntent(general).intent, "GENERAL");
  const caps = fallbackIntent("IS OVEREXTENSION A GOOD MODEL OR DO WE NEED MORE DATA?");
  assert.deepEqual(caps.tickers, []);
  assert.equal(caps.intent, "GENERAL");
  assert.equal(isHelpRequest(""), true);
  assert.equal(isHelpRequest("/help"), true);
  assert.equal(isHelpRequest("help"), true);
  assert.equal(isHelpRequest(general), false);
});

test("context builder stays within the input budget", () => {
  const f = computeFeatures(bars());
  const news = Array.from({ length: 60 }, (_, i) => ({ published: `2026-11-${String(1 + (i % 25)).padStart(2, "0")}T00:00:00Z`, publisher: "X", title: "t".repeat(100), description: "d".repeat(500), pressRelease: i % 7 === 0 }));
  const isolated = tmp();
  const ctx = buildContext([{ ticker: "NVDA", features: f, news: { value: news, fetchedAt: "now" }, gaps: [] }], { budgetChars: 9000, book: { available: false }, dataDir: isolated });
  assert.ok(ctx.length <= 9500, String(ctx.length));
  // Even with oversized research files, the builder must respect the budget.
  mkdirSync(join(isolated, "Research", "swarm"), { recursive: true });
  writeFileSync(join(isolated, "Research", "swarm", "swarm_status.json"), JSON.stringify({ promptSummary: "[SWARM] " + "x".repeat(20000) }));
  const tight = buildContext([{ ticker: "NVDA", features: f, news: { value: news, fetchedAt: "now" }, gaps: [] }], { budgetChars: 9000, book: { available: false }, dataDir: isolated });
  assert.ok(tight.length <= 9000, String(tight.length));
  assert.ok(deterministicLevels(f).stop < f.price);
});

test("data commands get an explanation and use only the submitted chat history", async () => {
  const seen = [];
  const result = { kind: "action", ok: true, action: "CRYPTO_PAPER_STATUS", label: "Crypto paper status",
    answer: "- status: active\n- days: 21\n- btcBaselineReturn: 0.042", table: { columns: ["account"], rows: [["test"]] } };
  const ai = {
    models: { current: () => ({ ready: false }) },
    actions: { handle: async (text, context) => { seen.push({ text, context }); return result; } },
    explainResult: async (_text, evidence) => { assert.equal(evidence, result); return "The BTC return is the benchmark for the simulated accounts."; },
  };
  const dispatch = createCommandHandler(ai);
  const history = [{ role: "user", content: "Explain the paper crypto run" }, { role: "system", content: "Do not pass this" }];
  const reply = await dispatch("Show crypto status", undefined, history);
  assert.deepEqual(reply.table, result.table);
  assert.match(reply.explanation, /benchmark/);
  assert.match(seen[0].context, /Explain the paper crypto run/);
  assert.doesNotMatch(seen[0].context, /Do not pass this/);
  await dispatch("Show crypto status", undefined, []);
  assert.equal(seen[1].context, "");

  const offline = await createCommandHandler({ ...ai, explainResult: undefined })("Show crypto status");
  assert.match(offline.explanation, /simulated crypto accounts.*21 days.*0\.042/);
});

test("retrieved data goes to the configured model as evidence for a written reply", async (t) => {
  const root = tmp(), calls = [];
  const ai = createAi({ root, dataDir: root, env: { OPENAI_API_KEY: KEY },
    fetchImpl: seq([toolResponse({ explanation: "The 21 days are a short simulation window." }, "tool_calls", "explainRetrievedData")], calls) });
  t.after(() => { ai.store.close(); ai.memory?.close(); ai.prices?.close?.(); });
  const explanation = await ai.explainResult("How is crypto doing?", {
    action: "CRYPTO_PAPER_STATUS", label: "Crypto paper status", answer: "- days: 21",
    table: { columns: ["account"], rows: [[200]] },
  });
  assert.equal(explanation, "The 21 days are a short simulation window.");
  assert.equal(calls.length, 1);
  assert.match(calls[0].body.messages.at(-1).content, /How is crypto doing\?.*days: 21/s);
});

test("the chart can find the last model verdict without changing which verdict is latest", (t) => {
  const store = openStore(tmp(), ":memory:");
  t.after(() => store.close());
  const first = { ticker: "NVDA", createdAt: "2026-11-25T20:00:00Z", asof: "2026-11-25", source: "LLM", flag: "YELLOW" };
  const latest = { ...first, createdAt: "2026-11-26T00:59:54Z", source: "FALLBACK" };
  store.saveVerdict(first);
  store.saveVerdict(latest);
  assert.deepEqual(store.latestVerdicts(["NVDA"]), [latest]);
  assert.deepEqual(store.latestVerdicts(["NVDA"], "LLM"), [first]);
});

test("dispatch routes no-ticker questions to ask, not help", async () => {
  const asked = [];
  const dispatch = createCommandHandler({
    models: { current: () => ({ ready: false }) },
    prices: { has: (t) => ["NVDA", "AMD", "AVGO", "SPY", "QQQ", "IWM", "AAPL"].includes(t) },
    sources: { search: async () => ({ value: [] }) },
    advisor: {
      ask: async (q, tickers) => { asked.push({ q, tickers }); return { source: "FALLBACK", answer: "local research: overextension_c35_linear needs more data", tickers }; },
      decision: async () => [], runTool: async () => ({}), research: async () => ({}), targets: () => [],
    },
    memory: { ingest: () => ({ duplicate: false }), recall: async () => ({ text: "" }) },
    store: { latestVerdicts: () => [] },
  });
  const general = await dispatch("Is overextension a good model or do we need more data?");
  assert.equal(general.kind, "answer");
  assert.notEqual(general.kind, "help");
  assert.equal(general.intent.intent, "GENERAL");
  assert.deepEqual(asked[0].tickers, []);
  assert.match(general.answer, /overextension_c35_linear/);

  const help = await dispatch("/help");
  assert.equal(help.kind, "help");
  assert.match(help.text, /Ask anything in plain English/);
  const empty = await dispatch("  ");
  assert.equal(empty.kind, "help");
  const bare = await dispatch("help");
  assert.equal(bare.kind, "help");

  const stock = await dispatch("What do you think of Nvidia today?");
  assert.equal(stock.kind, "answer");
  assert.deepEqual(asked.at(-1).tickers, ["NVDA"]);
  const compare = await dispatch("Compare AMD and AVGO");
  assert.equal(compare.kind, "answer");
  assert.deepEqual(asked.at(-1).tickers, ["AMD", "AVGO"]);
});

test("advisor.ask: no-ticker question with no AI key surfaces local research summaries", async () => {
  const dir = tmp();
  mkdirSync(join(dir, "Research", "regime_allocation"), { recursive: true });
  writeFileSync(join(dir, "Research", "regime_allocation", "current_state.json"), JSON.stringify({
    available: true, asOf: "2026-11-24", overextension: 0.12, volBucket: "high", volDirection: "rising",
    recommendation: {
      decision: "stay_static_with_interest", champion: "overextension_c35_linear", baseline: "static_100_0_0",
      robust: false, livePaperUnchanged: true,
      gates: { deflatedSharpeProb: 0.96, pbo: 0.73, spaPVsTsmom: 1.0 },
      text: "Keep the live paper account on plain TSMOM. The pre-holdout leader is overextension_c35_linear (Sharpe 2.03) but it does not clear the robustness gates. Need more data.",
    },
  }));
  const store = openStore(dir);
  const models = createModelService({ root: dir, env: {}, settingsPath: join(dir, "s.json") });
  const advisor = createAdvisor({ dataDir: dir, models, store, sources: { search: async () => ({ value: [] }) }, prices: null, env: {},
    fetchImpl: async () => ({ ok: false, status: 401, text: async () => "" }), sleep: noSleep });
  const r = await advisor.ask("Is overextension a good model or do we need more data?", []);
  assert.equal(r.source, "FALLBACK");
  assert.deepEqual(r.tickers, []);
  assert.match(r.answer, /AI key is needed/i);
  assert.match(r.answer, /overextension_c35_linear/);
  assert.match(r.answer, /more data/i);
  const snippet = researchFallbackText("overextension hedging", { dataDir: dir });
  assert.match(snippet, /REGIME_ALLOCATION/);
  assert.match(snippet, /overextension_c35_linear/);
});
