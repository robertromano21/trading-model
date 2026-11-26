// Tool loop + quant tools tests: mocked model tool calls (no network) and real engine/quant_tools.py runs
// (skipped when the Python environment / local price data are missing).
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { contract } from "../retortSchema.js";
import { TOOL_STEP } from "../schemas.js";
import * as E from "../enums.js";
import { createModelService } from "../models.js";
import { openStore } from "../store.js";
import { createAdvisor } from "../advisor.js";
import { planCall, summarize, createTools, parseToolCommand, toolCatalog, TOOLS } from "../tools.js";
import { createQuantRunner, pythonFor } from "../quant.js";
import { createStrategyCard, renderMaster } from "../strategyCard.js";
import { parseSlash } from "../command.js";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const KEY = "sk-test-0123456789abcdefghijklmnop";
const tmp = () => mkdtempSync(join(tmpdir(), "tm-tools-"));
const noSleep = async () => {};
const call = (tool, extra = {}) => ({ tool, method: "DEFAULT", family: "NONE", tickers: [], start: "", end: "", horizonDays: null, options: [], query: "", purpose: "", ...extra });
const fnResponse = (name, args) => ({ ok: true, status: 200, text: async () => JSON.stringify({ model: "m", usage: { prompt_tokens: 10, completion_tokens: 5 },
  choices: [{ finish_reason: "tool_calls", message: { tool_calls: [{ id: "1", function: { name, arguments: JSON.stringify(args) } }] } }] }) });
const ANSWER_VALUE = { answer: "NVDA daily returns are fat-tailed (JB p<0.001); size with t-VaR.", stance: "YELLOW", conviction: "MEDIUM",
  keyPoints: ["fat tails"], levels: [], citations: [{ source: "TOOL_RESULT", asOf: "2026-11-24", detail: "STATS/NORMALITY" }], followUps: [] };

function bars(n = 260) {
  const out = []; let c = 100; const d0 = Date.now() - n * 86400000;
  for (let i = 0; i < n; i++) { const p = c; c *= 1 + 0.001 + 0.01 * Math.sin(i / 3); out.push({ date: new Date(d0 + i * 86400000).toISOString().slice(0, 10), open: p, high: Math.max(p, c) * 1.01, low: Math.min(p, c) * 0.99, close: c, volume: 1e6 }); }
  return out;
}
function harness({ fetchImpl, quantImpl, env = {}, withKey = true, card = null }) {
  const dir = tmp(), store = openStore(dir);
  const b = bars();
  const prices = { bars: () => b, has: () => true };
  const sources = {
    details: async () => ({ value: { name: "Nvidia Corp", type: "CS" } }), news: async () => ({ value: [] }),
    dividends: async () => ({ value: [] }), splits: async () => ({ value: [] }), schwabQuotes: async () => { throw new Error("auth_required"); },
    secFinancials: async () => ({ value: { derived: {} } }), search: async () => ({ value: [] }), hasMassive: () => true,
  };
  const models = createModelService({ root: dir, env: withKey ? { OPENAI_API_KEY: KEY } : {}, settingsPath: join(dir, "s.json") });
  const quantCalls = [];
  const quant = { python: "fake", run: async (op, args, opts) => { quantCalls.push({ op, args, opts }); return quantImpl ? quantImpl(op, args) : { ok: true, result: { op, n: 250, jb_p: 0.0001, excess_kurtosis: 6.1234567 } }; } };
  const tools = createTools({ root: dir, dataDir: dir, quant, store, memory: null, research: null });
  const advisor = createAdvisor({ dataDir: dir, models, store, sources, prices, tools, card, env: { ADVISORY_VERDICTS_PATH: join(dir, "a.json"), ...env }, fetchImpl, sleep: noSleep });
  return { dir, store, advisor, quantCalls };
}
const rowsOf = (store, kind) => store.db ? store.db.prepare("SELECT * FROM ai_log WHERE kind=?").all(kind) : null;

test("tool step schema: strict, Retort enums for tool/method/family, no character limits", () => {
  const { wireSchema, answerSchema } = contract(TOOL_STEP);
  const s = JSON.stringify(wireSchema);
  assert.ok(!/maxLength|minLength|maxItems/.test(s));
  const item = answerSchema.properties.calls.items;
  assert.deepEqual(item.properties.tool.enum, E.TOOL);
  assert.deepEqual(item.properties.method.enum, E.TOOL_METHOD);
  assert.deepEqual(answerSchema.properties.decision.enum, ["CALL_TOOLS", "ANSWER"]);
  for (const t of E.TOOL) assert.ok(TOOLS[t], `spec for ${t}`);
  for (const t of E.TOOL) assert.match(toolCatalog(), new RegExp(`\\b${t}\\b`));
});

test("planCall maps enums + options to quant ops and rejects bad input", () => {
  const wf = planCall(call("WALK_FORWARD", { family: "MEANREV", options: [{ name: "grid.entry_z", value: "[-2,-2.5]" }, { name: "test_len", value: "42" }] }));
  assert.deepEqual(wf, { ...wf, op: "walk_forward", timeoutMs: 900000 });
  assert.equal(wf.args.family, "meanrev"); assert.deepEqual(wf.args.grid, { entry_z: [-2, -2.5] }); assert.equal(wf.args.test_len, 42);
  const bt = planCall(call("RUN_BACKTEST", { family: "TSMOM", options: [{ name: "max_positions", value: "5" }, { name: "horizons", value: "[20,60]" }] }));
  assert.equal(bt.op, "backtest"); assert.deepEqual(bt.args.params, { max_positions: 5, horizons: [20, 60] });
  const st = planCall(call("STATS", { method: "NORMALITY", tickers: ["nvda"], horizonDays: 5 }));
  assert.equal(st.op, "normality"); assert.deepEqual(st.args.tickers, ["NVDA"]); assert.equal(st.args.horizon, 5);
  const ar = planCall(call("ECONOMETRICS", { method: "ARIMA", tickers: ["SPY"], horizonDays: 10 }));
  assert.equal(ar.op, "arima"); assert.equal(ar.args.steps, 10);
  assert.equal(planCall(call("MEMORY_SEARCH", { query: "NVDA earnings" })).args.query, "NVDA earnings");
  assert.equal(planCall(call("MACRO_CALENDAR")).op, "macroCalendar");
  assert.equal(planCall(call("MACRO_CALENDAR")).kind, "node");
  const attr = planCall(call("ATTRIBUTION", { options: [{ name: "strategy", value: "tsmom" }, { name: "source", value: "backtest" }] }));
  assert.equal(attr.op, "attribute_returns");
  assert.equal(attr.args.strategy, "tsmom");
  assert.equal(attr.args.source, "backtest");
  const pr = planCall(call("PROB_MODEL", { tickers: ["NVDA"] }));
  assert.equal(pr.op, "prob_forecast");
  assert.deepEqual(pr.args.tickers, ["NVDA"]);
  assert.equal(planCall(call("BLACK_SWAN")).op, "blackSwan");
  assert.equal(planCall(call("BLACK_SWAN")).kind, "node");
  assert.throws(() => planCall(call("RM_RF")), /Unknown tool/);
});

test("summarize keeps prompts compact; parseToolCommand + /tool slash", () => {
  const big = { values: Array.from({ length: 5000 }, (_, i) => i / 7), stat: 1.234567891 };
  const s = summarize(big, 500);
  assert.ok(s.length < 700); assert.match(s, /truncated|1\.235/);
  const c = parseToolCommand("/tool STATS/NORMALITY NVDA SPY start=2025-01-01 horizon=5 horizons=[1,5,20]");
  assert.equal(c.tool, "STATS"); assert.equal(c.method, "NORMALITY"); assert.deepEqual(c.tickers, ["NVDA", "SPY"]);
  assert.equal(c.start, "2025-01-01"); assert.equal(c.horizonDays, 5); assert.deepEqual(c.options, [{ name: "horizons", value: "[1,5,20]" }]);
  assert.equal(parseToolCommand("/tool RUN_BACKTEST/TSMOM max_positions=5").family, "TSMOM");
  assert.equal(parseToolCommand("/tool MEMORY_SEARCH nvda earnings guidance").query, "nvda earnings guidance");
  assert.throws(() => parseToolCommand("/tool NOPE"), /Unknown tool/);
  assert.equal(parseSlash("/tool STATS NVDA").intent, "TOOL");
});

test("tools.execute: caches python results, logs the FULL result, never throws", async () => {
  const dir = tmp(), store = openStore(dir);
  let runs = 0;
  const long = Array.from({ length: 3000 }, (_, i) => i);
  const quant = { run: async () => { runs++; return { ok: true, result: { long } }; } };
  const tools = createTools({ root: dir, dataDir: dir, quant, store });
  const a = await tools.execute(call("STATS", { method: "DESCRIBE", tickers: ["NVDA"] }), { promptChars: 300 });
  const b = await tools.execute(call("STATS", { method: "DESCRIBE", tickers: ["NVDA"] }));
  assert.equal(a.ok, true); assert.equal(b.cached, true); assert.equal(runs, 1);
  assert.ok(a.summary.length < 450);
  const logged = rowsOf(store, "tool");
  if (logged) assert.ok(logged.some((r) => (r.output || "").includes("2999")), "full result in ai_log");
  const bad = await tools.execute(call("RUN_BACKTEST", { family: "TSMOM", options: [{ name: "grid.x", value: "{bad json" }] }));
  assert.equal(typeof bad.ok, "boolean");
  const failing = createTools({ root: dir, dataDir: dir, quant: { run: async () => ({ ok: false, error: { code: "TIMEOUT", message: "slow" } }) }, store });
  const f = await failing.execute(call("STATS", { method: "NORMALITY", tickers: ["SPY"] }));
  assert.equal(f.ok, false); assert.match(f.summary, /TIMEOUT/);
  store.close?.();
});

test("advisor.ask tool loop: model calls STATS/NORMALITY, results reach the final prompt, loop is bounded", async () => {
  const calls = [];
  let steps = 0;
  const fetchImpl = async (url, init) => {
    const body = JSON.parse(init.body); calls.push(body);
    const name = body.tool_choice.function.name;
    if (name === "analystToolStep") {
      steps++;
      if (steps === 1) return fnResponse(name, { decision: "CALL_TOOLS", reasoning: "check tails", calls: [
        call("STATS", { method: "NORMALITY", tickers: ["NVDA"], purpose: "fat tails?" }),
        call("STATS", { method: "NORMALITY", tickers: ["NVDA"], purpose: "duplicate" }),
        call("RISK", { tickers: ["NVDA"], horizonDays: 5, purpose: "5d VaR" })] });
      return fnResponse(name, { decision: "ANSWER", reasoning: "enough", calls: [] });
    }
    return fnResponse("analystAnswer", ANSWER_VALUE);
  };
  const h = harness({ fetchImpl });
  const a = await h.advisor.ask("Is NVDA fat-tailed and what is my 5-day VaR?", ["NVDA"]);
  assert.equal(a.source, "LLM");
  assert.equal(a.tools.length, 2, "duplicate call de-duplicated");
  assert.deepEqual(a.tools.map((t) => t.tool), ["STATS", "RISK"]);
  assert.deepEqual(h.quantCalls.map((q) => q.op), ["normality", "var"]);
  assert.deepEqual(a.toolSteps.map((s) => s.decision), ["CALL_TOOLS", "ANSWER"]);
  const final = calls.at(-1);
  assert.equal(final.tool_choice.function.name, "analystAnswer");
  const text = final.messages.map((m) => m.content).join("\n");
  assert.match(text, /\[TOOL_RESULTS\]/); assert.match(text, /STATS\/NORMALITY/); assert.match(text, /6\.123/);
  assert.ok(!("max_tokens" in final) && !("max_completion_tokens" in final));
  const stepPrompt = calls[0].messages.map((m) => m.content).join("\n");
  assert.match(stepPrompt, /Tool catalog/); assert.match(stepPrompt, /WALK_FORWARD/);
  // bounded: a model that always asks for (new) tools stops at TRADING_AI_TOOL_STEPS
  let n = 0;
  const greedy = harness({ env: { TRADING_AI_TOOL_STEPS: "2" }, fetchImpl: async (url, init) => {
    const name = JSON.parse(init.body).tool_choice.function.name;
    return name === "analystToolStep" ? fnResponse(name, { decision: "CALL_TOOLS", reasoning: "more", calls: [call("STATS", { method: "DESCRIBE", tickers: ["NVDA"], horizonDays: ++n })] })
      : fnResponse("analystAnswer", ANSWER_VALUE);
  } });
  const g = await greedy.advisor.ask("NVDA?", ["NVDA"]);
  assert.equal(g.toolSteps.length, 2); assert.equal(g.tools.length, 2); assert.equal(g.source, "LLM");
});

test("tool loop fallbacks: planner failure -> answer without tools; no key -> no model calls; useTools=false", async () => {
  const fetchImpl = async (url, init) => {
    const name = JSON.parse(init.body).tool_choice.function.name;
    return name === "analystToolStep" ? fnResponse(name, { decision: "MAYBE", reasoning: "", calls: [] }) : fnResponse("analystAnswer", ANSWER_VALUE);
  };
  const h = harness({ fetchImpl });
  const a = await h.advisor.ask("NVDA?", ["NVDA"]);
  assert.equal(a.source, "LLM"); assert.equal(a.tools.length, 0); assert.equal(a.toolSteps[0].status, "failed");
  let fetched = 0;
  const nk = harness({ withKey: false, fetchImpl: async () => { fetched++; return fnResponse("x", {}); } });
  const b = await nk.advisor.ask("NVDA?", ["NVDA"]);
  assert.equal(b.source, "FALLBACK"); assert.equal(fetched, 0); assert.equal(nk.quantCalls.length, 0);
  let stepCalls = 0;
  const off = harness({ fetchImpl: async (u, init) => { if (JSON.parse(init.body).tool_choice.function.name === "analystToolStep") stepCalls++; return fnResponse("analystAnswer", ANSWER_VALUE); } });
  await off.advisor.ask("NVDA?", ["NVDA"], { useTools: false });
  assert.equal(stepCalls, 0);
});

test("master system prompt = rules + generated reference (and survives a missing card)", async () => {
  const card = { get: async () => ({ generatedAt: "2026-11-25T00:00:00Z", strategies: [], results: {} }), cached: () => null };
  const h = harness({ fetchImpl: async () => fnResponse("analystAnswer", ANSWER_VALUE), card });
  const sys = await h.advisor.system();
  assert.match(sys, /Grounding rules/); assert.match(sys, /# TRADING MODEL REFERENCE/);
  const broken = harness({ fetchImpl: async () => fnResponse("analystAnswer", ANSWER_VALUE), card: { get: async () => { throw new Error("boom"); }, cached: () => null } });
  const s2 = await broken.advisor.system();
  assert.match(s2, /Grounding rules/); assert.ok(!/# TRADING MODEL REFERENCE/.test(s2));
});

// ---------------------------------------------------------------- real quant_tools.py runs
const PY = process.env.QUANT_PYTHON || pythonFor(ROOT);
const SCRIPT = join(ROOT, "engine", "quant_tools.py");
const HAVE_PY = existsSync(SCRIPT) && (PY.includes("/") || PY.includes("\\") ? existsSync(PY) : true);
const HAVE_DATA = existsSync(join(ROOT, "Massive", "prices.sqlite3"));
const quantReal = createQuantRunner({ root: ROOT, python: PY });
function synth(n = 600, seed = 7) {
  let s = seed; const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
  return Array.from({ length: n }, () => { const u = rnd(), v = rnd(); const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); return (rnd() < 0.05 ? 4 : 1) * 0.01 * z; });
}

test("real quant: describe / normality / garch / var / sizing on a series", { skip: !HAVE_PY && "python/quant_tools.py not found" }, async () => {
  const series = synth();
  const d = await quantReal.run("describe", { series, input: "returns" });
  assert.equal(d.ok, true, JSON.stringify(d.error)); assert.equal(d.result.n ?? d.result.count ?? d.result.series?.n ?? 600, 600);
  const nm = await quantReal.run("normality", { series, input: "returns", horizons: [1, 5, 20] });
  assert.equal(nm.ok, true, JSON.stringify(nm.error));
  assert.match(JSON.stringify(nm.result), /jarque|jb/i);
  const g = await quantReal.run("garch", { series, input: "returns" });
  assert.equal(g.ok, true, JSON.stringify(g.error));
  const v = await quantReal.run("var", { series, input: "returns", confidences: [0.95, 0.99], horizon: 5 });
  assert.equal(v.ok, true, JSON.stringify(v.error)); assert.match(JSON.stringify(v.result), /cvar|es/i);
  const z = await quantReal.run("sizing", { series, input: "returns", equity: 10000 });
  assert.equal(z.ok, true, JSON.stringify(z.error));
  const bad = await quantReal.run("no_such_op", {});
  assert.equal(bad.ok, false); assert.equal(bad.error.code, "UNKNOWN_OP");
});

test("real quant: TSMOM backtest matches research, strategy card renders the master prompt", { skip: (!HAVE_PY || !HAVE_DATA) && "needs .venv + Massive/prices.sqlite3", timeout: 240000 }, async () => {
  const bt = await quantReal.run("backtest", { family: "tsmom" }, { timeoutMs: 200000 });
  assert.equal(bt.ok, true, JSON.stringify(bt.error));
  const m = bt.result.performance || bt.result.metrics || bt.result;
  assert.ok(Math.abs((m.cagr ?? m.CAGR) - 0.143) < 0.01, `cagr ${m.cagr ?? m.CAGR}`);
  const dir = tmp();
  const card = createStrategyCard({ root: ROOT, dataDir: join(ROOT, "Massive"), quant: quantReal, cacheFile: join(dir, "card.json") });
  const text = renderMaster(await card.get({ refresh: true }));
  for (const s of ["TRADING MODEL REFERENCE", "STRATEGIES", "MEASURED RESULTS", "RISK", "NORMALITY"]) assert.match(text, new RegExp(s));
  for (const s of ["TSMOM|tsmom", "mean.?rev", "ORB|opening.range", "200", "RSI", "Donchian", "pairs", "momentum"]) assert.match(text, new RegExp(s, "i"));
});
