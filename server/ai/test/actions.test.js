// Adjudicated analyst actions: stubbed LLM + stubbed routes (no network, no model, no engine).
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createActions, legalize, fallbackAdjudicate, APP_ACTIONS, CONFIRM_TTL_MS, isForbidden, ORIGIN } from "../actions.js";
import { ROUTE_ACTIONS, MATH_ACTIONS } from "../actionRegistry.js";
import { HUMAN_ONLY } from "../../human-only-routes.js";
import { createExperiments } from "../../experiments.js";

function harness({ llmValue = null, routes = {}, clock = { t: Date.parse("2026-11-26T15:00:00Z") } } = {}) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    const u = new URL(url); const key = `${init.method} ${u.pathname}`;
    calls.push({ key, query: Object.fromEntries(u.searchParams), body: init.body ? JSON.parse(init.body) : undefined, origin: init.headers["x-tm-origin"] });
    const h = routes[key];
    const value = h ? await h(calls.at(-1)) : { ok: true };
    return { ok: !(value && value.__status >= 400), status: value?.__status || 200, text: async () => JSON.stringify(value) };
  };
  const toolCalls = [];
  const tools = { list: () => ({ RUN_BACKTEST: { methods: { DEFAULT: "" } }, STATS: { methods: { SHARPE_INFERENCE: "" } } }),
    execute: async (call) => { toolCalls.push(call); return { ok: true, result: { performance: { sharpe: 0.8, totalReturn: 0.2 }, rows: [{ a: 1, b: 2 }] } }; } };
  const llm = llmValue ? async () => (typeof llmValue === "function" ? llmValue() : llmValue) : null;
  const memoryItems = [], logs = [];
  const actions = createActions({ fetchImpl, tools, llm, memory: { ingest: (i) => memoryItems.push(i) }, store: { log: (e) => logs.push(e) }, dataDir: mkdtempSync(join(tmpdir(), "act-")), now: () => clock.t, pollMs: 5 });
  actions.attach({ baseUrl: "http://127.0.0.1:1" });
  return { actions, calls, toolCalls, memoryItems, logs, clock };
}
const A = (o) => Object.entries(o).map(([name, value]) => ({ name, value: typeof value === "string" ? value : JSON.stringify(value) }));
const pickOf = (action, args = {}, extra = {}) => ({ action, args: A(args), confidence: "HIGH", needsConfirm: "NO", manualTopic: "NONE", clarifyQuestion: "", ...extra });

test("model picks a read action: runs now, no confirm, tagged origin", async () => {
  const h = harness({ llmValue: pickOf("QUOTES", { symbols: ["SPY", "QQQ"] }), routes: { "GET /api/market/quotes": () => ({ quotes: [{ symbol: "SPY", last: 600 }] }) } });
  const r = await h.actions.handle("quote spy and qqq");
  assert.equal(r.kind, "action"); assert.equal(r.action, "QUOTES");
  assert.equal(h.calls[0].query.symbols, "SPY,QQQ"); assert.equal(h.calls[0].origin, ORIGIN);
  assert.ok(r.table.rows.length);
});

test("bad enum from the model is rejected -> clarify", async () => {
  const h = harness({ llmValue: pickOf("PLACE_SCHWAB_ORDER", { ticker: "SPY" }) });
  const r = await h.actions.handle("do something");
  assert.equal(r.kind, "clarify"); assert.equal(h.calls.length, 0);
});

test("bad args are rejected by the legalizer (no call made)", async () => {
  const h = harness({ llmValue: pickOf("EXPERIMENT_ORDER", { side: "buy", ticker: "spy!!", qty: "-3" }) });
  const r = await h.actions.handle("paper buy");
  assert.equal(r.kind, "clarify"); assert.match(r.text, /ticker|qty/); assert.equal(h.calls.length, 0);
  assert.equal(legalize("EXPERIMENT_ORDER", { side: "buy", ticker: "SPY", qty: 10, bogus: 1 }).ok, false);
});

test("ambiguous / low confidence -> clarifying question", async () => {
  const h = harness({ llmValue: pickOf("CLARIFY", {}, { clarifyQuestion: "Which strategy?" }) });
  assert.deepEqual((await h.actions.handle("run a backtest")).text, "Which strategy?");
  const h2 = harness({ llmValue: pickOf("MATH_BACKTEST", { family: "TSMOM" }, { confidence: "LOW" }) });
  assert.equal((await h2.actions.handle("maybe test something")).kind, "clarify");
});

test("human-only is refused with a pointer to the UI, never called", async () => {
  const h = harness({ llmValue: pickOf("MANUAL_STEP", {}, { manualTopic: "LIVE_ORDER" }) });
  const r = await h.actions.handle("buy 10 SPY in my Schwab account");
  assert.equal(r.kind, "manual"); assert.match(r.text, /manual step/); assert.equal(h.calls.length, 0);
  for (const route of Object.keys(HUMAN_ONLY)) assert.ok(isForbidden(route));
  assert.ok(isForbidden("POST /api/live/order/preview")); assert.ok(isForbidden("POST /api/keys/fred")); assert.ok(isForbidden("GET /api/live/status")); assert.ok(!isForbidden("GET /api/keys/fred")); assert.ok(!isForbidden("POST /api/experiments/1000/order"));
  assert.ok(!APP_ACTIONS.some((a) => /LIVE_ORDER|LIVE_TOGGLE|KEY_SET|SCHWAB_ORDER/.test(a)));
});

test("Allow / Cancel: write returns a prompt with a summary; Allow runs once, Cancel never runs, expiry", async () => {
  const h = harness({ llmValue: pickOf("EXPERIMENT_ORDER", { side: "buy", ticker: "SPY", qty: "10" }), routes: { "POST /api/experiments/1000/order": () => ({ order: { fill: 600 } }) } });
  const r = await h.actions.handle("paper buy 10 SPY");
  assert.equal(r.kind, "confirm"); assert.equal(h.calls.length, 0);
  assert.equal(r.summary.action, "EXPERIMENT_ORDER"); assert.ok(r.summary.affects.some((s) => /PAPER BUY 10 SPY in experiment account 1000/.test(s)));
  assert.ok(r.expiresAt);
  const ok = await h.actions.confirm(r.token);
  assert.equal(ok.kind, "action"); assert.equal(h.calls.length, 1); assert.equal(h.calls[0].body.confirm, true); assert.equal(h.calls[0].origin, ORIGIN);
  assert.equal((await h.actions.confirm(r.token)).code, "EXPIRED");   // single use
  const r2 = await h.actions.handle("paper buy 10 SPY");
  assert.equal(h.actions.cancel(r2.token).kind, "cancelled"); assert.equal((await h.actions.confirm(r2.token)).code, "EXPIRED"); assert.equal(h.calls.length, 1);
  const r3 = await h.actions.handle("paper buy 10 SPY");
  h.clock.t += CONFIRM_TTL_MS + 1;
  assert.equal((await h.actions.confirm(r3.token)).code, "EXPIRED"); assert.equal(h.calls.length, 1);
});

test("every write action needs Allow; every read/math action does not", async () => {
  for (const [id, s] of Object.entries(ROUTE_ACTIONS)) assert.ok(["read", "write"].includes(s.kind), id);
  const h = harness({ routes: {} });
  const r = await h.actions.run("AUTOPILOT_TOGGLE", { enabled: "off" });
  assert.equal(r.kind, "confirm"); assert.match(r.summary.affects.join(" "), /OFF/);
  assert.equal(h.calls.length, 0);
});

test("math: backtest is clamped before the holdout, trial recorded, memory fed", async () => {
  const h = harness({ llmValue: pickOf("MATH_BACKTEST", { family: "TSMOM", tickers: ["QQQ"], lookback_days: "126" }) });
  const trialsSeen = [];
  h.actions.attach({ experiments: { recordTrial: (t) => trialsSeen.push(t), trialCount: () => 3, list: () => ({ accounts: [], active: 0, cap: 200 }) } });
  const r = await h.actions.handle("backtest TSMOM on QQQ with a 6-month lookback");
  assert.equal(r.kind, "action");
  const call = h.toolCalls[0];
  assert.equal(call.tool, "RUN_BACKTEST"); assert.equal(call.family, "TSMOM"); assert.deepEqual(call.tickers, ["QQQ"]);
  assert.ok(call.options.some((o) => o.name === "horizons" && o.value === "[126]"));
  assert.equal(call.end, "2026-07-08");
  assert.equal(trialsSeen[0].origin, ORIGIN); assert.equal(trialsSeen[0].holdoutLook, false);
  assert.ok(h.memoryItems.some((m) => m.meta.origin === ORIGIN));
  assert.ok(h.logs.some((l) => l.input.origin === ORIGIN));
});

test("fallback adjudicator (no model) covers the headline prompts", () => {
  const cases = {
    "backtest TSMOM on QQQ with a 6-month lookback": ["MATH_BACKTEST", { family: "TSMOM", lookback_days: "126" }],
    "try a dollar filter on the crypto trend": ["OPS_RUN", { job: "crypto_research_pre" }],
    "open a paper account running tsmom with $10k": ["EXPERIMENT_CREATE", { strategy: "tsmom", equity: "10000" }],
    "paper buy 10 SPY": ["EXPERIMENT_ORDER", { side: "buy", qty: "10", ticker: "SPY" }],
    "10 $10k accounts sweeping lookback 3-12m": ["EXPERIMENT_CREATE", { count: "10", equity: "10000" }],
    "what's the deflated Sharpe of account 12": ["MATH_ACCOUNT_SHARPE", { account: "12" }],
    "run the regime model on QQQ": ["MATH_REGIME", {}],
    "price a 30-day SPY 600 call": ["MATH_OPTION_PRICE", { dte: "30", strike: "600", kind: "call" }],
    "delete experiment account 1003": ["EXPERIMENT_DELETE", { id: "1003" }],
    "buy 10 SPY in my schwab account": ["MANUAL_STEP", {}],
    "what do you think of NVDA?": ["ASK_ANALYST", {}],
  };
  for (const [text, [action, args]] of Object.entries(cases)) {
    const r = fallbackAdjudicate(text);
    assert.equal(r.action, action, text);
    const got = Object.fromEntries(r.args.map((a) => [a.name, a.value]));
    for (const [k, v] of Object.entries(args)) assert.equal(got[k], v, `${text}: ${k}`);
    if (!["MANUAL_STEP", "ASK_ANALYST"].includes(action)) assert.ok(legalize(action, r.args).ok, `${text}: ${legalize(action, r.args).errors}`);
  }
});

test("experiments: batch sweep, id range, cap, list/retire/delete, origin + trial registry, holdout look needs confirm", async () => {
  const dataDir = mkdtempSync(join(tmpdir(), "exp-"));
  mkdirSync(join(dataDir, "Research", "swarm"), { recursive: true });
  writeFileSync(join(dataDir, "Research", "swarm", "design.json"), JSON.stringify({ holdoutStart: "2026-07-09" }));
  const quantCalls = [];
  const quant = { run: async (op, args) => { quantCalls.push({ op, args }); return { ok: true, result: { performance: { sharpe: 0.5, totalReturn: 0.1 } } }; } };
  const ex = createExperiments({ dataDir, quant, priceOf: async () => ({ price: 100, source: "local" }), env: { TM_MAX_USER_ACCOUNTS: "12" } });
  const b = ex.create({ strategy: "tsmom", equity: 10000, count: 10, sweep: { param: "lookback_days", from: 63, to: 252 } }, ORIGIN);
  assert.equal(b.created.length, 10); assert.ok(b.batchId);
  assert.deepEqual(b.created.map((a) => a.id), [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]);
  assert.equal(b.created[0].params.lookback_days, 63); assert.equal(b.created[9].params.lookback_days, 252);
  assert.throws(() => ex.create({ strategy: "tsmom", count: 3 }, "user_ui"), /cap is 12/);
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(quantCalls.length, 10);
  assert.ok(quantCalls.every((c) => c.args.end === "2026-07-08" && Array.isArray(c.args.params.horizons)));
  const o = await ex.order(1000, { side: "buy", ticker: "SPY", qty: 10 }, ORIGIN);
  assert.equal(o.account.positions[0].qty, 10);
  await assert.rejects(ex.order(1001, { side: "buy", ticker: "SPY", qty: 1 }, ORIGIN), /manual account/);
  await assert.rejects(ex.order(1000, { side: "sell", ticker: "SPY", qty: 11 }, ORIGIN), /cannot short/);
  ex.retire(1002, ORIGIN); ex.remove(1003, ORIGIN);
  const l = ex.list();
  assert.equal(l.accounts.find((a) => a.id === 1002).status, "retired"); assert.ok(!l.accounts.some((a) => a.id === 1003));
  assert.equal(ex.trialCount(), 10);  // deleted accounts still counted
  const reg = readFileSync(join(dataDir, "Research", "experiments", "trial_registry.jsonl"), "utf8").trim().split("\n").map(JSON.parse);
  assert.ok(reg.every((t) => t.origin === ORIGIN || t.origin === "user_ui"));
  await assert.rejects(ex.holdoutLook({ strategy: "tsmom", reason: "x" }, ORIGIN), /confirm/);
  await ex.holdoutLook({ strategy: "tsmom", reason: "final check", confirm: true }, ORIGIN);
  const ledger = JSON.parse(readFileSync(join(dataDir, "Research", "swarm", "holdout_ledger.json"), "utf8"));
  assert.equal(ledger.entries.at(-1).origin, ORIGIN);
  assert.ok(!ex.trials().some((t) => t.holdoutLook));     // holdout looks never reach learning-side reads
  assert.ok(existsSync(join(dataDir, "Research", "experiments", "accounts.json")));
});
