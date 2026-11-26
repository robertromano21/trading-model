// Mocked tests for workbench.js, logging.js and autopilot.js. No network, no real Schwab, no real orders.
// Run: node --test workbench.test.mjs
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createAutopilot, TASKS } from "./autopilot.js";
import { createLogKit } from "./logging.js";
import { checkSchwabUrl, createWorkbench, easternClock, JOBS, pidIsAlive } from "./workbench.js";

const T0 = Date.UTC(2026, 10, 25, 15, 0, 0);   // 10:00 ET Wednesday
function fakeSpawn(calls, { code = 0, out = "{}" } = {}) {
  return (command, args, options) => {
    calls.push({ command, args, cwd: options?.cwd, env: options?.env, detached: Boolean(options?.detached), stdio: options?.stdio });
    const child = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.kill = () => {};
    child.unref = () => {};
    child.pid = 4242;
    setImmediate(() => { child.stdout.emit("data", typeof out === "function" ? out(args) : out); child.emit("close", typeof code === "function" ? code(args) : code); });
    return child;
  };
}
function tempRoot(token) {
  const root = mkdtempSync(join(tmpdir(), "wb-"));
  mkdirSync(join(root, "Massive", "Research"), { recursive: true });
  if (token) writeFileSync(join(root, "schwab_token.json"), JSON.stringify(token));
  return root;
}

test("Schwab URL guard allows market data only", () => {
  assert.doesNotThrow(() => checkSchwabUrl("https://api.schwabapi.com/marketdata/v1/quotes?symbols=SPY"));
  assert.throws(() => checkSchwabUrl("https://api.schwabapi.com/trader/v1/accounts/x/orders"), /market-data/);
  assert.throws(() => checkSchwabUrl("https://evil.example.com/marketdata/v1/quotes"), /non-Schwab/);
});

test("quotes use Schwab when the token is valid, cache them, and fall back cleanly when expired", async () => {
  const fetched = [];
  const fetchImpl = async (url) => { fetched.push(url); return { ok: true, json: async () => ({ SPY: { quote: { lastPrice: 700, closePrice: 690, bidPrice: 699.9, askPrice: 700.1 } } }) }; };
  let clock = T0;
  const root = tempRoot({ creation_timestamp: T0 / 1000 - 3600, token: { access_token: "abc", refresh_token: "r", expires_at: T0 / 1000 + 1800 } });
  const wb = createWorkbench({ root, dataDir: join(root, "Massive"), fetchImpl, now: () => clock, env: {} });
  const first = await wb.quotes(["SPY"]);
  assert.equal(first.source, "schwab");
  assert.equal(first.quotes.SPY.last, 700);
  assert.ok(Math.abs(first.quotes.SPY.changePct - (700 / 690 - 1) * 100) < 1e-9);
  await wb.quotes(["SPY"]);                       // within the 4 s cache: no second request
  assert.equal(fetched.length, 1);
  assert.ok(fetched[0].includes("/marketdata/v1/quotes"));
  clock += 3600_000;                               // access token expired, no app key -> no refresh, cached data only
  const later = await wb.quotes(["SPY"]);
  assert.equal(later.schwab.usable, false);
  assert.match(later.schwab.reason, /SCHWAB_APP_KEY/);
  assert.equal(fetched.length, 1);
});

test("rate limiter caps Schwab requests per minute", async () => {
  let n = 0;
  const fetchImpl = async () => { n += 1; return { ok: true, json: async () => ({}) }; };
  const root = tempRoot({ creation_timestamp: T0 / 1000, token: { access_token: "abc", expires_at: T0 / 1000 + 1800 } });
  const wb = createWorkbench({ root, dataDir: join(root, "Massive"), fetchImpl, now: () => T0, env: {} });
  for (let i = 0; i < 120; i += 1) await wb.quotes([`S${i}`]);
  assert.equal(n, 90);
});

test("jobs are allow-listed, confirmed when they change state, and run fixed scripts", () => {
  const calls = [];
  const root = tempRoot();
  const wb = createWorkbench({ root, dataDir: join(root, "Massive"), spawnImpl: fakeSpawn(calls), env: {} });
  assert.throws(() => wb.startJob("rm_rf"), /Unknown job/);
  assert.throws(() => wb.startJob("paper_run"), /confirm/);
  const job = wb.startJob("research_all");
  assert.equal(job.status, "running");
  assert.deepEqual(calls[0].args.slice(-2), ["research.py", "all"]);
  for (const spec of Object.values(JOBS)) assert.ok(!/live|order/i.test(`${spec.script} ${spec.args.join(" ")}`));
});

test("Schwab app settings are saved to .env.local and never returned", () => {
  const root = tempRoot();
  writeFileSync(join(root, ".env.local"), "MASSIVE_API_KEY=keepme\n");
  const env = {};
  const wb = createWorkbench({ root, dataDir: join(root, "Massive"), env });
  const status = wb.saveAppKeys("APPKEY1234567890", "SECRET123456", "https://127.0.0.1:8182");
  const text = JSON.stringify(status);
  assert.ok(!text.includes("APPKEY1234567890") && !text.includes("SECRET123456"));
  assert.equal(status.config.appKeySet, true);
  const file = readFileSync(join(root, ".env.local"), "utf8");
  assert.match(file, /MASSIVE_API_KEY=keepme/);
  assert.match(file, /SCHWAB_APP_SECRET=SECRET123456/);
  assert.throws(() => wb.saveAppKeys("", "", "http://example.com"), /callback/);
});

test("reauthorization needs app keys and redacts them from its output", async () => {
  const root = tempRoot();
  const calls = [];
  const env = {};
  const wb = createWorkbench({ root, dataDir: join(root, "Massive"), env,
    spawnImpl: fakeSpawn(calls, { out: (args) => (args.includes("schwab_reauth.py") ? "opening https://x/auth?client_id=APPKEY1234567890&code=abcdefgh\nDONE\n" : "") }) });
  await assert.rejects(wb.startReauth(), /app key/);
  wb.saveAppKeys("APPKEY1234567890", "SECRET123456");
  await wb.startReauth();
  await new Promise((r) => setTimeout(r, 20));
  const view = wb.reauthView();
  assert.equal(view.job.status, "complete");
  const lines = view.job.lines.join("\n");
  assert.ok(!lines.includes("APPKEY1234567890") && !lines.includes("abcdefgh"));
  assert.ok(calls.some((c) => c.args.includes("schwab_reauth.py")));
});

test("log kit redacts secrets, compacts and filters streams", async () => {
  const dir = mkdtempSync(join(tmpdir(), "logs-"));
  const kit = createLogKit({ logsDir: dir, env: { MASSIVE_API_KEY: "supersecretkey" } });
  assert.equal(kit.redact("x supersecretkey y"), "x [redacted] y");
  assert.match(kit.redact("Authorization: Bearer abcdefghijkl"), /Bearer \[redacted\]/);
  assert.match(kit.redact('{"accountNumber":"12345678"}'), /…5678/);
  assert.deepEqual(kit.scrub({ refresh_token: "zzz", ok: 1 }), { refresh_token: "[redacted]", ok: 1 });
  assert.ok(kit.compact("a".repeat(40_000)).length < 17_000);
  const day = "2026-11-25";
  writeFileSync(join(dir, `server.${day}.log`), [
    JSON.stringify({ timestamp: `${day}T01:00:00Z`, level: "info", message: "hello", requestId: "req-12345678" }),
    JSON.stringify({ timestamp: `${day}T01:00:01Z`, level: "error", message: "boom", source: "workbench" })].join("\n"));
  writeFileSync(join(dir, `engine.${day}.log`), JSON.stringify({ timestamp: `${day}T01:00:02Z`, level: "warn", message: "py", source: "engine:paper_trader" }) + "\n");
  const all = await kit.view({ date: day });
  assert.equal(all.entries.length, 3);
  assert.equal((await kit.view({ date: day, level: "warn" })).entries.length, 2);
  assert.equal((await kit.view({ date: day, source: "engine:paper" })).entries.length, 1);
  assert.equal((await kit.view({ date: day, requestId: "req-12345678" })).entries[0].message, "hello");
});

test("autopilot schedule: watch in market hours, close run after 16:10, swing after 21:30, nothing on weekends", () => {
  const root = tempRoot();
  let clock = T0;
  const ap = createAutopilot({ root, dataDir: join(root, "Massive"), now: () => clock, env: {}, spawnImpl: fakeSpawn([]) });
  assert.equal(ap.due(easternClock(clock)), "daytrade_watch");
  clock = Date.UTC(2026, 10, 25, 21, 15);           // 16:15 ET
  assert.equal(ap.due(easternClock(clock)), "daytrade_close");
  ap._state.last.daytrade_close = { startedAt: new Date(clock).toISOString() };
  clock = Date.UTC(2026, 10, 26, 2, 35);            // 21:35 ET same day
  assert.equal(ap.due(easternClock(clock)), "swing");
  clock = Date.UTC(2026, 10, 28, 16, 0);            // Saturday
  assert.equal(ap.due(easternClock(clock)), null);
});

test("autopilot runs only paper scripts, feeds the chat log, and a failing memory hook never fails a run", async () => {
  const root = tempRoot();
  const calls = [];
  const ap = createAutopilot({ root, dataDir: join(root, "Massive"), now: () => T0, env: {},
    spawnImpl: fakeSpawn(calls, { out: (args) => (args.includes("decision_feed.py") ? '{"added":0}' : '{"lastRun":{"message":"ok"}}') }),
    memoryIngest: () => () => { throw new Error("memory down"); } });
  await ap.addMessage({ at: "2026-11-25T21:00:00Z", role: "trader", kind: "fill", ticker: "IEF", text: "FILLED BUY 18 IEF", rationale: "entry" });
  const record = await ap.runTask("swing", "test");
  assert.equal(record.ok, true);
  const scripts = calls.map((c) => c.args[2]);
  assert.deepEqual(scripts, ["dilution.py", "analyst_scoring.py", "paper_trader.py", "decision_feed.py", "drift_monitor.py", "swarm.py", "crypto_paper.py",
    "macro_service.py", "black_swan.py", "options_data.py", "options_paper.py"]);
  assert.equal(record.memory.status, "ok");
  assert.equal(record.memory.failed, 1);
  for (const task of Object.values(TASKS)) for (const [script] of task.steps) assert.ok(["analyst_scoring.py", "paper_trader.py", "daytrader.py", "schwab_daily.py"].includes(script));
});

test("options paper steps never fail the swing run and TRADING_OPTIONS_PAPER=off skips them", async () => {
  const calls = [];
  const ap = createAutopilot({ root: tempRoot(), dataDir: join(tempRoot(), "Massive"), now: () => T0, env: {},
    spawnImpl: fakeSpawn(calls, { code: (args) => (/options_/.test(args[2]) ? 2 : 0), out: (args) => (args.includes("decision_feed.py") ? '{"added":0}' : '{"lastRun":{"message":"ok"}}') }),
    memoryIngest: () => null });
  const record = await ap.runTask("swing", "test");
  assert.equal(record.ok, true);
  assert.deepEqual(calls.filter((c) => /options_/.test(c.args[2])).map((c) => c.args.slice(2)), [["options_data.py", "snapshot"], ["options_paper.py", "run", "--refresh-vix"]]);
  assert.equal(record.options.paper.code, 2);
  const off = [];
  const ap2 = createAutopilot({ root: tempRoot(), dataDir: join(tempRoot(), "Massive"), now: () => T0, env: { TRADING_OPTIONS_PAPER: "off" },
    spawnImpl: fakeSpawn(off, { out: '{"lastRun":{"message":"ok"}}' }), memoryIngest: () => null });
  assert.equal((await ap2.runTask("swing", "test")).ok, true);
  assert.ok(!off.some((c) => /options_/.test(c.args[2])));
});

test("follow-up questions are stored with the AI answer and the decision context", async () => {
  const root = tempRoot();
  const asked = [];
  const ap = createAutopilot({ root, dataDir: join(root, "Massive"), env: {}, port: 3999,
    fetchImpl: async (url, init) => { asked.push({ url, body: JSON.parse(init.body) }); return { ok: true, json: async () => ({ answer: "Because the stop hit.", source: "LLM" }) }; } });
  const id = await ap.addMessage({ at: "2026-11-25T21:00:00Z", role: "trader", kind: "stop", ticker: "XLE", text: "STOP SELL 3 XLE", rationale: "ATR stop" });
  const result = await ap.ask({ question: "Why did this happen?", messageId: id });
  assert.equal(result.answer, "Because the stop hit.");
  assert.match(asked[0].body.question, /STOP SELL 3 XLE/);
  assert.deepEqual(asked[0].body.tickers, ["XLE"]);
  const { messages } = await ap.listChat({ ticker: "XLE" });
  assert.deepEqual(messages.map((m) => m.kind), ["stop", "followup_q", "followup_a"]);
  assert.equal((await ap.listChat({ kind: "followup_a" })).messages.length, 1);
});

test("no automated module references the live order path", () => {
  for (const file of ["autopilot.js", "workbench.js", "logging.js"]) {
    const text = readFileSync(new URL(file, import.meta.url), "utf8");
    assert.ok(!/live-trading|\/api\/live|previewOrder|placeOrder/.test(text), file);
    if (file !== "workbench.js") assert.ok(!text.includes("/trader/"), file);
  }
});

test("paper decisions reach AI memory as DECISION items with a dedupe key, once", async () => {
  const root = tempRoot();
  const items = [];
  const ap = createAutopilot({ root, dataDir: join(root, "Massive"), now: () => T0, env: {}, spawnImpl: fakeSpawn([], { out: "{}" }),
    memoryIngest: () => (item) => { items.push(item); return { id: items.length, duplicate: false }; } });
  await ap.addMessage({ at: "2026-11-25T21:00:00Z", role: "trader", kind: "run", text: "run summary" });
  await ap.addMessage({ at: "2026-11-25T21:00:01Z", role: "trader", kind: "fill", ticker: "IEF", side: "buy", qty: 18, price: 95.1,
    account: "default", text: "FILLED BUY 18 IEF", rationale: "entry", verdict: { verdict: "GREEN", summary: "ok" } });
  await ap.addMessage({ at: "2026-11-25T21:00:02Z", role: "user", kind: "followup_q", text: "why?" });
  const first = await ap.ingestMemory();
  assert.equal(first.sent, 1);
  assert.equal(items[0].kind, "DECISION");
  assert.equal(items[0].ticker, "IEF");
  assert.equal(items[0].meta.refPrice, 95.1);
  assert.match(items[0].meta.dedupeKey, /^chat:/);
  assert.match(items[0].text, /Why: entry[\s\S]*AI verdict used: GREEN/);
  assert.equal((await ap.ingestMemory()).sent, 0);
});

test("pidIsAlive reports this process and rejects invalid pids", () => {
  assert.equal(pidIsAlive(process.pid), true);
  assert.equal(pidIsAlive(0), false);
  assert.equal(pidIsAlive(-1), false);
  assert.equal(pidIsAlive("nope"), false);
});

test("schwab daily backfill is spawned detached so a server stop does not kill it", () => {
  const calls = [];
  const root = tempRoot();
  const wb = createWorkbench({ root, dataDir: join(root, "Massive"), spawnImpl: fakeSpawn(calls), env: {} });
  assert.equal(JOBS.schwab_daily_backfill.detached, true);
  const job = wb.startJob("schwab_daily_backfill");
  assert.equal(job.detached, true);
  assert.equal(job.pid, 4242);
  assert.equal(calls[0].detached, true);
  assert.deepEqual(calls[0].stdio[0], "ignore");
  const short = [];
  const wb2 = createWorkbench({ root, dataDir: join(root, "Massive"), spawnImpl: fakeSpawn(short), env: {} });
  wb2.startJob("research_all");
  assert.equal(short[0].detached, false);
});

test("a backfill still running after a restart is adopted; a dead pid is marked interrupted", () => {
  const root = tempRoot();
  const jobsDir = join(root, "Massive", "Research", "jobs");
  mkdirSync(jobsDir, { recursive: true });
  const live = { id: "live-1", name: "schwab_daily_backfill", label: "backfill", status: "running",
    startedAt: "2026-11-25T22:00:00.000Z", finishedAt: null, exitCode: null, pid: 424242, detached: true };
  const dead = { id: "dead-1", name: "schwab_daily_backfill", label: "backfill", status: "running",
    startedAt: "2026-11-25T20:00:00.000Z", finishedAt: null, exitCode: null, pid: 999999, detached: true };
  writeFileSync(join(jobsDir, "jobs.json"), JSON.stringify([dead, live]));
  const timers = [];
  let holderAlive = true;
  const wb = createWorkbench({
    root, dataDir: join(root, "Massive"), spawnImpl: fakeSpawn([]), env: {},
    pidAlive: (pid) => holderAlive && pid === 424242,
    setIntervalImpl: (fn, ms) => { const t = { fn, ms, unref() {} }; timers.push(t); return t; },
    clearIntervalImpl: (t) => { t.cleared = true; },
  });
  const listed = [...wb.jobs.values()];
  const adopted = listed.find((j) => j.id === "live-1");
  const ended = listed.find((j) => j.id === "dead-1");
  assert.equal(adopted.status, "running");
  assert.equal(ended.status, "interrupted");
  assert.throws(() => wb.startJob("schwab_daily_backfill"), /already running/);
  holderAlive = false;
  timers[0].fn();
  assert.equal(adopted.status, "exited");
  assert.equal(timers[0].cleared, true);
});
