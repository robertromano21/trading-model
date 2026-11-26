import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createAnalystScoring } from "./analyst-scoring.js";

function fakeApp() {
  const routes = {};
  const app = { get: (p, h) => { routes["GET " + p] = h; }, post: (p, h) => { routes["POST " + p] = h; } };
  const call = async (method, path, query = {}) => {
    let status = 200, body;
    const response = { status(s) { status = s; return this; }, json(b) { body = b; return this; } };
    await routes[`${method} ${path}`]({ query }, response);
    return { status, body };
  };
  return { app, call };
}

function fakeSpawn(stdout, code = 0, calls = []) {
  return (cmd, args, opts) => {
    calls.push({ cmd, args, opts });
    const child = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    setImmediate(() => { child.stdout.emit("data", stdout); child.emit("close", code); });
    return child;
  };
}

test("scorecard API: 404 before the first run, filtered summary after, fail-closed gate when stale", async () => {
  const dir = mkdtempSync(join(tmpdir(), "score-"));
  try {
    const { app, call } = fakeApp();
    const calls = [];
    const scoring = createAnalystScoring({ root: "/repo", dataDir: dir, python: "py",
      spawnImpl: fakeSpawn('progress\n{"status":"ok","calls":2,"gateOpen":[]}\n', 0, calls) });
    scoring.mount(app);
    assert.equal((await call("GET", "/api/analyst-scorecard")).status, 404);
    assert.deepEqual((await call("GET", "/api/analyst-scorecard/gate")).body.gateOpen, []);
    mkdirSync(join(dir, "Research"), { recursive: true });
    const doc = { schema: "analyst_scorecard/v1", generatedAt: new Date().toISOString(), gate: { RED: { 20: true } },
      gateOpen: ["RED@20d"], params: { maxAgeDays: 7 }, recent: [{ ticker: "NVDA" }, { ticker: "AAPL" }] };
    writeFileSync(join(dir, "Research", "analyst_scorecard.json"), JSON.stringify(doc));
    const all = await call("GET", "/api/analyst-scorecard");
    assert.equal(all.body.recent.length, 2);
    assert.deepEqual((await call("GET", "/api/analyst-scorecard", { ticker: "nvda" })).body.recent, [{ ticker: "NVDA" }]);
    assert.deepEqual((await call("GET", "/api/analyst-scorecard/gate")).body.gateOpen, ["RED@20d"]);
    writeFileSync(join(dir, "Research", "analyst_scorecard.json"),
      JSON.stringify({ ...doc, generatedAt: new Date(Date.now() - 9 * 86400000).toISOString() }));
    const stale = (await call("GET", "/api/analyst-scorecard/gate")).body;
    assert.equal(stale.fresh, false);
    assert.deepEqual(stale.gateOpen, []);
    const refreshed = await call("POST", "/api/analyst-scorecard/refresh");
    assert.equal(refreshed.status, 200);
    assert.equal(refreshed.body.calls, 2);
    assert.equal(calls.length, 1);
    assert.ok(calls[0].args.at(-2).endsWith("analyst_scoring.py") && calls[0].args.at(-1) === "run");
    assert.equal(calls[0].opts.env.PRICES_DATA_DIR, dir);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("refresh reports Python failures without throwing", async () => {
  const { app, call } = fakeApp();
  createAnalystScoring({ root: "/repo", dataDir: "/nowhere", spawnImpl: fakeSpawn("not json", 1) }).mount(app);
  const r = await call("POST", "/api/analyst-scorecard/refresh");
  assert.equal(r.status, 500);
  assert.equal(r.body.status, "error");
});
