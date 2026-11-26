// Learning feed: analyst-adjudicated actions write the same records as the UI path, tagged
// origin=analyst_user_request, and count toward multiple-testing trials; holdout looks never feed learning.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createExperiments } from "../../experiments.js";

function setup() {
  const dataDir = mkdtempSync(join(tmpdir(), "learn-"));
  mkdirSync(join(dataDir, "Research", "swarm"), { recursive: true });
  writeFileSync(join(dataDir, "Research", "swarm", "design.json"), JSON.stringify({ holdoutStart: "2026-07-09" }));
  const memory = [];
  const quant = { run: async () => ({ ok: true, result: { performance: { sharpe: 1.1, totalReturn: 0.3, maxDrawdown: -0.1 } } }) };
  const ex = createExperiments({ dataDir, quant, memory: { ingest: (i) => memory.push(i) }, priceOf: async () => ({ price: 50 }) });
  return { dataDir, ex, memory };
}
// Minimal express-like app to exercise the real route handlers with the origin header.
function fakeApp() {
  const routes = {};
  const app = { get: (p, h) => (routes["GET " + p] = h), post: (p, h) => (routes["POST " + p] = h) };
  const call = (key, { params = {}, body = {}, origin } = {}) => new Promise((resolve) => {
    const res = { code: 200, status(c) { this.code = c; return this; }, json(v) { resolve({ code: this.code, body: v }); } };
    routes[key]({ params, body, get: (h) => (h.toLowerCase() === "x-tm-origin" ? origin : undefined) }, res);
  });
  return { app, call };
}

test("UI and analyst paths write identical records except for origin", async () => {
  const { dataDir, ex, memory } = setup();
  const { app, call } = fakeApp(); ex.mount(app);
  const ui = await call("POST /api/experiments", { body: { strategy: "tsmom", params: { lookback_days: 126 } } });
  const an = await call("POST /api/experiments", { body: { strategy: "tsmom", params: { lookback_days: 189 } }, origin: "analyst_user_request" });
  assert.equal(ui.code, 200); assert.equal(an.code, 200);
  assert.equal(ui.body.created[0].origin, "user_ui"); assert.equal(an.body.created[0].origin, "analyst_user_request");
  await new Promise((r) => setTimeout(r, 20));
  const reg = readFileSync(join(dataDir, "Research", "experiments", "trial_registry.jsonl"), "utf8").trim().split("\n").map(JSON.parse);
  const kinds = (o) => reg.filter((t) => t.origin === o).map((t) => t.kind).sort();
  assert.deepEqual(kinds("user_ui"), kinds("analyst_user_request"));          // same records
  assert.deepEqual(kinds("analyst_user_request"), ["result", "strategy"]);
  assert.equal(ex.trialCount(), 2);                                             // both count as trials
  assert.ok(memory.some((m) => m.meta.origin === "analyst_user_request"));
  const accounts = JSON.parse(readFileSync(join(dataDir, "Research", "experiments", "accounts.json"), "utf8")).accounts;
  assert.ok(accounts.every((a) => a.id >= 1000 && a.id <= 9999));               // never a live / swarm / crypto / SBIT id
  assert.ok(accounts.every((a) => a.baseline?.end === "2026-07-08"));           // baselines stop before the holdout
});

test("holdout look: ledger entry, excluded from learning, needs confirm", async () => {
  const { dataDir, ex, memory } = setup();
  const before = memory.length;
  await assert.rejects(ex.holdoutLook({ strategy: "tsmom", reason: "check" }, "analyst_user_request"), /confirm/);
  await ex.holdoutLook({ strategy: "tsmom", reason: "final check", confirm: true }, "analyst_user_request");
  const ledger = JSON.parse(readFileSync(join(dataDir, "Research", "swarm", "holdout_ledger.json"), "utf8"));
  assert.equal(ledger.entries.length, 1); assert.equal(ledger.entries[0].origin, "analyst_user_request");
  assert.equal(memory.length, before);                     // nothing from the holdout reaches memory
  assert.equal(ex.trials().length, 0); assert.equal(ex.trialCount(), 0);
  assert.equal(ex.trials({ preHoldoutOnly: false }).length, 1);
});

test("deleting an account keeps its trial counted (no significance laundering)", async () => {
  const { ex } = setup();
  const { created } = ex.create({ strategy: "meanrev", count: 3, sweep: { param: "entry_z", from: 1, to: 2 } }, "analyst_user_request");
  for (const a of created) ex.remove(a.id, "analyst_user_request");
  assert.equal(ex.trialCount(), 3);
});
