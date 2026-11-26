// Options routes: validation, allow-listed ops only, and no order paths (mocked Python runner, no network).
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import express from "express";
import { cleanLegs, createOptions } from "./options.js";

async function serve(runner) {
  const app = express();
  app.use(express.json());
  const dir = mkdtempSync(join(tmpdir(), "opt-"));
  createOptions({ root: dir, dataDir: dir, python: "python-not-used", runner }).mount(app);
  const server = await new Promise((resolve) => { const s = app.listen(0, "127.0.0.1", () => resolve(s)); });
  const base = `http://127.0.0.1:${server.address().port}`;
  return { base, close: () => server.close() };
}

test("cleanLegs validates kinds, strikes, dates and sizes", () => {
  assert.deepEqual(cleanLegs([{ kind: "put", qty: 1, strike: 100, expiration: "2026-12-18" }]), [{ kind: "put", qty: 1, strike: 100, expiration: "2026-12-18" }]);
  assert.throws(() => cleanLegs([]), /1-6 legs/);
  assert.throws(() => cleanLegs([{ kind: "future", qty: 1 }]), /kind/);
  assert.throws(() => cleanLegs([{ kind: "call", qty: 1, strike: -5, expiration: "2026-12-18" }]), /strike/);
  assert.throws(() => cleanLegs([{ kind: "call", qty: 1, strike: 5, expiration: "12/18/26" }]), /expiration/);
  assert.throws(() => cleanLegs([{ kind: "call", qty: 1000, strike: 5, expiration: "2026-12-18" }]), /qty/);
});

test("routes map to fixed analytics ops and validate input", async () => {
  const calls = [];
  const s = await serve(async (op, args) => { calls.push([op, args]); return { ok: true, result: { op, args } }; });
  try {
    let r = await fetch(`${s.base}/api/options/chain?symbol=spy&prefer=model`);
    assert.equal(r.status, 200);
    assert.equal(calls.at(-1)[0], "options_chain");
    assert.equal(calls.at(-1)[1].symbol, "SPY");
    r = await fetch(`${s.base}/api/options/chain?symbol=${encodeURIComponent("SPY;rm -rf")}`);
    assert.equal(r.status, 400);
    r = await fetch(`${s.base}/api/options/analyze`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol: "SPY", strategy: "bear_put_spread", expiration: "2026-12-18", strikes: { k1: 600, evil: 1 } }) });
    assert.equal(r.status, 200);
    assert.deepEqual(calls.at(-1)[1].strikes, { k1: 600 });
    r = await fetch(`${s.base}/api/options/analyze`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ strategy: "x" }) });
    assert.equal(r.status, 400);
    r = await fetch(`${s.base}/api/options/research`);
    assert.equal(r.status, 200);
    const paper = await (await fetch(`${s.base}/api/options/paper`)).json();
    assert.equal(paper.book, null);
    assert.ok(calls.every(([op]) => ["options_chain", "options_payoff"].includes(op)));
  } finally { s.close(); }
});
