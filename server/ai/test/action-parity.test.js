// Parity: every HTTP route the dashboard exposes is either an analyst action (ROUTE_ACTIONS), explicitly
// human-only (HUMAN_ONLY), or plumbing (INTERNAL). A new route without a decision fails this test.
// Also: human-only / forbidden routes are never in the action enum, every action has a usable arg spec and example,
// and every write action goes through the Allow / Cancel prompt.
import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { ROUTE_ACTIONS, MATH_ACTIONS, INTERNAL, FORBIDDEN } from "../actionRegistry.js";
import { HUMAN_ONLY } from "../../human-only-routes.js";
import { APP_ACTION } from "../enums.js";
import { createActions } from "../actions.js";

const serverDir = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
function files(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    if (name === "node_modules" || name === "test" || name.startsWith(".")) continue;
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...files(p));
    else if (/\.(m?js)$/.test(name) && !/\.test\.m?js$|\.bak/.test(name)) out.push(p);
  }
  return out;
}
function serverRoutes() {
  const routes = new Set();
  for (const f of files(serverDir)) {
    const text = readFileSync(f, "utf8");
    for (const m of text.matchAll(/\bapp\.(get|post|put|delete|patch)\(\s*["'`](\/[^"'`]+)["'`]/g)) routes.add(`${m[1].toUpperCase()} ${m[2]}`);
  }
  return [...routes].sort();
}

test("every server route is an action, human-only or internal", () => {
  const routes = serverRoutes();
  assert.ok(routes.length > 100, `found only ${routes.length} routes`);
  const covered = new Set([...Object.values(ROUTE_ACTIONS).map((a) => a.route), ...Object.keys(HUMAN_ONLY), ...Object.keys(INTERNAL)]);
  const missing = routes.filter((r) => !covered.has(r));
  assert.deepEqual(missing, [], "routes with no analyst decision (add to actionRegistry.js)");
});

test("registry routes still exist (no stale actions)", () => {
  const routes = new Set(serverRoutes());
  const stale = Object.entries(ROUTE_ACTIONS).filter(([, a]) => !routes.has(a.route)).map(([id, a]) => `${id} ${a.route}`);
  assert.deepEqual(stale, []);
});

test("human-only and forbidden routes are never analyst actions", () => {
  for (const [id, a] of Object.entries(ROUTE_ACTIONS)) {
    assert.ok(!HUMAN_ONLY[a.route], `${id} is human-only`);
    assert.ok(!FORBIDDEN.some((re) => re.test(a.route)), `${id} ${a.route} is forbidden`);
    assert.ok(!/schwab.*order|\/live\/(?!status)|provider-key|app-keys|^POST .*reauth|^POST \/api\/keys\//i.test(a.route), `${id} ${a.route}`);
  }
  assert.ok(!APP_ACTION.some((x) => /LIVE_(ORDER|TOGGLE|PREVIEW)|SCHWAB_ORDER|KEY_SAVE|PROVIDER_KEY/.test(x)));
});

test("every action has a label, example, valid arg spec; enum = registry + meta", () => {
  for (const [id, a] of Object.entries({ ...ROUTE_ACTIONS, ...MATH_ACTIONS })) {
    assert.ok(a.label && a.example, id);
    for (const [name, s] of Object.entries(a.args)) {
      assert.ok(["ticker", "tickers", "coin", "int", "num", "date", "enum", "bool", "json", "text"].includes(s.type), `${id}.${name}: ${s.type}`);
      if (s.type === "enum") assert.ok(s.values.length, `${id}.${name}`);
    }
    assert.ok(APP_ACTION.includes(id), id);
  }
  assert.equal(APP_ACTION.length, Object.keys(ROUTE_ACTIONS).length + Object.keys(MATH_ACTIONS).length + 3);
});

test("every write action returns an Allow / Cancel prompt and never runs on its own", async () => {
  let calls = 0;
  const actions = createActions({ fetchImpl: async () => { calls++; return { ok: true, status: 200, text: async () => "{}" }; } }).attach({ baseUrl: "http://127.0.0.1:1" });
  for (const [id, a] of Object.entries(ROUTE_ACTIONS)) {
    if (a.kind !== "write") continue;
    const args = {};
    for (const [name, s] of Object.entries(a.args)) {
      if (!s.req) continue;
      args[name] = { ticker: "SPY", tickers: ["SPY"], coin: "BTC", int: s.min ?? 1, num: s.min ?? 1, date: "2025-01-02", enum: s.values?.[0], bool: true, json: {}, text: "x" }[s.type];
    }
    const r = await actions.run(id, args);
    assert.equal(r.kind, "confirm", `${id}: ${r.text || ""}`);
    assert.ok(r.summary.title && r.token && r.expiresAt, id);
  }
  assert.equal(calls, 0);
});
