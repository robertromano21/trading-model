// Offline tests for nowcast-routes.js: FRED key never returned, .env.local upsert, refresh spawns detached once.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import express from "express";
import { createNowcastRoutes, upsertEnvValue } from "./nowcast-routes.js";

const KEY = "0123456789abcdef0123456789abcdef";

async function withApp(opts, fn) {
  const root = mkdtempSync(join(tmpdir(), "nc-"));
  const dataDir = join(root, "Massive");
  mkdirSync(join(dataDir, "Research", "macro"), { recursive: true });
  writeFileSync(join(root, ".env.local"), "OTHER_KEY=keepme\nFRED_API_KEY=old\n");
  const env = { ...(opts.env || {}) };
  const spawned = [];
  const routes = createNowcastRoutes({ root, dataDir, env, setIntervalImpl: () => ({ unref() {} }),
    fetchImpl: opts.fetchImpl || (async () => ({ ok: true, status: 200, json: async () => ({ seriess: [{}] }) })),
    spawnImpl: (cmd, args, o) => { spawned.push({ cmd, args, o }); return { pid: process.pid, unref() {} }; } });
  const app = express();
  app.use(express.json());
  routes.mount(app);
  const server = app.listen(0);
  const base = `http://127.0.0.1:${server.address().port}`;
  try { await fn({ base, root, dataDir, env, spawned }); } finally { server.close(); routes.stop(); }
}

test("FRED key: saved to .env.local, status only, never echoed", async () => {
  await withApp({ env: { MACRO_NOWCAST_AUTO: "0" } }, async ({ base, root, env }) => {
    let r = await fetch(`${base}/api/keys/fred`).then((x) => x.json());
    assert.equal(r.set, false);
    const bad = await fetch(`${base}/api/keys/fred`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ key: "short" }) });
    assert.equal(bad.status, 400);
    const res = await fetch(`${base}/api/keys/fred`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ key: KEY }) });
    const text = await res.text();
    assert.equal(res.status, 200);
    assert.ok(!text.includes(KEY), "key must never be sent back");
    assert.equal(JSON.parse(text).set, true);
    const file = readFileSync(join(root, ".env.local"), "utf8");
    assert.match(file, /OTHER_KEY=keepme/);
    assert.equal((file.match(/FRED_API_KEY=/g) || []).length, 1);
    assert.ok(file.includes(`FRED_API_KEY=${KEY}`));
    assert.equal(env.FRED_API_KEY, KEY);
    const t = await fetch(`${base}/api/keys/fred/test`, { method: "POST" }).then((x) => x.text());
    assert.ok(!t.includes(KEY));
    assert.equal(JSON.parse(t).ok, true);
  });
});

test("FRED key rejected by FRED is not saved", async () => {
  await withApp({ env: { MACRO_NOWCAST_AUTO: "0" }, fetchImpl: async () => ({ ok: false, status: 400, json: async () => ({}) }) }, async ({ base, root }) => {
    const res = await fetch(`${base}/api/keys/fred`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ key: KEY }) });
    assert.equal(res.status, 400);
    assert.ok(!(await res.text()).includes(KEY));
    assert.ok(readFileSync(join(root, ".env.local"), "utf8").includes("FRED_API_KEY=old"));
  });
});

test("nowcast endpoints and detached refresh", async () => {
  await withApp({ env: { MACRO_NOWCAST_AUTO: "0" } }, async ({ base, dataDir, spawned }) => {
    let r = await fetch(`${base}/api/macro/nowcast`).then((x) => x.json());
    assert.equal(r.available, false);
    const out = join(dataDir, "Research", "macro");
    writeFileSync(join(out, "nowcast_latest.json"), JSON.stringify({ asof: "2026-11-26", nowcasts: [{ target: "CPI_MOM", nowcast: 0.2 }] }));
    writeFileSync(join(out, "nowcast_history.jsonl"), '{"target":"CPI_MOM","nowcast":0.2}\n{"target":"UNRATE","nowcast":4.1}\n');
    r = await fetch(`${base}/api/macro/nowcast`).then((x) => x.json());
    assert.equal(r.latest.nowcasts[0].nowcast, 0.2);
    const h = await fetch(`${base}/api/macro/nowcast/history?target=cpi_mom`).then((x) => x.json());
    assert.equal(h.live.length, 1);
    const a = await fetch(`${base}/api/macro/nowcast/refresh`, { method: "POST" }).then((x) => x.json());
    assert.equal(a.started, true);
    assert.equal(spawned.length, 1);
    assert.equal(spawned[0].o.detached, true);
    assert.deepEqual(spawned[0].args.slice(-2), ["macro_nowcast.py", "refresh"]);
    const b = await fetch(`${base}/api/macro/nowcast/refresh`, { method: "POST" }).then((x) => x.json());
    assert.equal(b.started, false, "second refresh while running is refused");
    assert.ok(existsSync(join(out, "nowcast_refresh.lock")));
  });
});

test("upsertEnvValue keeps other lines", () => {
  const dir = mkdtempSync(join(tmpdir(), "env-"));
  const f = join(dir, ".env.local");
  writeFileSync(f, "A=1\nexport FRED_API_KEY=x\nB=2\n");
  upsertEnvValue(f, "FRED_API_KEY", "y");
  assert.equal(readFileSync(f, "utf8"), "A=1\nB=2\nFRED_API_KEY=y\n");
});
