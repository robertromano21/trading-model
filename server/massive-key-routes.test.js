// Massive key routes: saved to .env.local, never echoed, format + live-test gate. No network (fetch is faked).
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import express from "express";
import { createMassiveKeyRoutes } from "./massive-key-routes.js";

const root = mkdtempSync(join(tmpdir(), "tm-massive-"));
writeFileSync(join(root, ".env.local"), "OTHER_KEY=keepme\nMASSIVE_API_KEY=oldoldoldoldold1\n");
const env = {};
let changed = 0;
const KEY = "Abcdefghij_klmnop-1234";
const fetchImpl = async (url) => ({ ok: url.includes(encodeURIComponent(KEY)), status: url.includes(encodeURIComponent(KEY)) ? 200 : 401 });
const app = express();
app.use(express.json());
createMassiveKeyRoutes({ root, env, fetchImpl, onChange: () => { changed++; } }).mount(app);
const server = createServer(app);
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;
after(() => { server.close(); rmSync(root, { recursive: true, force: true }); });
const post = (p, body) => fetch(base + p, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

test("status starts not set and never includes the key", async () => {
  const s = await fetch(base + "/api/keys/massive").then((r) => r.json());
  assert.deepEqual(s, { set: false, envFile: ".env.local" });
});

test("bad format and rejected keys are not saved", async () => {
  assert.equal((await post("/api/keys/massive", { key: "short" })).status, 400);
  const r = await post("/api/keys/massive", { key: "Wrongwrongwrong_key-99" });
  assert.equal(r.status, 400);
  assert.match(readFileSync(join(root, ".env.local"), "utf8"), /MASSIVE_API_KEY=oldoldoldoldold1/);
});

test("a good key is saved once, applied to env, and not echoed", async () => {
  const r = await post("/api/keys/massive", { key: KEY });
  const body = await r.text();
  assert.equal(r.status, 200);
  assert.ok(!body.includes(KEY));
  const file = readFileSync(join(root, ".env.local"), "utf8");
  assert.equal((file.match(/MASSIVE_API_KEY=/g) || []).length, 1);
  assert.ok(file.includes(`MASSIVE_API_KEY=${KEY}`) && file.includes("OTHER_KEY=keepme"));
  assert.equal(env.MASSIVE_API_KEY, KEY);
  assert.equal(changed, 1);
  const t = await post("/api/keys/massive/test", {}).then((x) => x.json());
  assert.equal(t.ok, true);
  assert.equal(t.set, true);
});
