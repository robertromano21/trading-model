// Alert API and speech-queue tests. No real orders, no dashboard restart.
// Run from server/:  node --test alerts/alerts.test.mjs
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { after, test } from "node:test";
import express from "express";
import { emitAlert, mountAlerts } from "./index.js";
import { createAlertQueue, inQuietHours, shouldPop, shouldSpeak } from "./queue.js";
import { normalizeType, templateSpeech } from "./phrase.js";

const app = express();
app.use(express.json());
mountAlerts(app, { log: () => {}, voice: false });
const server = createServer(app);
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const { port } = server.address();
const base = `http://127.0.0.1:${port}`;
after(() => new Promise((resolve) => server.close(resolve)));

async function post(path, body) {
  const res = await fetch(base + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: res.status, json: await res.json() };
}

test("normalizeType accepts aliases used by paper trader and AI", () => {
  assert.equal(normalizeType("fill"), "fill");
  assert.equal(normalizeType("order-fill"), "fill");
  assert.equal(normalizeType("paper_trade"), "decision");
  assert.equal(normalizeType("AI flag"), "flag_change");
  assert.equal(normalizeType("Schwab auth needed"), "schwab_auth");
  assert.equal(normalizeType("nope"), null);
  assert.equal(normalizeType("drift"), "drift");
  assert.equal(normalizeType("backtest_drift"), "drift");
  assert.equal(normalizeType("macro_release"), "macro_release");
  assert.equal(normalizeType("economic_release"), "macro_release");
  assert.equal(normalizeType("dilution"), "dilution");
  assert.equal(normalizeType("share_dilution"), "dilution");
  assert.equal(normalizeType("black_swan"), "black_swan");
  assert.equal(normalizeType("geopolitical"), "black_swan");
});

test("template speech is short, first-person, and uses ticker + flag", () => {
  const fill = templateSpeech({ type: "fill", ticker: "IEF", meta: { side: "buy", qty: 18, price: 96.2 } });
  assert.match(fill, /IEF/);
  assert.match(fill, /buy/);
  assert.match(fill, /18/);
  const flag = templateSpeech({ type: "flag_change", ticker: "NVDA", meta: { flag: "RED" } });
  assert.match(flag, /NVDA/);
  assert.match(flag, /red/);
  const halt = templateSpeech({ type: "circuit_breaker", text: "Daily loss limit reached." });
  assert.match(halt, /halted/i);
  const warn = templateSpeech({ type: "drift", severity: "warning", meta: { asof: "2026-11-24" }, text: "Turnover is high." });
  assert.match(warn, /usual range/i);
  assert.match(warn, /2026-11-24/);
  const severe = templateSpeech({ type: "drift", severity: "critical", meta: { asof: "2026-11-24" }, text: "Daily return is outside the 1 to 99 percent band." });
  assert.match(severe, /serious drift/i);
  const macro = templateSpeech({ type: "macro_release", text: "Large surprise. CPI is out at 0.4 versus 0.2 (z +2.20).", meta: { eventId: "cpi", surpriseZ: 2.2 } });
  assert.match(macro, /CPI|surprise/i);
  const dil = templateSpeech({ type: "dilution", ticker: "XYZ", text: "Split-adjusted share count is up 12.0% year over year.", meta: { shareChangeYoY: 0.12 } });
  assert.match(dil, /XYZ/);
  assert.match(dil, /dilution/i);
  const swan = templateSpeech({ type: "black_swan", text: "Confirmed negative shock (war, severity 5/5).", meta: { state: "SHOCK" } });
  assert.match(swan, /shock|war|black swan/i);
});

test("POST /api/alerts rejects unknown types and accepts a fill", async () => {
  const bad = await post("/api/alerts", { type: "explode", text: "nope" });
  assert.equal(bad.status, 400);
  const ok = await post("/api/alerts", {
    type: "fill", ticker: "ief", text: "Bought 18 IEF",
    meta: { side: "buy", qty: 18, price: 96.2 },
  });
  assert.equal(ok.status, 200);
  assert.equal(ok.json.ok, true);
  assert.equal(ok.json.alert.type, "fill");
  assert.equal(ok.json.alert.ticker, "IEF");
  assert.equal(ok.json.alert.href, "/ticker.html?symbol=IEF");
  assert.match(ok.json.alert.speech, /IEF/);
});

test("POST /api/alerts/emit is an alias and recent list contains the alert", async () => {
  const ok = await post("/api/alerts/emit", { type: "stop_hit", ticker: "NVDA", meta: { price: 121.49 } });
  assert.equal(ok.status, 200);
  const recent = await fetch(base + "/api/alerts/recent?limit=10").then((r) => r.json());
  assert.ok(recent.alerts.some((a) => a.id === ok.json.alert.id));
});

test("SSE /api/alerts/stream sends hello and a live alert", async () => {
  const ac = new AbortController();
  const res = await fetch(base + "/api/alerts/stream", { signal: ac.signal, headers: { Accept: "text/event-stream" } });
  assert.equal(res.status, 200);
  assert.match(res.headers.get("content-type"), /text\/event-stream/);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const saw = { hello: false, live: false };
  const wait = (async () => {
    while (!saw.live) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      if (buf.includes("event: hello")) saw.hello = true;
      if (buf.includes("circuit_breaker") && buf.includes("event: alert")) saw.live = true;
    }
  })();
  await new Promise((r) => setTimeout(r, 150));
  const emitted = await post("/api/alerts", { type: "circuit_breaker", text: "Daily loss limit reached." });
  assert.equal(emitted.status, 200);
  await Promise.race([wait, new Promise((_, rej) => setTimeout(() => rej(new Error("sse timeout: " + buf.slice(0, 400))), 3000))]);
  ac.abort();
  assert.equal(saw.hello, true);
  assert.equal(saw.live, true);
});

test("emitAlert helper matches the HTTP path", async () => {
  const result = await emitAlert({ type: "schwab_auth" });
  assert.equal(result.ok, true);
  assert.match(result.alert.speech, /Schwab authorization/i);
});

test("queue speaks jobs in order and never overlaps", async () => {
  const seen = [];
  let overlapping = 0;
  let active = 0;
  const queue = createAlertQueue({
    speak: async (alert) => {
      active += 1;
      if (active > 1) overlapping += 1;
      seen.push(alert.id);
      await new Promise((r) => setTimeout(r, 30));
      active -= 1;
    },
  });
  await Promise.all([
    queue.enqueue({ id: "a", type: "fill" }, { speak: true }),
    queue.enqueue({ id: "b", type: "fill" }, { speak: true }),
    queue.enqueue({ id: "c", type: "fill" }, { speak: true }),
  ]);
  assert.deepEqual(seen, ["a", "b", "c"]);
  assert.equal(overlapping, 0);
  assert.equal(queue.busy, false);
  assert.equal(queue.length, 0);
});

test("shouldSpeak respects Voice off, quiet hours, and per-type flags", () => {
  const alert = { type: "fill" };
  assert.equal(shouldSpeak(alert, { voiceMode: "speech" }), true);
  assert.equal(shouldSpeak(alert, { voiceMode: "off" }), false);
  assert.equal(shouldSpeak(alert, { voiceMode: "speech", speakTypes: { fill: false } }), false);
  assert.equal(shouldPop(alert, { popTypes: { fill: false } }), false);
  const quiet = { voiceMode: "speech", quietHoursEnabled: true, quietStart: "21:00", quietEnd: "07:00" };
  assert.equal(inQuietHours(quiet, new Date(2026, 10, 25, 22, 0, 0)), true);
  assert.equal(inQuietHours(quiet, new Date(2026, 10, 25, 10, 0, 0)), false);
  assert.equal(shouldSpeak(alert, quiet, new Date(2026, 10, 25, 22, 0, 0)), false);
  assert.equal(shouldSpeak(alert, quiet, new Date(2026, 10, 25, 10, 0, 0)), true);
});

test("POST /api/alerts accepts a severe drift event and phrases it in the first person", async () => {
  const ok = await post("/api/alerts", {
    type: "drift",
    severity: "critical",
    text: "Daily return is outside the 1 to 99 percent band.",
    meta: { asof: "2026-11-24", breaches: ["daily_return"] },
    href: "/paper2.html#drift",
  });
  assert.equal(ok.status, 200);
  assert.equal(ok.json.alert.type, "drift");
  assert.equal(ok.json.alert.severity, "critical");
  assert.equal(ok.json.alert.href, "/paper2.html#drift");
  assert.match(ok.json.alert.speech, /serious drift/i);
});

test("status endpoint reports voice without throwing when models may be absent", async () => {
  const json = await fetch(base + "/api/alerts/status").then((r) => r.json());
  assert.equal(json.ok, true);
  assert.ok(json.voice);
  assert.ok("available" in json.voice);
  assert.ok("female" in json.voice.models);
  assert.ok("male" in json.voice.models);
});
