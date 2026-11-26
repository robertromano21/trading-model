// Unit tests for the dashboard's last-window auto-shutdown.
// Run: node --test dashboard-lifecycle.test.mjs
import assert from "node:assert/strict";
import http from "node:http";
import test from "node:test";
import express from "express";
import {
  parseAutoShutdown,
  validClientId,
  createPresenceTracker,
  installDashboardLifecycle,
  CRITICAL_JOB_NAMES,
} from "./dashboard-lifecycle.js";

function fakeClock() {
  let t = 1_000_000;
  const timers = new Map();
  let nextId = 1;
  const now = () => t;
  const setTimeoutFn = (fn, ms) => {
    const id = nextId++;
    timers.set(id, { fn, at: t + Number(ms) });
    return id;
  };
  const clearTimeoutFn = (id) => { timers.delete(id); };
  const advance = (ms) => {
    const target = t + ms;
    let guard = 0;
    for (;;) {
      const due = [...timers.entries()].filter(([, timer]) => timer.at <= target).sort((a, b) => a[1].at - b[1].at);
      if (!due.length) { t = target; break; }
      if (++guard > 1000) throw new Error("timer loop");
      const [id, timer] = due[0];
      timers.delete(id);
      t = timer.at;
      timer.fn();
    }
  };
  return { now, setTimeoutFn, clearTimeoutFn, advance, get t() { return t; }, pending: () => timers.size };
}

function trackerOf(clock, extra = {}) {
  const shutdowns = [];
  const logs = [];
  const state = { busy: null };
  const tracker = createPresenceTracker({
    now: clock.now,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
    graceMs: 45_000,
    busyWaitMs: 120_000,
    heartbeatTtlMs: 45_000,
    isBusy: () => state.busy,
    onShutdown: (reason) => { shutdowns.push(reason); },
    log: (level, message, meta) => logs.push({ level, message, meta }),
    ...extra,
  });
  return { tracker, shutdowns, logs, state };
}

test("parseAutoShutdown reads env knobs and clamps", () => {
  assert.deepEqual(parseAutoShutdown({}), { enabled: true, graceMs: 45_000, busyWaitMs: 1_800_000 });
  assert.equal(parseAutoShutdown({ DASHBOARD_AUTO_SHUTDOWN: "off" }).enabled, false);
  assert.equal(parseAutoShutdown({ DASHBOARD_AUTO_SHUTDOWN: "0" }).enabled, false);
  assert.equal(parseAutoShutdown({ DASHBOARD_AUTO_SHUTDOWN: "false" }).enabled, false);
  assert.equal(parseAutoShutdown({ DASHBOARD_SHUTDOWN_GRACE_SECONDS: "30" }).graceMs, 30_000);
  assert.equal(parseAutoShutdown({ DASHBOARD_SHUTDOWN_BUSY_WAIT_SECONDS: "60" }).busyWaitMs, 60_000);
  assert.equal(parseAutoShutdown({ DASHBOARD_SHUTDOWN_GRACE_SECONDS: "1" }).graceMs, 5_000);
  assert.ok(CRITICAL_JOB_NAMES.has("schwab_daily_backfill"));
  assert.ok(CRITICAL_JOB_NAMES.has("paper_run"));
  assert.equal(validClientId("a1b2-c3"), true);
  assert.equal(validClientId(""), false);
  assert.equal(validClientId("no spaces"), false);
});

test("does not shut down before the first client connects", () => {
  const clock = fakeClock();
  const { tracker, shutdowns } = trackerOf(clock);
  clock.advance(600_000);
  tracker.sweep();
  assert.equal(tracker.hasSeenClient(), false);
  assert.equal(tracker.snapshot().shutdownArmed, false);
  assert.equal(shutdowns.length, 0);
});

test("last client close arms grace, then shuts down with a reason", () => {
  const clock = fakeClock();
  const { tracker, shutdowns } = trackerOf(clock);
  tracker.openSse("tab-1");
  assert.equal(tracker.connectedCount(), 1);
  tracker.closeSse("tab-1");
  assert.equal(tracker.snapshot().shutdownArmed, true);
  assert.equal(shutdowns.length, 0);
  clock.advance(44_000);
  assert.equal(shutdowns.length, 0);
  clock.advance(1_000);
  assert.deepEqual(shutdowns, ["last-dashboard-window-closed"]);
  assert.equal(tracker.isStopping(), true);
});

test("a reconnect during the grace period cancels shutdown", () => {
  const clock = fakeClock();
  const { tracker, shutdowns } = trackerOf(clock);
  tracker.openSse("tab-1");
  tracker.closeSse("tab-1");
  clock.advance(20_000);
  tracker.openSse("tab-2");
  assert.equal(tracker.snapshot().shutdownArmed, false);
  clock.advance(60_000);
  assert.equal(shutdowns.length, 0);
  assert.equal(tracker.connectedCount(), 1);
});

test("closing one of two tabs does not shut down", () => {
  const clock = fakeClock();
  const { tracker, shutdowns } = trackerOf(clock);
  tracker.openSse("tab-1");
  tracker.openSse("tab-2");
  tracker.closeSse("tab-1");
  clock.advance(60_000);
  assert.equal(shutdowns.length, 0);
  assert.equal(tracker.connectedCount(), 1);
  tracker.leave("tab-2");
  clock.advance(45_000);
  assert.deepEqual(shutdowns, ["last-dashboard-window-closed"]);
});

test("heartbeat leave and stale-heartbeat sweep both release the last client", () => {
  const clock = fakeClock();
  const { tracker, shutdowns } = trackerOf(clock);
  tracker.heartbeat("hb-1");
  tracker.leave("hb-1");
  clock.advance(45_000);
  assert.deepEqual(shutdowns, ["last-dashboard-window-closed"]);

  const clock2 = fakeClock();
  const second = trackerOf(clock2);
  second.tracker.heartbeat("hb-2");
  clock2.advance(45_001);
  second.tracker.sweep();
  assert.equal(second.tracker.connectedCount(), 0);
  clock2.advance(45_000);
  assert.deepEqual(second.shutdowns, ["last-dashboard-window-closed"]);
});

test("an SSE client stays connected without heartbeats", () => {
  const clock = fakeClock();
  const { tracker, shutdowns } = trackerOf(clock);
  tracker.openSse("sse-1");
  clock.advance(120_000);
  tracker.sweep();
  assert.equal(tracker.connectedCount(), 1);
  assert.equal(shutdowns.length, 0);
});

test("DASHBOARD_AUTO_SHUTDOWN=off never shuts down", () => {
  const clock = fakeClock();
  const { tracker, shutdowns } = trackerOf(clock, { enabled: false });
  tracker.openSse("tab-1");
  tracker.closeSse("tab-1");
  clock.advance(600_000);
  assert.equal(shutdowns.length, 0);
  assert.equal(tracker.snapshot().shutdownArmed, false);
});

test("critical backfill defers shutdown past the busy-wait cap", () => {
  const clock = fakeClock();
  const { tracker, shutdowns, state, logs } = trackerOf(clock);
  state.busy = { reason: "Schwab: long daily history backfill", critical: true };
  tracker.openSse("tab-1");
  tracker.closeSse("tab-1");
  clock.advance(45_000);
  assert.equal(shutdowns.length, 0);
  clock.advance(120_000);
  clock.advance(45_000);
  clock.advance(45_000);
  assert.equal(shutdowns.length, 0);
  assert.ok(logs.some((row) => row.message.includes("still deferred") || row.message.includes("Deferred") || row.message.includes("deferred")));
  state.busy = null;
  clock.advance(45_000);
  assert.deepEqual(shutdowns, ["last-dashboard-window-closed"]);
});

test("a non-critical job defers until the busy-wait elapses, then shuts down", () => {
  const clock = fakeClock();
  const { tracker, shutdowns, state } = trackerOf(clock);
  state.busy = { reason: "Research: walk-forward", critical: false };
  tracker.openSse("tab-1");
  tracker.closeSse("tab-1");
  clock.advance(45_000);
  assert.equal(shutdowns.length, 0);
  clock.advance(134_000);
  assert.equal(shutdowns.length, 0);
  clock.advance(1_000);
  assert.deepEqual(shutdowns, ["last-dashboard-window-closed"]);
});

test("lifecycle HTTP routes accept heartbeat and leave, and refuse remote hosts", async () => {
  const app = express();
  const server = http.createServer(app);
  const exits = [];
  const processRef = { pid: 4242, exit: (code) => exits.push(code), on() {}, removeListener() {} };
  app.use(express.json({ limit: "1kb" }));
  const installed = installDashboardLifecycle(app, {
    server,
    processRef,
    env: { DASHBOARD_AUTO_SHUTDOWN: "on", DASHBOARD_SHUTDOWN_GRACE_SECONDS: "45", PRICES_PORT: "3000" },
    isLocal: (request) => String(request.headers.host || "").startsWith("127.0.0.1:") || request.headers.host === "localhost:3000",
    sameOrigin: (request) => !request.headers.origin || request.headers.origin === "http://localhost:3000" || request.headers.origin.startsWith("http://127.0.0.1:"),
    log: { info() {}, warn() {}, error() {} },
    timeoutMs: 50,
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  const base = `http://127.0.0.1:${port}`;
  try {
    const remote = await new Promise((resolve, reject) => {
      const req = http.request({ hostname: "127.0.0.1", port, path: "/api/lifecycle", method: "GET",
        headers: { host: "evil.example:80" } }, (res) => { res.resume(); res.on("end", () => resolve(res)); });
      req.on("error", reject);
      req.end();
    });
    assert.equal(remote.statusCode, 403);

    const beat = await fetch(`${base}/api/lifecycle/heartbeat`, {
      method: "POST",
      headers: { origin: `http://127.0.0.1:${port}`, "content-type": "application/json" },
      body: JSON.stringify({ clientId: "client-1" }),
    });
    assert.equal(beat.status, 204);
    const status = await fetch(`${base}/api/lifecycle`);
    const body = await status.json();
    assert.equal(body.connected, 1);
    assert.equal(body.autoShutdown, true);
    assert.equal(body.hasOwnProperty("running"), true);

    const leave = await fetch(`${base}/api/lifecycle/leave`, {
      method: "POST",
      headers: { origin: `http://127.0.0.1:${port}`, "content-type": "application/json" },
      body: JSON.stringify({ clientId: "client-1" }),
    });
    assert.equal(leave.status, 204);
    const after = await (await fetch(`${base}/api/lifecycle`)).json();
    assert.equal(after.connected, 0);
    assert.equal(after.shutdownArmed, true);
  } finally {
    installed.dispose();
    await new Promise((resolve) => server.close(resolve));
  }
  assert.equal(exits.length, 0);
});
