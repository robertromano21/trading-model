// Last-window auto-shutdown for the Prices dashboard.
// Clients open an EventSource at /api/lifecycle/presence (and send a heartbeat
// plus sendBeacon on pagehide). The server shuts down only after the last
// client has been gone for a grace period, and never before the first client
// connects. A running Schwab daily backfill or paper daily job defers exit.
const OFF_VALUES = new Set(["off", "0", "false", "no"]);
export const CRITICAL_JOB_NAMES = new Set([
  "schwab_daily_backfill",
  "paper_run",
  "daytrade_run",
  "history",
  "import",
]);

export function parseAutoShutdown(env = process.env) {
  const raw = String(env.DASHBOARD_AUTO_SHUTDOWN ?? "on").trim().toLowerCase();
  const enabled = !OFF_VALUES.has(raw);
  const graceSeconds = Number(env.DASHBOARD_SHUTDOWN_GRACE_SECONDS);
  const busyWaitSeconds = Number(env.DASHBOARD_SHUTDOWN_BUSY_WAIT_SECONDS);
  const graceMs = Math.round((Number.isFinite(graceSeconds) && graceSeconds > 0 ? graceSeconds : 45) * 1000);
  const busyWaitMs = Math.round((Number.isFinite(busyWaitSeconds) && busyWaitSeconds > 0 ? busyWaitSeconds : 1800) * 1000);
  return {
    enabled,
    graceMs: Math.min(Math.max(graceMs, 5_000), 600_000),
    busyWaitMs: Math.min(Math.max(busyWaitMs, 5_000), 86_400_000),
  };
}

export function validClientId(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 80 && /^[A-Za-z0-9._-]+$/.test(value);
}

export function createPresenceTracker({
  now = () => Date.now(),
  graceMs = 45_000,
  busyWaitMs = 1_800_000,
  heartbeatTtlMs = 45_000,
  enabled = true,
  isBusy = () => null,
  onShutdown = () => {},
  log = () => {},
  setTimeoutFn = setTimeout,
  clearTimeoutFn = clearTimeout,
} = {}) {
  const clients = new Map();
  let firstConnectAt = 0;
  let closeTimer = null;
  let busyStartedAt = 0;
  let stopping = false;
  let lastShutdownReason = null;

  function connectedCount() {
    const t = now();
    let n = 0;
    for (const client of clients.values()) {
      if (client.sseCount > 0 || t - client.lastSeen <= heartbeatTtlMs) n += 1;
    }
    return n;
  }

  function snapshot() {
    return {
      enabled,
      stopping,
      connected: connectedCount(),
      firstConnectAt,
      graceMs,
      busyWaitMs,
      shutdownArmed: Boolean(closeTimer),
      lastShutdownReason,
    };
  }

  function cancelTimer() {
    if (!closeTimer) return;
    clearTimeoutFn(closeTimer);
    closeTimer = null;
  }

  function ensure(id) {
    let client = clients.get(id);
    if (!client) {
      client = { sseCount: 0, lastSeen: 0 };
      clients.set(id, client);
    }
    return client;
  }

  function markSeen(id) {
    if (!id || stopping) return snapshot();
    const client = ensure(id);
    client.lastSeen = now();
    firstConnectAt ||= now();
    busyStartedAt = 0;
    cancelTimer();
    return snapshot();
  }

  function openSse(id) {
    if (!id || stopping) return snapshot();
    const client = ensure(id);
    client.sseCount += 1;
    return markSeen(id);
  }

  function closeSse(id) {
    const client = clients.get(id);
    if (!client) {
      maybeArm();
      return snapshot();
    }
    client.sseCount = Math.max(0, client.sseCount - 1);
    if (client.sseCount === 0) clients.delete(id);
    maybeArm();
    return snapshot();
  }

  function heartbeat(id) {
    return markSeen(id);
  }

  function leave(id) {
    if (id) clients.delete(id);
    maybeArm();
    return snapshot();
  }

  function sweep() {
    if (stopping) return snapshot();
    const t = now();
    for (const [id, client] of clients) {
      if (client.sseCount === 0 && t - client.lastSeen > heartbeatTtlMs) clients.delete(id);
    }
    maybeArm();
    return snapshot();
  }

  function maybeArm() {
    if (!enabled || stopping || !firstConnectAt || connectedCount() > 0 || closeTimer) return;
    log("info", "No dashboard clients; scheduling shutdown", {
      shutdownInMs: graceMs, reason: "last-dashboard-window-closed",
    });
    closeTimer = setTimeoutFn(() => { void fire(); }, graceMs);
    closeTimer?.unref?.();
  }

  async function fire() {
    closeTimer = null;
    if (!enabled || stopping || connectedCount() > 0) return;
    const busy = isBusy();
    if (busy) {
      busyStartedAt ||= now();
      const waitedMs = now() - busyStartedAt;
      const critical = busy.critical !== false;
      if (critical || waitedMs < busyWaitMs) {
        if (critical && waitedMs >= busyWaitMs) {
          log("warn", "Shutdown still deferred; a job is running", {
            job: busy.reason, waitedMs, busyWaitMs,
          });
          busyStartedAt = now();
        } else {
          log("info", "Shutdown deferred while a job is running", {
            job: busy.reason, waitedMs, busyWaitMs, critical,
          });
        }
        closeTimer = setTimeoutFn(() => { void fire(); }, graceMs);
        closeTimer?.unref?.();
        return;
      }
      log("warn", "Busy-wait elapsed; shutting down with a non-critical job still running", {
        job: busy.reason, waitedMs,
      });
    }
    busyStartedAt = 0;
    await shutdown("last-dashboard-window-closed");
  }

  async function shutdown(reason = "shutdown") {
    if (stopping) return lastShutdownReason;
    stopping = true;
    lastShutdownReason = reason;
    cancelTimer();
    await onShutdown(reason);
    return reason;
  }

  function dispose() {
    stopping = true;
    cancelTimer();
    clients.clear();
  }

  return {
    openSse,
    closeSse,
    heartbeat,
    leave,
    sweep,
    shutdown,
    dispose,
    snapshot,
    connectedCount,
    isStopping: () => stopping,
    hasSeenClient: () => firstConnectAt > 0,
  };
}

function readClientId(body) {
  if (body && typeof body === "object") return body.clientId;
  if (typeof body === "string" && body.trim()) {
    try { return JSON.parse(body).clientId; } catch { return null; }
  }
  return null;
}

export function installDashboardLifecycle(app, {
  server,
  processRef = process,
  env = process.env,
  isLocal = () => true,
  sameOrigin = () => true,
  isBusy = () => null,
  beforeClose = async () => {},
  flush = async () => {},
  log = { info() {}, warn() {}, error() {} },
  timeoutMs = 8_000,
  setTimeoutFn = setTimeout,
  clearTimeoutFn = clearTimeout,
  now = () => Date.now(),
} = {}) {
  const config = parseAutoShutdown(env);
  const httpSockets = new Set();
  let shutdownPromise = null;
  let disposed = false;
  const listenPort = () => {
    try { return server?.address?.()?.port ?? Number(env.PRICES_PORT || 3003); } catch { return Number(env.PRICES_PORT || 3003); }
  };

  if (server) {
    server.on("connection", (socket) => {
      httpSockets.add(socket);
      socket.once("close", () => httpSockets.delete(socket));
    });
  }

  const tracker = createPresenceTracker({
    now,
    graceMs: config.graceMs,
    busyWaitMs: config.busyWaitMs,
    enabled: config.enabled,
    isBusy,
    log: (level, message, meta) => (log[level] || log.info || (() => {}))(message, meta),
    setTimeoutFn,
    clearTimeoutFn,
    onShutdown: (reason) => shutdown(reason),
  });

  const sweepTimer = setTimeoutFn ? setInterval(() => tracker.sweep(), 15_000) : null;
  sweepTimer?.unref?.();

  function refuse(request, response) {
    if (!isLocal(request)) {
      response.status(403).json({ error: "Use the local dashboard address." });
      return true;
    }
    if (request.headers.origin && !sameOrigin(request)) {
      response.status(403).json({ error: "Lifecycle requests must come from the dashboard." });
      return true;
    }
    return false;
  }

  function shutdown(reason = "shutdown", code = 0) {
    if (shutdownPromise) return shutdownPromise;
    if (sweepTimer) clearInterval(sweepTimer);
    log.info?.("Dashboard shutting down", { reason, pid: processRef.pid, port: listenPort() });
    shutdownPromise = new Promise((resolve) => {
      let finished = false;
      let flushing = false;
      const cleanup = Promise.resolve().then(beforeClose).catch((error) => {
        log.error?.("Dashboard shutdown cleanup failed", { error: error.message, reason });
      });
      const deadline = setTimeoutFn(() => finish(), timeoutMs);
      function finish() {
        if (finished) return;
        finished = true;
        clearTimeoutFn(deadline);
        for (const socket of httpSockets) {
          try { socket.destroy(); } catch { /* already closed */ }
        }
        resolve();
        processRef.exit(code);
      }
      function drain() {
        if (flushing || finished) return;
        flushing = true;
        for (const socket of httpSockets) {
          try { socket.destroy(); } catch { /* already closed */ }
        }
        cleanup.then(flush).catch((error) => {
          log.error?.("Dashboard shutdown flush failed", { error: error.message, reason });
        }).finally(finish);
      }
      if (server) {
        server.close(drain);
        server.closeIdleConnections?.();
        server.closeAllConnections?.();
      } else {
        drain();
      }
    });
    return shutdownPromise;
  }

  app.get("/api/lifecycle", (request, response) => {
    if (!isLocal(request)) return response.status(403).json({ error: "Use the local dashboard address." });
    response.json({
      running: !tracker.isStopping(),
      pid: processRef.pid,
      port: listenPort(),
      autoShutdown: config.enabled,
      ...tracker.snapshot(),
    });
  });

  app.get("/api/lifecycle/presence", (request, response) => {
    if (refuse(request, response)) return;
    if (tracker.isStopping() || disposed) return response.status(503).end();
    const id = validClientId(request.query.clientId) ? request.query.clientId : `sse-${now()}-${Math.random().toString(36).slice(2, 10)}`;
    response.setHeader("Content-Type", "text/event-stream");
    response.setHeader("Cache-Control", "no-store");
    response.setHeader("X-Accel-Buffering", "no");
    response.write("retry: 1000\n: dashboard window connected\n\n");
    tracker.openSse(id);
    const heartbeat = setInterval(() => {
      try { response.write(": dashboard window alive\n\n"); } catch { /* closed */ }
    }, 15_000);
    heartbeat.unref?.();
    response.once("close", () => {
      clearInterval(heartbeat);
      tracker.closeSse(id);
    });
  });

  function handlePing(kind) {
    return (request, response) => {
      if (refuse(request, response)) return;
      if (tracker.isStopping() || disposed) return response.status(503).end();
      const id = validClientId(readClientId(request.body)) ? readClientId(request.body) : null;
      if (!id) return response.status(400).json({ error: "clientId is required." });
      if (kind === "leave") tracker.leave(id);
      else tracker.heartbeat(id);
      response.sendStatus(204);
    };
  }
  app.post("/api/lifecycle/heartbeat", handlePing("heartbeat"));
  app.post("/api/lifecycle/leave", handlePing("leave"));

  const onSignal = (signal) => { void shutdown(signal); };
  const handlers = Object.fromEntries(["SIGINT", "SIGTERM", "SIGHUP"].map((signal) => [signal, () => onSignal(signal)]));
  for (const [signal, handler] of Object.entries(handlers)) processRef.on(signal, handler);

  function dispose() {
    disposed = true;
    if (sweepTimer) clearInterval(sweepTimer);
    tracker.dispose();
    for (const [signal, handler] of Object.entries(handlers)) processRef.removeListener(signal, handler);
  }

  if (config.enabled) {
    log.info?.("Dashboard auto-shutdown armed", {
      graceSeconds: Math.round(config.graceMs / 1000),
      busyWaitSeconds: Math.round(config.busyWaitMs / 1000),
    });
  } else {
    log.info?.("Dashboard auto-shutdown disabled", { env: "DASHBOARD_AUTO_SHUTDOWN" });
  }

  return { shutdown, dispose, tracker, config, isStopping: () => tracker.isStopping() };
}
