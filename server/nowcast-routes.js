// Macro nowcast (our own GDPNow-style forecast) + FRED key management. Read-only market/macro data:
// nothing here places or previews orders. The FRED key is saved to .env.local (gitignored) and is never logged,
// returned to the browser, or echoed in an error message.
import { spawn } from "node:child_process";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const REFRESH_EVERY_MS = 6 * 3600 * 1000;      // republish if the latest forecast is older than this
const CHECK_EVERY_MS = 30 * 60 * 1000;
export const FRED_KEY_RE = /^[a-z0-9]{32}$/i;

function readJson(path) {
  try { return JSON.parse(readFileSync(path, "utf8")); } catch { return null; }
}

function readJsonl(path, filter = () => true, last = 4000) {
  if (!existsSync(path)) return [];
  const rows = [];
  for (const line of readFileSync(path, "utf8").split(/\r?\n/).filter(Boolean).slice(-last)) {
    try { const row = JSON.parse(line); if (filter(row)) rows.push(row); } catch { /* skip */ }
  }
  return rows;
}

function pidAlive(pid) {
  if (!pid) return false;
  try { process.kill(pid, 0); return true; } catch (error) { return error.code === "EPERM"; }
}

/** Upsert one KEY=value line in .env.local (atomic). Value is never logged. */
export function upsertEnvValue(file, name, value) {
  const lines = existsSync(file) ? readFileSync(file, "utf8").split(/\r?\n/) : [];
  const kept = lines.filter((l) => !new RegExp(`^\\s*(export\\s+)?${name}\\s*=`).test(l));
  while (kept.length && !kept[kept.length - 1].trim()) kept.pop();
  kept.push(`${name}=${value}`, "");
  writeFileSync(`${file}.tmp`, kept.join("\n"), { mode: 0o600 });
  renameSync(`${file}.tmp`, file);
}

export function createNowcastRoutes({ root, dataDir, python = "python", log = () => {}, env = process.env,
  spawnImpl = spawn, fetchImpl = globalThis.fetch, now = () => Date.now(), setIntervalImpl = setInterval } = {}) {
  const outDir = join(dataDir, "Research", "macro");
  const latestPath = join(outDir, "nowcast_latest.json");
  const lockPath = join(outDir, "nowcast_refresh.lock");
  const engineDir = join(root, "engine");
  let timer = null;

  function running() {
    const lock = readJson(lockPath);
    return Boolean(lock && pidAlive(lock.pid) && now() - (lock.startedMs || 0) < 3600 * 1000);
  }

  function refresh(reason = "manual") {
    if (running()) return { started: false, running: true, reason: "A refresh is already running." };
    mkdirSync(outDir, { recursive: true });
    const logFile = join(outDir, "nowcast_refresh.log");
    const fd = openSync(logFile, "a");
    let child;
    try {
      child = spawnImpl(python, ["-W", "ignore", "macro_nowcast.py", "refresh"], {
        cwd: engineDir, detached: true, windowsHide: true, stdio: ["ignore", fd, fd],
        env: { ...env, PRICES_DATA_DIR: dataDir, PYTHONUTF8: "1" },
      });
    } finally {
      try { closeSync(fd); } catch { /* inherited */ }
    }
    child.unref?.();
    writeFileSync(lockPath, JSON.stringify({ pid: child.pid ?? null, startedMs: now(), reason }));
    log("info", "Macro nowcast refresh started", { pid: child.pid, reason });
    return { started: true, running: true, pid: child.pid ?? null };
  }

  function staleMs() {
    try { return now() - statSync(latestPath).mtimeMs; } catch { return Infinity; }
  }

  function startAutoRefresh() {
    if (timer || String(env.MACRO_NOWCAST_AUTO || "1") === "0") return;
    const tick = () => { try { if (staleMs() > REFRESH_EVERY_MS && env.FRED_API_KEY?.trim()) refresh("auto"); } catch (error) { log("warn", "Nowcast auto-refresh failed", { error: error.message }); } };
    timer = setIntervalImpl(tick, CHECK_EVERY_MS);
    timer.unref?.();
    setTimeout(tick, 60 * 1000).unref?.();
  }

  function fredStatus() {
    return { set: Boolean(env.FRED_API_KEY?.trim()), envFile: ".env.local" };
  }

  async function testFred(candidate) {
    const key = String(candidate ?? env.FRED_API_KEY ?? "").trim();
    if (!key) return { ok: false, message: "No FRED key saved yet." };
    if (!FRED_KEY_RE.test(key)) return { ok: false, message: "That does not look like a FRED key (32 letters/digits)." };
    try {
      const url = `https://api.stlouisfed.org/fred/series?series_id=GDP&file_type=json&api_key=${encodeURIComponent(key)}`;
      const response = await fetchImpl(url, { signal: AbortSignal.timeout(15000) });
      const body = await response.json().catch(() => ({}));
      if (response.ok && Array.isArray(body.seriess)) return { ok: true, message: "FRED accepted the key." };
      return { ok: false, message: `FRED rejected the key (HTTP ${response.status}).` };
    } catch (error) {
      return { ok: false, message: `Could not reach FRED (${error.name}).` };
    }
  }

  function mount(app) {
    app.get("/api/macro/nowcast", (_request, response) => {
      const latest = readJson(latestPath);
      const accuracy = readJson(join(outDir, "nowcast_accuracy.json"));
      response.json({
        available: Boolean(latest), latest, running: running(), ageMinutes: Number.isFinite(staleMs()) ? Math.round(staleMs() / 60000) : null,
        fredKey: fredStatus(),
        accuracy: accuracy ? { generatedAt: accuracy.generatedAt, pointInTime: accuracy.pointInTime,
          byTarget: Object.fromEntries(Object.entries(accuracy.accuracy || {}).map(([k, v]) => [k, v.exPandemic || v.final])),
          outside: accuracy.outsideComparison || null } : null,
      });
    });
    app.get("/api/macro/nowcast/history", (request, response) => {
      const target = String(request.query.target || "").toUpperCase().replace(/[^A-Z_]/g, "");
      const live = readJsonl(join(outDir, "nowcast_history.jsonl"), (r) => !target || r.target === target);
      const evo = readJson(join(outDir, "nowcast_evolution.json")) || {};
      response.json({ target: target || null, live: live.slice(-1500),
        backfilled: target ? (evo.targets || {})[target] || [] : evo.targets || {}, note: evo.note || null });
    });
    app.post("/api/macro/nowcast/refresh", (_request, response) => {
      try { response.status(202).json(refresh("dashboard")); } catch (error) { response.status(500).json({ error: error.message }); }
    });
    app.get("/api/keys/fred", (_request, response) => response.json(fredStatus()));
    app.post("/api/keys/fred", async (request, response) => {
      const key = String(request.body?.key ?? "").trim();
      if (!FRED_KEY_RE.test(key)) return response.status(400).json({ error: "A FRED API key is 32 letters and digits (fredaccount.stlouisfed.org → API Keys)." });
      const test = await testFred(key);
      if (!test.ok && request.body?.force !== true) return response.status(400).json({ error: test.message, ...fredStatus() });
      upsertEnvValue(join(root, ".env.local"), "FRED_API_KEY", key);
      env.FRED_API_KEY = key;
      log("info", "FRED API key saved to .env.local", { field: "FRED_API_KEY", tested: test.ok });
      return response.json({ ...fredStatus(), tested: test.ok, message: test.message });
    });
    app.post("/api/keys/fred/test", async (_request, response) => response.json({ ...(await testFred()), ...fredStatus() }));
    startAutoRefresh();
  }

  return { mount, refresh, running, fredStatus, testFred, stop: () => { if (timer) clearInterval(timer); timer = null; } };
}
