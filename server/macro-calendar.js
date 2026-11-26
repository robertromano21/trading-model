// Macro calendar + release-watcher API. Mounted from server.js.
// The watcher is a long-running Python process (engine/macro_watcher.py watch) started with
// the dashboard. It never places broker orders. Times in the JSON are America/New_York (ET).
import { spawn } from "node:child_process";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const MACRO = "Macro";
const RESEARCH = "Research";

function readJson(path) {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return null;
  }
}

function sendFile(res, paths, notFound = "No macro calendar yet. It is built on the first dashboard start or by python macro_service.py refresh.") {
  for (const p of paths) {
    if (existsSync(p)) {
      try {
        res.setHeader("Content-Type", "application/json");
        res.send(readFileSync(p));
        return;
      } catch (error) {
        res.status(500).json({ error: "Could not read macro file." });
        return;
      }
    }
  }
  res.status(404).json({ error: notFound });
}

export function createMacroCalendar({ root, dataDir, python, log = () => {} } = {}) {
  let child = null;
  let starting = false;

  function runPython(script, args, { timeoutMs = 120000 } = {}) {
    return new Promise((resolve, reject) => {
      const proc = spawn(python, ["-u", join(root, "engine", script), ...args], {
        cwd: join(root, "engine"),
        env: { ...process.env, PRICES_DATA_DIR: dataDir, PYTHONUTF8: "1" },
        windowsHide: true,
      });
      let out = "", err = "";
      const timer = setTimeout(() => {
        try { proc.kill(); } catch { /* already exited */ }
        reject(new Error(`${script} timed out`));
      }, timeoutMs);
      proc.stdout.on("data", (d) => { out += d; });
      proc.stderr.on("data", (d) => { err += d; });
      proc.once("error", (error) => {
        clearTimeout(timer);
        reject(new Error(`Could not start Python: ${error.message}`));
      });
      proc.once("close", (code) => {
        clearTimeout(timer);
        resolve({ code, out, err });
      });
    });
  }

  function startWatcher() {
    if (child || starting || process.env.TM_MACRO_WATCHER === "off") return;
    starting = true;
    const script = join(root, "engine", "macro_watcher.py");
    if (!existsSync(script)) {
      starting = false;
      log("warn", "macro watcher skipped: engine/macro_watcher.py is not on disk yet");
      return;
    }
    // Make sure a calendar exists (fallback is enough; official fetch can happen in refresh).
    runPython("macro_service.py", ["refresh"], { timeoutMs: 180000 })
      .catch((error) => log("warn", "macro calendar refresh failed", { error: error.message }))
      .finally(() => {
        const proc = spawn(python, ["-u", script, "watch"], {
          cwd: join(root, "engine"),
          env: { ...process.env, PRICES_DATA_DIR: dataDir, PYTHONUTF8: "1" },
          windowsHide: true, stdio: ["ignore", "pipe", "pipe"],
        });
        child = proc;
        starting = false;
        proc.stdout.on("data", (d) => log("info", "macro watcher", { line: String(d).trim().slice(0, 400) }));
        proc.stderr.on("data", (d) => log("warn", "macro watcher", { line: String(d).trim().slice(0, 400) }));
        proc.once("error", (error) => {
          log("warn", "macro watcher failed to start", { error: error.message });
          child = null;
        });
        proc.once("exit", (code, signal) => {
          log("warn", "macro watcher exited", { code, signal });
          child = null;
        });
        log("info", "macro release watcher started");
      });
  }

  function stopWatcher() {
    if (!child) return;
    try { child.kill(); } catch { /* already gone */ }
    child = null;
  }

  function mount(app) {
    app.get("/api/macro/status", (_req, res) => {
      sendFile(res, [
        join(dataDir, RESEARCH, "macro_status.json"),
        join(dataDir, MACRO, "status_calendar.json"),
      ], "No macro status yet.");
    });
    app.get("/api/macro/calendar", (req, res) => {
      const doc = readJson(join(dataDir, MACRO, "calendar.json"));
      if (!doc) return res.status(404).json({ error: "No calendar cache yet." });
      const from = req.query.from ? String(req.query.from) : null;
      const to = req.query.to ? String(req.query.to) : null;
      let events = doc.events || [];
      if (from || to) {
        events = events.filter((e) => (!from || e.date >= from) && (!to || e.date <= to));
      }
      res.json({ ...doc, events, eventCount: events.length, timezone: "America/New_York" });
    });
    app.get("/api/macro/upcoming", (req, res) => {
      const days = Math.max(1, Math.min(30, Number(req.query.days) || 7));
      const doc = readJson(join(dataDir, MACRO, "calendar.json"));
      if (!doc) return res.status(404).json({ error: "No calendar cache yet." });
      const now = new Date();
      // Compare using the ET ISO strings already stored on each row.
      const upcoming = (doc.events || []).filter((e) => e.datetimeEt && new Date(e.datetimeEt) >= new Date(now.getTime() - 2 * 3600 * 1000)
        && new Date(e.datetimeEt) <= new Date(now.getTime() + days * 86400 * 1000));
      res.json({ timezone: "America/New_York", count: upcoming.length, events: upcoming });
    });
    app.get("/api/macro/releases", (_req, res) => {
      sendFile(res, [join(dataDir, MACRO, "surprises.json")], "No scored releases yet.");
    });
    app.get("/api/macro/surprises", (_req, res) => {
      sendFile(res, [join(dataDir, MACRO, "surprises.json")], "No scored surprises yet.");
    });
    app.get("/api/macro/event-study", (_req, res) => {
      sendFile(res, [
        join(dataDir, RESEARCH, "macro_event_study.json"),
        join(dataDir, MACRO, "event_study.json"),
      ], "No event study yet. Run python macro_service.py study.");
    });
    app.get("/api/macro/overnight", (_req, res) => {
      sendFile(res, [
        join(dataDir, RESEARCH, "macro_overnight_study.json"),
        join(dataDir, MACRO, "overnight_study.json"),
      ], "No overnight study yet.");
    });
    app.get("/api/macro/features", (_req, res) => {
      sendFile(res, [join(dataDir, MACRO, "overnight_features.json")], "No overnight features yet.");
    });
    app.get("/api/macro/watcher", (_req, res) => {
      const status = readJson(join(dataDir, MACRO, "watcher_status.json")) || { state: "never_started" };
      res.json({ ...status, processRunning: Boolean(child) });
    });
    app.get("/api/macro/context", (_req, res) => {
      const ctxPath = join(dataDir, MACRO, "context.txt");
      // Prefer a live Python render so the analyst tool and the UI stay in sync; fall back to files.
      runPython("macro_service.py", ["context"], { timeoutMs: 15000 })
        .then(({ code, out, err }) => {
          if (code !== 0) return res.json({ context: out || err, error: err || null });
          res.json({ context: out, timezone: "America/New_York" });
        })
        .catch((error) => res.status(500).json({ error: error.message }));
    });
    app.post("/api/macro/consensus", (req, res) => {
      const id = String(req.body?.id || req.body?.eventId || "").trim();
      const period = String(req.body?.period || "").trim();
      const value = Number(req.body?.value);
      const field = String(req.body?.field || "headline");
      if (!id || !period || !Number.isFinite(value)) {
        return res.status(400).json({ error: "Need id, period and a numeric value." });
      }
      runPython("macro_service.py", ["consensus", "--id", id, "--period", period, "--value", String(value),
        "--source", "manual", "--field", field], { timeoutMs: 20000 })
        .then(({ code, out, err }) => {
          if (code !== 0) return res.status(500).json({ error: (err || out || "consensus failed").slice(-400) });
          try { res.json(JSON.parse(out.slice(out.indexOf("{")))); }
          catch { res.json({ ok: true, raw: out }); }
        })
        .catch((error) => res.status(500).json({ error: error.message }));
    });
    app.post("/api/macro/refresh", (_req, res) => {
      runPython("macro_service.py", ["refresh"], { timeoutMs: 180000 })
        .then(({ code, out, err }) => {
          if (code !== 0) return res.status(500).json({ error: (err || out || "refresh failed").slice(-400) });
          try { res.json(JSON.parse(out.slice(out.indexOf("{")))); }
          catch { res.json({ ok: true, raw: out }); }
        })
        .catch((error) => res.status(500).json({ error: error.message }));
    });
  }

  return { mount, startWatcher, stopWatcher, runPython };
}

export default createMacroCalendar;
