// Analyst scorecard API (engine/analyst_scoring.py): how the AI analyst's GREEN/YELLOW/RED flags and stops actually
// did 5/20/60 trading days later, and which flag types (if any) the risk engine lets change position sizes.
// Read-only plus an allow-listed refresh job; never touches orders or Schwab. Data stays in Massive/ (gitignored).
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";

export function createAnalystScoring({ root, dataDir, python = "python", log = () => {}, spawnImpl = spawn, env = process.env } = {}) {
  const file = join(dataDir, "Research", "analyst_scorecard.json");
  let running = null;
  let last = null;

  function read() {
    try { return JSON.parse(readFileSync(file, "utf8")); } catch { return null; }
  }

  function refresh() {
    if (running) return running;
    const startedAt = new Date().toISOString();
    running = new Promise((resolve) => {
      let out = "", err = "";
      let child;
      try {
        child = spawnImpl(python, ["-W", "ignore", join(root, "engine", "analyst_scoring.py"), "run"], {
          cwd: join(root, "engine"), env: { ...env, PRICES_DATA_DIR: dataDir, PYTHONUTF8: "1" }, windowsHide: true });
      } catch (error) {
        resolve({ status: "error", message: `Could not start Python: ${error.message}` });
        return;
      }
      child.stdout?.on("data", (d) => { out += d; if (out.length > 1e6) out = out.slice(-5e5); });
      child.stderr?.on("data", (d) => { err += d; if (err.length > 1e5) err = err.slice(-5e4); });
      child.on("error", (error) => resolve({ status: "error", message: `Could not start Python: ${error.message}` }));
      child.on("close", (code) => {
        const line = out.trim().split(/\r?\n/).filter((l) => l.startsWith("{")).pop();
        try { resolve({ ...JSON.parse(line), exitCode: code }); }
        catch { resolve({ status: "error", exitCode: code, message: err.trim().split(/\r?\n/).slice(-1)[0] || "no output" }); }
      });
    }).then((result) => {
      last = { startedAt, finishedAt: new Date().toISOString(), ...result };
      log(result.status === "ok" ? "info" : "warn", "Analyst scorecard refresh finished",
        { status: result.status, calls: result.calls, gateOpen: result.gateOpen, message: result.message });
      return last;
    }).finally(() => { running = null; });
    return running;
  }

  function summary(doc, { ticker = null, limit = 50 } = {}) {
    const { recent = [], ...rest } = doc;
    const list = ticker ? recent.filter((r) => r.ticker === ticker) : recent;
    return { ...rest, recent: list.slice(0, limit), refreshing: Boolean(running), lastRefresh: last };
  }

  function mount(app) {
    app.get("/api/analyst-scorecard", (request, response) => {
      const doc = read();
      if (!doc) {
        response.status(404).json({ error: "No analyst scorecard yet. It is rebuilt before each evening swing paper run, or click Refresh.",
          refreshing: Boolean(running), lastRefresh: last });
        return;
      }
      const ticker = typeof request.query.ticker === "string" && /^[A-Z][A-Z0-9.\-]{0,9}$/.test(request.query.ticker.toUpperCase())
        ? request.query.ticker.toUpperCase() : null;
      const limit = Math.min(200, Math.max(1, Number(request.query.limit) || 50));
      response.json(summary(doc, { ticker, limit }));
    });
    app.get("/api/analyst-scorecard/gate", (_request, response) => {
      const doc = read();
      const ageDays = doc?.generatedAt ? (Date.now() - Date.parse(doc.generatedAt)) / 86400000 : null;
      const maxAge = doc?.params?.maxAgeDays ?? 7;
      const fresh = ageDays != null && ageDays <= maxAge;
      response.json({ generatedAt: doc?.generatedAt ?? null, fresh, gate: fresh ? doc.gate : null,
        gateOpen: fresh ? doc.gateOpen : [], note: "Verdicts change position sizes only for flag types listed in gateOpen; everything else is log-only." });
    });
    app.post("/api/analyst-scorecard/refresh", async (_request, response) => {
      const result = await refresh();
      response.status(result.status === "ok" ? 200 : 500).json(result);
    });
  }

  return { read, refresh, mount, file };
}
