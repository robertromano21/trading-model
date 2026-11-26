// Runner for engine/quant_tools.py: JSON in on stdin -> JSON out on stdout, one short-lived process per call.
// Not a code runner: Python only accepts an operation name from its fixed OPS table plus plain arguments.
// Timeouts kill the process; output size is bounded; every failure resolves to { ok: false, error } (never throws).
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";

export const MAX_OUTPUT_BYTES = 32 * 1024 * 1024;

export function pythonFor(root, fallback) {
  const venv = join(root, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  return existsSync(venv) ? venv : fallback || (process.platform === "win32" ? "python" : "python3");
}

export function createQuantRunner({ root, python, script = join(root, "engine", "quant_tools.py"), spawnImpl = spawn } = {}) {
  const exe = pythonFor(root, python);
  function run(op, args = {}, { timeoutMs = 120000 } = {}) {
    const started = Date.now();
    return new Promise((resolve) => {
      let child;
      try {
        child = spawnImpl(exe, [script], { cwd: join(root, "engine"), windowsHide: true,
          env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" } });
      } catch (error) {
        resolve({ ok: false, op, error: { code: "SPAWN_FAILED", message: error.message }, ms: Date.now() - started });
        return;
      }
      const out = [], err = [];
      let size = 0, done = false;
      const finish = (value) => { if (done) return; done = true; clearTimeout(timer); resolve({ ...value, ms: Date.now() - started }); };
      const timer = setTimeout(() => {
        try { child.kill(); } catch { /* already gone */ }
        finish({ ok: false, op, error: { code: "TIMEOUT", message: `${op} did not finish within ${Math.round(timeoutMs / 1000)} s` } });
      }, timeoutMs);
      child.stdout.on("data", (chunk) => {
        size += chunk.length;
        if (size > MAX_OUTPUT_BYTES) { try { child.kill(); } catch { /* ignore */ } finish({ ok: false, op, error: { code: "OUTPUT_TOO_LARGE", message: "result exceeded 32 MB" } }); }
        else out.push(chunk);
      });
      child.stderr.on("data", (chunk) => { if (err.length < 200) err.push(chunk); });
      child.on("error", (error) => finish({ ok: false, op, error: { code: "SPAWN_FAILED", message: error.message } }));
      child.on("close", () => {
        const text = Buffer.concat(out).toString("utf8").trim();
        try {
          const parsed = JSON.parse(text);
          finish(parsed && typeof parsed === "object" ? parsed : { ok: false, op, error: { code: "BAD_OUTPUT", message: "non-object output" } });
        } catch {
          const stderr = Buffer.concat(err).toString("utf8").trim().split(/\r?\n/).slice(-3).join(" | ");
          finish({ ok: false, op, error: { code: "BAD_OUTPUT", message: stderr || text.slice(0, 300) || "no output" } });
        }
      });
      child.stdin.on("error", () => { /* process died early; close handler reports */ });
      child.stdin.end(JSON.stringify({ op, args }));
    });
  }
  return { run, python: exe, script };
}
