// Massive (market data) API key: status / save / test. The key is saved to the gitignored .env.local as MASSIVE_API_KEY,
// applied to this process's environment immediately (new engine jobs inherit it), and is never logged or sent back
// to the browser: responses only say whether a key is set.
import { join } from "node:path";
import { upsertEnvValue } from "./nowcast-routes.js";

export const MASSIVE_KEY_RE = /^[A-Za-z0-9_\-]{16,80}$/;

export function createMassiveKeyRoutes({ root, env = process.env, fetchImpl = globalThis.fetch, log = () => {}, onChange = () => {} } = {}) {
  const status = () => ({ set: Boolean(env.MASSIVE_API_KEY?.trim()), envFile: ".env.local" });

  async function test(candidate) {
    const key = String(candidate ?? env.MASSIVE_API_KEY ?? "").trim();
    if (!key) return { ok: false, message: "No Massive key saved yet." };
    if (!MASSIVE_KEY_RE.test(key)) return { ok: false, message: "That does not look like a Massive API key." };
    try {
      const url = `https://api.massive.com/v3/reference/tickers?ticker=SPY&limit=1&apiKey=${encodeURIComponent(key)}`;
      const response = await fetchImpl(url, { signal: AbortSignal.timeout(15000) });
      if (response.ok) return { ok: true, message: "Massive accepted the key." };
      if (response.status === 401 || response.status === 403) return { ok: false, message: `Massive rejected the key (HTTP ${response.status}).` };
      return { ok: false, message: `Massive answered HTTP ${response.status}; the key was not confirmed.` };
    } catch (error) {
      return { ok: false, message: `Could not reach Massive (${error.name}).` };
    }
  }

  function mount(app) {
    app.get("/api/keys/massive", (_request, response) => response.json(status()));
    app.post("/api/keys/massive", async (request, response) => {
      const key = String(request.body?.key ?? "").trim();
      if (!MASSIVE_KEY_RE.test(key)) return response.status(400).json({ error: "Paste the API key from your Massive dashboard (letters, digits, - or _).", ...status() });
      const result = await test(key);
      if (!result.ok && request.body?.force !== true) return response.status(400).json({ error: result.message, ...status() });
      upsertEnvValue(join(root, ".env.local"), "MASSIVE_API_KEY", key);
      env.MASSIVE_API_KEY = key;
      try { onChange(true); } catch { /* status refresh only */ }
      log("info", "Massive API key saved to .env.local", { field: "MASSIVE_API_KEY", tested: result.ok });
      return response.json({ ...status(), tested: result.ok, message: result.message });
    });
    app.post("/api/keys/massive/test", async (_request, response) => response.json({ ...(await test()), ...status() }));
  }

  return { mount, status, test };
}
