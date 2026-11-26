// Model registry + provider key storage for the AI analyst layer.
// Model catalog (non-reasoning models) and provider key handling. Key order: an environment key (OPENAI_API_KEY / XAI_API_KEY / GROQ_API_KEY, incl. .env.local) always
// wins; otherwise a key entered in the UI is saved to <TRADING_AI_KEY_DIR or <project>/keys>/<provider>.key
// (gitignored, mode 0600, atomic write) and loaded automatically next start. Keys are never logged, never
// returned to the browser, and never written to browser storage.
import { existsSync, lstatSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { join } from "node:path";

export const CATALOG = Object.freeze([
  // [id, name, provider, apiModel, reasoningNone, note]
  ["gpt-6-luna", "GPT-6 Luna", "openai", "gpt-6-luna", true, "Efficient, high-volume model (default). Reasoning off."],
  ["gpt-6-sol", "GPT-6 Sol", "openai", "gpt-6-sol", true, "Higher-capability GPT-6. Reasoning off."],
  ["gpt-5.6-luna", "GPT-5.6 Luna", "openai", "gpt-5.6-luna", true, "Cost-sensitive GPT-5.6. Reasoning off."],
  ["gpt-5.6-terra", "GPT-5.6 Terra", "openai", "gpt-5.6-terra", true, "Balanced GPT-5.6. Reasoning off."],
  ["gpt-4.1", "GPT-4.1", "openai", "gpt-4.1", false, "Legacy non-reasoning model."],
  ["gpt-4.1-mini", "GPT-4.1 Mini", "openai", "gpt-4.1-mini", false, "Legacy non-reasoning model."],
  ["grok-4.20-0309-non-reasoning", "Grok 4.20 (non-reasoning)", "xai", "grok-4.20-0309-non-reasoning", false,
    "xAI non-reasoning model; needs an xAI key."],
  ["groq-gpt-oss-120b", "GPT-OSS 120B (Groq)", "groq", "openai/gpt-oss-120b", false,
    "Groq-hosted open-weight model (low reasoning effort); needs a Groq key.", "low"],
].map(([id, name, provider, apiModel, reasoningNone, note, reasoningEffort]) =>
  Object.freeze({ id, name, provider, apiModel, reasoningNone, note, reasoningEffort: reasoningEffort || null })));

export const KEY_ENV = Object.freeze({ openai: "OPENAI_API_KEY", xai: "XAI_API_KEY", groq: "GROQ_API_KEY" });
export const BASE_URL = Object.freeze({
  openai: () => process.env.TRADING_AI_OPENAI_URL || "https://api.openai.com/v1",
  xai: () => process.env.TRADING_AI_XAI_URL || "https://api.x.ai/v1",
  groq: () => "https://api.groq.com/openai/v1",
});
export const DEFAULT_MODEL = "gpt-6-luna";
export const PROVIDER_ORDER = Object.freeze(["openai", "xai", "groq"]);

const problem = (message, status = 400) => Object.assign(new Error(message), { status });

export function createModelService({ root, env = process.env, settingsPath } = {}) {
  const keySource = {};
  const keyDir = () => env.TRADING_AI_KEY_DIR || join(root, "keys");
  const keyPath = (provider) => join(keyDir(), provider + ".key");
  let selected = null;

  function loadSavedKeys() {
    for (const [provider, name] of Object.entries(KEY_ENV)) {
      if (env[name] && keySource[provider] !== "file") { keySource[provider] = "env"; continue; }
      try {
        const stat = lstatSync(keyPath(provider));
        if (stat.isSymbolicLink()) { keySource[provider] = keySource[provider] || "none"; continue; }
        const key = readFileSync(keyPath(provider), "utf8").trim();
        if (key) { env[name] = key; keySource[provider] = "file"; continue; }
      } catch { /* no saved key */ }
      keySource[provider] = env[name] ? "env" : "none";
    }
  }

  function saveKey(provider, key, { persistOnly = false } = {}) {
    const name = KEY_ENV[provider];
    if (!name) throw problem("Unknown provider.");
    const environmentWins = keySource[provider] === "env";
    if (environmentWins && !persistOnly) throw problem("This key is set by the server environment; change it there.", 409);
    const value = String(key || "").trim();
    if (!/^[\x21-\x7e]{20,300}$/.test(value)) throw problem("That does not look like an API key.");
    mkdirSync(keyDir(), { recursive: true, mode: 0o700 });
    if (lstatSync(keyDir()).isSymbolicLink() || (existsSync(keyPath(provider)) && lstatSync(keyPath(provider)).isSymbolicLink()))
      throw problem("Unsafe local key storage.", 409);
    const temporary = keyPath(provider) + "." + randomUUID() + ".tmp";
    try {
      writeFileSync(temporary, value + "\n", { mode: 0o600, flag: "wx" });
      renameSync(temporary, keyPath(provider));
    } finally { if (existsSync(temporary)) unlinkSync(temporary); }
    if (!environmentWins) { env[name] = value; keySource[provider] = "file"; }
    return keys();
  }

  const keys = () => Object.fromEntries(Object.keys(KEY_ENV).map((p) => [p, keySource[p] || "none"]));
  const storedKeys = () => Object.fromEntries(Object.keys(KEY_ENV).map((p) => {
    try { const s = lstatSync(keyPath(p)); return [p, s.isFile() && !s.isSymbolicLink() && s.size > 0]; } catch { return [p, false]; }
  }));
  const hasKey = (provider) => Boolean(env[KEY_ENV[provider]]);
  const apiKey = (provider) => env[KEY_ENV[provider]] || null; // used only to build the Authorization header

  function readSettings() {
    try { return JSON.parse(readFileSync(settingsPath, "utf8")); } catch { return {}; }
  }
  function writeSettings(values) {
    if (!settingsPath) return;
    try {
      const temporary = settingsPath + "." + randomUUID() + ".tmp";
      writeFileSync(temporary, JSON.stringify({ ...readSettings(), ...values }, null, 2));
      renameSync(temporary, settingsPath);
    } catch { /* settings are a convenience only */ }
  }

  function profile(id) {
    const row = CATALOG.find((r) => r.id === id);
    if (!row) throw problem("Unsupported model. Choose one from the model list.");
    return row;
  }

  // Selected model: explicit UI choice > TRADING_AI_MODEL > default Luna; if that provider has no key, the first
  // model whose provider has one (so a single xAI key works without extra setup).
  function current() {
    const wanted = selected || readSettings().model || env.TRADING_AI_MODEL || DEFAULT_MODEL;
    let row = CATALOG.find((r) => r.id === wanted) || profile(DEFAULT_MODEL);
    if (!hasKey(row.provider)) {
      const usable = CATALOG.find((r) => hasKey(r.provider));
      if (usable && !selected) row = usable;
    }
    return { ...row, ready: hasKey(row.provider) };
  }
  function select(id) {
    const row = profile(id);
    selected = row.id;
    writeSettings({ model: row.id });
    return current();
  }
  function view() {
    return {
      current: current(),
      keys: keys(),
      storedKeys: storedKeys(),
      all: CATALOG.map((r) => ({ id: r.id, name: r.name, provider: r.provider, note: r.note,
        status: hasKey(r.provider) ? "ready" : "needs key" })),
    };
  }

  loadSavedKeys();
  return { profile, current, select, view, saveKey, keys, storedKeys, hasKey, apiKey, loadSavedKeys, keyDir };
}
