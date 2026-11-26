// Local, gitignored storage for the analyst layer: <dataDir>/ai_research.sqlite3 (Massive/ is gitignored and so is
// *.sqlite3). Holds the timestamped source cache, every verdict (full JSON) and the Q&A log.
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";

const SCHEMA = `
CREATE TABLE IF NOT EXISTS source_cache (
  key TEXT PRIMARY KEY, source TEXT NOT NULL, ticker TEXT, fetched_at TEXT NOT NULL, expires_at TEXT NOT NULL,
  payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ai_verdicts (
  ticker TEXT NOT NULL, created_at TEXT NOT NULL, asof TEXT NOT NULL, source TEXT NOT NULL, model TEXT,
  flag TEXT NOT NULL, conviction TEXT, conviction_score REAL, action TEXT, horizon TEXT, trend TEXT,
  overextension TEXT, overextension_score REAL, fundamental_health TEXT, opportunity TEXT, size_hint TEXT,
  ref_price REAL, entry_low REAL, entry_high REAL, limit_price REAL, order_type TEXT, stop REAL, take_profit REAL,
  atr REAL, diversification TEXT, clamps TEXT, payload TEXT NOT NULL, PRIMARY KEY (ticker, created_at));
CREATE INDEX IF NOT EXISTS ai_verdicts_latest ON ai_verdicts (ticker, created_at DESC);
CREATE TABLE IF NOT EXISTS ai_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, kind TEXT NOT NULL, model TEXT, status TEXT NOT NULL,
  input TEXT, output TEXT, error TEXT, attempts INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER);
`;

export function openStore(dataDir, file = "ai_research.sqlite3") {
  const path = file === ":memory:" ? file : join(dataDir, file);
  if (path !== ":memory:") mkdirSync(dirname(path), { recursive: true });
  const db = new DatabaseSync(path);
  db.exec("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;");
  db.exec(SCHEMA);
  const nowIso = () => new Date().toISOString();

  function cacheGet(key, { allowStale = false } = {}) {
    const row = db.prepare("SELECT fetched_at, expires_at, payload FROM source_cache WHERE key=?").get(key);
    if (!row) return null;
    const stale = row.expires_at < nowIso();
    if (stale && !allowStale) return null;
    return { fetchedAt: row.fetched_at, stale, value: JSON.parse(row.payload) };
  }
  function cachePut(key, source, ticker, value, ttlMs) {
    const fetchedAt = nowIso();
    db.prepare("INSERT OR REPLACE INTO source_cache VALUES (?, ?, ?, ?, ?, ?)").run(key, source, ticker || null, fetchedAt,
      new Date(Date.now() + ttlMs).toISOString(), JSON.stringify(value));
    return { fetchedAt, stale: false, value };
  }
  // Fetch-through cache: fresh hit -> cached; miss -> loader(); loader failure -> stale copy if any, else throw.
  async function cached(key, source, ticker, ttlMs, loader, { refresh = false } = {}) {
    if (!refresh) { const hit = cacheGet(key); if (hit) return { ...hit, cached: true }; }
    try {
      return { ...cachePut(key, source, ticker, await loader(), ttlMs), cached: false };
    } catch (error) {
      const stale = cacheGet(key, { allowStale: true });
      if (stale) return { ...stale, cached: true, stale: true, warning: error.message };
      throw error;
    }
  }

  function saveVerdict(v) {
    db.prepare(`INSERT OR REPLACE INTO ai_verdicts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
      v.ticker, v.createdAt, v.asof, v.source, v.model ?? null, v.flag, v.conviction ?? null, v.convictionScore ?? null,
      v.action ?? null, v.horizon ?? null, v.trend ?? null, v.overextension ?? null, v.overextensionScore ?? null,
      v.fundamentalHealth ?? null, v.opportunity ?? null, v.sizeHint ?? null, v.refPrice ?? null, v.entryLow ?? null,
      v.entryHigh ?? null, v.limitPrice ?? null, v.orderType ?? null, v.stop ?? null, v.takeProfit ?? null, v.atr ?? null,
      v.diversification ?? null, JSON.stringify(v.clamps || []), JSON.stringify(v));
  }
  function latestVerdicts(tickers = null) {
    const rows = db.prepare(`SELECT payload FROM ai_verdicts v WHERE created_at =
      (SELECT MAX(created_at) FROM ai_verdicts w WHERE w.ticker = v.ticker) ORDER BY ticker`).all();
    const all = rows.map((r) => JSON.parse(r.payload));
    return tickers ? all.filter((v) => tickers.includes(v.ticker)) : all;
  }
  function log(entry) {
    db.prepare(`INSERT INTO ai_log (at, kind, model, status, input, output, error, attempts, prompt_tokens, completion_tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(nowIso(), entry.kind, entry.model ?? null, entry.status,
      entry.input == null ? null : JSON.stringify(entry.input), entry.output == null ? null : JSON.stringify(entry.output),
      entry.error ?? null, entry.attempts ?? null, entry.usage?.prompt_tokens ?? null, entry.usage?.completion_tokens ?? null);
  }
  return { db, path, cacheGet, cachePut, cached, saveVerdict, latestVerdicts, log, close: () => db.close() };
}
