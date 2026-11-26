// Analyst memory: local MiniLM vector store in SQLite (<dataDir>/ai_memory.sqlite3, gitignored).
// Retrieval: the last few exchanges verbatim + older relevant pieces
// chosen by MiniLM similarity (lexical fallback after RECALL_TIMEOUT_MS) inside a FIXED character budget (~2,500
// tokens), so prompt size never grows with history. Added for trading: ticker/kind/time filters, recency weighting
// per kind, near-duplicate suppression, and "how it played out" (price then -> price now) for past views/decisions.
//
// Public API (also used by the paper-trader decision / chat-log writer):
//   const memory = getMemory({ dataDir, modelDir });
//   await memory.ingest({ kind, ticker, text, ts, meta })   -> { id, duplicate }
//   await memory.recall({ query, tickers, kinds, since, budgetChars, keepRecent, prices })
//   memory.search({ ... })  (same as recall but returns ranked rows, no rendering)
import { createHash } from "node:crypto";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { createEmbedder, DIMS } from "./embedder.js";
import { lexical, passages } from "./lexical.js";

export const KINDS = Object.freeze(["QA", "CHAT", "VERDICT", "DECISION", "NEWS", "FUNDAMENTALS", "NOTE", "EVENT"]);
export const HALF_LIFE_DAYS = Object.freeze({ QA: 30, CHAT: 21, VERDICT: 45, DECISION: 60, NEWS: 7, FUNDAMENTALS: 120, NOTE: 365, EVENT: 30 });
const envInt = (name, dflt) => { const v = Number.parseInt(process.env[name] ?? "", 10); return Number.isFinite(v) && v >= 0 ? v : dflt; };
export const TRACK_RECORD_VERDICTS = envInt("TRADING_AI_TRACK_RECORD_VERDICTS", 15); // latest N verdicts per ticker, always shown
export const TRACK_RECORD_DECISIONS = envInt("TRADING_AI_TRACK_RECORD_DECISIONS", 3); // latest N paper-trader decisions per ticker
export const TRACK_LINE_REASON_CHARS = 90;  // one-line reason on each compact track-record line (~140 chars/line)
export const BUDGET_CHARS = 10000;       // ~2,500 tokens TOTAL for the whole [MEMORY] block (fixed; never grows with history)
export const PINNED_BUDGET_CHARS = 4000; // part of it: per-ticker track record (15 verdicts + 3 decisions ≈ 2,500 chars for one ticker)
export const RECENT_BUDGET_CHARS = 2000; // part of it: last few Q&A exchanges, verbatim (trimmed); older relevant items get the rest (≥ 4,000)
export const KEEP_RECENT = 3;
export const RECALL_TIMEOUT_MS = 2500;

const SCHEMA = `
CREATE TABLE IF NOT EXISTS memory_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, ticker TEXT, ts TEXT NOT NULL, text TEXT NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}', hash TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
  embedded INTEGER NOT NULL DEFAULT 0, model_key TEXT);
CREATE INDEX IF NOT EXISTS memory_items_ticker ON memory_items (ticker, ts);
CREATE INDEX IF NOT EXISTS memory_items_kind ON memory_items (kind, ts);
CREATE INDEX IF NOT EXISTS memory_items_pending ON memory_items (embedded);
CREATE TABLE IF NOT EXISTS memory_chunks (
  item_id INTEGER NOT NULL, idx INTEGER NOT NULL, text TEXT NOT NULL, vector BLOB NOT NULL,
  PRIMARY KEY (item_id, idx)) WITHOUT ROWID;
`;

// Same formula as server/ai/memory_ingest.py (so Python writers dedupe identically).
export function memoryHash(kind, ticker, text, dedupeKey) {
  const key = dedupeKey != null && dedupeKey !== "" ? String(dedupeKey) : String(text).replace(/\s+/g, " ").trim();
  return createHash("sha256").update(`${kind}|${ticker || ""}|${key}`).digest("hex");
}
const toBlob = (v) => Buffer.from(v.buffer, v.byteOffset, v.byteLength);
const fromBlob = (b) => new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const pct = (x) => (x >= 0 ? "+" : "") + (x * 100).toFixed(1) + "%";

export function createMemory({ dataDir, file = "ai_memory.sqlite3", modelDir, embed = null, embedder = null,
  logger = () => {}, now = () => Date.now(), recallTimeoutMs = RECALL_TIMEOUT_MS, autoEmbed = true } = {}) {
  const path = file === ":memory:" ? file : join(dataDir, file);
  if (path !== ":memory:") mkdirSync(dataDir, { recursive: true });
  const db = new DatabaseSync(path);
  db.exec("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;");
  db.exec(SCHEMA);
  const enc = embedder || createEmbedder({ modelDir, embed, logger });
  let embedding = null, timer = null;

  function ingest({ kind, ticker = null, text, ts = null, meta = {} } = {}) {
    kind = String(kind || "").toUpperCase();
    if (!KINDS.includes(kind)) throw Object.assign(Error(`Unknown memory kind ${kind}. Use one of ${KINDS.join(", ")}.`), { status: 400 });
    const body = String(text ?? "").trim();
    if (!body) throw Object.assign(Error("Memory text is empty."), { status: 400 });
    ticker = ticker ? String(ticker).toUpperCase().trim() : null;
    const when = ts ? new Date(ts) : new Date(now());
    if (Number.isNaN(when.getTime())) throw Object.assign(Error("Invalid ts."), { status: 400 });
    const hash = memoryHash(kind, ticker, body, meta?.dedupeKey);
    const result = db.prepare(`INSERT OR IGNORE INTO memory_items (kind, ticker, ts, text, meta, hash, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)`).run(kind, ticker, when.toISOString(), body, JSON.stringify(meta || {}), hash, new Date(now()).toISOString());
    const duplicate = result.changes === 0;
    const id = duplicate ? db.prepare("SELECT id FROM memory_items WHERE hash=?").get(hash).id : Number(result.lastInsertRowid);
    if (!duplicate && autoEmbed) schedule();
    return { id, duplicate };
  }

  function schedule() {
    if (timer || !enc.configured) return;
    timer = setTimeout(() => { timer = null; embedPending().catch(() => {}); }, 50);
    timer.unref?.();
  }

  // Embed items that have no vectors yet (also rows written directly by Python). Safe to call any time.
  async function embedPending(limit = 256) {
    if (!enc.configured) return { embedded: 0, reason: "model not installed" };
    if (embedding) return embedding;
    embedding = (async () => {
      let total = 0;
      for (;;) {
        await enc.start();
        const key = enc.key();
        const rows = db.prepare("SELECT id, kind, ticker, ts, text FROM memory_items WHERE embedded=0 OR model_key IS NOT ? ORDER BY id LIMIT ?").all(key, 32);
        if (!rows.length || total >= limit) break;
        const pieces = rows.flatMap((r) => passages(r.text).map((p, idx) => ({ id: r.id, idx, text: `${r.ticker || ""} ${r.kind} ${r.ts.slice(0, 10)}: ${p}` })));
        const vectors = await enc.vectorize(pieces.map((p) => p.text));
        db.exec("BEGIN");
        try {
          for (const r of rows) db.prepare("DELETE FROM memory_chunks WHERE item_id=?").run(r.id);
          pieces.forEach((p, i) => db.prepare("INSERT INTO memory_chunks VALUES (?, ?, ?, ?)").run(p.id, p.idx, p.text, toBlob(vectors[i])));
          for (const r of rows) db.prepare("UPDATE memory_items SET embedded=1, model_key=? WHERE id=?").run(key, r.id);
          db.exec("COMMIT");
        } catch (error) { db.exec("ROLLBACK"); throw error; }
        total += rows.length;
      }
      return { embedded: total };
    })().finally(() => { embedding = null; });
    return embedding;
  }

  function candidates({ tickers = [], kinds = null, since = null, excludeIds = [] }) {
    const where = [], params = [];
    if (tickers.length) { where.push(`(ticker IN (${tickers.map(() => "?").join(",")}) OR ticker IS NULL)`); params.push(...tickers); }
    if (kinds?.length) { where.push(`kind IN (${kinds.map(() => "?").join(",")})`); params.push(...kinds); }
    if (since) { where.push("ts >= ?"); params.push(new Date(since).toISOString()); }
    const rows = db.prepare(`SELECT id, kind, ticker, ts, text, meta, embedded FROM memory_items ${where.length ? "WHERE " + where.join(" AND ") : ""}
      ORDER BY ts DESC LIMIT 3000`).all(...params);
    const skip = new Set(excludeIds);
    return rows.filter((r) => !skip.has(r.id)).map((r) => ({ ...r, meta: JSON.parse(r.meta || "{}") }));
  }

  async function semanticScores(query, rows) {
    if (!enc.configured || !rows.some((r) => r.embedded)) return null;
    const work = (async () => {
      const [q] = await enc.vectorize([query.slice(0, 1200)]);
      const ids = rows.filter((r) => r.embedded).map((r) => r.id);
      const scores = new Map(), vecs = new Map();
      for (let i = 0; i < ids.length; i += 500) {
        const slice = ids.slice(i, i + 500);
        for (const c of db.prepare(`SELECT item_id, vector FROM memory_chunks WHERE item_id IN (${slice.map(() => "?").join(",")})`).all(...slice)) {
          const v = fromBlob(c.vector);
          if (v.length !== DIMS) continue;
          const s = dot(q, v);
          if (s > (scores.get(c.item_id) ?? -1)) { scores.set(c.item_id, s); vecs.set(c.item_id, v); }
        }
      }
      return { scores, vecs };
    })();
    let t;
    const result = await Promise.race([work, new Promise((r) => { t = setTimeout(() => r(null), recallTimeoutMs); })]).catch(() => null);
    clearTimeout(t);
    return result;
  }

  // Ranked rows: relevance (MiniLM cosine, else normalised lexical) + recency (per-kind half-life) + ticker match.
  async function search({ query = "", tickers = [], kinds = null, since = null, limit = 40, excludeIds = [] } = {}) {
    tickers = tickers.map((t) => String(t).toUpperCase());
    const rows = candidates({ tickers, kinds, since, excludeIds });
    if (!rows.length) return { rows: [], method: "none" };
    let method = "semantic", sem = query.trim() ? await semanticScores(query, rows) : null;
    const lex = lexical(query, rows), maxLex = Math.max(1e-9, ...lex.values());
    if (!sem) method = query.trim() ? "lexical" : "recency";
    const t0 = now();
    const scored = rows.map((r) => {
      const ageDays = Math.max(0, (t0 - Date.parse(r.ts)) / 86400000);
      const recency = Math.pow(0.5, ageDays / (HALF_LIFE_DAYS[r.kind] || 30));
      const relevance = sem?.scores.has(r.id) ? sem.scores.get(r.id) : (lex.get(r.id) || 0) / maxLex * (sem ? 0.5 : 0.8);
      const tickerBoost = r.ticker && tickers.includes(r.ticker) ? 0.08 : 0;
      return { ...r, relevance, recency, score: relevance + 0.15 * recency + tickerBoost, vector: sem?.vecs.get(r.id) || null };
    }).filter((r) => !query.trim() || r.relevance >= (sem ? 0.2 : 0.05));
    scored.sort((a, b) => b.score - a.score);
    return { rows: scored.slice(0, limit), method };
  }

  function outcome(r, prices) {
    const ref = Number(r.meta?.refPrice), nowPrice = r.ticker ? Number(prices?.[r.ticker]) : NaN;
    if (!(ref > 0) || !(nowPrice > 0)) return "";
    return ` -> then ${ref.toFixed(2)}, now ${nowPrice.toFixed(2)} (${pct(nowPrice / ref - 1)})`;
  }
  // Compact track-record line: date, flag/action, price then -> now %, one-line reason.
  function trackLine(r, prices) {
    const m = r.meta || {};
    const tag = [m.flag, m.action || m.side].filter(Boolean).join("/");
    const ref = Number(m.refPrice), nowPrice = Number(prices?.[r.ticker]);
    const move = ref > 0 ? (nowPrice > 0 ? ` ${ref.toFixed(2)} -> ${nowPrice.toFixed(2)} (${pct(nowPrice / ref - 1)})` : ` @ ${ref.toFixed(2)}`) : "";
    let reason = String(m.reason || r.text).replace(/\s+/g, " ").replace(/^Flagged .*?stop \S+?\.\s+/, "").trim();
    reason = reason.split(/(?<=[.!?])\s/)[0];
    if (reason.length > TRACK_LINE_REASON_CHARS) reason = reason.slice(0, TRACK_LINE_REASON_CHARS - 1).trimEnd() + "…";
    return `- ${[r.ts.slice(0, 10), r.kind === "VERDICT" ? "" : r.kind, tag].filter(Boolean).join(" ")}${move}${reason ? ": " + reason : ""}`;
  }
  const line = (r, prices, maxText) => `- [${r.ts.slice(0, 10)} ${r.kind}${r.ticker ? " " + r.ticker : ""}]${outcome(r, prices)} ${r.text.replace(/\s+/g, " ").slice(0, maxText)}`;

  /** Fixed-budget memory block for a prompt. Never throws: any failure returns an empty block. */
  async function recall({ query = "", tickers = [], kinds = null, since = null, budgetChars = BUDGET_CHARS,
    keepRecent = KEEP_RECENT, recentBudgetChars = RECENT_BUDGET_CHARS, trackVerdicts = TRACK_RECORD_VERDICTS,
    trackDecisions = TRACK_RECORD_DECISIONS, pinnedBudgetChars = PINNED_BUDGET_CHARS, prices = {}, maxItemChars = 700 } = {}) {
    try {
      tickers = [...new Set(tickers.map((t) => String(t).toUpperCase()))];
      // 0) Track record per ticker: latest N verdicts (+ latest decisions), compact lines with then -> now.
      //    The pinned budget is split evenly across tickers, so many tickers never grow the block.
      const pinned = [], pinnedLines = []; let usedPinned = 0;
      const pinCap = Math.min(pinnedBudgetChars, budgetChars);
      const perTicker = tickers.length ? Math.floor(pinCap / tickers.length) : 0;
      for (const t of tickers) {
        const v = trackVerdicts > 0 ? candidates({ tickers: [t], kinds: ["VERDICT"] }).filter((r) => r.ticker === t).slice(0, trackVerdicts) : [];
        const d = trackDecisions > 0 ? candidates({ tickers: [t], kinds: ["DECISION"] }).filter((r) => r.ticker === t).slice(0, trackDecisions) : [];
        if (!v.length && !d.length) continue;
        const head = `${t} track record (latest first):`;
        let used = head.length + 1; const lines = [];
        for (const r of [...v, ...d]) {
          const l = trackLine(r, prices);
          if (used + l.length + 1 > perTicker) break;
          lines.push(l); pinned.push(r); used += l.length + 1;
        }
        if (lines.length) { pinnedLines.push(head, ...lines); usedPinned += used; }
      }
      // 1) Most recent Q&A exchanges about these tickers, verbatim (trimmed), a last-turns window.
      const recent = keepRecent > 0 ? candidates({ tickers, kinds: ["QA"] }).filter((r) => !tickers.length || r.ticker == null || tickers.includes(r.ticker)).slice(0, keepRecent) : [];
      const recentLines = []; let usedRecent = 0;
      const recentCap = Math.min(recentBudgetChars, budgetChars - usedPinned);
      for (const r of recent) {
        const l = line(r, prices, Math.max(200, Math.floor(recentCap / Math.max(1, recent.length)) - 60));
        if (usedRecent + l.length + 1 > recentCap) break;
        recentLines.push(l); usedRecent += l.length + 1;
      }
      const olderBudget = Math.max(0, budgetChars - usedPinned - usedRecent);
      // 2) Older relevant pieces under the fixed budget, near-duplicates suppressed.
      const { rows, method } = await search({ query, tickers, kinds, since, excludeIds: [...recent, ...pinned].map((r) => r.id), limit: 80 });
      const picked = []; let used = 0;
      for (const r of rows) {
        if (picked.some((p) => (p.vector && r.vector && dot(p.vector, r.vector) > 0.92) || p.text === r.text)) continue;
        const l = line(r, prices, maxItemChars);
        if (used + l.length + 1 > olderBudget) continue;
        picked.push({ ...r, rendered: l }); used += l.length + 1;
      }
      picked.sort((a, b) => a.ts.localeCompare(b.ts));
      const parts = [];
      if (pinnedLines.length) parts.push(pinnedLines.join("\n"));
      if (picked.length) parts.push(`Older relevant memory (${method}; oldest first):\n` + picked.map((p) => p.rendered).join("\n"));
      if (recentLines.length) parts.push("Most recent exchanges:\n" + recentLines.join("\n"));
      const text = parts.length ? `[MEMORY] Your earlier views, paper-trader decisions, notes and news (dated; "then/now" shows the price move since). Current data blocks take priority.\n${parts.join("\n")}` : "";
      return { text, method, items: [...pinned, ...picked, ...recent].map((r) => ({ id: r.id, kind: r.kind, ticker: r.ticker, ts: r.ts, score: r.score ?? null })), chars: text.length };
    } catch (error) {
      logger("warn", "AI memory recall skipped", { error: error.message });
      return { text: "", method: "error", items: [], chars: 0, error: error.message };
    }
  }

  function stats() {
    const byKind = Object.fromEntries(db.prepare("SELECT kind, COUNT(*) n FROM memory_items GROUP BY kind").all().map((r) => [r.kind, r.n]));
    const pending = db.prepare("SELECT COUNT(*) n FROM memory_items WHERE embedded=0").get().n;
    return { path, items: Object.values(byKind).reduce((a, b) => a + b, 0), byKind, pending, embedder: enc.status() };
  }
  function list({ ticker, kind, limit = 50 } = {}) {
    const where = [], params = [];
    if (ticker) { where.push("ticker=?"); params.push(String(ticker).toUpperCase()); }
    if (kind) { where.push("kind=?"); params.push(String(kind).toUpperCase()); }
    return db.prepare(`SELECT id, kind, ticker, ts, text, meta, embedded FROM memory_items ${where.length ? "WHERE " + where.join(" AND ") : ""} ORDER BY ts DESC LIMIT ?`)
      .all(...params, Math.min(500, limit)).map((r) => ({ ...r, meta: JSON.parse(r.meta) }));
  }
  function remove(id) { db.prepare("DELETE FROM memory_chunks WHERE item_id=?").run(id); return db.prepare("DELETE FROM memory_items WHERE id=?").run(id).changes; }

  if (autoEmbed && enc.configured) { const t = setInterval(() => embedPending().catch(() => {}), 60000); t.unref?.(); }
  return { ingest, recall, search, embedPending, stats, list, remove, db, path,
    close: async () => { await enc.close?.(); db.close(); } };
}

const instances = new Map();
/** Process-wide singleton per data directory (dashboard routes, flag jobs and the chat-log writer share it). */
export function getMemory({ dataDir, modelDir, logger } = {}) {
  const key = join(dataDir, "ai_memory.sqlite3");
  if (!instances.has(key)) instances.set(key, createMemory({ dataDir, modelDir, logger }));
  return instances.get(key);
}
