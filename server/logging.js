// Logging toolkit (server logger with rotation, request ids and redaction):
// * request ids: every request gets X-Request-Id (client-supplied if valid) kept in AsyncLocalStorage, added to
//   every server log line written while handling it, and passed to Python children as TM_REQUEST_ID;
// * redaction of secrets (env key values, bearer/OAuth tokens, sk-/xai-/gsk- keys, Schwab account numbers) and
//   compaction (media payloads dropped, >16 KB messages truncated head+tail);
// * console capture: console.* also goes to the rotating server log;
// * process error capture: uncaughtExceptionMonitor + unhandledRejection;
// * browser capture endpoint POST /api/client-logs (console errors, unhandled rejections, fetch failures, UI actions)
//   written to logs/browser.YYYY-MM-DD.log; Python scripts write logs/engine.YYYY-MM-DD.log (engine/structured_log.py);
// * viewer API GET /api/log-viewer: one or all streams, level/source/request-id/text filters.
// Rotation/retention: winston-daily-rotate-file (10 MB parts, 14 days) for server/browser; the Python logger mirrors it.
import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";
import { createReadStream, readdirSync } from "node:fs";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { formatWithOptions } from "node:util";

export const MAX_MESSAGE_LENGTH = 16384;
const SECRET_ENV = ["MASSIVE_API_KEY", "SCHWAB_APP_KEY", "SCHWAB_APP_SECRET", "OPENAI_API_KEY", "XAI_API_KEY",
  "GROQ_API_KEY", "ANTHROPIC_API_KEY"];
const SECRET_KEY = /(token|secret|password|api_?key|authorization|cookie|hash_?value|account_?number)/i;
const PATTERNS = [
  [/(bearer\s+)[A-Za-z0-9._~+/=-]{8,}/gi, "$1[redacted]"],
  [/((?:api[_-]?key|apikey|access_token|refresh_token|id_token|client_secret|code|token|secret)["']?\s*[:=]\s*["']?)[^"'&\s,}]{6,}/gi, "$1[redacted]"],
  [/\b(sk|xai|gsk)-[A-Za-z0-9_-]{10,}/g, "$1-[redacted]"],
  [/(account(?:number|_number|hash|id)?["']?\s*[:=]\s*["']?)[A-Za-z0-9]{4,}([A-Za-z0-9]{4})\b/gi, "$1…$2"],
];
export const STREAMS = ["server", "browser", "engine", "browser-sync"];
const LEVEL_RANK = { debug: 10, verbose: 10, info: 20, http: 20, warn: 30, error: 40 };

export function createLogKit({ logsDir, env = process.env } = {}) {
  const context = new AsyncLocalStorage();

  function redact(value) {
    let text = String(value);
    for (const name of SECRET_ENV) {
      const secret = env[name]?.trim();
      if (secret && secret.length >= 6) text = text.split(secret).join("[redacted]");
    }
    for (const [pattern, replacement] of PATTERNS) text = text.replace(pattern, replacement);
    return text;
  }
  function compact(text) {
    const clean = String(text).replace(/data:(?:image|audio)\/[a-z0-9.+-]+;base64,[a-z0-9+/=]+/gi, "[media payload omitted]");
    if (clean.length <= MAX_MESSAGE_LENGTH) return clean;
    const note = `\n[Log truncated: ${clean.length} characters]\n`;
    return clean.slice(0, MAX_MESSAGE_LENGTH - 2048 - note.length) + note + clean.slice(-2048);
  }
  function scrub(value, depth = 0) {
    if (value == null || typeof value === "number" || typeof value === "boolean") return value;
    if (typeof value === "string") return compact(redact(value));
    if (value instanceof Error) return { name: value.name, message: scrub(value.message), stack: scrub(value.stack ?? "") };
    if (depth > 4) return "[depth]";
    if (Array.isArray(value)) return value.slice(0, 100).map((v) => scrub(v, depth + 1));
    if (typeof value === "object") {
      return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, SECRET_KEY.test(k) ? "[redacted]" : scrub(v, depth + 1)]));
    }
    return scrub(String(value));
  }

  // winston format: adds requestId from the async context and redacts every field
  function format(winston) {
    return winston.format((info) => {
      const store = context.getStore();
      if (store?.requestId && !info.requestId) info.requestId = store.requestId;
      for (const key of Object.keys(info)) {
        if (key === "level" || key === "timestamp") continue;
        info[key] = SECRET_KEY.test(key) ? "[redacted]" : scrub(info[key]);
      }
      return info;
    })();
  }

  function requestIds(request, response, next) {
    const supplied = request.headers["x-request-id"];
    const id = typeof supplied === "string" && /^[\w-]{8,64}$/.test(supplied) ? supplied : randomUUID();
    request.id = id;
    response.setHeader("X-Request-Id", id);
    context.run({ requestId: id }, next);
  }
  const currentRequestId = () => context.getStore()?.requestId;

  // console.* also lands in the rotating server log (fileLogger has no Console transport)
  function captureConsole(fileLogger) {
    const methods = { log: "info", info: "info", warn: "warn", error: "error", debug: "debug" };
    const originals = {};
    for (const [method, level] of Object.entries(methods)) {
      originals[method] = console[method];
      console[method] = (...args) => {
        originals[method].apply(console, args);
        try { fileLogger.log(level, compact(formatWithOptions({ colors: false }, ...args)), { source: "console" }); } catch { /* never throw */ }
      };
    }
    return () => Object.assign(console, originals);
  }

  function captureProcessErrors(logger) {
    const onException = (error, origin) => logger.error("Uncaught exception", { source: "process", origin, error });
    const onRejection = (reason) => logger.error("Unhandled promise rejection", { source: "process",
      error: reason instanceof Error ? reason : { message: String(reason) } });
    process.on("uncaughtExceptionMonitor", onException);
    process.on("unhandledRejection", onRejection);
    return () => { process.off("uncaughtExceptionMonitor", onException); process.off("unhandledRejection", onRejection); };
  }

  // ------------------------------------------------------------------ browser log intake
  const ALLOWED_KINDS = new Set(["console", "error", "rejection", "fetch", "action", "navigation", "info"]);
  function clientLogsHandler(browserLogger) {
    return (request, response) => {
      const entries = Array.isArray(request.body?.entries) ? request.body.entries : [request.body];
      if (!entries.length || entries.length > 50) return response.status(400).json({ error: "Send 1-50 log entries." });
      let written = 0;
      for (const e of entries) {
        if (!e || typeof e.message !== "string") continue;
        const level = ["debug", "info", "warn", "error"].includes(e.level) ? e.level : "info";
        const kind = ALLOWED_KINDS.has(e.kind) ? e.kind : "info";
        browserLogger.log(level, e.message.slice(0, 4000), {
          source: `browser:${String(e.page ?? "").replace(/[^\w./-]/g, "").slice(0, 60) || "?"}`, kind,
          browserTimestamp: typeof e.timestamp === "string" ? e.timestamp.slice(0, 40) : undefined,
          session: typeof e.session === "string" ? e.session.slice(0, 40) : undefined,
          requestId: typeof e.requestId === "string" ? e.requestId.slice(0, 64) : undefined,
          detail: e.detail === undefined ? undefined : scrub(e.detail),
        });
        written += 1;
      }
      response.status(202).json({ written });
    };
  }

  // ------------------------------------------------------------------ viewer
  function files(stream) {
    const pattern = new RegExp(`^${stream}\\.(\\d{4}-\\d{2}-\\d{2})\\.log(?:\\.(\\d+))?$`);
    try {
      return readdirSync(logsDir, { withFileTypes: true }).flatMap((entry) => {
        const match = entry.isFile() && pattern.exec(entry.name);
        return match ? [{ stream, filename: entry.name, date: match[1], part: Number(match[2] || 0) }] : [];
      });
    } catch { return []; }
  }
  async function view({ streams = STREAMS, date, level = "", source = "", requestId = "", q = "", limit = 500 }) {
    const all = streams.flatMap(files);
    const dates = [...new Set(all.map((f) => f.date))].sort().reverse();
    const day = date ?? dates[0] ?? null;
    const minRank = LEVEL_RANK[level] ?? 0;
    const needle = q.toLowerCase();
    const out = [];
    let scanned = 0;
    for (const f of all.filter((x) => x.date === day).sort((a, b) => a.part - b.part)) {
      const lines = createInterface({ input: createReadStream(join(logsDir, f.filename), { encoding: "utf8" }), crlfDelay: Infinity });
      for await (const line of lines) {
        if (!line.trim()) continue;
        scanned += 1;
        let entry;
        try { entry = JSON.parse(line); } catch { entry = { message: line, level: "info" }; }
        entry.stream = f.stream;
        entry.source = entry.source ?? f.stream;
        if ((LEVEL_RANK[entry.level] ?? 20) < minRank) continue;
        if (source && !String(entry.source).toLowerCase().includes(source.toLowerCase())) continue;
        if (requestId && entry.requestId !== requestId) continue;
        if (needle && !line.toLowerCase().includes(needle)) continue;
        out.push(entry);
        if (out.length > limit * 4) out.splice(0, out.length - limit);
      }
    }
    out.sort((a, b) => String(a.timestamp ?? "").localeCompare(String(b.timestamp ?? "")));
    return { dates, date: day, scanned, total: out.length, entries: out.slice(-limit) };
  }
  function viewerHandler() {
    return async (request, response) => {
      const qs = request.query;
      const streams = qs.stream && qs.stream !== "all" ? [String(qs.stream)] : STREAMS;
      if (!streams.every((s) => STREAMS.includes(s)) || (qs.date && !/^\d{4}-\d{2}-\d{2}$/.test(qs.date)) ||
          (qs.level && !(qs.level in LEVEL_RANK))) {
        return response.status(400).json({ error: "Choose a valid stream, date and level." });
      }
      const text = (v, n) => (typeof v === "string" ? v.slice(0, n) : "");
      try {
        response.json(await view({ streams, date: qs.date, level: text(qs.level, 10), source: text(qs.source, 80),
          requestId: text(qs.requestId, 64), q: text(qs.q, 200), limit: Math.max(10, Math.min(2000, Number(qs.limit) || 500)) }));
      } catch (error) {
        response.status(500).json({ error: error.message });
      }
    };
  }

  return { redact, compact, scrub, format, requestIds, currentRequestId, captureConsole, captureProcessErrors,
    clientLogsHandler, viewerHandler, view, context };
}
