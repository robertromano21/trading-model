import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { createReadStream, existsSync, mkdirSync, readdirSync, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { basename, join, resolve } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import express from "express";
import winston from "winston";
import DailyRotateFile from "winston-daily-rotate-file";
import { WebSocket, WebSocketServer } from "ws";

const root = fileURLToPath(new URL("../", import.meta.url));
const localEnv = join(root, ".env.local");
if (existsSync(localEnv)) process.loadEnvFile(localEnv);
const serverVersion = createHash("sha256").update(readFileSync(fileURLToPath(import.meta.url))).digest("hex");
const publicDir = join(root, "public");
const logsDir = join(root, "logs");
const dataDir = resolve(process.env.PRICES_DATA_DIR || join(root, "Massive"));
const workbook = resolve(process.env.ADX_WORKBOOK_PATH || join(root, "ADX worksheet 2.0.xlsm"));
const python = process.env.PRICES_PYTHON || (process.platform === "win32" ? "python" : "python3");
const port = Number(process.env.PRICES_PORT || 3000);
const host = "127.0.0.1";
const url = `http://localhost:${port}`;
const allowedHosts = new Set([`localhost:${port}`, `${host}:${port}`]);
if (port === 80) {
  allowedHosts.add("localhost");
  allowedHosts.add(host);
}
const hasApiKey = Boolean(process.env.MASSIVE_API_KEY?.trim());
const marketClock = new Intl.DateTimeFormat("en-CA", {
  timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit",
});
const marketHour = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York", hour: "numeric", hourCycle: "h23",
});
const jobLabels = { import: "Price import", strategies: "Strategy scan", history: "History backfill" };

if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error("PRICES_PORT must be a port from 1 to 65535.");
}

mkdirSync(logsDir, { recursive: true });
const format = winston.format.combine(winston.format.timestamp(), winston.format.json());

function rotatingLog(name) {
  const file = new DailyRotateFile({
    dirname: logsDir,
    filename: `${name}.%DATE%.log`,
    datePattern: "YYYY-MM-DD",
    maxSize: "10m",
    maxFiles: "14d",
    auditFile: join(logsDir, `.${name}-rotation.json`),
    utc: true,
  });
  file.on("error", (error) => console.error(`[${name}] ${error.message}`));
  return file;
}

const serverLog = winston.createLogger({
  format,
  transports: [new winston.transports.Console(), rotatingLog("server")],
});
const browserLog = winston.createLogger({ format, transports: [rotatingLog("browser-sync")] });
const app = express();
const server = createServer(app);
const sockets = new WebSocketServer({ noServer: true, maxPayload: 1024 });
let job = null;
let syncAfterJob = false;
let modelRows = null;
let adxSummaryCache;
let activeAdxReplay = false;
let backtestRegistry;
let activeBacktest = false;
let syncingDividends = false;

function localHost(request) {
  return allowedHosts.has(request.headers.host);
}

function sameOrigin(request) {
  return request.headers.origin === `http://${request.headers.host}`;
}

function parseDate(value) {
  if (typeof value !== "string" || !/^(?!0000)\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const time = Date.parse(value);
  return Number.isFinite(time) && new Date(time).toISOString().startsWith(`${value}T`) ? time : null;
}

function latestLocalDate() {
  const dates = [dataDir, join(dataDir, "Processed"), join(dataDir, "History", "JSON")].flatMap((directory) => {
    try {
      return readdirSync(directory, { withFileTypes: true })
        .filter((entry) => entry.isFile())
        .map((entry) => /^prices_(\d{4}-\d{2}-\d{2})\.json$/i.exec(entry.name)?.[1])
        .filter((date) => parseDate(date) !== null);
    } catch (error) {
      if (error.code !== "ENOENT") serverLog.warn("Could not list JSON files", { directory, error: error.message });
      return [];
    }
  });
  try {
    const document = JSON.parse(readFileSync(join(dataDir, "prices.json"), "utf8").replace(/^\uFEFF/, ""));
    const timestamp = document.results?.find((record) => Number.isFinite(record.t))?.t;
    if (timestamp !== undefined) dates.push(new Date(timestamp).toISOString().slice(0, 10));
  } catch (error) {
    if (error.code !== "ENOENT") serverLog.warn("Could not read legacy JSON date", { error: error.message });
  }
  return dates.sort().at(-1) ?? null;
}

function yesterdayEastern(now = new Date()) {
  const parts = Object.fromEntries(marketClock.formatToParts(now).map(({ type, value }) => [type, value]));
  const today = Date.UTC(Number(parts.year), Number(parts.month) - 1, Number(parts.day));
  return new Date(today - 86_400_000).toISOString().slice(0, 10);
}

function logFiles(stream) {
  const pattern = new RegExp(`^${stream}\\.(\\d{4}-\\d{2}-\\d{2})\\.log(?:\\.(\\d+))?$`);
  return readdirSync(logsDir, { withFileTypes: true }).flatMap((entry) => {
    const match = entry.isFile() && pattern.exec(entry.name);
    return match ? [{ filename: entry.name, date: match[1], part: Number(match[2] || 0) }] : [];
  });
}

async function readLogFiles(files, level, query) {
  const limit = 1000;
  const recent = [];
  let matches = 0;
  for (const file of files.sort((a, b) => a.part - b.part)) {
    const input = createReadStream(join(logsDir, file.filename), { encoding: "utf8" });
    const lines = createInterface({ input, crlfDelay: Infinity });
    for await (const line of lines) {
      if (!line.toLowerCase().includes(query)) continue;
      let entry;
      try {
        entry = JSON.parse(line);
      } catch {
        continue;
      }
      if (typeof entry?.message === "string" && (!level || entry.level === level)) {
        recent[matches % limit] = entry;
        matches += 1;
      }
    }
  }
  const first = matches > limit ? matches % limit : 0;
  return { entries: [...recent.slice(first), ...recent.slice(0, first)].reverse(), matches };
}

function send(socket, message) {
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(message, (error) => {
      if (error) socket.terminate();
    });
  }
}

function broadcastJob() {
  const message = JSON.stringify({ job });
  for (const socket of sockets.clients) send(socket, message);
}

function runEngine({ script = "fetch_and_model.py", args = [], options, id }) {
  return new Promise((fulfill, reject) => {
    const child = spawn(python, ["-u", join(root, "engine", script), ...args], {
      cwd: root,
      env: { ...process.env, PRICES_DATA_DIR: dataDir, ADX_WORKBOOK_PATH: workbook, PYTHONUTF8: "1" },
      windowsHide: true,
    });
    let output = "";
    child.stdout.setEncoding("utf8").on("data", (chunk) => { output += chunk; });
    createInterface({ input: child.stderr }).on("line", (line) => {
      if (line.startsWith("PROGRESS ") && job?.id === id && job.kind === "history") {
        try {
          job = { ...job, progress: JSON.parse(line.slice(9)) };
          broadcastJob();
          return;
        } catch {
          // Preserve the original line for diagnosis.
        }
      }
      const level = line.startsWith("ERROR ") ? "error" : line.startsWith("WARNING ") ? "warn" : "info";
      const message = line.replace(/^(INFO|WARNING|ERROR) /, "");
      serverLog.log(level, message, { jobId: id, component: "python" });
    });
    child.once("error", (error) => reject(new Error(`Could not start Python: ${error.message}`)));
    child.stdin.on("error", () => { /* The process exit reports an early close of stdin. */ });
    child.stdin.end(options === undefined ? undefined : JSON.stringify(options));
    child.once("close", (code, signal) => {
      if (id && script === "backtester.py") {
        serverLog.info("Backtest Python process exited", { jobId: id, exitCode: code, signal });
      }
      if (code !== 0) {
        reject(new Error(`Python exited with ${signal || `code ${code}`}. See the server log for details.`));
        return;
      }
      try {
        fulfill(JSON.parse(output));
      } catch {
        reject(new Error("Python returned an invalid result. See the server log for details."));
      }
    });
  });
}

async function completeJob(options, id, kind) {
  let result;
  const isModel = kind === "strategies";
  try {
    const script = kind === "history" ? "history_backfill.py" : "fetch_and_model.py";
    result = await runEngine({ script, args: isModel ? ["--strategies"] : [], options, id });
    if (kind === "history") {
      if (!["complete", "partial", "failed"].includes(result?.status)) throw new Error("Python returned an unexpected history result.");
    } else if (isModel && result?.status === "complete" && Array.isArray(result.rows)) {
      const { rows, ...summary } = result;
      modelRows = { id, rows };
      result = summary;
    } else if (isModel && result?.status === "failed") {
      result = { status: "failed", message: result.message };
    } else if (isModel || !["complete", "partial", "failed"].includes(result?.status) ||
               !Number.isSafeInteger(result.barsCount) || !Array.isArray(result.sessions)) {
      throw new Error("Python returned an unexpected run result.");
    }
  } catch (error) {
    result = { status: "failed", message: error.message, barsCount: 0, sessions: [] };
  }
  job = { ...job, status: result.status, finishedAt: new Date().toISOString(), result };
  const level = result.status === "failed" ? "error" : result.status === "partial" ? "warn" : "info";
  serverLog.log(level, `${jobLabels[kind]} finished`, {
    jobId: id, status: result.status, bars: result.barsCount ?? result.bars,
    sessions: result.sessionCount ?? (Array.isArray(result.sessions) ? result.sessions.length : result.sessions),
    excel: result.excel, summary: result.summary, missing: result.missing, error: result.message,
  });
  broadcastJob();
  if (syncAfterJob) {
    syncAfterJob = false;
    syncPricesOnLaunch();
  }
}

function startJob(options, kind, response) {
  if (job?.status === "running") {
    response?.status(409).json({ error: "A job is already in progress.", job });
    return;
  }
  job = { id: randomUUID(), kind, status: "running", startedAt: new Date().toISOString(), ...options };
  serverLog.info(`${jobLabels[kind]} started`, { jobId: job.id, ...options });
  response?.status(202).json({ job });
  broadcastJob();
  void completeJob(options, job.id, kind);
}

function syncPricesOnLaunch(response) {
  const now = new Date();
  const options = { date: yesterdayEastern(now), mode: hasApiKey ? "massive" : "local", startup: true };
  // Paid plans may serve today's bars after the extended session has ended.
  if (hasApiKey && Number(marketHour.format(now)) >= 21) options.today = marketClock.format(now);
  if (job?.status === "running") {
    if (!job.startup || job.date !== options.date || job.today !== options.today) syncAfterJob = true;
    response?.status(202).json({ job, queued: syncAfterJob });
    return;
  }
  startJob(options, "history", response);
}

app.disable("x-powered-by");
app.use((request, response, next) => {
  if (!localHost(request)) {
    response.status(403).json({ error: "Use the local dashboard address." });
    return;
  }
  response.set({
    "Content-Security-Policy": "default-src 'self'; connect-src 'self' ws://localhost:" + port +
      " ws://127.0.0.1:" + port + "; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
  });
  if (request.path.startsWith("/api/")) {
    response.set("Cache-Control", "no-store");
    const started = Date.now();
    const path = request.path;
    response.on("finish", () => serverLog.info("HTTP request", {
      method: request.method, path, statusCode: response.statusCode, durationMs: Date.now() - started,
    }));
  }
  next();
});

app.get("/api/health", (_request, response) => response.json({ service: "prices-dashboard", serverVersion }));
app.get("/api/status", (_request, response) => response.json({
  job,
  dataDir,
  workbook: basename(workbook),
  workbookExists: existsSync(workbook),
  massiveAvailable: hasApiKey,
  latestLocalDate: latestLocalDate(),
  latestHistoryDownload: yesterdayEastern(),
}));

app.get("/api/logs", async (request, response) => {
  const { stream, date, level = "", q = "" } = request.query;
  if (!["server", "browser-sync"].includes(stream) ||
      (date !== undefined && parseDate(date) === null) ||
      !["", "info", "warn", "error"].includes(level) || typeof q !== "string" || q.length > 200) {
    response.status(400).json({ error: "Choose a valid log stream, date, level, and search term." });
    return;
  }
  const files = logFiles(stream);
  const dates = [...new Set(files.map((file) => file.date))].sort().reverse();
  const selected = date ?? dates[0] ?? null;
  const result = await readLogFiles(files.filter((file) => file.date === selected), level, q.toLowerCase());
  response.json({ dates, date: selected, ...result });
});

app.get("/api/market-data/dates", (_request, response) => sendEngineQuery(response, ["--market-dates"]));
app.get("/api/market-data/:date", async (request, response) => {
  const date = request.params.date;
  if (parseDate(date) === null) {
    response.status(400).json({ error: "Use a valid market date in YYYY-MM-DD format." });
    return;
  }
  try {
    const result = await runEngine({ args: ["--read-date", date] });
    if (result?.status === "missing") {
      response.status(404).json({ error: "No matching JSON file." });
    } else if (result?.status === "complete" && result.date === date && Array.isArray(result.bars)) {
      response.json({ date, bars: result.bars });
    } else {
      throw new Error(result?.message || "Python returned an unexpected session result.");
    }
  } catch (error) {
    serverLog.error("Could not read market session", { date, error: error.message });
    response.status(500).json({ error: error.message });
  }
});

app.get("/api/strategy-results/:id", (request, response) => {
  if (request.params.id !== modelRows?.id) {
    response.status(404).json({ error: "Strategy results are unavailable. Run the model again." });
    return;
  }
  response.json(modelRows);
});

async function sendEngineQuery(response, args, script) {
  try {
    const result = await runEngine({ args, script });
    if (result?.status !== "complete") throw new Error(result?.message || "Could not read SQLite history.");
    response.json(result);
  } catch (error) {
    serverLog.error("Could not read SQLite history", { error: error.message });
    response.status(500).json({ error: error.message });
  }
}

app.get("/api/history/status", (_request, response) => sendEngineQuery(response, ["--history-status"]));
app.get("/api/adx-summary/dates", (_request, response) =>
  sendEngineQuery(response, ["--dates"], "adx_summary.py"));

function historyVersion() {
  return ["prices.sqlite3", "prices.sqlite3-wal"].map((file) => {
    const stat = statSync(join(dataDir, file), { bigint: true, throwIfNoEntry: false });
    return stat ? `${stat.mtimeNs}:${stat.size}` : "";
  }).join("|");
}

async function adxSummary(date) {
  const key = `${date}:${historyVersion()}`;
  if (adxSummaryCache?.key !== key) {
    const started = Date.now();
    const result = runEngine({ script: "adx_summary.py", args: [date] });
    adxSummaryCache = { key, result };
    try {
      const report = await result;
      if (report.status !== "complete" && adxSummaryCache?.result === result) adxSummaryCache = undefined;
      if (report.status === "complete") {
        serverLog.info("ADX Summary calculated", { date, symbols: report.rows.length, durationMs: Date.now() - started });
      }
      return report;
    } catch (error) {
      if (adxSummaryCache?.result === result) adxSummaryCache = undefined;
      throw error;
    }
  }
  return adxSummaryCache.result;
}

app.get("/api/adx-summary", async (request, response) => {
  const { date } = request.query;
  if (parseDate(date) === null) {
    response.status(400).json({ error: "Choose a valid saved market date." });
    return;
  }
  const report = await adxSummary(date);
  if (report.status !== "complete") {
    response.status(400).json({ error: report.message });
    return;
  }
  response.json(report);
});

app.get("/api/history/ticker", (request, response) => {
  const { ticker, through = "9999-12-31" } = request.query;
  if (typeof ticker !== "string" || !ticker.trim() || ticker.trim().length > 64 || parseDate(through) === null) {
    response.status(400).json({ error: "Enter a valid ticker and optional end date." });
    return;
  }
  return sendEngineQuery(response, ["--ticker-history", ticker.trim().toUpperCase(), through]);
});

function strategies() {
  backtestRegistry ??= runEngine({ script: "backtester.py", args: ["--strategies"] }).then((result) => {
    if (result?.status !== "complete" || !Array.isArray(result.strategies)) {
      throw new Error("The backtest strategy registry is unavailable.");
    }
    return result;
  }).catch((error) => {
    backtestRegistry = undefined;
    throw error;
  });
  return backtestRegistry;
}

app.get("/api/strategies", async (_request, response) => {
  response.json(await strategies());
});

app.get("/api/universe", async (_request, response) => {
  const result = await runEngine({ script: "backtester.py", args: ["--universe"] });
  if (result?.status !== "complete") throw new Error(result?.message || "Could not inspect SQLite coverage.");
  response.json({ ...result, dividendSyncAvailable: hasApiKey });
});

function validBacktestNumbers(fields, definitions) {
  return fields && typeof fields === "object" && !Array.isArray(fields) &&
    Object.entries(fields).every(([key, value]) => {
      const field = Object.hasOwn(definitions, key) ? definitions[key] : undefined;
      return field && typeof value === "number" && Number.isFinite(value) &&
        value >= field.min && value <= field.max && (field.type !== "integer" || Number.isInteger(value));
    });
}

const standardApiJson = express.json({ limit: "64kb" });
const largeApiJson = express.json({ limit: "1mb" });
app.use("/api", (request, response, next) => {
  if (request.method !== "POST") {
    next();
  } else if (request.headers.origin !== undefined && !sameOrigin(request)) {
    response.status(403).json({ error: "Use the dashboard to make this request." });
  } else if (!request.is("application/json")) {
    response.status(415).json({ error: "Send an application/json request." });
  } else {
    next();
  }
}, (request, response, next) => {
  const largeBody = ["/adx-summary/forward-returns", "/papertrade/marks"].includes(request.path);
  const parseJson = largeBody ? largeApiJson : standardApiJson;
  parseJson(request, response, next);
});

app.post("/api/papertrade/marks", async (request, response) => {
  const result = await runEngine({ script: "papertrade.py", options: request.body });
  if (result?.status !== "complete") {
    response.status(400).json({ error: result?.message || "Papertrade prices could not be loaded." });
    return;
  }
  response.json(result);
});

app.post("/api/adx-summary/forward-returns", async (request, response) => {
  const { through, filters, tickers } = request.body ?? {};
  if (parseDate(through) === null || !filters || !Array.isArray(filters.columns) || !Array.isArray(tickers) || !tickers.length) {
    response.status(400).json({ error: "Choose a saved market date with at least one ticker in the filtered table." });
    return;
  }
  if (activeAdxReplay) {
    response.status(409).json({ error: "A historical return analysis is already running." });
    return;
  }
  activeAdxReplay = true;
  const started = Date.now();
  try {
    const result = await runEngine({ script: "adx_history.py", options: { through, filters, tickers } });
    if (result?.status !== "complete") {
      response.status(400).json({ error: result?.message || "The return analysis could not finish." });
      return;
    }
    serverLog.info("ADX historical returns calculated", {
      through, filteredTickers: tickers.length, observations: result.observations, durationMs: Date.now() - started,
    });
    response.json(result);
  } finally {
    activeAdxReplay = false;
  }
});

app.post("/api/backtest", async (request, response) => {
  const { strategy, mode, tickers, startDate, endDate, includeDividends = false, params = {}, costs = {} } = request.body ?? {};
  const first = parseDate(startDate);
  const last = parseDate(endDate);
  if (!["single", "portfolio"].includes(mode) || !Array.isArray(tickers) ||
      (mode === "single" && tickers.length !== 1) || new Set(tickers).size !== tickers.length ||
      tickers.some((ticker) => typeof ticker !== "string" || !/^[A-Z0-9][A-Z0-9.-]{0,63}$/.test(ticker)) ||
      first === null || last === null || first >= last || typeof includeDividends !== "boolean") {
    response.status(400).json({ error: "Choose valid tickers, dates, and dividend settings." });
    return;
  }
  const registry = await strategies();
  const definition = registry.strategies.find((item) => item.id === strategy);
  if (!definition || !validBacktestNumbers(params, definition.parameters) || !validBacktestNumbers(costs, registry.settings)) {
    response.status(400).json({ error: "Choose an available strategy and enter parameters within their ranges." });
    return;
  }
  if (activeBacktest) {
    response.status(409).json({ error: "A backtest is already running." });
    return;
  }
  activeBacktest = true;
  const id = randomUUID();
  const start = Date.now();
  const options = { strategy, mode, tickers, startDate, endDate, includeDividends, params, costs };
  serverLog.info("Backtest started", { jobId: id, strategy, mode, tickerCount: tickers.length, startDate, endDate, includeDividends });
  try {
    const result = await runEngine({ script: "backtester.py", options, id });
    if (result?.status === "failed") {
      serverLog.warn("Backtest failed", { jobId: id, durationMs: Date.now() - start, error: result.message });
      response.status(result.errorType === "validation" ? 400 : 500).json({ error: result.message });
    } else if (result?.status === "complete" && result.runId && Array.isArray(result.trades) && Array.isArray(result.equityCurve)) {
      serverLog.info("Backtest finished", { jobId: id, runId: result.runId, durationMs: Date.now() - start,
        symbols: result.symbolCount, trades: result.trades.length });
      response.json(result);
    } else {
      throw new Error("Python returned an unexpected backtest result.");
    }
  } catch (error) {
    serverLog.error("Backtest failed", { jobId: id, durationMs: Date.now() - start, error: error.message });
    response.status(500).json({ error: error.message });
  } finally {
    activeBacktest = false;
  }
});

app.post("/api/backtest/dividends", async (_request, response) => {
  if (!hasApiKey || syncingDividends) {
    response.status(hasApiKey ? 409 : 400).json({ error: hasApiKey ? "Dividend history is already syncing." :
      "Set MASSIVE_API_KEY in PowerShell and restart the server to sync dividends." });
    return;
  }
  syncingDividends = true;
  const id = randomUUID();
  const start = Date.now();
  serverLog.info("Dividend sync started", { jobId: id });
  try {
    const result = await runEngine({ script: "dividend_history.py", id });
    if (result?.status !== "complete") throw new Error(result?.message || "Dividend sync failed.");
    serverLog.info("Dividend sync finished", { jobId: id, durationMs: Date.now() - start, events: result.events });
    response.json(result);
  } catch (error) {
    serverLog.error("Dividend sync failed", { jobId: id, durationMs: Date.now() - start, error: error.message });
    response.status(502).json({ error: error.message });
  } finally {
    syncingDividends = false;
  }
});

app.post("/api/run-model", (request, response) => {
  const { startDate, endDate = startDate, source, updateWorkbook = false } = request.body ?? {};
  const first = parseDate(startDate);
  const last = parseDate(endDate);
  const pending = source === "pending";
  const validDates = first !== null && last !== null && last >= first && last - first <= 365 * 86_400_000;
  if (!["local", "massive", "pending"].includes(source) || (!pending && !validDates) ||
      typeof updateWorkbook !== "boolean") {
    response.status(400).json({ error: "Choose a valid source and an ordered date range of up to 366 days." });
    return;
  }
  if (source === "massive" && !hasApiKey) {
    response.status(400).json({ error: "Set MASSIVE_API_KEY in PowerShell before starting the server." });
    return;
  }
  const options = pending ? { source, updateWorkbook } : { startDate, endDate, source, updateWorkbook };
  startJob(options, "import", response);
});

app.post("/api/run-strategies", (request, response) => {
  const { date, baseline = "sma", fastEma = 5, trailing = "ma", breakoutStop = "atr", trendDays = 12 } = request.body ?? {};
  if (parseDate(date) === null || !["sma", "ema"].includes(baseline) || ![5, 10].includes(fastEma) ||
      !["ma", "atr"].includes(trailing) || !["atr", "swing"].includes(breakoutStop) ||
      !Number.isInteger(trendDays) || trendDays < 10 || trendDays > 15) {
    response.status(400).json({ error: "Choose a valid market date and model settings." });
    return;
  }
  startJob({ date, baseline, fastEma, trailing, breakoutStop, trendDays }, "strategies", response);
});

app.post("/api/backfill-history", (request, response) => {
  const { date, mode } = request.body ?? {};
  if (parseDate(date) === null || Number(date.slice(0, 4)) < 4 || !["local", "massive"].includes(mode)) {
    response.status(400).json({ error: "Choose a valid end date and source for the three-year history." });
    return;
  }
  if (mode === "massive" && (!hasApiKey || date > yesterdayEastern())) {
    response.status(400).json({ error: !hasApiKey ? "Set MASSIVE_API_KEY in PowerShell and restart the server." :
      "Choose a history end date before today in New York." });
    return;
  }
  startJob({ date, mode }, "history", response);
});

app.post("/api/sync-prices", (_request, response) => syncPricesOnLaunch(response));

app.post("/api/browser-logs", (request, response) => {
  const { level, message, timestamp } = request.body ?? {};
  if (level !== "error" || typeof message !== "string" || message.length > 4000 ||
      typeof timestamp !== "string" || !Number.isFinite(Date.parse(timestamp))) {
    response.status(400).json({ error: "Send a browser error with a timestamp and up to 4,000 characters." });
    return;
  }
  browserLog.error(message, { browserTimestamp: timestamp });
  response.sendStatus(204);
});

app.use("/api", (_request, response) => response.status(404).json({ error: "Unknown API endpoint." }));
app.get("/assets/chart.umd.js", (_request, response) => {
  response.sendFile(join(root, "server", "node_modules", "chart.js", "dist", "chart.umd.js"));
});
app.use(express.static(publicDir, { dotfiles: "deny" }));
app.use((error, _request, response, _next) => {
  const status = error.status || 500;
  if (status >= 500) serverLog.error("HTTP error", { error: error.message });
  response.status(status).json({ error: status === 413 ? "Request is too large." :
    status === 400 ? "Invalid JSON request." : "The server could not complete the request." });
});

server.on("upgrade", (request, socket, head) => {
  if (request.url !== "/live" || !localHost(request) || !sameOrigin(request)) {
    socket.destroy();
    return;
  }
  sockets.handleUpgrade(request, socket, head, (websocket) => {
    websocket.on("error", (error) => serverLog.warn("WebSocket error", { error: error.message }));
    send(websocket, JSON.stringify({ job }));
  });
});

server.on("error", (error) => {
  serverLog.error("Could not start server", { error: error.message });
  process.exitCode = 1;
});
server.listen(port, host, () => {
  serverLog.info("Prices dashboard listening", { url, dataDir, workbook: basename(workbook) });
  syncPricesOnLaunch();
  if (process.platform === "win32" && process.env.PRICES_OPEN_BROWSER === "1") {
    const browser = spawn("rundll32.exe", ["url.dll,FileProtocolHandler", url], {
      detached: true, stdio: "ignore", windowsHide: true,
    });
    browser.on("error", (error) => serverLog.warn("Could not open the browser", { error: error.message }));
    browser.unref();
  }
});
