// Workbench API: ticker charts (engine-computed indicators), real-time Schwab quotes (market data ONLY),
// data health, Schwab token status, paper-trading tables, and allow-listed research/paper jobs.
//
// Safety: the Schwab client here can only call https://api.schwabapi.com/marketdata/v1/* and /v1/oauth/token.
// Order/account endpoints (/trader/...) are refused in code; manual orders stay in the separate manual-ticket module.
// Nothing here writes market data except the Schwab quote cache (intraday_quotes) and allow-listed Python jobs.
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { appendFileSync, closeSync, existsSync, mkdirSync, openSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const SCHWAB = "https://api.schwabapi.com";
const ALLOWED_PREFIXES = ["/marketdata/v1/", "/v1/oauth/token"];
const TICKER = /^[A-Z][A-Z0-9.\-]{0,9}$/;
const QUOTE_TTL_MS = 4_000;
const QUOTE_PERSIST_MS = 60_000;
const SCHWAB_PER_MINUTE = 90;            // Schwab allows 120 market-data requests/minute; stay well below
const BAR_REFRESH_MS = 60_000;

export function pidIsAlive(pid) {
  const n = Number(pid);
  if (!Number.isInteger(n) || n <= 0) return false;
  try {
    process.kill(n, 0);
    return true;
  } catch (error) {
    return error?.code === "EPERM";
  }
}

export function checkSchwabUrl(url) {
  const parsed = new URL(url);
  if (parsed.protocol !== "https:" || parsed.host !== "api.schwabapi.com") throw new Error("Refusing non-Schwab URL.");
  if (parsed.pathname.startsWith("/trader/") || !ALLOWED_PREFIXES.some((p) => parsed.pathname.startsWith(p))) {
    throw new Error(`Refusing ${parsed.pathname}: the workbench may only call Schwab market-data endpoints.`);
  }
  return parsed;
}

// Jobs the UI may start. Each runs one engine script with fixed arguments (no user-supplied arguments).
export const JOBS = {
  research_all: { label: "Research: all (backtests, walk-forward, multi-horizon, normality)", script: "research.py", args: ["all"], minutes: 4 },
  research_backtest: { label: "Research: full-period backtests", script: "research.py", args: ["backtest"], minutes: 1 },
  research_walkforward: { label: "Research: walk-forward", script: "research.py", args: ["walkforward"], minutes: 3 },
  research_multihorizon: { label: "Research: multi-horizon trend", script: "research.py", args: ["multihorizon"], minutes: 1 },
  research_normality: { label: "Research: normality by horizon", script: "research.py", args: ["normality"], minutes: 1 },
  research_attribution: { label: "Research: return attribution (market / sectors / alpha)", script: "research.py", args: ["attribution"], minutes: 2 },
  research_prob: { label: "Research: probability walk-forward (high-probability entries)", script: "research.py", args: ["prob"], minutes: 8 },
  daytrade_research: { label: "Day-trading research (Schwab bars)", script: "daytrader.py", args: ["research"], minutes: 1 },
  paper_dry: { label: "Swing paper trader: dry run (nothing saved)", script: "paper_trader.py", args: ["run", "--dry-run"], minutes: 2 },
  daytrade_dry: { label: "Day-trading paper: dry run (nothing saved)", script: "daytrader.py", args: ["run", "--dry-run"], minutes: 1 },
  paper_run: { label: "Swing paper trader: run and SAVE paper state", script: "paper_trader.py", args: ["run"], minutes: 2, confirm: true },
  daytrade_run: { label: "Day-trading paper: fetch Schwab bars, run and SAVE", script: "daytrader.py", args: ["run", "--fetch"], minutes: 2, confirm: true },
  schwab_fetch: { label: "Schwab: refresh intraday bars + quotes for targets", script: "daytrader.py", args: ["fetch"], minutes: 3 },
  schwab_daily_backfill: { label: "Schwab: long daily history backfill (~20y, resumable)", script: "schwab_daily.py", args: ["backfill"], minutes: 60, detached: true },
  swarm_run: { label: "Swarm: 50 simulated $10k accounts (walk-forward + sealed holdout)", script: "swarm.py", args: ["run"], minutes: 4 },
  swarm_forward: { label: "Swarm: advance shadow accounts (forward mode)", script: "swarm.py", args: ["forward"], minutes: 3 },
  crypto_data: { label: "Crypto: refresh Coinbase / FRED / Schwab market data", script: "crypto_data.py", args: ["update"], minutes: 2 },
  crypto_research: { label: "Crypto: backtests + dollar-feedback study (walk-forward, sealed holdout)", script: "crypto_research.py", args: ["run"], minutes: 2 },
  crypto_paper: { label: "Crypto: advance the 20 crypto PAPER accounts (ids 200-219)", script: "crypto_paper.py", args: ["run", "--fetch"], minutes: 2 },
  macro_nowcast_refresh: { label: "Macro: refresh FRED data + publish our nowcasts (raw data only)", script: "macro_nowcast.py", args: ["refresh"], minutes: 3, detached: true },
  macro_nowcast_research: { label: "Macro: nowcast walk-forward accuracy + daily panel (pre-holdout)", script: "macro_nowcast.py", args: ["research"], minutes: 6, detached: true },
  macro_backtest: { label: "Macro: regime filters vs plain TSMOM (walk-forward, SPA, pre-holdout)", script: "macro_backtest.py", args: ["run"], minutes: 8, detached: true },
  index_long_update: { label: "Schwab: refresh long index history ($SPX, $DJI, ... since 1970)", script: "index_long.py", args: ["update"], minutes: 2 },
  data_rescan: { label: "Data: re-scan price anomalies / split report", script: "corporate_actions.py", args: ["--report"], minutes: 3, confirm: true },
};

export function easternClock(now = Date.now()) {
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", weekday: "short", hour: "numeric", minute: "numeric", hourCycle: "h23",
    year: "numeric", month: "2-digit", day: "2-digit",
  }).formatToParts(new Date(now)).map((p) => [p.type, p.value]));
  const minute = Number(parts.hour) * 60 + Number(parts.minute);
  const weekday = !["Sat", "Sun"].includes(parts.weekday);
  return { date: `${parts.year}-${parts.month}-${parts.day}`, minute, weekday,
    regular: weekday && minute >= 570 && minute < 960, extended: weekday && minute >= 420 && minute < 1200 };
}

export function createWorkbench({ root, dataDir, python = "python", log = () => {}, fetchImpl = globalThis.fetch,
  now = () => Date.now(), env = process.env, spawnImpl = spawn, sqlite = null, requestId = () => undefined, onJobDone = null,
  pidAlive = pidIsAlive, setIntervalImpl = setInterval, clearIntervalImpl = clearInterval } = {}) {
  const engineDir = join(root, "engine");
  const researchDir = join(dataDir, "Research");
  const jobsDir = join(researchDir, "jobs");
  const tokenPath = env.SCHWAB_TOKEN_PATH || join(root, "schwab_token.json");
  const quoteCache = new Map();          // symbol -> { at, quote }
  const persisted = new Map();           // symbol -> last persisted ms
  const barRefresh = new Map();          // symbol|freq -> ms
  const chartCache = new Map();          // key -> { at, stamp, body }
  const calls = [];                      // timestamps of Schwab requests (rate limiter)
  const jobs = new Map();
  let schwabError = null;
  let healthCache = null;
  let pythonSlots = 3;
  const waiting = [];

  // ------------------------------------------------------------------ python helper
  async function slot() {
    if (pythonSlots > 0) { pythonSlots -= 1; return; }
    await new Promise((resolve) => waiting.push(resolve));
  }
  function release() {
    const next = waiting.shift();
    if (next) next(); else pythonSlots += 1;
  }
  async function runPython(script, args, { timeoutMs = 90_000 } = {}) {
    await slot();
    try {
      return await new Promise((resolve, reject) => {
        const child = spawnImpl(python, ["-W", "ignore", script, ...args], {
          cwd: engineDir, env: { ...env, PYTHONUTF8: "1", TM_REQUEST_ID: requestId() ?? "" }, windowsHide: true });
        let out = "", err = "";
        const timer = setTimeout(() => { child.kill(); reject(new Error(`${script} timed out`)); }, timeoutMs);
        child.stdout.on("data", (d) => { out += d; });
        child.stderr.on("data", (d) => { err += d; });
        child.on("error", (e) => { clearTimeout(timer); reject(e); });
        child.on("close", (code) => {
          clearTimeout(timer);
          if (code !== 0) return reject(new Error(`${script} failed: ${err.trim().split("\n").slice(-1)[0] || `exit ${code}`}`));
          resolve(out);
        });
      });
    } finally {
      release();
    }
  }

  // ------------------------------------------------------------------ Schwab market data (read-only)
  function readToken() {
    if (!existsSync(tokenPath)) return null;
    try { return JSON.parse(readFileSync(tokenPath, "utf8")); } catch { return null; }
  }
  function canRefresh() { return Boolean(env.SCHWAB_APP_KEY?.trim() && env.SCHWAB_APP_SECRET?.trim()); }
  function fmt(ms) {
    return ms ? new Date(ms).toLocaleString("en-US", { timeZone: "America/New_York", hour12: false }) + " ET" : null;
  }
  function schwabStatus() {
    const config = { appKeySet: Boolean(env.SCHWAB_APP_KEY?.trim()), appSecretSet: Boolean(env.SCHWAB_APP_SECRET?.trim()),
      callbackUrl: env.SCHWAB_CALLBACK_URL?.trim() || "https://127.0.0.1:8182", envFile: ".env.local" };
    const data = readToken();
    if (!data) return { tokenFile: false, usable: false, setupNeeded: true, config,
      reason: config.appKeySet && config.appSecretSet ? "Schwab is not authorized yet (no schwab_token.json)." : "Schwab is not set up yet." };
    const token = data.token ?? {};
    const accessMs = (token.expires_at ?? 0) * 1000;
    const refreshMs = data.creation_timestamp ? (data.creation_timestamp + 7 * 86400) * 1000 : null;
    const accessValid = accessMs > now() + 60_000;
    const refreshValid = refreshMs ? refreshMs > now() : false;
    const usable = accessValid || (refreshValid && canRefresh());
    const reason = usable ? null : !refreshValid ? "The Schwab refresh token has expired (7-day limit): sign in again with schwab-py."
      : "The access token expired and SCHWAB_APP_KEY / SCHWAB_APP_SECRET are not set, so it cannot be refreshed.";
    return { tokenFile: true, usable, reason, config, setupNeeded: !config.appKeySet || !config.appSecretSet, accessTokenValid: accessValid, accessTokenExpires: fmt(accessMs),
      refreshTokenExpiresAbout: fmt(refreshMs), refreshTokenValid: refreshValid, canRefresh: canRefresh(),
      lastError: schwabError, requestsLastMinute: calls.filter((t) => t > now() - 60_000).length, limitPerMinute: SCHWAB_PER_MINUTE };
  }
  function takeRateSlot() {
    while (calls.length && calls[0] < now() - 60_000) calls.shift();
    if (calls.length >= SCHWAB_PER_MINUTE) return false;
    calls.push(now());
    return true;
  }
  async function accessToken() {
    const data = readToken();
    if (!data) throw new Error("No Schwab token file.");
    const token = data.token ?? {};
    if ((token.expires_at ?? 0) * 1000 > now() + 60_000) return token.access_token;
    const status = schwabStatus();
    if (!status.usable) throw new Error(status.reason);
    const url = `${SCHWAB}/v1/oauth/token`;
    checkSchwabUrl(url);
    if (!takeRateSlot()) throw new Error("Schwab rate limit reached; using cached data.");
    const basic = Buffer.from(`${env.SCHWAB_APP_KEY.trim()}:${env.SCHWAB_APP_SECRET.trim()}`).toString("base64");
    const response = await fetchImpl(url, { method: "POST", headers: { Authorization: `Basic ${basic}`,
      "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ grant_type: "refresh_token", refresh_token: token.refresh_token }) });
    if (!response.ok) throw new Error(`Schwab token refresh failed (HTTP ${response.status}).`);
    const fresh = await response.json();
    data.token = { ...token, ...fresh, refresh_token: fresh.refresh_token ?? token.refresh_token,
      expires_at: Math.floor(now() / 1000) + Number(fresh.expires_in ?? 1800) };
    writeFileSync(`${tokenPath}.tmp`, JSON.stringify(data));
    renameSync(`${tokenPath}.tmp`, tokenPath);
    return data.token.access_token;
  }
  async function schwabGet(path) {
    const url = `${SCHWAB}${path}`;
    checkSchwabUrl(url);
    const tokenValue = await accessToken();
    if (!takeRateSlot()) throw new Error("Schwab rate limit reached; using cached data.");
    const response = await fetchImpl(url, { headers: { Authorization: `Bearer ${tokenValue}`, Accept: "application/json" } });
    if (!response.ok) throw new Error(`Schwab market data HTTP ${response.status}`);
    return response.json();
  }

  function normalizeQuote(symbol, raw) {
    const q = raw?.quote ?? {};
    const ref = raw?.reference ?? {};
    const prev = q.closePrice ?? null;
    const last = q.lastPrice ?? q.mark ?? null;
    return { symbol, last, bid: q.bidPrice ?? null, ask: q.askPrice ?? null, bidSize: q.bidSize ?? null, askSize: q.askSize ?? null,
      open: q.openPrice ?? null, high: q.highPrice ?? null, low: q.lowPrice ?? null, prevClose: prev, volume: q.totalVolume ?? null,
      change: q.netChange ?? (last != null && prev ? last - prev : null),
      changePct: q.netPercentChange ?? (last != null && prev ? (last / prev - 1) * 100 : null),
      quoteTime: q.quoteTime ?? null, tradeTime: q.tradeTime ?? null, description: ref.description ?? null,
      week52High: q["52WeekHigh"] ?? null, week52Low: q["52WeekLow"] ?? null };
  }

  async function db(path, readOnly = true) {
    if (!existsSync(path)) return null;
    try {
      const { DatabaseSync } = sqlite ?? await import("node:sqlite");
      return new DatabaseSync(path, { readOnly });
    } catch (error) {
      log("warn", "node:sqlite unavailable", { error: error.message });
      return null;
    }
  }

  async function persistQuotes(list) {
    const due = list.filter((q) => q.last != null && (persisted.get(q.symbol) ?? 0) < now() - QUOTE_PERSIST_MS);
    if (!due.length) return;
    const connection = await db(join(dataDir, "intraday.sqlite3"), false);
    if (!connection) return;
    try {
      connection.exec("PRAGMA busy_timeout = 3000");
      const fetchedAt = new Date(now()).toISOString().replace("T", " ").slice(0, 19);
      const insert = connection.prepare("INSERT OR REPLACE INTO intraday_quotes (symbol, fetched_at, last, bid, ask, bid_size, ask_size, volume, quote_time) VALUES (?,?,?,?,?,?,?,?,?)");
      for (const q of due) {
        insert.run(q.symbol, fetchedAt, q.last, q.bid, q.ask, q.bidSize, q.askSize, q.volume, q.quoteTime);
        persisted.set(q.symbol, now());
      }
    } catch (error) {
      log("warn", "Could not cache Schwab quotes", { error: error.message });
    } finally {
      connection.close();
    }
  }

  async function quotes(symbols) {
    const clock = easternClock(now());
    const result = { fetchedAt: new Date(now()).toISOString(), market: clock, quotes: {}, source: "cache" };
    const stale = symbols.filter((s) => !(quoteCache.get(s)?.at > now() - QUOTE_TTL_MS));
    const status = schwabStatus();
    result.schwab = { usable: status.usable, reason: status.reason, accessTokenExpires: status.accessTokenExpires };
    if (stale.length && status.usable) {
      try {
        const data = await schwabGet(`/marketdata/v1/quotes?symbols=${encodeURIComponent(stale.join(","))}&fields=quote,reference`);
        const fresh = [];
        for (const s of stale) {
          if (!data[s]) continue;
          const q = normalizeQuote(s, data[s]);
          quoteCache.set(s, { at: now(), quote: q });
          fresh.push(q);
        }
        schwabError = null;
        result.source = "schwab";
        persistQuotes(fresh).catch(() => {});
      } catch (error) {
        schwabError = { at: new Date(now()).toISOString(), message: error.message };
        result.schwab.reason = error.message;
        result.schwab.usable = false;
      }
    }
    for (const s of symbols) {
      const hit = quoteCache.get(s);
      if (hit) result.quotes[s] = { ...hit.quote, ageSeconds: Math.round((now() - hit.at) / 1000) };
    }
    if (!Object.keys(result.quotes).length) result.source = "none";
    return result;
  }

  // ------------------------------------------------------------------ charts
  function stampOf(path) {
    try { return statSync(path).mtimeMs; } catch { return 0; }
  }
  async function chart(ticker, tf, sessions, refresh) {
    const intraday = tf !== "1d";
    const notes = [];
    if (intraday && refresh) {
      const key = `${ticker}|${tf}`;
      const status = schwabStatus();
      if (!status.usable) {
        notes.push(`Schwab unavailable (${status.reason}); showing cached bars.`);
      } else if ((barRefresh.get(key) ?? 0) < now() - BAR_REFRESH_MS && takeRateSlot()) {
        barRefresh.set(key, now());
        try {
          await runPython("schwab_data.py", ["fetch", ticker, "--days", "2", "--frequency", tf === "1m" ? "1" : "5"], { timeoutMs: 60_000 });
        } catch (error) {
          notes.push(`Schwab bar refresh failed: ${error.message}`);
        }
      }
    }
    const dbPath = join(dataDir, intraday ? "intraday.sqlite3" : "prices.sqlite3");
    const stamp = `${stampOf(dbPath)}|${stampOf(join(dataDir, "paper_trading.sqlite3"))}|${stampOf(join(dataDir, "Research", "ondemand", "updated.stamp"))}`;   // on-demand Schwab history (chart picker)
    const key = `${ticker}|${tf}|${sessions ?? ""}`;
    const hit = chartCache.get(key);
    let body;
    if (hit && hit.stamp === stamp && hit.at > now() - (intraday ? 15_000 : 300_000)) {
      body = hit.body;
    } else {
      const args = [ticker, "--tf", tf];
      if (sessions) args.push("--sessions", String(sessions));
      body = JSON.parse(await runPython("chart_data.py", args));
      chartCache.set(key, { at: now(), stamp, body });
      if (chartCache.size > 60) chartCache.delete(chartCache.keys().next().value);
    }
    return { ...body, notes, schwab: schwabStatus() };
  }

  // ------------------------------------------------------------------ paper tables (read-only)
  async function paperTables(account, limit) {
    const connection = await db(join(dataDir, "paper_trading.sqlite3"));
    if (!connection) return { available: false };
    const has = (name) => Boolean(connection.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?").get(name));
    const all = (sql, ...args) => connection.prepare(sql).all(...args).map((r) => ({ ...r }));
    try {
      return {
        available: true,
        accounts: has("accounts") ? all("SELECT name, initial_equity, cash, start_date, last_processed, breaker_remaining, blocked, settings FROM accounts") : [],
        fills: has("fills") ? all("SELECT fill_date, ticker, shares, price, effective_price, notional, cost_bps, slippage_bps FROM fills WHERE account=? ORDER BY fill_id DESC LIMIT ?", account, limit) : [],
        dtDaily: has("dt_daily") ? all("SELECT account, session_date, start_equity, pnl, equity, trades, skipped FROM dt_daily ORDER BY session_date DESC LIMIT ?", limit) : [],
        dtAccounts: has("dt_accounts") ? all("SELECT name, initial_equity, equity, frequency, start_session, last_session, params FROM dt_accounts") : [],
        dtTrades: has("dt_trades") ? all("SELECT account, session_date, symbol, direction, entry_minute, exit_minute, shares, entry_price, exit_price, stop, reason, pnl, r_multiple FROM dt_trades ORDER BY session_date DESC, entry_minute DESC LIMIT ?", limit) : [],
        runs: has("runs") ? all("SELECT run_id, account, asof, mode, finished_at, status, sessions_processed, equity, cash, gross, message, diagnostics FROM runs ORDER BY run_id DESC LIMIT 10") : [],
        advisory: has("advisory_verdicts") ? all("SELECT * FROM advisory_verdicts ORDER BY rowid DESC LIMIT 50") : [],
      };
    } finally {
      connection.close();
    }
  }

  // ------------------------------------------------------------------ jobs
  function jobView(j) {
    const { child, ...rest } = j;
    return rest;
  }
  function saveJobs() {
    try {
      mkdirSync(jobsDir, { recursive: true });
      writeFileSync(join(jobsDir, "jobs.json"), JSON.stringify([...jobs.values()].slice(-50).map(jobView), null, 1));
    } catch { /* informational */ }
  }
  const watches = new Map();
  function watchPid(job) {
    if (!job?.pid || watches.has(job.id)) return;
    const timer = setIntervalImpl(() => {
      if (pidAlive(job.pid)) return;
      clearIntervalImpl(timer);
      watches.delete(job.id);
      if (job.status !== "running") return;
      job.exitCode = null;
      job.status = "exited";
      job.finishedAt = new Date(now()).toISOString();
      saveJobs();
      onJobDone?.(job.name, null);
      log("info", "Detached workbench job process ended", { name: job.name, pid: job.pid });
    }, 5_000);
    timer?.unref?.();
    watches.set(job.id, timer);
  }
  try {
    for (const j of JSON.parse(readFileSync(join(jobsDir, "jobs.json"), "utf8"))) {
      if (j.status === "running" && j.pid && pidAlive(j.pid)) {
        const adopted = { ...j, status: "running" };
        jobs.set(j.id, adopted);
        watchPid(adopted);
      } else {
        jobs.set(j.id, j.status === "running" ? { ...j, status: "interrupted" } : j);
      }
    }
  } catch { /* first start */ }

  function startJob(name, confirm) {
    const spec = JOBS[name];
    if (!spec) throw Object.assign(new Error("Unknown job."), { status: 400 });
    if (spec.confirm && confirm !== true) throw Object.assign(new Error("This job changes saved state; confirm it first."), { status: 409 });
    const running = [...jobs.values()].filter((j) => j.status === "running");
    if (running.some((j) => j.name === name)) throw Object.assign(new Error("That job is already running."), { status: 409 });
    if (running.length >= 2) throw Object.assign(new Error("Two jobs are already running; wait for one to finish."), { status: 429 });
    mkdirSync(jobsDir, { recursive: true });
    const id = randomUUID();
    const stamp = new Date(now()).toISOString().replace(/[:T]/g, "-").slice(0, 19);
    const logFile = join(jobsDir, `${stamp}-${name}.log`);
    const job = { id, name, label: spec.label, status: "running", startedAt: new Date(now()).toISOString(),
      finishedAt: null, exitCode: null, logFile, expectedMinutes: spec.minutes };
    writeFileSync(logFile, `$ python ${spec.script} ${spec.args.join(" ")}\n`);
    const detached = Boolean(spec.detached);
    const spawnOpts = {
      cwd: engineDir,
      env: { ...env, PYTHONUTF8: "1", TM_JOB_ID: id, TM_REQUEST_ID: requestId() ?? "" },
      windowsHide: true,
    };
    let stdioFd;
    if (detached) {
      spawnOpts.detached = true;
      stdioFd = openSync(logFile, "a");
      spawnOpts.stdio = ["ignore", stdioFd, stdioFd];
    }
    let child;
    try {
      child = spawnImpl(python, ["-W", "ignore", spec.script, ...spec.args], spawnOpts);
    } finally {
      if (stdioFd !== undefined) {
        try { closeSync(stdioFd); } catch { /* the child inherited the handle */ }
      }
    }
    if (detached) child.unref?.();
    const write = (d) => { try { appendFileSync(logFile, d); } catch { /* ignore */ } };
    child.stdout?.on("data", write);
    child.stderr?.on("data", write);
    child.on("error", (e) => { write(`\n${e.message}\n`); job.status = "failed"; job.finishedAt = new Date(now()).toISOString(); saveJobs(); });
    child.on("close", (code) => {
      job.exitCode = code;
      job.status = code === 0 ? "complete" : "failed";
      job.finishedAt = new Date(now()).toISOString();
      healthCache = null;
      chartCache.clear();
      saveJobs();
      onJobDone?.(name, code);
      log(code === 0 ? "info" : "warn", "Workbench job finished", { name, code });
    });
    job.pid = child.pid ?? null;
    job.detached = detached;
    jobs.set(id, Object.defineProperty(job, "child", { value: child, enumerable: false }));
    saveJobs();
    log("info", "Workbench job started", { name, pid: job.pid, detached });
    return jobView(job);
  }
  function logTail(file, bytes = 6000) {
    try {
      const text = readFileSync(file, "utf8");
      return text.length > bytes ? text.slice(-bytes) : text;
    } catch { return ""; }
  }

  async function health(refresh, files) {
    if (!refresh && healthCache && healthCache.at > now() - 120_000) return healthCache.body;
    const body = JSON.parse(await runPython("data_health.py", ["--check-files", String(files)], { timeoutMs: 180_000 }));
    body.schwabLive = schwabStatus();
    healthCache = { at: now(), body };
    return body;
  }

  function readText(name) {
    const path = join(researchDir, name);
    return existsSync(path) ? readFileSync(path, "utf8") : null;
  }

  // ------------------------------------------------------------------ Schwab re-authorization (runs engine/schwab_reauth.py)
  let reauth = null;
  let authPython = null;
  const redact = (text) => {
    let out = String(text);
    for (const secret of [env.SCHWAB_APP_KEY, env.SCHWAB_APP_SECRET]) {
      if (secret && secret.trim().length >= 6) out = out.split(secret.trim()).join("[redacted]");
    }
    return out.replace(/(code|access_token|refresh_token|client_secret)=([^&\s"]+)/gi, "$1=[redacted]");
  };
  function probe(command, prefix) {
    return new Promise((resolve) => {
      try {
        const child = spawnImpl(command, [...prefix, "-c", "import schwab"], { cwd: engineDir, env, windowsHide: true });
        const timer = setTimeout(() => { child.kill(); resolve(false); }, 20_000);
        child.on("error", () => { clearTimeout(timer); resolve(false); });
        child.on("close", (code) => { clearTimeout(timer); resolve(code === 0); });
      } catch { resolve(false); }
    });
  }
  async function findAuthPython() {
    if (authPython) return authPython;
    const candidates = [];
    if (env.SCHWAB_AUTH_PYTHON) candidates.push([env.SCHWAB_AUTH_PYTHON, []]);
    // the repo .venv first (schwab-py is in engine/requirements.txt), then the server's Python, then system launchers
    const venv = join(root, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
    if (existsSync(venv)) candidates.push([venv, []]);
    candidates.push([python, []]);
    if (process.platform === "win32") candidates.push(["py", ["-3"]]);
    else candidates.push(["python3", []]);
    for (const [command, prefix] of candidates) {
      if (await probe(command, prefix)) { authPython = { command, prefix }; return authPython; }
    }
    return null;
  }
  async function startReauth() {
    if (reauth?.status === "running") return reauthView();
    if (!env.SCHWAB_APP_KEY?.trim() || !env.SCHWAB_APP_SECRET?.trim()) {
      throw Object.assign(new Error("Enter the Schwab app key and secret first (saved to .env.local)."), { status: 409 });
    }
    const chosen = await findAuthPython();
    if (!chosen) {
      throw Object.assign(new Error("No Python with schwab-py found. Set SCHWAB_AUTH_PYTHON to the Python you log in with."), { status: 500 });
    }
    reauth = { id: randomUUID(), status: "running", startedAt: new Date(now()).toISOString(), finishedAt: null,
      exitCode: null, lines: [], python: chosen.command };
    const job = reauth;
    const child = spawnImpl(chosen.command, [...chosen.prefix, "schwab_reauth.py"], {
      cwd: engineDir, env: { ...env, PYTHONUTF8: "1", PYTHONWARNINGS: "ignore" }, windowsHide: true });
    const add = (d) => {
      for (const line of redact(d).split(/\r?\n/)) if (line.trim()) job.lines.push(line);
      if (job.lines.length > 200) job.lines.splice(0, job.lines.length - 200);
    };
    child.stdout?.on("data", add);
    child.stderr?.on("data", add);
    child.on("error", (e) => { add(e.message); job.status = "failed"; job.finishedAt = new Date(now()).toISOString(); });
    child.on("close", (code) => {
      job.exitCode = code;
      job.status = code === 0 ? "complete" : "failed";
      job.finishedAt = new Date(now()).toISOString();
      schwabError = null;
      quoteCache.clear();
      log(code === 0 ? "info" : "warn", "Schwab re-authorization finished", { code });
      if (code === 0 && schwabStatus().usable) {
        try { startJob("schwab_daily_backfill"); log("info", "Started Schwab long daily backfill after reauthorization"); }
        catch (error) { log("warn", "Could not auto-start the Schwab daily backfill", { error: error.message }); }
      }
    });
    log("info", "Schwab re-authorization started", { python: chosen.command });
    return reauthView();
  }
  function reauthView() {
    return { job: reauth ? { ...reauth, lines: reauth.lines.slice(-60) } : null, token: schwabStatus() };
  }
  // Saves Schwab app settings to .env.local (gitignored). Values are never logged or returned; blank fields keep
  // the saved value. Only the callback URL (not secret) is reported back.
  function saveAppKeys(appKey, appSecret, callbackUrl) {
    const updates = {};
    const key = String(appKey ?? "").trim(), secret = String(appSecret ?? "").trim(), callback = String(callbackUrl ?? "").trim();
    if (key) {
      if (!/^[\w-]{8,128}$/.test(key)) throw Object.assign(new Error("The app key (client ID) should be letters and digits from developer.schwab.com."), { status: 400 });
      updates.SCHWAB_APP_KEY = key;
    }
    if (secret) {
      if (!/^[\w-]{8,128}$/.test(secret)) throw Object.assign(new Error("The app secret should be letters and digits from developer.schwab.com."), { status: 400 });
      updates.SCHWAB_APP_SECRET = secret;
    }
    if (callback) {
      let parsed;
      try { parsed = new URL(callback); } catch { parsed = null; }
      if (!parsed || parsed.protocol !== "https:" || !["127.0.0.1", "localhost"].includes(parsed.hostname)) {
        throw Object.assign(new Error("The callback URL must be the https://127.0.0.1:<port> address registered on your Schwab app."), { status: 400 });
      }
      updates.SCHWAB_CALLBACK_URL = callback.replace(/\/$/, "");
    }
    if (!Object.keys(updates).length) throw Object.assign(new Error("Nothing to save."), { status: 400 });
    const file = join(root, ".env.local");
    const lines = existsSync(file) ? readFileSync(file, "utf8").split(/\r?\n/) : [];
    const names = Object.keys(updates);
    const kept = lines.filter((l) => !names.some((n) => new RegExp(`^\\s*(export\\s+)?${n}\\s*=`).test(l)));
    while (kept.length && !kept[kept.length - 1].trim()) kept.pop();
    for (const [name, value] of Object.entries(updates)) kept.push(`${name}=${value}`);
    kept.push("");
    writeFileSync(`${file}.tmp`, kept.join("\n"), { mode: 0o600 });
    renameSync(`${file}.tmp`, file);
    Object.assign(env, updates);
    log("info", "Schwab app settings saved to .env.local", { fields: names });
    return schwabStatus();
  }

  // ------------------------------------------------------------------ routes
  function mount(app) {
    const fail = (response, error) => response.status(error.status || 500).json({ error: error.message });
    const cleanTicker = (value) => {
      const t = String(value ?? "").toUpperCase().trim();
      if (!TICKER.test(t)) throw Object.assign(new Error("Enter a valid ticker symbol."), { status: 400 });
      return t;
    };
    app.get("/api/chart/:ticker", async (request, response) => {
      try {
        const ticker = cleanTicker(request.params.ticker);
        const tf = ["1d", "5m", "1m"].includes(request.query.tf) ? request.query.tf : "1d";
        const sessions = request.query.sessions ? Math.max(1, Math.min(60, Number(request.query.sessions) || 0)) : null;
        response.json(await chart(ticker, tf, sessions, request.query.refresh === "1"));
      } catch (error) { fail(response, error); }
    });
    app.get("/api/market/quotes", async (request, response) => {
      try {
        const symbols = [...new Set(String(request.query.symbols ?? "").split(",").map((s) => s.trim().toUpperCase()).filter(Boolean))];
        if (!symbols.length || symbols.length > 40 || !symbols.every((s) => TICKER.test(s))) {
          throw Object.assign(new Error("Pass 1-40 ticker symbols."), { status: 400 });
        }
        response.json(await quotes(symbols));
      } catch (error) { fail(response, error); }
    });
    app.get("/api/schwab/status", (_request, response) => response.json({ ...schwabStatus(), market: easternClock(now()) }));
    app.get("/api/schwab/reauth", (_request, response) => response.json(reauthView()));
    app.post("/api/schwab/reauth", async (_request, response) => {
      try { response.status(202).json(await startReauth()); } catch (error) { fail(response, error); }
    });
    app.post("/api/schwab/app-keys", (request, response) => {
      try { response.json(saveAppKeys(request.body?.appKey, request.body?.appSecret, request.body?.callbackUrl)); } catch (error) { fail(response, error); }
    });
    app.get("/api/data-health", async (request, response) => {
      try {
        response.json(await health(request.query.refresh === "1", Math.max(0, Math.min(60, Number(request.query.files ?? 5) || 5))));
      } catch (error) { fail(response, error); }
    });
    app.get("/api/paper2/tables", async (request, response) => {
      try {
        const account = /^[\w-]{1,40}$/.test(String(request.query.account ?? "")) ? String(request.query.account) : "default";
        response.json(await paperTables(account, Math.max(1, Math.min(1000, Number(request.query.limit ?? 200) || 200))));
      } catch (error) { fail(response, error); }
    });
    app.get("/api/research/report", (_request, response) => {
      const text = readText("research_report.md");
      text == null ? response.status(404).json({ error: "No research report yet. Run Research: all." }) : response.type("text/plain").send(text);
    });
    app.get("/api/ops/jobs", (_request, response) => response.json({
      available: Object.entries(JOBS).map(([name, j]) => ({ name, label: j.label, confirm: Boolean(j.confirm), minutes: j.minutes })),
      jobs: [...jobs.values()].reverse().slice(0, 30).map(jobView),
    }));
    app.get("/api/ops/jobs/:id", (request, response) => {
      const job = jobs.get(request.params.id);
      if (!job) return response.status(404).json({ error: "Unknown job." });
      response.json({ ...jobView(job), log: logTail(job.logFile) });
    });
    app.post("/api/ops/run", (request, response) => {
      try { response.status(202).json(startJob(String(request.body?.job ?? ""), request.body?.confirm)); } catch (error) { fail(response, error); }
    });
    app.get("/api/schwab/daily-backfill", (_request, response) => {
      let summary = null;
      const text = readText("schwab_daily_backfill.json");
      if (text) { try { summary = JSON.parse(text); } catch { summary = null; } }
      const token = schwabStatus();
      response.json({
        token: { usable: token.usable, accessTokenValid: token.accessTokenValid, canRefresh: token.canRefresh,
          reason: token.reason, setupNeeded: token.setupNeeded, accessTokenExpires: token.accessTokenExpires,
          refreshTokenExpiresAbout: token.refreshTokenExpiresAbout },
        summary,
        running: [...jobs.values()].some((j) => j.name === "schwab_daily_backfill" && j.status === "running"),
      });
    });
    app.post("/api/schwab/daily-backfill", (request, response) => {
      try { response.status(202).json(startJob("schwab_daily_backfill", request.body?.confirm)); } catch (error) { fail(response, error); }
    });
  }

  return { mount, quotes, chart, schwabStatus, startJob, jobs, health, paperTables, startReauth, reauthView, saveAppKeys, _calls: calls };
}
