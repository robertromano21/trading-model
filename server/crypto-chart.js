// Chart-picker symbol lookup and crypto chart data. Pure local lookup: the fixed crypto universe below plus the local
// daily price database (Massive/prices.sqlite3). No AI/LLM calls, no network for the lookup. Crypto charts come from
// engine/crypto_chart.py (stored Coinbase public candles; market data only, nothing here can place or preview orders).
import { spawn } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

// The crypto universe the dashboard stores candles for (engine/crypto_data.py COINS). schwab: spot = Schwab Crypto
// trades it (0.75%/side), announced = on Schwab's announced list, no = not offered by Schwab.
export const CRYPTO_UNIVERSE = Object.freeze([
  { symbol: "BTC", name: "Bitcoin", aliases: ["BITCOIN", "XBT", "BTCUSD", "BTC-USD"], schwab: "spot" },
  { symbol: "ETH", name: "Ether (Ethereum)", aliases: ["ETHER", "ETHEREUM", "ETHUSD", "ETH-USD"], schwab: "spot" },
  { symbol: "SOL", name: "Solana", aliases: ["SOLANA", "SOLUSD", "SOL-USD"], schwab: "announced" },
  { symbol: "AVAX", name: "Avalanche", aliases: ["AVALANCHE", "AVAXUSD", "AVAX-USD"], schwab: "announced" },
  { symbol: "LINK", name: "Chainlink", aliases: ["CHAINLINK", "LINKUSD", "LINK-USD"], schwab: "announced" },
  { symbol: "XRP", name: "XRP", aliases: ["RIPPLE", "XRPUSD", "XRP-USD"], schwab: "no" },
  { symbol: "DOGE", name: "Dogecoin", aliases: ["DOGECOIN", "DOGEUSD", "DOGE-USD"], schwab: "no" },
  { symbol: "ADA", name: "Cardano", aliases: ["CARDANO", "ADAUSD", "ADA-USD"], schwab: "no" },
  { symbol: "DOT", name: "Polkadot", aliases: ["POLKADOT", "DOTUSD", "DOT-USD"], schwab: "no" },
  { symbol: "LTC", name: "Litecoin", aliases: ["LITECOIN", "LTCUSD", "LTC-USD"], schwab: "no" },
  { symbol: "BCH", name: "Bitcoin Cash", aliases: ["BITCOINCASH", "BITCOIN CASH", "BCHUSD", "BCH-USD"], schwab: "no" },
]);
// Listed securities that share a coin's symbol (local labels; the dropdown still works without a name).
const SAME_SYMBOL_SECURITY = Object.freeze({
  BTC: "Grayscale Bitcoin Mini Trust ETF", ETH: "Grayscale Ethereum Mini Trust ETF", SOL: "Emeren Group",
  LINK: "Interlink Electronics", LTC: "LTC Properties", BCH: "Banco de Chile (ADR)",
});
const TICKER = /^[A-Z][A-Z0-9.\-]{0,9}$/;
const LOOKUP_TTL_MS = 10 * 60_000;
const SEARCH_RE = /^[A-Z0-9.\-]{2,12}$/;
const SEARCH_TTL_MS = 30 * 60_000;
const MAX_FETCHES = 2;

export function cleanQuery(q) {
  return String(q ?? "").trim().toUpperCase().replace(/^\$/, "").replace(/\s+/g, " ").slice(0, 24);
}

export function cryptoMatches(q) {
  const s = cleanQuery(q);
  if (!s) return [];
  return CRYPTO_UNIVERSE.filter((c) => c.symbol === s || c.aliases.includes(s) || c.aliases.includes(s.replace(/ /g, "")));
}

/** Local stock lookup: { sessions, last } when the symbol has daily bars in prices.sqlite3, else null. */
export function createStockIndex(dataDir) {
  const path = join(dataDir, "prices.sqlite3");
  const cache = new Map();
  let db = null, dbStamp = 0;
  const open = () => {
    if (!existsSync(path)) return null;
    const stamp = statSync(path).mtimeMs;
    if (db && stamp === dbStamp) return db;
    try { db?.close(); } catch { /* ignore */ }
    db = new DatabaseSync(path, { readOnly: true });
    dbStamp = stamp;
    cache.clear();
    return db;
  };
  return {
    get(symbol) {
      const s = String(symbol).toUpperCase();
      if (!TICKER.test(s)) return null;
      const hit = cache.get(s);
      if (hit && hit.at > Date.now() - LOOKUP_TTL_MS) return hit.value;
      let value = null;
      try {
        const conn = open();
        const row = conn?.prepare("SELECT COUNT(*) AS n, MAX(date) AS last FROM daily_prices WHERE ticker = ?").get(s);
        value = row && row.n > 0 ? { sessions: Number(row.n), last: row.last } : null;
      } catch { value = null; }
      cache.set(s, { at: Date.now(), value });
      if (cache.size > 2000) cache.delete(cache.keys().next().value);
      return value;
    },
    close() { try { db?.close(); } catch { /* ignore */ } db = null; },
  };
}

/** Matches for the chart picker. One match opens directly; more than one shows a dropdown. */
export function lookupSymbols(q, stocks) {
  const s = cleanQuery(q);
  const matches = [];
  for (const c of cryptoMatches(s)) {
    matches.push({ kind: "crypto", symbol: c.symbol, name: c.name, schwab: c.schwab,
      detail: `Crypto · Coinbase spot${c.schwab === "spot" ? " · Schwab Crypto (paper only here)" : ""}`,
      href: `/ticker.html?symbol=${encodeURIComponent(c.symbol)}&asset=crypto` });
  }
  const ticker = s.replace(/ /g, "");
  if (TICKER.test(ticker)) {
    const local = stocks?.get(ticker) ?? null;
    if (local) {
      matches.push({ kind: "stock", symbol: ticker, name: SAME_SYMBOL_SECURITY[ticker] || "", sessions: local.sessions, last: local.last,
        detail: `Stock / ETF · ${local.sessions} daily bars to ${local.last}`, href: `/ticker.html?symbol=${encodeURIComponent(ticker)}` });
    } else if (!matches.length) {
      // not stored locally and not a coin: keep the old behaviour (the chart page tries Schwab / the local DB)
      matches.push({ kind: "stock", symbol: ticker, name: "", detail: "Stock / ETF (not in the local database)",
        href: `/ticker.html?symbol=${encodeURIComponent(ticker)}` });
    }
  }
  return { query: s, matches };
}

/** Add Schwab instrument-search hits (symbols not in the local data) to a local lookup result. */
export function withRemote(result, remote) {
  const have = new Set(result.matches.filter((m) => m.kind !== "stock" || m.sessions).map((m) => `${m.kind}:${m.symbol}`));
  const extra = (remote?.matches ?? []).filter((r) => !have.has(`stock:${r.symbol}`)).map((r) => ({
    kind: "schwab", symbol: r.symbol, name: r.description || "",
    detail: `Schwab · ${[r.exchange, r.assetType].filter(Boolean).join(" · ")} · history loads when picked`,
    href: `/ticker.html?symbol=${encodeURIComponent(r.symbol)}&fetch=1` }));
  if (!extra.length) return { ...result, remoteError: remote?.error };
  // the "not in the local database" placeholder is replaced by real Schwab hits
  const kept = result.matches.filter((m) => !(m.kind === "stock" && !m.sessions));
  return { ...result, matches: [...kept, ...extra], remote: true };
}

/** Express routes: GET /api/symbols/lookup?q=[&remote=1], GET|POST /api/symbols/fetch/:ticker,
 *  GET /api/crypto/chart/:coin?tf=1d|1h&refresh=1 */
export function mountCryptoChart(app, { root, dataDir, python, log = () => {} } = {}) {
  const stocks = createStockIndex(dataDir);
  const charts = new Map();
  const running = new Map();
  const exe = python || (process.platform === "win32" ? join(root, ".venv", "Scripts", "python.exe") : "python3");
  const runChart = (coin, tf, refresh) => {
    const key = `${coin}|${tf}|${refresh ? 1 : 0}`;
    if (running.has(key)) return running.get(key);
    const job = new Promise((resolve, reject) => {
      const args = ["-u", join(root, "engine", "crypto_chart.py"), coin, "--tf", tf, ...(refresh ? ["--refresh"] : [])];
      const child = spawn(exe, args, { cwd: join(root, "engine"), windowsHide: true, env: { ...process.env, PYTHONUTF8: "1" },
        stdio: ["ignore", "pipe", "pipe"] });
      let out = "", err = "";
      const timer = setTimeout(() => { try { child.kill(); } catch { /* gone */ } }, 90_000);
      child.stdout.on("data", (d) => { out += d; });
      child.stderr.on("data", (d) => { err = (err + d).slice(-2000); });
      child.on("error", (e) => { clearTimeout(timer); reject(e); });
      child.on("close", (code) => {
        clearTimeout(timer);
        if (code !== 0) return reject(new Error(`crypto_chart.py exited ${code}: ${err.trim().split(/\r?\n/).pop() || ""}`));
        try { resolve(JSON.parse(out)); } catch (e) { reject(new Error(`Bad chart JSON: ${e.message}`)); }
      });
    }).finally(() => running.delete(key));
    running.set(key, job);
    return job;
  };

  // Generic engine runner (async child process; never blocks the event loop).
  const runPy = (script, args, timeoutMs = 30_000) => new Promise((resolve, reject) => {
    const child = spawn(exe, ["-u", join(root, "engine", script), ...args], { cwd: join(root, "engine"), windowsHide: true,
      env: { ...process.env, PYTHONUTF8: "1" }, stdio: ["ignore", "pipe", "ignore"] });
    let out = "";
    const timer = setTimeout(() => { try { child.kill(); } catch { /* gone */ } }, timeoutMs);
    child.stdout.on("data", (d) => { out += d; });
    child.on("error", (e) => { clearTimeout(timer); reject(e); });
    child.on("close", () => { clearTimeout(timer); try { resolve(JSON.parse(out.trim().split(/\r?\n/).pop())); } catch { reject(new Error("no result")); } });
  });
  // Schwab instrument lookup for symbols not in the local data: cached, de-duplicated, one at a time.
  const searches = new Map();
  let searchChain = Promise.resolve();
  const searchSchwab = (q) => {
    const hit = searches.get(q);
    if (hit && hit.at > Date.now() - SEARCH_TTL_MS) return hit.promise;
    const promise = (searchChain = searchChain.catch(() => {}).then(() => runPy("schwab_ondemand.py", ["search", q], 20_000)))
      .catch((e) => ({ query: q, matches: [], error: e.message }));
    searches.set(q, { at: Date.now(), promise });
    if (searches.size > 300) searches.delete(searches.keys().next().value);
    return promise;
  };
  app.get("/api/symbols/lookup", async (request, response) => {
    try {
      const q = cleanQuery(request.query.q);
      if (!q) return response.status(400).json({ error: "Type a ticker or coin." });
      let result = lookupSymbols(q, stocks);
      const ticker = q.replace(/ /g, "");
      const localStock = result.matches.some((m) => m.kind === "stock" && m.sessions);
      if (request.query.remote === "1" && !localStock && SEARCH_RE.test(ticker)) result = withRemote(result, await searchSchwab(ticker));
      response.set("Cache-Control", "private, max-age=120");
      response.json(result);
    } catch (error) { response.status(500).json({ error: error.message }); }
  });
  // On-demand history for the one symbol the user picked (or whose local history is short). Detached child process;
  // per-symbol cooldown/lock live in engine/schwab_ondemand.py. Poll GET for the status.
  const fetching = new Set();
  const statusFile = (t) => join(dataDir, "Research", "ondemand", `${t.replace(/\//g, "_").replace(/^\$/, "IDX_")}.json`);
  const readStatus = (t) => { try { return JSON.parse(readFileSync(statusFile(t), "utf8")); } catch { return { status: "idle" }; } };
  const cleanSym = (v) => {
    const t = String(v ?? "").toUpperCase().trim();
    if (!TICKER.test(t)) throw Object.assign(new Error("Enter a valid ticker symbol."), { status: 400 });
    return t;
  };
  app.get("/api/symbols/fetch/:ticker", (request, response) => {
    try {
      const t = cleanSym(request.params.ticker);
      response.json({ symbol: t, ...readStatus(t), ...(fetching.has(t) ? { status: "running" } : {}) });
    } catch (error) { response.status(error.status || 500).json({ error: error.message }); }
  });
  app.post("/api/symbols/fetch/:ticker", (request, response) => {
    try {
      const t = cleanSym(request.params.ticker);
      if (fetching.has(t)) return response.status(202).json({ symbol: t, status: "running" });
      if (fetching.size >= MAX_FETCHES) return response.status(429).json({ symbol: t, status: "busy", error: "Other Schwab history fetches are running; try again shortly." });
      fetching.add(t);
      runPy("schwab_ondemand.py", ["fetch", t], 180_000)
        .catch((e) => log("warn", "On-demand Schwab fetch failed", { symbol: t, error: e.message }))
        .finally(() => fetching.delete(t));
      response.status(202).json({ symbol: t, status: "running" });
    } catch (error) { response.status(error.status || 500).json({ error: error.message }); }
  });
  app.get("/api/crypto/universe", (_request, response) => response.json({ coins: CRYPTO_UNIVERSE }));
  app.get("/api/crypto/chart/:coin", async (request, response) => {
    try {
      const coin = cleanQuery(request.params.coin);
      if (!CRYPTO_UNIVERSE.some((c) => c.symbol === coin)) return response.status(400).json({ error: "Unknown coin." });
      const tf = request.query.tf === "1h" ? "1h" : "1d";
      const refresh = request.query.refresh === "1";
      const key = `${coin}|${tf}`;
      const ttl = tf === "1h" ? 60_000 : 5 * 60_000;
      const hit = charts.get(key);
      let body;
      if (!refresh && hit && hit.at > Date.now() - ttl) body = hit.body;
      else {
        body = await runChart(coin, tf, refresh);
        charts.set(key, { at: Date.now(), body });
      }
      response.json({ ...body, notes: [] });
    } catch (error) {
      log("warn", "Crypto chart failed", { error: error.message });
      response.status(502).json({ error: error.message });
    }
  });
  return { stocks, lookup: (q) => lookupSymbols(q, stocks) };
}
