// Crypto awareness for the AI analyst: the [CRYPTO] context block, crypto-question detection, the CRYPTO_STATUS
// tool payload and a lightweight background refresher of engine/crypto_snapshot.py (Coinbase spot, Schwab
// market-data quotes, FRED fallback). Read-only market data; nothing here can place or preview orders.
import { spawn } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const massiveDir = (dataDir) => dataDir || process.env.PRICES_DATA_DIR || fileURLToPath(new URL("../../Massive", import.meta.url));
export const SNAPSHOT_REL = ["Research", "crypto", "crypto_snapshot.json"];
export const PAPER_REL = ["Research", "crypto", "paper", "crypto_paper_status.json"];
export const INTRADAY_REL = ["Research", "crypto", "paper", "crypto_intraday_status.json"];
export const REFRESH_MS = 5 * 60_000;

// Coin symbols are never stock tickers in a crypto question (BTC is also a Grayscale ETF, SOL/LINK are equities).
export const COIN_SYMBOLS = Object.freeze(["BTC", "ETH", "SOL", "AVAX", "LINK", "XRP", "DOGE", "ADA", "DOT", "LTC", "BCH"]);
const CRYPTO_RE = /\b(crypto(?:currency|currencies|s)?|bitcoin|btc|ether(?:eum)?|eth|solana|sol|avalanche|avax|chainlink|ripple|xrp|dogecoin|doge|cardano|litecoin|altcoins?|stablecoins?|coinbase prices?|crypto paper|crypto accounts?)\b/i;
// "sol"/"link"/"eth" alone are ambiguous in English; require a crypto-ish neighbour for those three.
const WEAK = /^(sol|link|eth|dot|ada)$/i;

export function isCryptoQuestion(text) {
  const t = String(text || "");
  const m = t.match(CRYPTO_RE);
  if (!m) return false;
  if (!WEAK.test(m[1])) return true;
  return /\b(crypto|coin|token|price|doing|bitcoin|ether|solana|chain|btc)\b/i.test(t) || /\$(sol|eth|link|dot|ada)\b/i.test(t)
    || /^\s*(sol|eth|link|dot|ada)\s*\??\s*$/i.test(t) || /\b(SOL|ETH)\b/.test(t);
}

const readJson = (p) => { try { return JSON.parse(readFileSync(p, "utf8")); } catch { return null; } };

export function snapshotAgeMs(dataDir) {
  const p = join(massiveDir(dataDir), ...SNAPSHOT_REL);
  try { return Date.now() - statSync(p).mtimeMs; } catch { return Infinity; }
}

/** [CRYPTO] context block (prices, trend, dollar/liquidity, nowcast regime, paper accounts, backtest verdict). */
export function cryptoBlock(dataDir, maxChars = 1800) {
  const s = readJson(join(massiveDir(dataDir), ...SNAPSHOT_REL));
  if (!s || !s.context) return null;
  const age = snapshotAgeMs(dataDir);
  const stale = Number.isFinite(age) && age > 6 * 3600_000 ? ` (snapshot is ${Math.round(age / 3600_000)} h old)` : "";
  const text = s.context + stale;
  return text.length <= maxChars ? text : text.slice(0, maxChars - 1) + "…";
}

/** Strip coin symbols from a ticker list for crypto questions (keeps real ETPs like IBIT/FBTC and COIN/MSTR). */
export function stripCoinTickers(tickers, text) {
  if (!isCryptoQuestion(text)) return tickers;
  return (tickers || []).filter((t) => !COIN_SYMBOLS.includes(String(t).toUpperCase().replace(/^\$/, "")));
}

let running = null;
/** Spawn engine/crypto_snapshot.py (detached-lite: no window, own lock; skips if fresh). Resolves when it exits. */
export function refreshSnapshot({ root, python, maxAgeMin = 4, timeoutMs = 45_000, log = () => {} } = {}) {
  if (running) return running;
  const script = join(root, "engine", "crypto_snapshot.py");
  if (!root || !existsSync(script)) return Promise.resolve({ status: "missing" });
  const exe = python || (process.platform === "win32" ? join(root, ".venv", "Scripts", "python.exe") : "python3");
  running = new Promise((resolve) => {
    let out = "";
    let child;
    try {
      child = spawn(exe, ["-u", script, "--max-age-min", String(maxAgeMin)], { cwd: join(root, "engine"), windowsHide: true,
        env: { ...process.env, PYTHONUTF8: "1" }, stdio: ["ignore", "pipe", "ignore"] });
    } catch (error) { running = null; resolve({ status: "error", error: error.message }); return; }
    const timer = setTimeout(() => { try { child.kill(); } catch { /* gone */ } }, timeoutMs);
    timer.unref?.();
    child.stdout.on("data", (d) => { out += d; });
    child.on("error", (error) => { clearTimeout(timer); running = null; resolve({ status: "error", error: error.message }); });
    child.on("close", (code) => {
      clearTimeout(timer); running = null;
      let parsed = { status: code === 0 ? "ok" : "error", code };
      try { parsed = JSON.parse(out.slice(out.lastIndexOf("{"))); } catch { /* keep */ }
      if (parsed.status === "error") log("warn", "Crypto snapshot refresh failed", parsed);
      resolve(parsed);
    });
  });
  return running;
}

/** Background refresher while the dashboard runs (TRADING_CRYPTO_SNAPSHOT=off disables). Returns a stop function. */
export function startCryptoRefresher({ root, python, env = process.env, log = () => {}, everyMs = REFRESH_MS } = {}) {
  if (String(env.TRADING_CRYPTO_SNAPSHOT ?? "").toLowerCase() === "off") return () => {};
  const tick = () => { refreshSnapshot({ root, python, maxAgeMin: Math.max(1, everyMs / 60_000 - 0.5), log }).catch(() => {}); };
  const first = setTimeout(tick, 20_000);
  first.unref?.();
  const timer = setInterval(tick, everyMs);
  timer.unref?.();
  return () => { clearTimeout(first); clearInterval(timer); };
}

/** CRYPTO_STATUS tool payload: refresh if older than 10 minutes, then summarize snapshot + paper accounts + verdict. */
export async function cryptoStatus({ root, dataDir, python, refresh = true, log } = {}) {
  const dir = massiveDir(dataDir);
  let refreshed = null;
  if (refresh && snapshotAgeMs(dir) > 10 * 60_000) refreshed = await refreshSnapshot({ root, python, maxAgeMin: 0, timeoutMs: 40_000, log });
  const snap = readJson(join(dir, ...SNAPSHOT_REL));
  const paper = readJson(join(dir, ...PAPER_REL));
  const intraday = readJson(join(dir, ...INTRADAY_REL));
  if (!snap && !paper && !intraday) return { available: false, context: "[CRYPTO] No crypto snapshot yet (engine/crypto_snapshot.py)." };
  const accounts = (paper?.accounts || []).map((a) => ({ id: a.id, label: a.label, equity: a.equity, totalReturn: a.totalReturn,
    maxDrawdown: a.maxDrawdown, targetWeights: a.targetWeights, cash: a.cash, schwabTradableNow: a.schwabTradableNow,
    dollarAware: a.dollarAware, backtestSharpe: a.preHoldoutBacktest?.sharpe, backtestMaxDD: a.preHoldoutBacktest?.maxDrawdown }));
  return {
    available: true, asOf: snap?.asOf, refreshed, coins: snap?.coins, macro: snap?.macro, nowcast: snap?.nowcast,
    schwabQuotes: snap?.schwab, verdict: snap?.verdict,
    short: snap?.short ?? null,   // SBIT (-2x) / BITI (-1x) inverse-ETF research + shadow accounts 220-223
    paper: paper && { forwardStart: paper.forwardStart, dataEnd: paper.dataEnd, days: paper.days, leader: paper.leader,
      btcBaselineReturn: paper.btcBaselineReturn, accounts },
    intraday,
    notes: "Spot from Coinbase public candles (UTC days, 24/7); Schwab market data for futures, $BTC/$ETH indexes and ETPs; "
      + "FRED CBBTCUSD/CBETHUSD fallback. Schwab Crypto trades BTC and ETH at 0.75% per side; SOL/AVAX/LINK announced. "
      + "Crypto accounts 200-223 are paper simulations only; 220-223 are SBIT shadow experiments. SBIT is a -2x daily "
      + "inverse BTC ETF (BITI is -1x): it trades only in US market hours, so 24/7 signals fill at the next session.",
    context: snap?.context || "[CRYPTO] snapshot missing; paper status only.",
  };
}
