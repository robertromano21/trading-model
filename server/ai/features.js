// Deterministic price features from the local Massive database (read-only) plus the live Schwab quote.
// These numbers are computed in code, never by the model, and they also drive the safe fallback verdict,
// the overextension classification and the stop / entry-band clamps.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const std = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, a.length - 1)); };
const sma = (a, n) => (a.length >= n ? mean(a.slice(-n)) : null);
const round = (x, d = 4) => (x == null || !Number.isFinite(x) ? null : Math.round(x * 10 ** d) / 10 ** d);

export function openPrices(dataDir) {
  const path = join(dataDir, "prices.sqlite3");
  if (!existsSync(path)) return null;
  const db = new DatabaseSync(path, { readOnly: true });
  db.exec("PRAGMA busy_timeout=5000");
  const stmt = db.prepare("SELECT date, open, high, low, close, volume FROM daily_prices WHERE ticker=? ORDER BY date DESC LIMIT ?");
  const exists = db.prepare("SELECT 1 FROM daily_prices WHERE ticker=? LIMIT 1");
  return {
    bars: (ticker, n = 300) => stmt.all(ticker, n).reverse(),
    has: (ticker) => Boolean(exists.get(ticker)),
    close: () => db.close(),
  };
}

function rsi(closes, n) {
  if (closes.length <= n) return null;
  let gain = 0, loss = 0;
  for (let i = closes.length - n; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    if (d > 0) gain += d; else loss -= d;
  }
  if (loss === 0) return 100;
  return 100 - 100 / (1 + gain / loss);
}

function moments(r) {
  if (r.length < 30) return { skew: null, exKurt: null };
  const m = mean(r), s = std(r);
  const skew = mean(r.map((x) => ((x - m) / s) ** 3));
  const kurt = mean(r.map((x) => ((x - m) / s) ** 4)) - 3;
  return { skew, exKurt: kurt };
}

export function classifyOverextension(f) {
  const z = f.zPrice20, a = f.atrDistance50, r14 = f.rsi14, r2 = f.rsi2;
  if ((z != null && z >= 2.5) || (a != null && a >= 6) || (r14 != null && r14 >= 80)) return "OVEREXTENDED";
  if ((z != null && z <= -2.5) || (r2 != null && r2 <= 5 && r14 != null && r14 <= 30)) return "DEEPLY_OVERSOLD";
  if ((z != null && z >= 1.5) || (a != null && a >= 4) || (r14 != null && r14 >= 70)) return "EXTENDED";
  if ((z != null && z <= -1.5) || (r14 != null && r14 <= 30)) return "OVERSOLD";
  return "NEUTRAL";
}

export function classifyTrend(f) {
  const p = f.price, s50 = f.sma50, s100 = f.sma100, s200 = f.sma200, r100 = f.ret100;
  if (p == null || s50 == null) return "SIDEWAYS";
  const above = [s50, s100, s200].filter((s) => s != null).map((s) => p > s);
  const up = above.every(Boolean), down = above.every((x) => !x);
  if (up && s100 != null && s50 > s100 && (s200 == null || s100 > s200) && r100 > 0.15) return "STRONG_UP";
  if (down && s100 != null && s50 < s100 && (s200 == null || s100 < s200) && r100 < -0.15) return "STRONG_DOWN";
  if (p > s50 && (r100 == null || r100 > 0)) return "UP";
  if (p < s50 && (r100 == null || r100 < 0)) return "DOWN";
  return "SIDEWAYS";
}

/** bars: [{date, open, high, low, close, volume}] ascending. live: optional Schwab quote (newer than the last bar). */
export function computeFeatures(bars, { live = null, benchmark = null } = {}) {
  if (!bars?.length) return null;
  const closes = bars.map((b) => b.close);
  const lastBar = bars.at(-1);
  let price = lastBar.close, priceSource = "massive_close", priceAsOf = lastBar.date;
  const liveTime = live?.quoteTimeMs ? new Date(live.quoteTimeMs) : null;
  const liveDate = liveTime ? liveTime.toLocaleDateString("en-CA", { timeZone: "America/New_York" }) : null;
  const livePrice = live?.last ?? live?.mark ?? null;
  if (livePrice > 0 && liveDate && liveDate >= lastBar.date) {
    price = livePrice; priceSource = "schwab_live"; priceAsOf = liveTime.toISOString();
  }
  const series = priceSource === "schwab_live" && liveDate > lastBar.date ? [...closes, price] : [...closes.slice(0, -1), price];
  const ret = (n) => (series.length > n ? price / series[series.length - 1 - n] - 1 : null);
  const daily = [];
  for (let i = 1; i < closes.length; i++) daily.push(Math.log(closes[i] / closes[i - 1]));
  const trs = [];
  for (let i = 1; i < bars.length; i++) {
    const b = bars[i], pc = bars[i - 1].close;
    trs.push(Math.max(b.high - b.low, Math.abs(b.high - pc), Math.abs(b.low - pc)));
  }
  const atr14 = trs.length >= 14 ? mean(trs.slice(-14)) : null;
  const s20 = sma(series, 20), s50 = sma(series, 50), s100 = sma(series, 100), s200 = sma(series, 200);
  const std20 = series.length >= 20 ? std(series.slice(-20)) : null;
  const vol = (n) => (daily.length >= n ? std(daily.slice(-n)) * Math.sqrt(252) : null);
  const window = series.slice(-252);
  const hi = Math.max(...window), lo = Math.min(...window);
  let peak = -Infinity, maxDD = 0;
  for (const c of window) { peak = Math.max(peak, c); maxDD = Math.min(maxDD, c / peak - 1); }
  const gaps = [];
  for (let i = 1; i < bars.length; i++) gaps.push(Math.log(bars[i].open / bars[i - 1].close));
  const gapSd = gaps.length > 30 ? std(gaps.slice(-252)) : null;
  const bigGaps = gapSd ? gaps.slice(-252).filter((g) => Math.abs(g) > 3 * gapSd).length : null;
  const worstGap = gaps.length ? Math.min(...gaps.slice(-252)) : null;
  const dollarVol20 = bars.length >= 20 ? mean(bars.slice(-20).map((b) => b.close * b.volume)) : null;
  const volRatio = bars.length >= 21 ? lastBar.volume / mean(bars.slice(-21, -1).map((b) => b.volume)) : null;
  const donchian = (n) => (series.length > n ? { high: Math.max(...series.slice(-n - 1, -1)), low: Math.min(...series.slice(-n - 1, -1)) } : null);
  const d20 = donchian(20), d55 = donchian(55);
  const m = moments(daily.slice(-252));
  let rs60 = null, rs120 = null, beta = null, corrSpy = null;
  if (benchmark?.length > 130) {
    const bmap = new Map(benchmark.map((b) => [b.date, b.close]));
    const paired = bars.filter((b) => bmap.has(b.date)).map((b) => [b.close, bmap.get(b.date)]);
    const rel = (n) => (paired.length > n ? (paired.at(-1)[0] / paired.at(-1 - n)[0]) / (paired.at(-1)[1] / paired.at(-1 - n)[1]) - 1 : null);
    rs60 = rel(60); rs120 = rel(120);
    const ra = [], rb = [];
    for (let i = Math.max(1, paired.length - 120); i < paired.length; i++) {
      ra.push(Math.log(paired[i][0] / paired[i - 1][0])); rb.push(Math.log(paired[i][1] / paired[i - 1][1]));
    }
    if (ra.length > 30) {
      const ma = mean(ra), mb = mean(rb);
      const cov = mean(ra.map((x, i) => (x - ma) * (rb[i] - mb)));
      beta = cov / (std(rb) ** 2); corrSpy = cov / (std(ra) * std(rb));
    }
  }
  const f = {
    lastBarDate: lastBar.date, sessions: bars.length, price, priceSource, priceAsOf,
    lastClose: lastBar.close, ret1: ret(1), ret5: ret(5), ret20: ret(20), ret60: ret(60), ret100: ret(100),
    ret120: ret(120), ret250: ret(250), sma20: s20, sma50: s50, sma100: s100, sma200: s200,
    distSma20: s20 ? price / s20 - 1 : null, distSma50: s50 ? price / s50 - 1 : null,
    distSma100: s100 ? price / s100 - 1 : null, distSma200: s200 ? price / s200 - 1 : null,
    zPrice20: std20 ? (price - s20) / std20 : null, atr14, atrPct: atr14 ? atr14 / price : null,
    atrDistance50: atr14 && s50 ? (price - s50) / atr14 : null,
    rsi2: rsi(series, 2), rsi14: rsi(series, 14), vol20: vol(20), vol60: vol(60),
    high252: hi, low252: lo, fromHigh252: price / hi - 1, fromLow252: price / lo - 1, maxDrawdown252: maxDD,
    donchian20: d20, donchian55: d55, breakout20: d20 ? price > d20.high : null, breakdown20: d20 ? price < d20.low : null,
    tsmom50: ret(50) == null ? null : Math.sign(ret(50)), tsmom100: ret(100) == null ? null : Math.sign(ret(100)),
    rs60VsSpy: rs60, rs120VsSpy: rs120, beta120: beta, corrSpy120: corrSpy,
    dollarVolume20: dollarVol20, volumeRatio: volRatio, gapSd, bigGaps252: bigGaps, worstGap252: worstGap,
    dailySkew252: m.skew, dailyExKurt252: m.exKurt,
  };
  f.overextension = classifyOverextension(f);
  f.trend = classifyTrend(f);
  for (const k of Object.keys(f)) if (typeof f[k] === "number") f[k] = round(f[k], Math.abs(f[k]) >= 1e6 ? 0 : 4);
  return f;
}

export function marketRegime(spyBars) {
  if (!spyBars?.length || spyBars.length < 60) return null;
  const f = computeFeatures(spyBars);
  const closes = spyBars.map((b) => b.close), r = [];
  for (let i = 1; i < closes.length; i++) r.push(Math.log(closes[i] / closes[i - 1]));
  const vols = [];
  for (let i = 20; i <= r.length; i++) vols.push(std(r.slice(i - 20, i)));
  const now = vols.at(-1);
  const pct = vols.filter((v) => v <= now).length / vols.length;
  return { asOf: f.lastBarDate, spyPrice: f.price, spyAbove200: f.sma200 ? f.price > f.sma200 : null,
    spyDistSma200: f.distSma200, spyRet20: f.ret20, spyVol20: f.vol20, spyVolPercentile1y: round(pct, 3),
    regime: f.sma200 && f.price > f.sma200 ? (pct > 0.8 ? "UPTREND_HIGH_VOL" : "UPTREND") : (pct > 0.8 ? "DOWNTREND_HIGH_VOL" : "DOWNTREND") };
}

// Read-only view of the paper trader's book (engine/paper_trader.py ledger), if it exists.
export function paperBook(dataDir, account = "default") {
  const path = join(dataDir, "paper_trading.sqlite3");
  if (!existsSync(path)) return { available: false, positions: [], targets: [], equity: null };
  let db;
  try {
    db = new DatabaseSync(path, { readOnly: true });
    const positions = db.prepare("SELECT ticker, shares, avg_cost, entry_date, last_price, last_price_date FROM positions WHERE account=? AND shares<>0").all(account);
    const acct = db.prepare("SELECT cash, initial_equity FROM accounts WHERE name=?").get(account);
    let targets = [];
    try {
      targets = db.prepare("SELECT ticker, target_weight, sleeve, stop FROM signal_state WHERE account=? AND asof=(SELECT MAX(asof) FROM signal_state WHERE account=?)").all(account, account);
    } catch { /* table may not exist yet */ }
    const invested = positions.reduce((s, p) => s + p.shares * (p.last_price ?? p.avg_cost), 0);
    const equity = acct ? acct.cash + invested : null;
    return { available: true, account, cash: acct?.cash ?? null, equity,
      positions: positions.map((p) => ({ ...p, weight: equity ? (p.shares * (p.last_price ?? p.avg_cost)) / equity : null })), targets };
  } catch (error) {
    return { available: false, error: error.message, positions: [], targets: [], equity: null };
  } finally { try { db?.close(); } catch { /* ignore */ } }
}

// Deterministic diversification check of a candidate against the book (sector weight and return correlation).
export function diversification({ ticker, industry, book, industries = {}, prices, maxSectorWeight = 0.35, maxCorr = 0.8 }) {
  const held = book.positions.filter((p) => p.ticker !== ticker);
  const sameSector = held.filter((p) => industry && industries[p.ticker] && sectorOf(industries[p.ticker]) === sectorOf(industry));
  const sectorWeight = sameSector.reduce((s, p) => s + (p.weight || 0), 0);
  let maxPair = null;
  if (prices && held.length) {
    const rets = (t) => { const b = prices.bars(t, 121); const out = new Map(); for (let i = 1; i < b.length; i++) out.set(b[i].date, Math.log(b[i].close / b[i - 1].close)); return out; };
    const mine = rets(ticker);
    for (const p of held) {
      const other = rets(p.ticker), a = [], b = [];
      for (const [d, r] of mine) if (other.has(d)) { a.push(r); b.push(other.get(d)); }
      if (a.length < 40) continue;
      const ma = mean(a), mb = mean(b);
      const c = mean(a.map((x, i) => (x - ma) * (b[i] - mb))) / (std(a) * std(b));
      if (!maxPair || c > maxPair.corr) maxPair = { ticker: p.ticker, corr: round(c, 3) };
    }
  }
  let status = "OK";
  if (sectorWeight >= maxSectorWeight) status = "WOULD_BREACH_CAP";
  else if (sameSector.length >= 2 || sectorWeight >= maxSectorWeight * 0.6) status = "SECTOR_HEAVY";
  else if (maxPair && maxPair.corr >= maxCorr) status = "CORRELATED_WITH_BOOK";
  return { status, positions: held.length, sectorPeers: sameSector.map((p) => p.ticker), sectorWeight: round(sectorWeight, 3),
    mostCorrelated: maxPair, limits: { maxSectorWeight, maxCorr } };
}
// SIC descriptions are very granular; group them coarsely so "SEMICONDUCTORS & RELATED DEVICES" peers match.
export function sectorOf(industry) {
  const s = String(industry || "").toUpperCase();
  const rules = [[/SEMICONDUCTOR|ELECTRONIC COMP/, "SEMIS"], [/SOFTWARE|PREPACKAGED|COMPUTER PROGRAM|DATA PROCESS|INFORMATION RETRIEVAL/, "SOFTWARE"],
    [/COMPUTER|COMMUNICATIONS EQUIP|ELECTRONIC/, "HARDWARE"], [/BANK|SAVINGS|CREDIT|FINANCE|SECURITY BROKERS|INSURANCE|INVESTMENT/, "FINANCIALS"],
    [/PHARMA|BIOLOG|MEDICAL|SURGICAL|HEALTH|HOSPITAL/, "HEALTHCARE"], [/PETROLEUM|CRUDE|OIL|GAS|NATURAL GAS/, "ENERGY"],
    [/RETAIL|STORES|CATALOG|MAIL-ORDER/, "RETAIL"], [/MOTOR VEHICLE|AUTO/, "AUTOS"], [/ELECTRIC SERVICES|UTILIT|WATER SUPPLY/, "UTILITIES"],
    [/REAL ESTATE|REIT/, "REAL_ESTATE"], [/TELEPHONE|TELECOM|CABLE|RADIOTELEPHONE/, "TELECOM"]];
  for (const [re, name] of rules) if (re.test(s)) return name;
  return s || "UNKNOWN";
}

// Walk-forward HMM snapshot written by the shared Python engine (paper trader / regime_compare).
export function loadHmmRegime(dataDir) {
  const paths = [join(dataDir, "regime_state.json"), join(dataDir, "Research", "regime_hmm_state.json")];
  for (const path of paths) {
    if (!existsSync(path)) continue;
    try { return JSON.parse(readFileSync(path, "utf8")); } catch { /* unreadable snapshot */ }
  }
  return null;
}
