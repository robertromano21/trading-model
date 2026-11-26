// Experiment PAPER accounts: a sandbox the user (or the AI analyst on the user's request) can open, trade by hand
// and sweep, without touching the live paper config (plain TSMOM) or any other account range.
//   Ids 1000-9999 (live/swing + swarm 0-49, crypto 200-219, SBIT shadows 220-223 and the study ranges stay separate).
//   Soft cap TM_MAX_USER_ACCOUNTS (default 200 active accounts); TM_MAX_EXPERIMENT_BACKTESTS (default 2) concurrent
//   engine backtests. Paper only: fills are simulated at the latest price; no broker endpoint is ever called.
//   Strategy accounts get a pre-holdout baseline backtest (engine/quant_tools.py "backtest", the same code path as
//   the analyst's RUN_BACKTEST tool). Looking into the sealed holdout goes through /holdout-look, which needs
//   confirm:true and appends to Massive/Research/swarm/holdout_ledger.json; those results never feed learning.
//   Every create / order / sweep is appended to Research/experiments/trial_registry.jsonl with its origin
//   (user_ui or analyst_user_request) so multiple-testing counts include user-prompted experiments.
import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync, appendFileSync, renameSync } from "node:fs";
import { join } from "node:path";

export const ID_MIN = 1000;
export const ID_MAX = 9999;
export const DEFAULT_MANUAL_ID = 1000;
export const ORIGIN_HEADER = "x-tm-origin";
export const ORIGINS = Object.freeze(["user_ui", "analyst_user_request"]);
const STRATEGIES = ["manual", "tsmom", "meanrev", "combined"];
const TICKER = /^[A-Z][A-Z0-9.-]{0,11}$/;
const SLIPPAGE_BPS = 2;          // simple paper fill model: last price +/- 2 bps, $0 commission (Schwab equities)

export function holdoutStartOf(dataDir) {
  try {
    const d = JSON.parse(readFileSync(join(dataDir, "Research", "swarm", "design.json"), "utf8"));
    if (/^\d{4}-\d{2}-\d{2}$/.test(d.holdoutStart || "")) return d.holdoutStart;
  } catch { /* default below */ }
  return "2026-07-09";
}
export function dayBefore(date) {
  const t = new Date(date + "T12:00:00Z"); t.setUTCDate(t.getUTCDate() - 1); return t.toISOString().slice(0, 10);
}
const etNow = (now) => new Date(now).toLocaleString("sv-SE", { timeZone: "America/New_York" });

export function createExperiments({ dataDir, quant = null, priceOf = async () => null, memory = null, log = () => {},
  env = process.env, now = () => Date.now() } = {}) {
  const dir = join(dataDir, "Research", "experiments");
  const file = join(dir, "accounts.json");
  const registry = join(dir, "trial_registry.jsonl");
  const cap = Math.max(1, Number(env.TM_MAX_USER_ACCOUNTS) || 200);
  let backtestSlots = Math.max(1, Number(env.TM_MAX_EXPERIMENT_BACKTESTS) || 2);
  const waiting = [];
  const slot = () => (backtestSlots > 0 ? (backtestSlots -= 1, Promise.resolve()) : new Promise((r) => waiting.push(r)));
  const release = () => { const next = waiting.shift(); if (next) next(); else backtestSlots += 1; };

  const load = () => { try { return JSON.parse(readFileSync(file, "utf8")); } catch { return { accounts: [] }; } };
  const save = (doc) => { mkdirSync(dir, { recursive: true }); writeFileSync(file + ".tmp", JSON.stringify(doc, null, 1)); renameSync(file + ".tmp", file); };
  const err = (status, message) => Object.assign(new Error(message), { status });

  function trial(entry) {
    mkdirSync(dir, { recursive: true });
    appendFileSync(registry, JSON.stringify({ at: etNow(now()), ...entry }) + "\n");
  }
  function trials({ preHoldoutOnly = true } = {}) {
    if (!existsSync(registry)) return [];
    return readFileSync(registry, "utf8").split(/\r?\n/).filter(Boolean).map((l) => { try { return JSON.parse(l); } catch { return null; } })
      .filter((t) => t && (!preHoldoutOnly || !t.holdoutLook));
  }
  // Number of strategy configurations tried (each distinct config counts once); used to deflate Sharpe ratios.
  const trialCount = () => new Set(trials().filter((t) => t.kind === "strategy").map((t) => t.configHash)).size;
  const remember = (item) => { try { memory?.ingest?.(item); } catch { /* memory is optional */ } };

  function view(a) {
    const pos = Object.entries(a.positions || {}).map(([ticker, p]) => ({ ticker, qty: p.qty, avgPrice: p.avgPrice, last: p.last ?? null }));
    const mv = pos.reduce((s, p) => s + p.qty * (p.last ?? p.avgPrice), 0);
    return { id: a.id, name: a.name, strategy: a.strategy, status: a.status, origin: a.origin, batchId: a.batchId || null,
      params: a.params || {}, tickers: a.tickers || [], equity0: a.equity0, cash: round(a.cash), equity: round(a.cash + mv),
      positions: pos, orders: (a.orders || []).slice(-10), baseline: a.baseline || null, forward: a.forward || null,
      createdAt: a.createdAt, retiredAt: a.retiredAt || null };
  }
  const round = (x) => (Number.isFinite(x) ? Math.round(x * 100) / 100 : x);
  const configHash = (strategy, tickers, params) => createHash("sha256").update(JSON.stringify([strategy, [...(tickers || [])].sort(), params || {}])).digest("hex").slice(0, 16);

  function list() {
    const doc = load();
    const active = doc.accounts.filter((a) => a.status === "active");
    return { range: [ID_MIN, ID_MAX], cap, active: active.length, holdoutStart: holdoutStartOf(dataDir), trialsTried: trialCount(),
      accounts: doc.accounts.map(view) };
  }
  function nextIds(doc, n) {
    const used = new Set(doc.accounts.map((a) => a.id));
    const ids = [];
    for (let id = ID_MIN + 1; id <= ID_MAX && ids.length < n; id++) if (!used.has(id)) ids.push(id);
    if (ids.length < n) throw err(409, "No free experiment ids left in 1000-9999.");
    return ids;
  }
  function cleanCreate(body) {
    const strategy = String(body.strategy || "manual").toLowerCase();
    if (!STRATEGIES.includes(strategy)) throw err(400, `strategy must be one of ${STRATEGIES.join(", ")}`);
    const equity = Number(body.equity ?? 10000);
    if (!(equity >= 1000 && equity <= 1e6)) throw err(400, "equity must be between $1,000 and $1,000,000");
    const count = Math.round(Number(body.count ?? 1));
    if (!(count >= 1 && count <= 50)) throw err(400, "count must be 1-50 per batch");
    const tickers = [...new Set((Array.isArray(body.tickers) ? body.tickers : String(body.tickers || "").split(/[\s,]+/))
      .map((t) => String(t).toUpperCase().trim()).filter(Boolean))];
    if (tickers.length > 60 || !tickers.every((t) => TICKER.test(t))) throw err(400, "tickers must be up to 60 valid symbols");
    const params = body.params && typeof body.params === "object" && !Array.isArray(body.params) ? body.params : {};
    if (JSON.stringify(params).length > 2000) throw err(400, "params too large");
    let sweep = null;
    if (body.sweep) {
      const s = body.sweep, from = Number(s.from), to = Number(s.to);
      if (!/^[a-z_][a-z0-9_.]{0,40}$/.test(String(s.param || "")) || !Number.isFinite(from) || !Number.isFinite(to)) throw err(400, "sweep needs {param, from, to}");
      if (strategy === "manual") throw err(400, "a sweep needs a strategy account (tsmom, meanrev or combined)");
      sweep = { param: s.param, from, to };
    }
    const name = String(body.name || "").replace(/[^\w .,$%()+-]/g, "").slice(0, 60);
    return { strategy, equity, count, tickers, params, sweep, name };
  }
  // lookback_days is the friendly name for a single TSMOM horizon.
  function engineParams(strategy, params) {
    const p = { ...params };
    if ("lookback_days" in p) { const v = Math.round(Number(p.lookback_days)); delete p.lookback_days; if (strategy === "tsmom") p.horizons = [v]; else if (strategy === "combined") p["tsmom.horizons"] = [v]; }
    return p;
  }

  async function baselineFor(a) {
    if (!quant || a.strategy === "manual") return null;
    const end = dayBefore(holdoutStartOf(dataDir));
    await slot();
    try {
      const out = await quant.run("backtest", { family: a.strategy, params: engineParams(a.strategy, a.params), ...(a.tickers.length ? { tickers: a.tickers } : {}), end }, { timeoutMs: 300000 });
      if (!out.ok) return { ok: false, error: out.error?.message || "backtest failed", end };
      const r = out.result || {}, perf = r.performance || {};
      return { ok: true, end, evaluationStart: r.evaluationStart, sharpe: perf.sharpe ?? null, cagr: perf.cagr ?? null,
        totalReturn: perf.totalReturn ?? null, maxDrawdown: perf.maxDrawdown ?? null, vol: perf.volatility ?? perf.vol ?? null,
        trades: perf.trades ?? null, benchmarkSPY: r.benchmarkSPY ? { sharpe: r.benchmarkSPY.sharpe ?? null, totalReturn: r.benchmarkSPY.totalReturn ?? null } : null };
    } finally { release(); }
  }

  function create(body, origin) {
    const c = cleanCreate(body || {});
    const doc = load();
    const active = doc.accounts.filter((a) => a.status === "active").length;
    if (active + c.count > cap) throw err(409, `That would make ${active + c.count} active experiment accounts; the cap is ${cap} (TM_MAX_USER_ACCOUNTS). Retire or delete some first.`);
    const ids = nextIds(doc, c.count);
    const batchId = c.count > 1 ? "b-" + randomUUID().slice(0, 8) : null;
    const created = ids.map((id, k) => {
      const params = { ...c.params };
      if (c.sweep) params[c.sweep.param] = c.count === 1 ? c.sweep.from : Math.round((c.sweep.from + (c.sweep.to - c.sweep.from) * k / (c.count - 1)) * 1000) / 1000;
      const a = { id, name: c.name ? (c.count > 1 ? `${c.name} #${k + 1}` : c.name) : `${c.strategy} ${id}`, strategy: c.strategy, params, tickers: c.tickers,
        equity0: c.equity, cash: c.equity, positions: {}, orders: [], status: "active", origin, batchId, createdAt: new Date(now()).toISOString() };
      doc.accounts.push(a);
      return a;
    });
    save(doc);
    for (const a of created) {
      const hash = configHash(a.strategy, a.tickers, a.params);
      trial({ kind: a.strategy === "manual" ? "manual_account" : "strategy", origin, account: a.id, batchId, strategy: a.strategy, params: a.params, tickers: a.tickers, configHash: hash, holdoutLook: false });
      remember({ kind: "EVENT", ticker: a.tickers[0] || null, text: `Experiment paper account ${a.id} opened (${a.strategy}, $${a.equity0}, params ${JSON.stringify(a.params)}) on ${origin === "analyst_user_request" ? "the user's request to the analyst" : "the dashboard"}.`, meta: { origin, account: a.id, batchId } });
    }
    // Pre-holdout baselines in the background (concurrency-limited); results are written back to the account.
    for (const a of created) {
      if (a.strategy === "manual") continue;
      baselineFor(a).then((baseline) => {
        const d = load(); const x = d.accounts.find((y) => y.id === a.id); if (!x) return;
        x.baseline = baseline; save(d);
        if (baseline?.ok) {
          trial({ kind: "result", origin, account: a.id, batchId, configHash: configHash(a.strategy, a.tickers, a.params), sharpe: baseline.sharpe, end: baseline.end, holdoutLook: false });
          remember({ kind: "EVENT", text: `Experiment ${a.id} (${a.strategy} ${JSON.stringify(a.params)}) pre-holdout baseline through ${baseline.end}: Sharpe ${baseline.sharpe}, total return ${baseline.totalReturn}, max drawdown ${baseline.maxDrawdown}.`, meta: { origin, account: a.id } });
        }
      }).catch((e) => log("warn", "experiment baseline failed", { account: a.id, error: e.message }));
    }
    log("info", "Experiment accounts created", { ids, origin, batchId });
    return { created: created.map(view), batchId, cap, active: active + c.count };
  }

  function account(doc, id, { autoManual = false, origin } = {}) {
    let a = doc.accounts.find((x) => x.id === id);
    if (!a && autoManual && id === DEFAULT_MANUAL_ID) {
      a = { id, name: "Manual paper", strategy: "manual", params: {}, tickers: [], equity0: 10000, cash: 10000, positions: {}, orders: [], status: "active", origin, createdAt: new Date(now()).toISOString() };
      doc.accounts.push(a);
      trial({ kind: "manual_account", origin, account: id, strategy: "manual", configHash: "manual", holdoutLook: false });
    }
    if (!a) throw err(404, `No experiment account ${id}.`);
    return a;
  }

  async function order(id, body, origin) {
    const side = String(body?.side || "").toLowerCase(), ticker = String(body?.ticker || "").toUpperCase().trim(), qty = Math.round(Number(body?.qty));
    if (!["buy", "sell"].includes(side)) throw err(400, "side must be buy or sell");
    if (!TICKER.test(ticker)) throw err(400, "Enter a valid ticker.");
    if (!(qty >= 1 && qty <= 100000)) throw err(400, "qty must be 1-100000 shares");
    const doc = load();
    const a = account(doc, id, { autoManual: true, origin });
    if (a.status !== "active") throw err(409, `Account ${id} is retired.`);
    if (a.strategy !== "manual") throw err(409, `Account ${id} runs ${a.strategy}; hand orders go to a manual account (e.g. ${DEFAULT_MANUAL_ID}).`);
    const px = await priceOf(ticker);
    if (!(px?.price > 0)) throw err(409, `No price for ${ticker} (local history or Schwab quote).`);
    const fill = round(px.price * (1 + (side === "buy" ? 1 : -1) * SLIPPAGE_BPS / 10000));
    const pos = a.positions[ticker] || { qty: 0, avgPrice: 0 };
    if (side === "buy") {
      if (fill * qty > a.cash + 1e-9) throw err(409, `Not enough cash: ${qty} ${ticker} at ${fill} needs $${round(fill * qty)}, account has $${round(a.cash)}.`);
      pos.avgPrice = round((pos.avgPrice * pos.qty + fill * qty) / (pos.qty + qty)); pos.qty += qty; a.cash -= fill * qty;
    } else {
      if (qty > pos.qty) throw err(409, `Account ${id} holds ${pos.qty} ${ticker}; paper accounts cannot short.`);
      pos.qty -= qty; a.cash += fill * qty;
    }
    pos.last = px.price;
    if (pos.qty > 0) a.positions[ticker] = pos; else delete a.positions[ticker];
    const rec = { at: new Date(now()).toISOString(), side, ticker, qty, fill, priceSource: px.source || "local", origin };
    a.orders.push(rec);
    save(doc);
    trial({ kind: "manual_order", origin, account: id, ticker, side, qty, fill, holdoutLook: false });
    remember({ kind: "DECISION", ticker, text: `Experiment account ${id} (paper, manual): ${side.toUpperCase()} ${qty} ${ticker} at ${fill} (${px.source || "local"} price) on ${origin === "analyst_user_request" ? "the user's request to the analyst" : "the dashboard"}.`, meta: { origin, account: id } });
    log("info", "Experiment paper order", { account: id, side, ticker, qty, fill, origin });
    return { order: rec, account: view(a) };
  }

  async function advance(id, origin) {
    const doc = load();
    const a = account(doc, id);
    for (const [t, p] of Object.entries(a.positions)) { const px = await priceOf(t); if (px?.price > 0) p.last = px.price; }
    if (a.strategy !== "manual" && quant) {
      // Forward shadow: same engine backtest, evaluation starting at the account's creation (after the sealed window).
      const created = a.createdAt.slice(0, 10);
      const start = new Date(Date.parse(created) - 175 * 86400000).toISOString().slice(0, 10);
      await slot();
      try {
        const out = await quant.run("backtest", { family: a.strategy, params: engineParams(a.strategy, a.params), ...(a.tickers.length ? { tickers: a.tickers } : {}), start }, { timeoutMs: 300000 });
        a.forward = out.ok ? { at: new Date(now()).toISOString(), evaluationStart: out.result?.evaluationStart, totalReturn: out.result?.performance?.totalReturn ?? null,
          equity: round(a.equity0 * (1 + (out.result?.performance?.totalReturn ?? 0))), note: "forward shadow from creation (warm-up before it)" } : { at: new Date(now()).toISOString(), error: out.error?.message };
      } finally { release(); }
    }
    save(doc);
    return { account: view(a) };
  }

  function retire(id, origin) {
    const doc = load(); const a = account(doc, id);
    a.status = "retired"; a.retiredAt = new Date(now()).toISOString(); save(doc);
    trial({ kind: "retire", origin, account: id, holdoutLook: false });
    return { account: view(a) };
  }
  function remove(id, origin) {
    const doc = load(); const a = account(doc, id);
    doc.accounts = doc.accounts.filter((x) => x.id !== id); save(doc);
    trial({ kind: "delete", origin, account: id, strategy: a.strategy, holdoutLook: false });  // the trial stays counted
    return { deleted: id };
  }

  async function holdoutLook(body, origin) {
    if (body?.confirm !== true) throw err(400, "A holdout look needs confirm:true (it is logged in the holdout ledger).");
    const strategy = String(body.strategy || "").toLowerCase();
    if (!["tsmom", "meanrev", "combined"].includes(strategy)) throw err(400, "strategy must be tsmom, meanrev or combined");
    const reason = String(body.reason || "").slice(0, 300);
    if (!reason) throw err(400, "Say why you are looking at the holdout (reason).");
    const tickers = Array.isArray(body.tickers) ? body.tickers.map((t) => String(t).toUpperCase()).filter((t) => TICKER.test(t)).slice(0, 60) : [];
    const params = body.params && typeof body.params === "object" ? body.params : {};
    const hs = holdoutStartOf(dataDir);
    const ledgerPath = join(dataDir, "Research", "swarm", "holdout_ledger.json");
    let ledger = { views: 0, entries: [] };
    try { ledger = JSON.parse(readFileSync(ledgerPath, "utf8")); } catch { /* new ledger */ }
    ledger.entries = Array.isArray(ledger.entries) ? ledger.entries : [];
    ledger.views = (Number(ledger.views) || ledger.entries.length) + 1;
    const hash = "experiment-" + configHash(strategy, tickers, params);
    ledger.entries.push({ at: etNow(now()), designHash: hash, study: "user_experiment", origin, holdoutStart: hs, reason, strategy, params, tickers });
    mkdirSync(join(dataDir, "Research", "swarm"), { recursive: true });
    writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2));
    trial({ kind: "holdout_look", origin, configHash: hash, holdoutLook: true });   // excluded from learning / memory
    if (!quant) return { ledgerViews: ledger.views, result: null };
    await slot();
    try {
      const out = await quant.run("backtest", { family: strategy, params: engineParams(strategy, params), ...(tickers.length ? { tickers } : {}), start: dayBefore(hs) }, { timeoutMs: 300000 });
      return { ledgerViews: ledger.views, holdoutStart: hs, ok: out.ok, performance: out.result?.performance || null, error: out.ok ? undefined : out.error?.message };
    } finally { release(); }
  }

  const originOf = (request) => (request.get?.(ORIGIN_HEADER) === "analyst_user_request" ? "analyst_user_request" : "user_ui");
  const idOf = (request) => { const id = Number(request.params.id); if (!Number.isInteger(id) || id < ID_MIN || id > ID_MAX) throw err(400, `Experiment ids are ${ID_MIN}-${ID_MAX}.`); return id; };
  const send = (response, fn) => Promise.resolve().then(fn).then((v) => response.json(v)).catch((e) => response.status(e.status || 500).json({ error: e.message }));
  function mount(app) {
    app.get("/api/experiments", (_q, r) => send(r, () => list()));
    app.post("/api/experiments", (q, r) => send(r, () => create(q.body, originOf(q))));
    app.post("/api/experiments/holdout-look", (q, r) => send(r, () => holdoutLook(q.body, originOf(q))));
    app.post("/api/experiments/:id/order", (q, r) => send(r, () => order(idOf(q), q.body, originOf(q))));
    app.post("/api/experiments/:id/advance", (q, r) => send(r, () => advance(idOf(q), originOf(q))));
    app.post("/api/experiments/:id/retire", (q, r) => send(r, () => retire(idOf(q), originOf(q))));
    app.post("/api/experiments/:id/delete", (q, r) => send(r, () => remove(idOf(q), originOf(q))));
    return api;
  }
  const recordTrial = (entry) => trial(entry);
  const api = { mount, list, create, order, advance, retire, remove, holdoutLook, trials, trialCount, recordTrial, cap };
  return api;
}
