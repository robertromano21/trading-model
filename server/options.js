// Options analytics and the options PAPER book for the dashboard: chains (Schwab market data when authorized, else the
// latest local snapshot, else a MODELED chain), payoff / greeks / probability of profit, the vol-sleeve and overlay
// paper state, and the options / downside research results. Market data and local files only: nothing in this module
// talks to an account or order endpoint (manual live tickets are a separate module behind the typed-confirm flow).
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const SYMBOL = /^[A-Z][A-Z0-9.\-]{0,9}$/;
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const STRATEGY = /^[a-z_]{3,30}$/;
const MAX_LEGS = 6;

function readJson(path) {
  if (!existsSync(path)) return null;
  try { return JSON.parse(readFileSync(path, "utf8")); } catch { return null; }
}

export function cleanLegs(legs) {
  if (!Array.isArray(legs) || legs.length < 1 || legs.length > MAX_LEGS) throw Object.assign(new Error(`legs must be a list of 1-${MAX_LEGS} legs`), { status: 400 });
  return legs.map((l) => {
    const kind = String(l?.kind || "").toLowerCase();
    if (!["call", "put", "stock"].includes(kind)) throw Object.assign(new Error("leg kind must be call, put or stock"), { status: 400 });
    const qty = Number(l.qty);
    if (!Number.isFinite(qty) || qty === 0 || Math.abs(qty) > (kind === "stock" ? 10000 : 100)) throw Object.assign(new Error("leg qty out of range"), { status: 400 });
    const out = { kind, qty };
    if (kind !== "stock") {
      const strike = Number(l.strike);
      if (!(strike > 0 && strike < 1e6)) throw Object.assign(new Error("leg strike out of range"), { status: 400 });
      if (!DATE.test(String(l.expiration || ""))) throw Object.assign(new Error("leg expiration must be YYYY-MM-DD"), { status: 400 });
      Object.assign(out, { strike, expiration: l.expiration });
    }
    if (l.price !== undefined && l.price !== null && l.price !== "") {
      const price = Number(l.price);
      if (!(price >= 0 && price < 1e6)) throw Object.assign(new Error("leg price out of range"), { status: 400 });
      out.price = price;
    }
    return out;
  });
}

export function createOptions({ root, dataDir, python, log = () => {}, runner } = {}) {
  const engine = join(root, "engine");
  const research = join(dataDir, "Research");
  const paperDir = join(dataDir, "PaperOptions");
  let paperRun = null;

  // Run one quant_tools op (fixed, allow-listed op names) and resolve its JSON envelope.
  const runOp = runner || ((op, args, timeoutMs = 120000) => new Promise((resolve) => {
    const child = spawn(python, [join(engine, "quant_tools.py")], { cwd: engine, windowsHide: true });
    let out = "", err = "";
    const timer = setTimeout(() => { child.kill(); resolve({ ok: false, error: { code: "TIMEOUT", message: `${op} timed out` } }); }, timeoutMs);
    child.stdout.on("data", (d) => { out += d; });
    child.stderr.on("data", (d) => { err += d; });
    child.on("close", () => {
      clearTimeout(timer);
      try { resolve(JSON.parse(out)); } catch { resolve({ ok: false, error: { code: "BAD_OUTPUT", message: (err || out).slice(-400) } }); }
    });
    child.stdin.end(JSON.stringify({ op, args }));
  }));

  const OPS = new Set(["options_chain", "options_greeks", "options_payoff", "options_pop", "expected_move", "top_risk", "vol_regime"]);
  async function op(name, args, response, timeoutMs) {
    if (!OPS.has(name)) return response.status(400).json({ error: "unknown op" });
    const result = await runOp(name, args, timeoutMs);
    if (!result?.ok) {
      log("warn", "options op failed", { op: name, code: result?.error?.code });
      return response.status(result?.error?.code === "BAD_ARGS" ? 400 : 502).json({ error: result?.error?.message || "failed", code: result?.error?.code });
    }
    return response.json(result.result);
  }

  function symbolOf(value) {
    const s = String(value || "").toUpperCase().trim();
    if (!SYMBOL.test(s)) throw Object.assign(new Error("symbol is required (e.g. SPY)"), { status: 400 });
    return s;
  }
  const num = (v, lo, hi, dflt) => { const n = v === undefined || v === "" ? dflt : Number(v); return Number.isFinite(n) && n >= lo && n <= hi ? n : dflt; };
  const wrap = (fn) => async (req, res) => {
    try { await fn(req, res); } catch (error) { res.status(error.status || 500).json({ error: error.message }); }
  };

  function analysisArgs(body) {
    const args = {};
    if (body.symbol) args.symbol = symbolOf(body.symbol);
    for (const k of ["spot", "vol"]) if (body[k] !== undefined && body[k] !== "") args[k] = num(body[k], 0.0001, 1e6, undefined);
    if (body.legs) args.legs = cleanLegs(body.legs);
    else {
      if (!STRATEGY.test(String(body.strategy || ""))) throw Object.assign(new Error("give legs, or a strategy name"), { status: 400 });
      if (!DATE.test(String(body.expiration || ""))) throw Object.assign(new Error("expiration must be YYYY-MM-DD"), { status: 400 });
      Object.assign(args, { strategy: body.strategy, expiration: body.expiration, qty: num(body.qty, 1, 100, 1) });
      if (body.back_expiration && DATE.test(body.back_expiration)) args.back_expiration = body.back_expiration;
      if (body.strikes && typeof body.strikes === "object") {
        args.strikes = {};
        for (const [k, v] of Object.entries(body.strikes)) if (/^(k|k[1-4]|put|call)$/.test(k) && Number(v) > 0) args.strikes[k] = Number(v);
      }
    }
    if (!args.symbol && (args.spot === undefined || args.vol === undefined)) throw Object.assign(new Error("give a symbol, or spot and vol"), { status: 400 });
    return args;
  }

  function mount(app) {
    app.get("/api/options/chain", wrap(async (req, res) => {
      const prefer = ["schwab", "snapshot", "model"].includes(req.query.prefer) ? req.query.prefer : "schwab";
      await op("options_chain", { symbol: symbolOf(req.query.symbol), prefer, strikes: num(req.query.strikes, 1, 40, 8),
        expirations: num(req.query.expirations, 1, 12, 4) }, res, 60000);
    }));
    app.post("/api/options/analyze", wrap(async (req, res) => op("options_payoff", analysisArgs(req.body || {}), res, 60000)));
    app.get("/api/options/greeks", wrap(async (req, res) => {
      const q = req.query;
      await op("options_greeks", { ...(q.symbol ? { symbol: symbolOf(q.symbol) } : {}), spot: q.spot, vol: q.vol,
        strike: q.strike, dte: q.dte, kind: q.kind === "put" ? "put" : "call" }, res, 30000);
    }));
    app.get("/api/options/expected-move", wrap(async (req, res) => op("expected_move", { symbol: symbolOf(req.query.symbol),
      days: num(req.query.days, 1, 730, 30), ...(req.query.straddle ? { straddle: num(req.query.straddle, 0, 1e6, undefined) } : {}) }, res, 30000)));
    app.get("/api/options/vol-regime", wrap(async (_req, res) => op("vol_regime", {}, res, 30000)));
    app.get("/api/options/top-risk", wrap(async (_req, res) => op("top_risk", {}, res, 30000)));
    app.get("/api/options/paper", (_req, res) => {
      res.json({ book: readJson(join(paperDir, "book.json")), volSleeve: readJson(join(paperDir, "vol_sleeve_paper.json")),
        overlay: readJson(join(paperDir, "overlay_paper.json")), config: readJson(join(paperDir, "config.json")),
        running: Boolean(paperRun), note: "PAPER only; option prices are MODELED unless a Schwab chain snapshot was available." });
    });
    app.post("/api/options/paper/run", (_req, res) => {
      if (paperRun) return res.status(409).json({ error: "an options paper run is already in progress" });
      const child = spawn(python, [join(engine, "options_paper.py"), "run"], { cwd: engine, windowsHide: true });
      let out = "", err = "";
      paperRun = child;
      child.stdout.on("data", (d) => { out += d; });
      child.stderr.on("data", (d) => { err += d; });
      child.on("close", (code) => {
        paperRun = null;
        log(code === 0 ? "info" : "warn", "options paper run finished", { code, tail: (err || "").slice(-300) });
      });
      res.status(202).json({ started: true });
    });
    app.get("/api/options/research", (_req, res) => {
      const ov = readJson(join(research, "options_overlays.json"));
      const vs = readJson(join(research, "vol_sleeve.json"));
      const td = readJson(join(research, "top_detector.json"));
      res.json({
        overlays: ov && { generated: ov.generated, honesty: ov.honesty, base: ov.base, results: (ov.results || []).map((r) => ({
          mode: r.mode, sizing: r.sizing, oosStart: r.oosStart, overlay: r.overlayOOS, base: r.baseOOS, sharpeDiff: r.sharpeDiff,
          p: r.pOverlayBeatsBase, holmP: r.holmP, trades: r.avgTradesPerCombo, exits: r.exitReasons, skips: r.topSkipReasons })) },
        volSleeve: vs && { generated: vs.generated, rule: vs.rule, honesty: vs.honesty, walkForward: vs.walkForward && {
          oosStart: vs.walkForward.oosStart, outOfSample: vs.walkForward.outOfSample, exitReasons: vs.walkForward.oosExitReasons,
          trades: vs.walkForward.oosTrades, folds: vs.walkForward.folds }, pOosMeanReturnGt0: vs.pOosMeanReturnGt0,
          pFullExitsBeatTimeOnly: vs.pFullExitsBeatTimeOnly, ablation: vs.ablation, buyHold: vs.buyHoldReferences },
        topDetector: td && { generated: td.generated, verdict: td.verdict, baseRates: td.baseRates, latest: td.latest,
          survivingHolm05: td.survivingHolm05, robustSurvivors: td.robustSurvivors, walkForwardComposite: td.walkForwardComposite,
          tests: (td.tests || []).slice(0, 60) },
      });
    });
  }
  return { mount, _runOp: runOp };
}
