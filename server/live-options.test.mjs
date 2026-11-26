// Mocked tests for manual OPTIONS and short-sale tickets. No network: fetch is replaced; no real order or preview is sent.
// Run: node --test live-options.test.mjs
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createLiveTrading } from "./live-trading.js";
import { analyzeOptionOrder, buildOptionOrder, occSymbol, parseOcc, validateOptionTicket, UNDEFINED_RISK_PHRASE, SHORT_SALE_PHRASE } from "./live-options.js";

// Reference JSON generated with schwab-py 1.6.0 on Robert's PC (orders/options.py, orders/equities.py), 2026-11-25.
const REF_SINGLE = '{"session": "NORMAL", "duration": "DAY", "orderType": "LIMIT", "price": "1.25", "orderLegCollection": [{"instruction": "BUY_TO_OPEN", "instrument": {"assetType": "OPTION", "symbol": "SPY   261218C00600000"}, "quantity": 2}], "orderStrategyType": "SINGLE"}';
const REF_VERTICAL = '{"session": "NORMAL", "duration": "DAY", "orderType": "NET_DEBIT", "complexOrderStrategyType": "VERTICAL", "quantity": 2, "price": "3.10", "orderLegCollection": [{"instruction": "BUY_TO_OPEN", "instrument": {"assetType": "OPTION", "symbol": "SPY   261218C00600000"}, "quantity": 2}, {"instruction": "SELL_TO_OPEN", "instrument": {"assetType": "OPTION", "symbol": "SPY   261218C00610000"}, "quantity": 2}], "orderStrategyType": "SINGLE"}';
const REF_SHORT = '{"session": "NORMAL", "duration": "DAY", "orderType": "LIMIT", "price": "100.50", "orderLegCollection": [{"instruction": "SELL_SHORT", "instrument": {"assetType": "EQUITY", "symbol": "AAPL"}, "quantity": 5}], "orderStrategyType": "SINGLE"}';
const py = (s) => JSON.stringify(JSON.parse(s));

test("OCC symbols match schwab-py OptionSymbol", () => {
  assert.equal(occSymbol("SPY", "2026-12-18", "C", 600), "SPY   261218C00600000");
  assert.equal(occSymbol("SPXW", "2026-12-18", "PUT", 5040.5), "SPXW  261218P05040500");
  assert.deepEqual(parseOcc("SPY   261218P00595500"), { underlying: "SPY", expiration: "2026-12-18", kind: "put", strike: 595.5, symbol: "SPY   261218P00595500" });
  assert.throws(() => occSymbol("SPY", "12/18/2026", "C", 600), /YYYY-MM-DD/);
});

test("order JSON matches schwab-py byte for byte (single leg and vertical)", () => {
  const single = validateOptionTicket({ underlying: "SPY", accountHash: "H", price: 1.25,
    legs: [{ instruction: "BUY_TO_OPEN", kind: "call", strike: 600, expiration: "2026-12-18", quantity: 2 }] });
  assert.equal(JSON.stringify(buildOptionOrder(single)), py(REF_SINGLE));
  const vertical = validateOptionTicket({ underlying: "SPY", accountHash: "H", price: 3.1, orderType: "NET_DEBIT", complexType: "VERTICAL",
    legs: [{ instruction: "BUY_TO_OPEN", kind: "call", strike: 600, expiration: "2026-12-18", quantity: 2 },
      { instruction: "SELL_TO_OPEN", kind: "call", strike: 610, expiration: "2026-12-18", quantity: 2 }] });
  assert.equal(JSON.stringify(buildOptionOrder(vertical)), py(REF_VERTICAL));
  const fly = validateOptionTicket({ underlying: "SPY", accountHash: "H", price: 1, orderType: "NET_DEBIT", complexType: "BUTTERFLY",
    legs: [{ instruction: "BUY_TO_OPEN", kind: "call", strike: 590, expiration: "2026-12-18", quantity: 1 },
      { instruction: "SELL_TO_OPEN", kind: "call", strike: 600, expiration: "2026-12-18", quantity: 2 },
      { instruction: "BUY_TO_OPEN", kind: "call", strike: 610, expiration: "2026-12-18", quantity: 1 }] });
  assert.equal(buildOptionOrder(fly).quantity, 1);
});

test("market orders, bad types and bad legs are refused", () => {
  const legs = [{ instruction: "BUY_TO_OPEN", kind: "put", strike: 500, expiration: "2026-12-18", quantity: 1 }];
  assert.throws(() => validateOptionTicket({ underlying: "SPY", accountHash: "H", legs, orderType: "MARKET", price: 1 }), /Market orders are not allowed/);
  assert.throws(() => validateOptionTicket({ underlying: "SPY", accountHash: "H", legs, orderType: "NET_DEBIT", price: 1 }), /Single-leg/);
  assert.throws(() => validateOptionTicket({ underlying: "SPY", accountHash: "H", legs: [{ ...legs[0], quantity: 1.5 }], price: 1 }), /whole number/);
  assert.throws(() => validateOptionTicket({ underlying: "SPY", accountHash: "H", legs: [{ ...legs[0], instruction: "BUY" }], price: 1 }), /instruction/);
  assert.throws(() => validateOptionTicket({ underlying: "SPY", accountHash: "H", legs, price: 0 }), /positive/);
});

test("payoff analysis: verticals defined, naked calls unbounded, covered calls covered by held shares", () => {
  const base = { kind: "call", expiration: "2026-12-18" };
  const vertical = analyzeOptionOrder([{ ...base, instruction: "BUY_TO_OPEN", strike: 600, quantity: 1, price: 10 },
    { ...base, instruction: "SELL_TO_OPEN", strike: 610, quantity: 1, price: 6 }], { spot: 600 });
  assert.equal(vertical.definedRisk, true);
  assert.equal(Math.round(vertical.maxLoss), -400);
  assert.equal(Math.round(vertical.maxGain), 600);
  assert.deepEqual(vertical.breakevens, [604]);
  const naked = analyzeOptionOrder([{ ...base, instruction: "SELL_TO_OPEN", strike: 610, quantity: 1, price: 6 }], { spot: 600 });
  assert.equal(naked.nakedCalls, true);
  assert.equal(naked.maxLoss, null);
  const covered = analyzeOptionOrder([{ ...base, instruction: "SELL_TO_OPEN", strike: 610, quantity: 1, price: 6 }], { spot: 600, heldShares: 100 });
  assert.equal(covered.nakedCalls, false);
});

// ---------------------------------------------------------------- mocked Schwab
function setup({ allowed = true, shorts = false, type = "MARGIN", positions = [], clock = { t: 1_800_000_000_000 } } = {}) {
  const root = mkdtempSync(join(tmpdir(), "liveopt-"));
  writeFileSync(join(root, "schwab_token.json"), JSON.stringify({ token: { access_token: "fake", refresh_token: "r", expires_at: Math.floor(clock.t / 1000) + 1800 } }));
  const calls = [];
  const opt = (bid, ask, delta) => ({ quote: { bidPrice: bid, askPrice: ask, mark: (bid + ask) / 2, delta, gamma: 0.01, theta: -0.05, vega: 0.3, volatility: 18 } });
  const fetchImpl = async (url, options = {}) => {
    const u = new URL(url);
    calls.push({ path: u.pathname, method: options.method ?? "GET", body: options.body });
    const reply = (status, body, headers = {}) => ({ ok: status < 300, status, text: async () => (body ? JSON.stringify(body) : ""),
      json: async () => body, headers: { get: (k) => headers[k.toLowerCase()] ?? null } });
    if (u.pathname.endsWith("/accountNumbers")) return reply(200, [{ accountNumber: "12345678", hashValue: "HASH1" }]);
    if (u.pathname.startsWith("/marketdata/v1/quotes")) {
      return reply(200, { SPY: { quote: { lastPrice: 600, bidPrice: 599.99, askPrice: 600.01 } }, AAPL: { quote: { lastPrice: 100, bidPrice: 99.9, askPrice: 100.1 } },
        "SPY   261218C00600000": opt(9.9, 10.1, 0.52), "SPY   261218C00610000": opt(5.9, 6.1, 0.38), "SPY   261218P00580000": opt(4.9, 5.1, -0.3),
        XYZ: { quote: { lastPrice: 50, bidPrice: 49.99, askPrice: 50.01 } }, "XYZ   261218C00055000": opt(0.9, 1.0, 0.25) });
    }
    if (u.pathname.endsWith("/previewOrder")) return reply(200, { orderStrategy: { orderBalance: { orderValue: 400 } } });
    if (u.pathname === "/trader/v1/accounts/HASH1") return reply(200, { securitiesAccount: { type, positions } });
    if (u.pathname.endsWith("/orders") && options.method === "POST") return reply(201, null, { location: `${url}/555` });
    return reply(404, { message: "unexpected" });
  };
  const env = { LIVE_TRADING_ALLOWED: allowed ? "1" : "0", ...(shorts ? { LIVE_SHORT_SELLING_ALLOWED: "1" } : {}) };
  const live = createLiveTrading({ root, dataDir: join(root, "Massive"), fetchImpl, env, now: () => clock.t, log: {} });
  return { live, calls, root, clock };
}
const orderPosts = (calls) => calls.filter((c) => c.path.endsWith("/orders") && c.method === "POST");
const vertical = { underlying: "SPY", accountHash: "HASH1", orderType: "NET_DEBIT", complexType: "VERTICAL", price: 4.1,
  legs: [{ instruction: "BUY_TO_OPEN", kind: "call", strike: 600, expiration: "2026-12-18", quantity: 1 },
    { instruction: "SELL_TO_OPEN", kind: "call", strike: 610, expiration: "2026-12-18", quantity: 1 }] };

test("option preview shows max loss/gain, breakevens, greeks and fees; confirm needs the typed CONFIRM", async () => {
  const { live, calls } = setup();
  live.setEnabled(true, "ENABLE LIVE TRADING");
  const p = await live.previewOption(vertical);
  assert.equal(p.risk.definedRisk, true);
  assert.equal(p.risk.maxLoss, -410);
  assert.equal(p.risk.maxGain, 590);
  assert.deepEqual(p.risk.breakevens, [604.1]);
  assert.equal(p.greeks.delta, 14);
  assert.ok(p.estimatedFees.commissionAndRegulatory >= 1.3);
  assert.equal(p.extraPhrase, null);
  assert.equal(orderPosts(calls).length, 0);
  await assert.rejects(live.confirm({ previewId: p.previewId, idempotencyKey: "key-opt-1" }), /Type CONFIRM/);
  const ok = await live.confirm({ previewId: p.previewId, idempotencyKey: "key-opt-2", typedConfirm: "CONFIRM" });
  assert.equal(ok.orderId, "555");
  assert.equal(orderPosts(calls).length, 1);
  assert.deepEqual(JSON.parse(orderPosts(calls)[0].body), p.order);
});

test("options confirm is refused while live trading is OFF", async () => {
  const { live, calls } = setup();
  const p = await live.previewOption(vertical);
  await assert.rejects(live.confirm({ previewId: p.previewId, idempotencyKey: "key-opt-3", typedConfirm: "CONFIRM" }), /OFF/);
  assert.equal(orderPosts(calls).length, 0);
});

test("naked short call needs the undefined-risk phrase; cash accounts refuse it", async () => {
  const naked = { underlying: "XYZ", accountHash: "HASH1", price: 0.9,
    legs: [{ instruction: "SELL_TO_OPEN", kind: "call", strike: 55, expiration: "2026-12-18", quantity: 1 }] };
  await assert.rejects(setup().live.previewOption({ ...naked, underlying: "SPY", price: 5.9, legs: [{ ...naked.legs[0], strike: 610 }] }),
    /per-order cap/);                                       // a naked SPY call's notional ($61k) is over the $25k cap
  const { live, calls } = setup();
  live.setEnabled(true, "ENABLE LIVE TRADING");
  const p = await live.previewOption(naked);
  assert.equal(p.risk.definedRisk, false);
  assert.equal(p.extraPhrase, UNDEFINED_RISK_PHRASE);
  assert.ok(p.warnings.some((w) => /NAKED SHORT CALL/.test(w)));
  await assert.rejects(live.confirm({ previewId: p.previewId, idempotencyKey: "key-opt-4", typedConfirm: "CONFIRM" }), /extra confirmation/);
  assert.equal(orderPosts(calls).length, 0);
  const cash = setup({ type: "CASH" });
  await assert.rejects(cash.live.previewOption(naked), /cash account/);
  const covered = setup({ positions: [{ instrument: { symbol: "XYZ" }, longQuantity: 100 }] });
  const c = await covered.live.previewOption(naked);
  assert.equal(c.extraPhrase, null);                       // covered by 100 held shares
});

test("per-order cap uses max loss or notional; closes cannot exceed contracts held", async () => {
  const { live } = setup();
  await assert.rejects(live.previewOption({ ...vertical, legs: vertical.legs.map((l) => ({ ...l, quantity: 70 })) }), /per-order cap/);
  await assert.rejects(live.previewOption({ underlying: "SPY", accountHash: "HASH1", price: 9.9,
    legs: [{ instruction: "SELL_TO_CLOSE", kind: "call", strike: 600, expiration: "2026-12-18", quantity: 1 }] }), /you hold 0/);
  await assert.rejects(live.previewOption({ underlying: "SPY", accountHash: "HASH1", price: 5, orderType: "MARKET",
    legs: [{ instruction: "BUY_TO_OPEN", kind: "put", strike: 580, expiration: "2026-12-18", quantity: 1 }] }), /Market orders/);
});

test("short selling is OFF by default; when allowed it is limit-only with the unlimited-loss phrase", async () => {
  const off = setup();
  await assert.rejects(off.live.preview({ symbol: "AAPL", side: "SELL_SHORT", quantity: 5, orderType: "LIMIT", limitPrice: 100.5, accountHash: "HASH1" }), /Short selling is OFF/);
  const on = setup({ shorts: true });
  on.live.setEnabled(true, "ENABLE LIVE TRADING");
  await assert.rejects(on.live.preview({ symbol: "AAPL", side: "SELL_SHORT", quantity: 5, orderType: "MARKET", accountHash: "HASH1" }), /limit orders only/);
  const p = await on.live.preview({ symbol: "AAPL", side: "SELL_SHORT", quantity: 5, orderType: "LIMIT", limitPrice: 100.5, accountHash: "HASH1" });
  assert.equal(JSON.stringify(p.order), JSON.stringify({ session: "NORMAL", duration: "DAY", orderType: "LIMIT", price: "100.50",
    orderLegCollection: [{ instruction: "SELL_SHORT", instrument: { assetType: "EQUITY", symbol: "AAPL" }, quantity: 5 }], orderStrategyType: "SINGLE" }));
  assert.equal(JSON.stringify(p.order), py(REF_SHORT));
  assert.match(p.shortSale.warning, /unlimited/);
  await assert.rejects(on.live.confirm({ previewId: p.previewId, idempotencyKey: "key-sh-1", typedConfirm: "CONFIRM" }), /extra confirmation/);
  const done = await on.live.confirm({ previewId: p.previewId, idempotencyKey: "key-sh-2", typedConfirm: "CONFIRM", extraConfirm: SHORT_SALE_PHRASE });
  assert.equal(done.status, "submitted");
  await assert.rejects(on.live.preview({ symbol: "AAPL", side: "BUY_TO_COVER", quantity: 5, orderType: "LIMIT", limitPrice: 100, accountHash: "HASH1" }), /short 0/);
  const cash = setup({ shorts: true, type: "CASH" });
  await assert.rejects(cash.live.preview({ symbol: "AAPL", side: "SELL_SHORT", quantity: 5, orderType: "LIMIT", limitPrice: 100.5, accountHash: "HASH1" }), /margin account/);
});

test("no automated, AI, options-data or worker module references order or preview endpoints", () => {
  const here = new URL(".", import.meta.url);
  const files = ["autopilot.js", "workbench.js", "logging.js", "options.js", "analyst-scoring.js",
    ...readdirSync(new URL("ai/", here)).filter((f) => /\.(m?js|py)$/.test(f)).map((f) => `ai/${f}`),
    ...readdirSync(new URL("alerts/", here)).filter((f) => /\.m?js$/.test(f) && !f.includes("test")).map((f) => `alerts/${f}`)];
  for (const file of files) {
    let text;
    try { text = readFileSync(new URL(file, here), "utf8"); } catch { continue; }
    assert.ok(!/live-trading|live-options|\/api\/live|previewOrder|placeOrder|place_order|\/orders\b/.test(text), `${file} references the live order path`);
  }
  const engine = new URL("../engine/", here);
  for (const f of readdirSync(engine).filter((x) => /^(options_|vol_sleeve|vix_data|top_detector)/.test(x) && x.endsWith(".py"))) {
    const text = readFileSync(new URL(f, engine), "utf8");
    assert.ok(!/\/trader\/|previewOrder|placeOrder|place_order|\/api\/live/.test(text), `engine/${f} references order endpoints`);
  }
});
