// Manual Schwab OPTIONS (and short-sale) order building, validation and risk analysis. Pure functions: no network,
// no order submission. server/live-trading.js calls these from its ticket -> preview -> typed-Confirm flow.
//
// Order JSON follows schwab-py 1.6 (schwab/orders/options.py and generic.OrderBuilder.build), checked by generating
// reference orders with schwab-py on this PC (2026-11-25):
//   single leg:  { session, duration, orderType: "LIMIT", price: "1.25",
//                  orderLegCollection: [{ instruction: "BUY_TO_OPEN", instrument: { assetType: "OPTION", symbol: "SPY   261218C00600000" }, quantity: 2 }],
//                  orderStrategyType: "SINGLE" }
//   multi leg:   adds complexOrderStrategyType (e.g. "VERTICAL") and an order-level quantity; orderType NET_DEBIT / NET_CREDIT / NET_ZERO.
//   equity short: instruction "SELL_SHORT" / "BUY_TO_COVER" (schwab-py equity_sell_short_limit / equity_buy_to_cover_limit).
// OCC symbol: underlying left-justified to 6, YYMMDD, C|P, strike * 1000 zero-padded to 8 (schwab-py OptionSymbol).
// Market orders are never built for options (limit / net-debit / net-credit only).

export const OPTION_INSTRUCTIONS = Object.freeze(["BUY_TO_OPEN", "SELL_TO_OPEN", "BUY_TO_CLOSE", "SELL_TO_CLOSE"]);
export const EQUITY_INSTRUCTIONS = Object.freeze(["BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER"]);
// schwab-py 1.6 ComplexOrderStrategyType values usable for option tickets.
export const COMPLEX_TYPES = Object.freeze(["NONE", "COVERED", "VERTICAL", "BACK_RATIO", "CALENDAR", "DIAGONAL", "STRADDLE",
  "STRANGLE", "COLLAR_SYNTHETIC", "BUTTERFLY", "CONDOR", "IRON_CONDOR", "VERTICAL_ROLL", "COLLAR_WITH_STOCK", "DOUBLE_DIAGONAL",
  "UNBALANCED_BUTTERFLY", "UNBALANCED_CONDOR", "UNBALANCED_IRON_CONDOR", "UNBALANCED_VERTICAL_ROLL", "CUSTOM"]);
export const OPTION_ORDER_TYPES = Object.freeze(["LIMIT", "NET_DEBIT", "NET_CREDIT", "NET_ZERO"]);
export const UNDEFINED_RISK_PHRASE = "I ACCEPT UNDEFINED RISK";
export const SHORT_SALE_PHRASE = "I UNDERSTAND SHORT SELLING HAS UNLIMITED LOSS";
export const TYPED_CONFIRM = "CONFIRM";
export const FEES = Object.freeze({ perContract: 0.65, orfPerContract: 0.011, tafPerContractSold: 0.00329, secPerMillion: 20.6, btcWaiver: 0.05 });

export class OptionTicketError extends Error {
  constructor(message) { super(message); this.status = 400; }
}
const fail = (m) => { throw new OptionTicketError(m); };

export function occSymbol(underlying, expiration, kind, strike) {
  const root = String(underlying ?? "").trim().toUpperCase();
  if (!/^[A-Z][A-Z0-9.]{0,5}$/.test(root)) fail("Underlying must be 1-6 letters.");
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(expiration ?? ""));
  if (!m) fail("Expiration must be YYYY-MM-DD.");
  const k = String(kind ?? "").toUpperCase();
  const cp = k === "C" || k === "CALL" ? "C" : k === "P" || k === "PUT" ? "P" : fail("Option type must be CALL or PUT.");
  const milli = Math.round(Number(strike) * 1000);
  if (!(milli > 0 && milli < 100_000_000)) fail("Strike out of range.");
  return `${root.padEnd(6, " ")}${m[1].slice(2)}${m[2]}${m[3]}${cp}${String(milli).padStart(8, "0")}`;
}

export function parseOcc(symbol) {
  const s = String(symbol ?? "");
  const m = /^([A-Z][A-Z0-9. ]{5})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$/.exec(s);
  if (!m) fail(`Not an OCC option symbol: ${s.trim()}`);
  return { underlying: m[1].trim(), expiration: `20${m[2]}-${m[3]}-${m[4]}`, kind: m[5] === "C" ? "call" : "put", strike: Number(m[6]) / 1000, symbol: s };
}

const gcd = (a, b) => (b ? gcd(b, a % b) : a);

/** Validate a multi-leg option ticket. input: { underlying, legs: [{ instruction, kind, strike, expiration, quantity } | { instruction, equity: true, quantity }],
 *  orderType, price, duration, complexType, accountHash }. Returns a normalized ticket. */
export function validateOptionTicket(input = {}) {
  const underlying = String(input.underlying ?? "").trim().toUpperCase();
  if (!/^[A-Z][A-Z.]{0,5}$/.test(underlying)) fail("Enter a valid underlying symbol.");
  const rawLegs = Array.isArray(input.legs) ? input.legs : [];
  if (!rawLegs.length) fail("Add at least one leg.");
  const legs = rawLegs.map((l, i) => {
    const quantity = Number(l.quantity);
    if (!(Number.isInteger(quantity) && quantity > 0)) fail(`Leg ${i + 1}: quantity must be a whole number.`);
    const instruction = String(l.instruction ?? "").toUpperCase();
    if (l.equity) {
      if (!["BUY", "SELL"].includes(instruction)) fail(`Leg ${i + 1}: stock legs are BUY or SELL.`);
      return { equity: true, instruction, quantity, symbol: underlying, kind: "stock" };
    }
    if (!OPTION_INSTRUCTIONS.includes(instruction)) fail(`Leg ${i + 1}: instruction must be one of ${OPTION_INSTRUCTIONS.join(", ")}.`);
    const symbol = l.symbol ? parseOcc(l.symbol).symbol : occSymbol(underlying, l.expiration, l.kind, l.strike);
    const p = parseOcc(symbol);
    if (p.underlying !== underlying && !p.underlying.startsWith(underlying)) fail(`Leg ${i + 1}: ${symbol.trim()} is not an option on ${underlying}.`);
    return { equity: false, instruction, quantity, symbol, kind: p.kind, strike: p.strike, expiration: p.expiration };
  });
  const optionLegs = legs.filter((l) => !l.equity);
  if (!optionLegs.length) fail("An options ticket needs at least one option leg (use the stock ticket for shares).");
  if (optionLegs.length > 4 || legs.length - optionLegs.length > 1) fail("Up to 4 option legs and 1 stock leg.");
  if (new Set(legs.map((l) => l.symbol)).size !== legs.length) fail("Each leg must be a different contract.");
  const orderType = String(input.orderType ?? (optionLegs.length === 1 && legs.length === 1 ? "LIMIT" : "NET_DEBIT")).toUpperCase();
  if (orderType === "MARKET") fail("Market orders are not allowed for options. Use a limit (or net debit / net credit) price.");
  if (!OPTION_ORDER_TYPES.includes(orderType)) fail(`Order type must be ${OPTION_ORDER_TYPES.join(", ")}.`);
  const single = legs.length === 1;
  if (single && orderType !== "LIMIT") fail("Single-leg option orders use LIMIT.");
  if (!single && orderType === "LIMIT") fail("Multi-leg orders use NET_DEBIT, NET_CREDIT or NET_ZERO.");
  const price = Number(input.price);
  if (orderType === "NET_ZERO" ? !(price === 0 || Number.isNaN(price) || input.price === undefined) : !(price > 0)) fail("Enter a positive limit / net price.");
  const duration = String(input.duration ?? "DAY").toUpperCase();
  if (!["DAY", "GOOD_TILL_CANCEL"].includes(duration)) fail("Time in force must be DAY or GOOD_TILL_CANCEL.");
  const complexType = single ? "NONE" : String(input.complexType ?? "CUSTOM").toUpperCase();
  if (!COMPLEX_TYPES.includes(complexType)) fail(`Strategy type must be one of ${COMPLEX_TYPES.join(", ")}.`);
  const accountHash = String(input.accountHash ?? "");
  if (!accountHash) fail("Choose an account.");
  return { underlying, legs, orderType, price: orderType === "NET_ZERO" ? 0 : price, duration, complexType, accountHash };
}

const money = (x) => (x < 1 ? x.toFixed(4).replace(/0{1,2}$/, "") : x.toFixed(2));

/** Schwab order JSON (field order and shape as schwab-py builds it). */
export function buildOptionOrder(t) {
  const legs = t.legs.map((l) => ({ instruction: l.instruction, instrument: { assetType: l.equity ? "EQUITY" : "OPTION", symbol: l.symbol }, quantity: l.quantity }));
  const order = { session: "NORMAL", duration: t.duration, orderType: t.orderType };
  if (t.legs.length > 1) {
    order.complexOrderStrategyType = t.complexType;
    order.quantity = t.legs.filter((l) => !l.equity).map((l) => l.quantity).reduce(gcd);
  }
  if (t.orderType !== "NET_ZERO") order.price = money(t.price);
  order.orderLegCollection = legs;
  order.orderStrategyType = "SINGLE";
  return order;
}

export function buildShortSaleOrder({ symbol, instruction, quantity, limitPrice, duration = "DAY" }) {
  if (!["SELL_SHORT", "BUY_TO_COVER"].includes(instruction)) fail("Short-sale tickets use SELL_SHORT or BUY_TO_COVER.");
  if (!(Number.isInteger(quantity) && quantity > 0)) fail("Quantity must be a whole number of shares.");
  if (!(limitPrice > 0)) fail("Short sales and covers are limit orders only: enter a limit price.");
  return { session: "NORMAL", duration, orderType: "LIMIT", price: money(limitPrice),
    orderLegCollection: [{ instruction, instrument: { assetType: "EQUITY", symbol }, quantity }], orderStrategyType: "SINGLE" };
}

// ---------------------------------------------------------------- math (Black-Scholes, payoff)
const ncdf = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
function erf(x) { // Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7)
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
export function bsPrice(S, K, T, vol, r = 0.04, kind = "call") {
  if (T <= 0 || vol <= 0) return kind === "call" ? Math.max(0, S - K) : Math.max(0, K - S);
  const sq = vol * Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r + 0.5 * vol * vol) * T) / sq;
  const d2 = d1 - sq;
  return kind === "call" ? S * ncdf(d1) - K * Math.exp(-r * T) * ncdf(d2) : K * Math.exp(-r * T) * ncdf(-d2) - S * ncdf(-d1);
}
const signed = (l) => (/^BUY/.test(l.instruction) ? 1 : -1) * l.quantity;
const yearsBetween = (a, b) => Math.max(0, (Date.parse(`${b}T21:00:00Z`) - Date.parse(`${a}T21:00:00Z`)) / (365 * 86400e3));

/** Max loss / gain / breakevens / defined risk for legs priced at `price` per share (debit > 0, credit < 0 per leg).
 *  heldShares: shares of the underlying already in the account (cover short calls). Calendars/diagonals are valued
 *  at the front expiry with Black-Scholes for later legs (approximate: true). */
export function analyzeOptionOrder(legs, { spot, heldShares = 0, vol = 0.3, rate = 0.04 } = {}) {
  const opts = legs.filter((l) => !l.equity);
  const front = opts.map((l) => l.expiration).sort()[0];
  const approximate = new Set(opts.map((l) => l.expiration)).size > 1;
  const stockQty = legs.filter((l) => l.equity).reduce((a, l) => a + signed(l), 0);
  const entry = legs.reduce((a, l) => a + signed(l) * (l.equity ? 1 : 100) * (l.price ?? 0), 0);
  const value = (S) => {
    let v = stockQty * S;
    for (const l of opts) {
      const q = signed(l) * 100;
      if (!approximate || l.expiration === front) v += q * (l.kind === "call" ? Math.max(0, S - l.strike) : Math.max(0, l.strike - S));
      else v += q * bsPrice(S, l.strike, Math.max(yearsBetween(front, l.expiration), 1 / 365), l.iv ?? vol, rate, l.kind);
    }
    return v - entry;
  };
  const strikes = [...new Set(opts.map((l) => l.strike))].sort((a, b) => a - b);
  const hi = Math.max(spot || 0, ...strikes) * 3 + 1;
  const grid = [];
  for (let i = 0; i <= 3000; i++) grid.push((hi * i) / 3000);
  grid.push(...strikes);
  grid.sort((a, b) => a - b);
  const vals = grid.map(value);
  const netCalls = opts.filter((l) => l.kind === "call").reduce((a, l) => a + signed(l) * 100, 0) + stockQty + Math.max(0, heldShares);
  const unboundedLoss = netCalls < 0;
  const unboundedGain = netCalls > 0;
  const breakevens = [];
  for (let i = 1; i < grid.length; i++) {
    const [a, b, va, vb] = [grid[i - 1], grid[i], vals[i - 1], vals[i]];
    if (va === 0) breakevens.push(a); else if (va * vb < 0) breakevens.push(a + ((b - a) * -va) / (vb - va));
  }
  const nakedCalls = opts.filter((l) => l.kind === "call").reduce((a, l) => a + signed(l), 0) * 100 + stockQty + Math.max(0, heldShares) < 0;
  const shortPuts = opts.filter((l) => l.kind === "put" && signed(l) < 0);
  const longPutsQty = opts.filter((l) => l.kind === "put" && signed(l) > 0).reduce((a, l) => a + l.quantity, 0);
  const uncoveredPuts = shortPuts.reduce((a, l) => a + l.quantity, 0) > longPutsQty;
  return {
    entryCash: -entry, netDebit: entry > 0 ? entry : 0, netCredit: entry < 0 ? -entry : 0,
    maxLoss: unboundedLoss ? null : Math.min(...vals), maxGain: unboundedGain ? null : Math.max(...vals),
    unboundedLoss, unboundedGain, definedRisk: !unboundedLoss && !uncoveredPuts, nakedCalls, uncoveredPuts,
    breakevens: [...new Set(breakevens.map((x) => Math.round(x * 100) / 100))], approximate, frontExpiration: front,
    stressLoss: Math.min(0, value(0), value(2 * (spot || Math.max(...strikes)))),
    curve: grid.filter((_, i) => i % 30 === 0).map((S) => [Math.round(S * 100) / 100, Math.round(value(S) * 100) / 100]),
  };
}

export function estimateFees(legs) {
  let total = 0;
  for (const l of legs) {
    if (l.equity) continue;
    const sell = /^SELL/.test(l.instruction);
    const waived = l.instruction === "BUY_TO_CLOSE" && (l.price ?? 1) <= FEES.btcWaiver;
    total += (waived ? 0 : FEES.perContract * l.quantity) + FEES.orfPerContract * l.quantity;
    if (sell) total += FEES.tafPerContractSold * l.quantity + (FEES.secPerMillion * l.quantity * 100 * (l.price ?? 0)) / 1e6;
  }
  return Math.round(total * 100) / 100;
}

/** Dollar amount checked against the per-order cap: defined risk -> max loss; undefined risk -> the larger of the
 *  notional (strike * 100 * contracts summed over legs, plus stock legs) and the stress loss (underlying at 0 or at
 *  2x today's price; true max loss on a naked call is unlimited). */
export function capExposure(analysis, legs, spot) {
  const notional = legs.reduce((a, l) => a + (l.equity ? l.quantity * spot : l.quantity * 100 * (l.strike || 0)), 0);
  if (analysis.definedRisk && analysis.maxLoss !== null) return { basis: "max loss", amount: Math.abs(Math.min(0, analysis.maxLoss)), notional };
  return { basis: "max(notional, stress loss at 0 / 2x price) - undefined risk", amount: Math.max(notional, Math.abs(analysis.stressLoss)), notional };
}
