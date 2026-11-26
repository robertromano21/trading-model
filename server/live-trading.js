// Manual live trading through Schwab: ticket -> preview -> explicit Confirm. Nothing else may send orders.
//
// Safety design
// - Only the four routes below, driven by a person clicking in the dashboard, can reach Schwab's order endpoint.
//   The Python engine (paper trader, research, day trader, AI layer) has no order code; engine/schwab_data.py refuses
//   /trader/ URLs and a unit test checks that no engine file references order endpoints.
// - Two switches, both default OFF: the environment variable LIVE_TRADING_ALLOWED=1 (set by Robert before starting
//   the dashboard) AND the in-app toggle (Massive/LiveTrading/settings.json, reset to OFF whenever the dashboard
//   restarts). Previews work with live trading OFF; Confirm refuses unless both are ON.
// - Confirm only accepts a previewId issued by this server within the last 120 seconds, for exactly the previewed
//   order, once. A client idempotency key and the previewId both guard against double submits.
// - Sells cannot exceed the long position in the selected account (no shorting from the stock ticket).
// - Options (live-options.js): limit / net-debit / net-credit only (never market), single or multi-leg in schwab-py's
//   order format; closes cannot exceed the contracts held; max loss, max gain, breakevens, greeks and fees are shown at
//   preview; the per-order cap applies to max loss (defined risk) or max(notional, max loss) (undefined risk); naked
//   short calls and other undefined-risk shorts need the extra phrase I ACCEPT UNDEFINED RISK at confirm and are
//   refused outright in cash accounts. Option and short-sale confirms also need the typed word CONFIRM.
// - Short sales (SELL_SHORT / BUY_TO_COVER, equity only): OFF unless LIVE_SHORT_SELLING_ALLOWED=1, limit orders only,
//   extra phrase I UNDERSTAND SHORT SELLING HAS UNLIMITED LOSS at confirm; covers cannot exceed the short position.
// - Every preview, confirm, refusal, error and status check is appended to Massive/LiveTrading/orders.jsonl
//   (gitignored). Tokens, secrets and full account numbers are never logged or sent to the browser.
import { randomUUID, createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync, appendFileSync } from "node:fs";
import { join } from "node:path";
import { OptionTicketError, SHORT_SALE_PHRASE, TYPED_CONFIRM, UNDEFINED_RISK_PHRASE, analyzeOptionOrder, buildOptionOrder,
  buildShortSaleOrder, capExposure, estimateFees, validateOptionTicket } from "./live-options.js";

const API = "https://api.schwabapi.com";
const PREVIEW_TTL_MS = 120_000;
const MAX_NOTIONAL = Number(process.env.LIVE_TRADING_MAX_ORDER_USD || 25_000);

export function createLiveTrading({ root, dataDir, log = console, fetchImpl = globalThis.fetch, now = () => Date.now(),
  env = process.env } = {}) {
  const dir = join(dataDir, "LiveTrading");
  const settingsPath = join(dir, "settings.json");
  const journal = join(dir, "orders.jsonl");
  const tokenPath = env.SCHWAB_TOKEN_PATH || join(root, "schwab_token.json");
  const previews = new Map();          // previewId -> { order, accountHash, expires, used, result }
  const idempotency = new Map();       // client key -> result
  let accountCache = null;
  mkdirSync(dir, { recursive: true });
  writeSettings({ enabled: false, changedAt: new Date(now()).toISOString(), reason: "dashboard start" });

  function writeSettings(value) {
    writeFileSync(`${settingsPath}.tmp`, JSON.stringify(value, null, 2));
    renameSync(`${settingsPath}.tmp`, settingsPath);
  }
  function settings() {
    try { return JSON.parse(readFileSync(settingsPath, "utf8")); } catch { return { enabled: false }; }
  }
  const envAllowed = () => env.LIVE_TRADING_ALLOWED === "1";
  const shortsAllowed = () => env.LIVE_SHORT_SELLING_ALLOWED === "1";
  const liveEnabled = () => envAllowed() && settings().enabled === true;
  function record(event) {
    appendFileSync(journal, `${JSON.stringify({ at: new Date(now()).toISOString(), ...event })}\n`);
  }

  // ---------------------------------------------------------------- auth
  function readToken() {
    if (!existsSync(tokenPath)) throw new LiveError(401, "No Schwab token file. Create schwab_token.json with schwab-py first.");
    return JSON.parse(readFileSync(tokenPath, "utf8"));
  }
  async function accessToken() {
    const data = readToken();
    const token = data.token ?? {};
    if ((token.expires_at ?? 0) * 1000 > now() + 60_000) return token.access_token;
    const key = env.SCHWAB_APP_KEY?.trim();
    const secret = env.SCHWAB_APP_SECRET?.trim();
    if (!key || !secret) {
      throw new LiveError(401, "The Schwab access token expired and SCHWAB_APP_KEY / SCHWAB_APP_SECRET are not set, so it cannot be refreshed.");
    }
    const response = await fetchImpl(`${API}/v1/oauth/token`, {
      method: "POST",
      headers: { Authorization: `Basic ${Buffer.from(`${key}:${secret}`).toString("base64")}`,
        "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ grant_type: "refresh_token", refresh_token: token.refresh_token }),
    });
    if (!response.ok) throw new LiveError(401, `Schwab token refresh failed (HTTP ${response.status}). Re-create the token with schwab-py.`);
    const fresh = await response.json();
    data.token = { ...token, ...fresh, refresh_token: fresh.refresh_token ?? token.refresh_token,
      expires_at: Math.floor(now() / 1000) + Number(fresh.expires_in ?? 1800) };
    writeFileSync(`${tokenPath}.tmp`, JSON.stringify(data));
    renameSync(`${tokenPath}.tmp`, tokenPath);
    return data.token.access_token;
  }
  async function schwab(path, { method = "GET", body } = {}) {
    const response = await fetchImpl(`${API}${path}`, {
      method,
      headers: { Authorization: `Bearer ${await accessToken()}`, Accept: "application/json",
        ...(body ? { "Content-Type": "application/json" } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    });
    const text = await response.text();
    let json = null;
    try { json = text ? JSON.parse(text) : null; } catch { json = null; }
    if (!response.ok) {
      const message = json?.message || json?.errors?.[0]?.detail || json?.error || `HTTP ${response.status}`;
      throw new LiveError(response.status >= 500 ? 502 : 400, `Schwab: ${message}`);
    }
    return { json, location: response.headers.get("location") };
  }

  // ---------------------------------------------------------------- helpers
  async function accounts() {
    if (!accountCache) {
      const { json } = await schwab("/trader/v1/accounts/accountNumbers");
      accountCache = (json ?? []).map((a) => ({ hash: a.hashValue, label: `…${String(a.accountNumber).slice(-4)}` }));
    }
    return accountCache;
  }
  async function quote(symbol) {
    const { json } = await schwab(`/marketdata/v1/quotes?symbols=${encodeURIComponent(symbol)}&fields=quote`);
    const q = json?.[symbol]?.quote;
    if (!q) throw new LiveError(400, `No Schwab quote for ${symbol}.`);
    return { last: q.lastPrice, bid: q.bidPrice, ask: q.askPrice, time: q.quoteTime };
  }
  async function accountPositions(accountHash) {
    const { json } = await schwab(`/trader/v1/accounts/${encodeURIComponent(accountHash)}?fields=positions`);
    return { positions: json?.securitiesAccount?.positions ?? [], type: json?.securitiesAccount?.type ?? null };
  }
  async function longQuantity(accountHash, symbol) {
    const { positions } = await accountPositions(accountHash);
    return positions.filter((p) => p.instrument?.symbol === symbol).reduce((sum, p) => sum + Number(p.longQuantity || 0), 0);
  }
  const held = (positions, symbol, side) => positions.filter((p) => p.instrument?.symbol === symbol)
    .reduce((sum, p) => sum + Number((side === "short" ? p.shortQuantity : p.longQuantity) || 0), 0);
  async function quotesFor(symbols) {
    const { json } = await schwab(`/marketdata/v1/quotes?symbols=${encodeURIComponent(symbols.join(","))}&fields=quote`);
    return json ?? {};
  }

  function validateTicket(input) {
    const symbol = String(input.symbol ?? "").trim().toUpperCase();
    if (!/^[A-Z][A-Z.]{0,5}$/.test(symbol)) throw new LiveError(400, "Enter a valid US stock or ETF symbol.");
    const side = String(input.side ?? "").toUpperCase();
    if (["SELL_SHORT", "BUY_TO_COVER"].includes(side)) {
      if (!shortsAllowed()) throw new LiveError(403, "Short selling is OFF. Start the dashboard with LIVE_SHORT_SELLING_ALLOWED=1 to allow SELL_SHORT / BUY_TO_COVER tickets.");
      if (String(input.orderType ?? "").toUpperCase() !== "LIMIT") throw new LiveError(400, "Short sales and covers are limit orders only.");
    } else if (!["BUY", "SELL"].includes(side)) throw new LiveError(400, "Side must be BUY or SELL.");
    const orderType = String(input.orderType ?? "MARKET").toUpperCase();
    if (!["MARKET", "LIMIT"].includes(orderType)) throw new LiveError(400, "Order type must be MARKET or LIMIT.");
    const duration = String(input.duration ?? "DAY").toUpperCase();
    if (!["DAY", "GOOD_TILL_CANCEL"].includes(duration)) throw new LiveError(400, "Time in force must be DAY or GOOD_TILL_CANCEL.");
    if (orderType === "MARKET" && duration !== "DAY") throw new LiveError(400, "Market orders must be DAY orders.");
    const limitPrice = orderType === "LIMIT" ? Number(input.limitPrice) : null;
    if (orderType === "LIMIT" && !(limitPrice > 0)) throw new LiveError(400, "Enter a positive limit price.");
    const quantity = input.quantity === undefined || input.quantity === "" ? null : Number(input.quantity);
    const dollars = input.dollars === undefined || input.dollars === "" ? null : Number(input.dollars);
    if ((quantity === null) === (dollars === null)) throw new LiveError(400, "Enter either a share quantity or a dollar amount.");
    if (quantity !== null && !(Number.isInteger(quantity) && quantity > 0)) throw new LiveError(400, "Quantity must be a whole number of shares.");
    if (dollars !== null && !(dollars > 0)) throw new LiveError(400, "Dollar amount must be positive.");
    const accountHash = String(input.accountHash ?? "");
    if (!accountHash) throw new LiveError(400, "Choose an account.");
    return { symbol, side, orderType, duration, limitPrice, quantity, dollars, accountHash };
  }

  function schwabOrder(t, quantity) {
    if (t.side === "SELL_SHORT" || t.side === "BUY_TO_COVER") {
      return buildShortSaleOrder({ symbol: t.symbol, instruction: t.side, quantity, limitPrice: t.limitPrice, duration: t.duration });
    }
    return {
      orderType: t.orderType, session: "NORMAL", duration: t.duration, orderStrategyType: "SINGLE",
      ...(t.orderType === "LIMIT" ? { price: t.limitPrice.toFixed(t.limitPrice < 1 ? 4 : 2) } : {}),
      orderLegCollection: [{ instruction: t.side, quantity, instrument: { symbol: t.symbol, assetType: "EQUITY" } }],
    };
  }

  // ---------------------------------------------------------------- operations
  async function preview(input) {
    const t = validateTicket(input);
    const known = await accounts();
    if (!known.some((a) => a.hash === t.accountHash)) throw new LiveError(400, "Unknown account.");
    const q = await quote(t.symbol);
    const buying = t.side === "BUY" || t.side === "BUY_TO_COVER";
    const reference = t.orderType === "LIMIT" ? t.limitPrice : (buying ? q.ask || q.last : q.bid || q.last);
    const quantity = t.quantity ?? Math.floor(t.dollars / reference);
    if (!(quantity >= 1)) throw new LiveError(400, `$${t.dollars} buys less than one share at ${reference}.`);
    const notional = quantity * reference;
    if (notional > MAX_NOTIONAL) throw new LiveError(400, `Order value $${notional.toFixed(2)} exceeds the $${MAX_NOTIONAL} per-order cap (LIVE_TRADING_MAX_ORDER_USD).`);
    if (t.side === "SELL") {
      const held = await longQuantity(t.accountHash, t.symbol);
      if (quantity > held) throw new LiveError(400, `You hold ${held} ${t.symbol}; this ticket cannot sell more than that (no short sales).`);
    }
    let shortSale = null;
    if (t.side === "SELL_SHORT" || t.side === "BUY_TO_COVER") {
      const { positions, type } = await accountPositions(t.accountHash);
      if (type && String(type).toUpperCase() === "CASH") throw new LiveError(400, "Short selling needs a margin account; this is a cash account.");
      const short = held(positions, t.symbol, "short");
      if (t.side === "BUY_TO_COVER" && quantity > short) throw new LiveError(400, `You are short ${short} ${t.symbol}; cannot cover more than that.`);
      shortSale = { warning: "SHORT SALE: losses are unlimited if the price rises; stops are not guaranteed through gaps; "
        + "borrow fees and dividends owed apply; Schwab may recall borrowed shares.", currentShort: short,
        extraPhrase: t.side === "SELL_SHORT" ? SHORT_SALE_PHRASE : null, initialRequirementRegT: t.side === "SELL_SHORT" ? notional * 1.5 : null };
    }
    const order = schwabOrder(t, quantity);
    let schwabPreview = null;
    try {
      const { json } = await schwab(`/trader/v1/accounts/${encodeURIComponent(t.accountHash)}/previewOrder`, { method: "POST", body: order });
      schwabPreview = json;
    } catch (error) {
      schwabPreview = { unavailable: error.message };
    }
    const secFee = t.side === "SELL" || t.side === "SELL_SHORT" ? notional * 20.6e-6 + Math.min(quantity * 0.000195, 9.79) : 0;
    const previewId = randomUUID();
    const digest = createHash("sha256").update(JSON.stringify([t.accountHash, order])).digest("hex");
    const result = {
      previewId, expiresInSeconds: PREVIEW_TTL_MS / 1000, liveTradingEnabled: liveEnabled(),
      ticket: { ...t, accountHash: undefined, account: known.find((a) => a.hash === t.accountHash).label, quantity },
      quote: q, referencePrice: reference, estimatedNotional: notional,
      estimatedFees: { commission: 0, regulatory: Number(secFee.toFixed(2)) },
      estimatedTotal: buying ? notional : notional - secFee,
      schwabPreview, order, ...(shortSale ? { shortSale, typedConfirm: TYPED_CONFIRM } : {}),
    };
    previews.set(previewId, { order, accountHash: t.accountHash, digest, expires: now() + PREVIEW_TTL_MS, used: false,
      typed: Boolean(shortSale), extraPhrase: shortSale?.extraPhrase ?? null });
    record({ event: "preview", previewId, symbol: t.symbol, side: t.side, quantity, orderType: t.orderType,
      limitPrice: t.limitPrice, duration: t.duration, account: result.ticket.account, notional,
      schwabPreview: schwabPreview?.unavailable ? "unavailable" : "ok" });
    return result;
  }

  // ---------------------------------------------------------------- options
  async function previewOption(input) {
    let t;
    try { t = validateOptionTicket(input); } catch (error) {
      if (error instanceof OptionTicketError) throw new LiveError(400, error.message);
      throw error;
    }
    const known = await accounts();
    if (!known.some((a) => a.hash === t.accountHash)) throw new LiveError(400, "Unknown account.");
    const symbols = [t.underlying, ...t.legs.filter((l) => !l.equity).map((l) => l.symbol)];
    const quotes = await quotesFor(symbols);
    const uq = quotes[t.underlying]?.quote;
    if (!uq) throw new LiveError(400, `No Schwab quote for ${t.underlying}.`);
    const spot = uq.lastPrice || uq.mark || (uq.bidPrice + uq.askPrice) / 2;
    const { positions, type } = await accountPositions(t.accountHash);
    const cash = type && String(type).toUpperCase() === "CASH";
    const shares = held(positions, t.underlying, "long");
    const greeks = { delta: 0, gamma: 0, theta: 0, vega: 0 };
    const legs = t.legs.map((l) => {
      if (l.equity) return { ...l, price: spot };
      const q = quotes[l.symbol]?.quote ?? quotes[l.symbol] ?? null;
      if (!q) throw new LiveError(400, `No Schwab quote for ${l.symbol.trim()}. Check the strike and expiration on the chain.`);
      const buy = /^BUY/.test(l.instruction);
      const sign = (buy ? 1 : -1) * l.quantity * 100;
      for (const k of Object.keys(greeks)) if (Number.isFinite(Number(q[k]))) greeks[k] += Number(q[k]) * sign;
      if (/_TO_CLOSE$/.test(l.instruction)) {
        const have = held(positions, l.symbol, l.instruction === "SELL_TO_CLOSE" ? "long" : "short");
        if (l.quantity > have) throw new LiveError(400, `${l.instruction} ${l.quantity} ${l.symbol.trim()}: you hold ${have} (${l.instruction === "SELL_TO_CLOSE" ? "long" : "short"}).`);
      }
      const iv = Number(q.volatility);
      return { ...l, bid: q.bidPrice, ask: q.askPrice, mark: q.mark, price: buy ? (q.askPrice ?? q.mark) : (q.bidPrice ?? q.mark),
        iv: Number.isFinite(iv) && iv > 0 ? iv / 100 : undefined };
    });
    // the ticket's net price is what Robert pays/receives; analyse at that price spread across legs proportionally
    const opening = t.legs.some((l) => /_TO_OPEN$/.test(l.instruction));
    const analysis = analyzeOptionOrder(legs, { spot, heldShares: opening ? shares : 0 });
    const credit = t.orderType === "NET_CREDIT" || (t.legs.length === 1 && /^SELL/.test(t.legs[0].instruction));
    const netAtLimit = credit ? -t.price : t.price;
    const scale = analysis.entryCash !== 0 ? (-netAtLimit * 100 * (buildOptionOrder(t).quantity ?? t.legs[0].quantity)) / analysis.entryCash : 1;
    const atLimit = Number.isFinite(scale) && scale > 0 ? analyzeOptionOrder(legs.map((l) => (l.equity ? l : { ...l, price: l.price * scale })),
      { spot, heldShares: opening ? shares : 0 }) : analysis;
    const fees = estimateFees(legs);
    const cap = capExposure(atLimit, legs, spot);
    if (cap.amount > MAX_NOTIONAL) throw new LiveError(400, `Order risk $${cap.amount.toFixed(2)} (${cap.basis}) exceeds the $${MAX_NOTIONAL} per-order cap.`);
    const undefinedRisk = !atLimit.definedRisk;
    if (undefinedRisk && cash) throw new LiveError(400, "Naked / undefined-risk short options are not allowed in a cash account.");
    const order = buildOptionOrder(t);
    let schwabPreview = null;
    try {
      const { json } = await schwab(`/trader/v1/accounts/${encodeURIComponent(t.accountHash)}/previewOrder`, { method: "POST", body: order });
      schwabPreview = json;
    } catch (error) {
      schwabPreview = { unavailable: error.message };
    }
    const warnings = [];
    if (atLimit.nakedCalls) warnings.push("NAKED SHORT CALL: unlimited loss if the stock rises. Blocked unless you type the undefined-risk phrase.");
    if (atLimit.uncoveredPuts && !atLimit.nakedCalls) warnings.push("Uncovered short put: loss up to the strike minus the credit (needs margin or the cash to buy 100 shares per contract).");
    if (legs.some((l) => !l.equity && /^SELL_TO_OPEN/.test(l.instruction) && l.kind === "call")) warnings.push("Short calls can be assigned early, especially the day before an ex-dividend date when the call is in the money.");
    if (atLimit.approximate) warnings.push("Calendar/diagonal: max loss/gain are approximate (later legs valued with Black-Scholes at the front expiration).");
    warnings.push("Schwab's Trader API does not report your option approval level; Schwab rejects orders above your approved tier at preview/submit.");
    const previewId = randomUUID();
    const digest = createHash("sha256").update(JSON.stringify([t.accountHash, order])).digest("hex");
    const label = known.find((a) => a.hash === t.accountHash).label;
    const round = (x) => (x === null || x === undefined ? null : Math.round(x * 100) / 100);
    const result = {
      previewId, expiresInSeconds: PREVIEW_TTL_MS / 1000, liveTradingEnabled: liveEnabled(), kind: "option",
      ticket: { ...t, accountHash: undefined, account: label }, underlyingPrice: spot, accountType: type, sharesHeld: shares,
      legs: legs.map(({ symbol, instruction, quantity, kind, strike, expiration, bid, ask, mark, iv, equity }) =>
        ({ symbol: symbol.trim(), instruction, quantity, kind, strike, expiration, bid, ask, mark, iv, equity })),
      risk: { maxLoss: round(atLimit.maxLoss), maxGain: round(atLimit.maxGain), breakevens: atLimit.breakevens,
        definedRisk: atLimit.definedRisk, unboundedLoss: atLimit.unboundedLoss, netDebit: round(atLimit.netDebit), netCredit: round(atLimit.netCredit),
        capBasis: cap.basis, capAmount: round(cap.amount), maxOrderUsd: MAX_NOTIONAL, payoff: atLimit.curve },
      greeks: Object.fromEntries(Object.entries(greeks).map(([k, v]) => [k, round(v)])),
      estimatedFees: { commissionAndRegulatory: fees, note: "$0.65/contract + ORF $0.011 + TAF $0.00329/contract sold + SEC on sale premium (estimate)" },
      warnings, typedConfirm: TYPED_CONFIRM, extraPhrase: undefinedRisk ? UNDEFINED_RISK_PHRASE : null, schwabPreview, order,
    };
    previews.set(previewId, { order, accountHash: t.accountHash, digest, expires: now() + PREVIEW_TTL_MS, used: false,
      typed: true, extraPhrase: undefinedRisk ? UNDEFINED_RISK_PHRASE : null });
    record({ event: "preview_option", previewId, underlying: t.underlying, orderType: t.orderType, price: t.price,
      complexType: t.complexType, legs: result.legs.map((l) => `${l.instruction} ${l.quantity} ${l.symbol}`), account: label,
      maxLoss: result.risk.maxLoss, capAmount: result.risk.capAmount, undefinedRisk, schwabPreview: schwabPreview?.unavailable ? "unavailable" : "ok" });
    return result;
  }

  async function confirm({ previewId, idempotencyKey, typedConfirm, extraConfirm }) {
    if (!idempotencyKey || String(idempotencyKey).length < 8) throw new LiveError(400, "Missing idempotency key.");
    if (idempotency.has(idempotencyKey)) return { ...idempotency.get(idempotencyKey), duplicate: true };
    if (!envAllowed()) throw refused(previewId, "Live trading is not allowed: start the dashboard with LIVE_TRADING_ALLOWED=1.");
    if (!liveEnabled()) throw refused(previewId, "Live trading is OFF. Turn it on in the live trading panel first.");
    const entry = previews.get(previewId);
    if (!entry) throw refused(previewId, "Unknown preview. Preview the order again.");
    if (entry.used) throw refused(previewId, "This preview was already submitted.");
    if (now() > entry.expires) throw refused(previewId, "The preview expired. Preview the order again for a fresh quote.");
    if (entry.typed && typedConfirm !== TYPED_CONFIRM) throw refused(previewId, `Type ${TYPED_CONFIRM} to send this order.`);
    if (entry.extraPhrase && extraConfirm !== entry.extraPhrase) throw refused(previewId, `This order needs the extra confirmation: type ${entry.extraPhrase}.`);
    entry.used = true;                                   // before the network call: blocks double submits
    idempotency.set(idempotencyKey, { status: "submitting", previewId });
    record({ event: "submit", previewId, idempotencyKey, order: entry.order });
    try {
      const { location } = await schwab(`/trader/v1/accounts/${encodeURIComponent(entry.accountHash)}/orders`, { method: "POST", body: entry.order });
      const orderId = location ? location.split("/").pop() : null;
      const result = { status: "submitted", previewId, orderId };
      idempotency.set(idempotencyKey, result);
      record({ event: "submitted", previewId, orderId });
      entry.orderId = orderId;
      return result;
    } catch (error) {
      const result = { status: "error", previewId, message: error.message };
      idempotency.set(idempotencyKey, result);
      record({ event: "submit_error", previewId, message: error.message });
      throw error;
    }
  }

  function refused(previewId, message) {
    record({ event: "refused", previewId, message });
    return new LiveError(403, message);
  }

  async function orderStatus(previewId) {
    const entry = previews.get(previewId);
    if (!entry?.orderId) throw new LiveError(404, "No submitted order for this preview.");
    const { json } = await schwab(`/trader/v1/accounts/${encodeURIComponent(entry.accountHash)}/orders/${encodeURIComponent(entry.orderId)}`);
    const status = { orderId: entry.orderId, status: json?.status, filledQuantity: json?.filledQuantity,
      remainingQuantity: json?.remainingQuantity, enteredTime: json?.enteredTime, closeTime: json?.closeTime,
      executions: (json?.orderActivityCollection ?? []).flatMap((a) => a.executionLegs ?? [])
        .map((l) => ({ quantity: l.quantity, price: l.price, time: l.time })) };
    record({ event: "status", previewId, ...status });
    return status;
  }

  function journalTail(limit = 50) {
    if (!existsSync(journal)) return [];
    return readFileSync(journal, "utf8").trim().split("\n").filter(Boolean).slice(-limit).reverse().map((l) => JSON.parse(l));
  }

  function setEnabled(enabled, confirmText) {
    if (enabled && !envAllowed()) throw new LiveError(403, "Start the dashboard with LIVE_TRADING_ALLOWED=1 to allow live trading.");
    if (enabled && confirmText !== "ENABLE LIVE TRADING") throw new LiveError(400, "Type ENABLE LIVE TRADING to turn it on.");
    const value = { enabled: Boolean(enabled), changedAt: new Date(now()).toISOString(), reason: "user toggle" };
    writeSettings(value);
    record({ event: enabled ? "enabled" : "disabled" });
    return { ...value, envAllowed: envAllowed() };
  }

  function mount(app) {
    const wrap = (handler) => async (request, response) => {
      try {
        response.json(await handler(request));
      } catch (error) {
        const status = error instanceof LiveError ? error.status : 500;
        if (!(error instanceof LiveError)) log.error?.("Live trading error", { error: error.message });
        response.status(status).json({ error: error instanceof LiveError ? error.message : "Live trading request failed." });
      }
    };
    app.get("/api/live/status", wrap(async () => ({ envAllowed: envAllowed(), ...settings(),
      liveTradingEnabled: liveEnabled(), tokenFile: existsSync(tokenPath), maxOrderUsd: MAX_NOTIONAL,
      shortSellingAllowed: shortsAllowed(), optionsTicket: true })));
    app.get("/api/live/accounts", wrap(async () => ({ accounts: await accounts() })));
    app.get("/api/live/orders", wrap(async () => ({ journal: journalTail() })));
    app.post("/api/live/toggle", wrap(async (request) => setEnabled(request.body?.enabled === true, request.body?.confirmText)));
    app.post("/api/live/preview", wrap(async (request) => preview(request.body ?? {})));
    app.post("/api/live/options/preview", wrap(async (request) => previewOption(request.body ?? {})));
    app.post("/api/live/confirm", wrap(async (request) => confirm(request.body ?? {})));
    app.post("/api/live/order-status", wrap(async (request) => orderStatus(request.body?.previewId)));
  }

  return { mount, preview, previewOption, confirm, orderStatus, setEnabled, settings, journalTail, _previews: previews };
}

export class LiveError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}
