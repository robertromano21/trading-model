// Options UI: chain view + strategy builder (ticker page), options paper book / vol sleeve / research (paper2), and the
// manual live options + short-sale ticket. Live tickets go ticket -> Schwab preview -> typed CONFIRM (plus an extra phrase
// for undefined-risk shorts and short sales) through /api/live/*; nothing here sends an order without that.
const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => { const n = Object.assign(document.createElement(tag), props); n.append(...kids.filter((k) => k !== null && k !== undefined)); return n; };
const money = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(Number(x)) ? "–" : `${Number(x) < 0 ? "-" : ""}$${Math.abs(Number(x)).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })}`);
const num = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(Number(x)) ? "–" : Number(x).toFixed(d));
const pct = (x, d = 1) => (x === null || x === undefined || !Number.isFinite(Number(x)) ? "–" : `${(Number(x) * 100).toFixed(d)}%`);

async function call(url, body) {
  const response = await fetch(url, body === undefined ? { cache: "no-store" } : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function table(headers, rows) {
  const t = el("table", { className: "opt-table" });
  t.append(el("thead", {}, el("tr", {}, ...headers.map((h) => el("th", { textContent: h })))));
  const body = el("tbody");
  for (const r of rows) body.append(el("tr", {}, ...r.map((c) => (c instanceof Node ? el("td", {}, c) : el("td", { textContent: String(c ?? "–") })))));
  t.append(body);
  return t;
}

// ------------------------------------------------------------------------------------------------ payoff chart
export function drawPayoff(canvas, curve, { spot, breakevens = [] } = {}) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, pad = 34;
  ctx.clearRect(0, 0, w, h);
  if (!curve?.length) return;
  const xs = curve.map((c) => c[0]), ys = curve.map((c) => c[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(0, ...ys), y1 = Math.max(0, ...ys);
  const X = (x) => pad + (x - x0) / (x1 - x0 || 1) * (w - 2 * pad);
  const Y = (y) => h - pad - (y - y0) / (y1 - y0 || 1) * (h - 2 * pad);
  ctx.strokeStyle = "#888"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, Y(0)); ctx.lineTo(w - pad, Y(0)); ctx.stroke();
  for (let i = 1; i < curve.length; i++) {
    ctx.strokeStyle = (ys[i] + ys[i - 1]) / 2 >= 0 ? "#2e9d57" : "#c8463c"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X(xs[i - 1]), Y(ys[i - 1])); ctx.lineTo(X(xs[i]), Y(ys[i])); ctx.stroke();
  }
  ctx.fillStyle = "#666"; ctx.font = "11px sans-serif";
  ctx.fillText(money(y1, 0), 2, Y(y1) + 10); ctx.fillText(money(y0, 0), 2, Y(y0));
  ctx.fillText(num(x0, 0), pad, h - 8); ctx.fillText(num(x1, 0), w - pad - 24, h - 8);
  if (spot) { ctx.strokeStyle = "#3b6fd8"; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(X(spot), pad / 2); ctx.lineTo(X(spot), h - pad); ctx.stroke(); ctx.setLineDash([]); ctx.fillText(`spot ${num(spot)}`, X(spot) + 3, pad / 2 + 8); }
  for (const b of breakevens || []) { ctx.fillStyle = "#a36d00"; ctx.fillRect(X(b) - 1, Y(0) - 5, 2, 10); ctx.fillText(`BE ${num(b)}`, X(b) + 3, Y(0) - 6); }
}

// ------------------------------------------------------------------------------------------------ chain + builder (ticker page)
const STRATEGIES = ["long_call", "long_put", "bull_call_spread", "bear_put_spread", "bull_put_spread", "bear_call_spread", "covered_call",
  "cash_secured_put", "collar", "protective_put", "straddle", "strangle", "iron_condor", "iron_butterfly", "call_butterfly", "put_butterfly",
  "calendar", "diagonal", "short_straddle", "short_strangle", "naked_put", "naked_call"];

function mountTickerOptions(symbol) {
  const host = el("section", { className: "box options-box", id: "options-box" });
  host.append(el("h2", { textContent: "Options" }),
    el("p", { className: "muted small", textContent: "Chain: live Schwab market data when authorized, else the latest local snapshot, else MODELED (Black-Scholes) prices. Builder values are model estimates." }));
  const status = el("p", { className: "muted small" });
  const expSel = el("select", { id: "opt-exp" });
  const chainWrap = el("div", { className: "opt-chain" });
  const reload = el("button", { type: "button", textContent: "Load chain" });
  host.append(el("div", { className: "opt-row" }, reload, el("label", {}, "Expiration ", expSel), status), chainWrap);
  // builder
  const strat = el("select", { id: "opt-strategy" }, ...STRATEGIES.map((s) => el("option", { value: s, textContent: s.replaceAll("_", " ") })));
  strat.value = "bear_put_spread";
  const bExp = el("input", { id: "opt-b-exp", type: "date" });
  const bBack = el("input", { id: "opt-b-back", type: "date", title: "Back expiration (calendar / diagonal)" });
  const strikes = el("input", { id: "opt-b-strikes", placeholder: "strikes e.g. k1=95,k2=100 (blank = auto)" });
  const qty = el("input", { id: "opt-b-qty", type: "number", min: 1, max: 100, value: 1, style: "width:4em" });
  const analyze = el("button", { type: "button", textContent: "Analyze" });
  const toTicket = el("button", { type: "button", textContent: "Live ticket…", title: "Open a manual Schwab option ticket with these legs" });
  const canvas = el("canvas", { width: 640, height: 220, className: "opt-payoff" });
  const out = el("div", { className: "opt-metrics" });
  host.append(el("h3", { textContent: "Strategy builder" }),
    el("div", { className: "opt-row" }, strat, el("label", {}, "Exp ", bExp), el("label", {}, "Back ", bBack), strikes, el("label", {}, "Qty ", qty), analyze, toTicket),
    canvas, out);
  const anchor = document.querySelector(".layout") || document.querySelector("main");
  anchor.after(host);
  let chain = null, lastLegs = null;

  function renderChain() {
    chainWrap.replaceChildren();
    if (!chain) return;
    const exp = expSel.value;
    const rows = {};
    for (const c of chain.contracts.filter((c) => c.expiration === exp)) (rows[c.strike] ||= {})[c.kind] = c;
    const spot = chain.underlyingPrice;
    const body = Object.keys(rows).map(Number).sort((a, b) => a - b).map((k) => {
      const c = rows[k].call || {}, p = rows[k].put || {};
      return [num(c.bid), num(c.ask), num(c.delta), pct(c.iv), el("strong", { textContent: num(k), className: spot && Math.abs(k - spot) / spot < 0.01 ? "atm" : "" }), pct(p.iv), num(p.delta), num(p.bid), num(p.ask)];
    });
    chainWrap.append(table(["Call bid", "Call ask", "Δ", "IV", "Strike", "IV", "Δ", "Put bid", "Put ask"], body));
  }
  async function loadChain() {
    status.textContent = "Loading…";
    try {
      chain = await call(`/api/options/chain?symbol=${encodeURIComponent(symbol)}&strikes=10&expirations=6`);
      expSel.replaceChildren(...chain.expirations.map((e) => el("option", { value: e, textContent: e })));
      const src = chain.source === "schwab" ? "LIVE Schwab chain" : chain.source === "snapshot" ? `local snapshot ${chain.asof || ""}` : "MODELED chain (no live data)";
      status.textContent = `${src} · underlying ${money(chain.underlyingPrice)}`;
      status.className = chain.modeled ? "warn small" : "muted small";
      if (!bExp.value && chain.expirations[1]) bExp.value = chain.expirations[1];
      if (!bBack.value && chain.expirations[3]) bBack.value = chain.expirations[3];
      renderChain();
    } catch (error) { status.textContent = `Chain unavailable: ${error.message}`; }
  }
  function strikeMap() {
    const m = {};
    for (const part of strikes.value.split(",").map((s) => s.trim()).filter(Boolean)) {
      const [k, v] = part.split("=").map((s) => s.trim());
      if (/^(k|k[1-4]|put|call)$/.test(k) && Number(v) > 0) m[k] = Number(v);
    }
    return m;
  }
  async function runAnalyze() {
    out.textContent = "Analyzing…";
    try {
      const r = await call("/api/options/analyze", { symbol, ...(chain?.underlyingPrice ? { spot: chain.underlyingPrice } : {}), strategy: strat.value,
        expiration: bExp.value, back_expiration: bBack.value || undefined, strikes: strikeMap(), qty: Number(qty.value) || 1 });
      lastLegs = r.legs;
      drawPayoff(canvas, r.curve, { spot: r.spot, breakevens: r.breakevens });
      out.replaceChildren(
        table(["Leg", "Qty", "Strike", "Expiration", "Model fill"], r.legs.map((l) => [l.kind, l.qty, l.strike ?? "–", l.expiration ?? "–", num(l.price)])),
        el("p", {}, `Max loss ${r.unboundedLoss ? "UNLIMITED" : money(r.maxLoss)} · max gain ${r.maxGain === null ? "UNLIMITED" : money(r.maxGain)} · breakevens ${(r.breakevens || []).map((b) => num(b)).join(", ") || "–"}`
          + ` · POP ${pct(r.probabilityOfProfit)} · est. fees ${money(r.estimatedFees)}${r.approximate ? " · (multi-expiry: approximate)" : ""}`),
        el("p", {}, `Greeks (position): Δ ${num(r.greeks?.delta)} · Γ ${num(r.greeks?.gamma, 4)} · vega ${num(r.greeks?.vega)} /vol pt · θ ${num(r.greeks?.theta)} /day`),
        el("p", { className: "muted small", textContent: `${r.note} Vol used ${pct(r.vol)}.` }));
    } catch (error) { out.textContent = `Analysis failed: ${error.message}`; }
  }
  reload.addEventListener("click", loadChain);
  expSel.addEventListener("change", renderChain);
  analyze.addEventListener("click", runAnalyze);
  toTicket.addEventListener("click", () => openOptionTicket(symbol, lastLegs));
  loadChain();
}

// ------------------------------------------------------------------------------------------------ live option / short ticket
let optPreview = null, optSubmitting = false;
function ticketDialog() {
  if ($("opt-ticket")) return $("opt-ticket");
  const d = el("dialog", { id: "opt-ticket", className: "opt-ticket" });
  const legRows = el("tbody", { id: "ot-legs" });
  d.append(el("form", { method: "dialog" },
    el("h3", { textContent: "Options / short-sale ticket (Schwab, manual)" }),
    el("p", { className: "muted small", textContent: "Limit / net-price orders only. Preview first; Confirm needs the typed word CONFIRM (and an extra phrase for undefined-risk shorts and short sales). $25,000 per-order cap on max loss (or the larger of notional and stress loss when risk is undefined)." }),
    el("label", {}, "Mode ", el("select", { id: "ot-mode" }, el("option", { value: "option", textContent: "Options" }), el("option", { value: "short", textContent: "Stock: SELL SHORT / BUY TO COVER" }))),
    el("label", {}, "Underlying ", el("input", { id: "ot-underlying", required: true, autocomplete: "off", style: "width:6em" })),
    el("div", { id: "ot-option-fields" },
      el("table", { className: "opt-table" }, el("thead", {}, el("tr", {}, ...["Instruction", "Type", "Strike", "Expiration", "Qty", ""].map((h) => el("th", { textContent: h })))), legRows),
      el("button", { type: "button", id: "ot-add", textContent: "+ leg" }), " ",
      el("button", { type: "button", id: "ot-add-stock", textContent: "+ stock leg" }),
      el("label", {}, " Order type ", el("select", { id: "ot-type" }, ...["LIMIT", "NET_DEBIT", "NET_CREDIT", "NET_ZERO"].map((v) => el("option", { value: v, textContent: v })))),
      el("label", {}, " Strategy type ", el("select", { id: "ot-complex" }, ...["CUSTOM", "VERTICAL", "COVERED", "STRADDLE", "STRANGLE", "IRON_CONDOR", "BUTTERFLY", "CONDOR", "CALENDAR", "DIAGONAL", "COLLAR_WITH_STOCK", "COLLAR_SYNTHETIC", "BACK_RATIO", "VERTICAL_ROLL"].map((v) => el("option", { value: v, textContent: v }))))),
    el("div", { id: "ot-short-fields", hidden: true },
      el("label", {}, "Side ", el("select", { id: "ot-short-side" }, el("option", { value: "SELL_SHORT", textContent: "SELL_SHORT" }), el("option", { value: "BUY_TO_COVER", textContent: "BUY_TO_COVER" }))),
      el("label", {}, " Shares ", el("input", { id: "ot-shares", type: "number", min: 1, step: 1, style: "width:6em" }))),
    el("label", {}, "Limit / net price ", el("input", { id: "ot-price", type: "number", min: 0, step: 0.01, style: "width:6em" })),
    el("label", {}, " Time in force ", el("select", { id: "ot-tif" }, el("option", { value: "DAY", textContent: "DAY" }), el("option", { value: "GOOD_TILL_CANCEL", textContent: "GTC" }))),
    el("label", {}, " Account ", el("select", { id: "ot-account" })),
    el("div", { id: "ot-preview", hidden: true }, el("pre", { id: "ot-preview-text" }), el("canvas", { id: "ot-payoff", width: 560, height: 180 }),
      el("label", {}, "Type CONFIRM ", el("input", { id: "ot-typed", autocomplete: "off" })),
      el("label", { id: "ot-extra-row", hidden: true }, " Extra confirmation: type ", el("code", { id: "ot-extra-phrase" }), " ", el("input", { id: "ot-extra", autocomplete: "off", size: 40 }))),
    el("p", { id: "ot-error", className: "error", hidden: true }),
    el("div", { className: "buttons" },
      el("button", { type: "button", id: "ot-preview-btn", textContent: "Preview" }),
      el("button", { type: "button", id: "ot-confirm", className: "danger", disabled: true, textContent: "Confirm & send to Schwab" }),
      el("button", { type: "button", id: "ot-cancel", textContent: "Cancel" }))));
  document.body.append(d);
  const err = (m) => { $("ot-error").textContent = m || ""; $("ot-error").hidden = !m; };
  $("ot-mode").addEventListener("change", () => { const s = $("ot-mode").value === "short"; $("ot-short-fields").hidden = !s; $("ot-option-fields").hidden = s; resetPreview(); });
  $("ot-add").addEventListener("click", () => addLeg({}));
  $("ot-add-stock").addEventListener("click", () => addLeg({ kind: "stock", instruction: "BUY", qty: 100 }));
  $("ot-cancel").addEventListener("click", () => d.close());
  function resetPreview() { optPreview = null; $("ot-preview").hidden = true; $("ot-confirm").disabled = true; err(""); }
  d.addEventListener("input", (e) => { if (!["ot-typed", "ot-extra"].includes(e.target.id)) resetPreview(); });
  $("ot-preview-btn").addEventListener("click", async () => {
    err(""); $("ot-confirm").disabled = true;
    try {
      const base = { accountHash: $("ot-account").value, duration: $("ot-tif").value };
      if ($("ot-mode").value === "short") {
        optPreview = await call("/api/live/preview", { ...base, symbol: $("ot-underlying").value, side: $("ot-short-side").value, orderType: "LIMIT",
          limitPrice: $("ot-price").value, quantity: $("ot-shares").value });
        const p = optPreview;
        $("ot-preview-text").textContent = [`${p.ticket.side} ${p.ticket.quantity} ${p.ticket.symbol} LIMIT @ ${money(p.ticket.limitPrice)} ${p.ticket.duration}, account ${p.ticket.account}`,
          `Quote: last ${money(p.quote.last)} bid ${money(p.quote.bid)} ask ${money(p.quote.ask)} · value ${money(p.estimatedNotional)} · reg. fees ${money(p.estimatedFees.regulatory)}`,
          ...(p.shortSale?.warnings || []).map((w) => `! ${w}`),
          `Schwab preview: ${p.schwabPreview?.unavailable ? `unavailable (${p.schwabPreview.unavailable})` : "accepted"}`,
          p.liveTradingEnabled ? "Type CONFIRM (and the extra phrase) within 2 minutes to send." : "Live trading is OFF: this order cannot be sent."].join("\n");
        $("ot-payoff").hidden = true;
      } else {
        const legs = [...$("ot-legs").rows].map((r) => {
          const v = (n) => r.querySelector(`[data-f=${n}]`)?.value;
          return v("kind") === "stock" ? { equity: true, instruction: v("instruction"), quantity: Number(v("qty")) }
            : { instruction: v("instruction"), kind: v("kind"), strike: Number(v("strike")), expiration: v("expiration"), quantity: Number(v("qty")) };
        });
        optPreview = await call("/api/live/options/preview", { ...base, underlying: $("ot-underlying").value, legs, orderType: $("ot-type").value,
          price: $("ot-price").value === "" ? undefined : Number($("ot-price").value), complexType: $("ot-complex").value });
        const p = optPreview, r = p.risk;
        $("ot-preview-text").textContent = [
          `${p.ticket.orderType} ${p.ticket.price} ${p.ticket.duration} ${p.ticket.complexType}, account ${p.ticket.account} (${p.accountType || "?"}) · ${p.ticket.underlying} ${money(p.underlyingPrice)}`,
          ...p.legs.map((l) => `  ${l.instruction} ${l.quantity} ${l.symbol}${l.equity ? "" : `  bid ${money(l.bid)} ask ${money(l.ask)}${l.iv ? ` IV ${pct(l.iv)}` : ""}`}`),
          `Max loss ${r.unboundedLoss ? "UNLIMITED" : money(r.maxLoss)} · max gain ${r.maxGain === null ? "UNLIMITED" : money(r.maxGain)} · breakevens ${(r.breakevens || []).map((b) => num(b)).join(", ") || "–"}`,
          `Cap check: ${money(r.capAmount)} (${r.capBasis}) vs ${money(r.maxOrderUsd)} cap`,
          `Greeks: Δ ${num(p.greeks.delta)} Γ ${num(p.greeks.gamma, 4)} vega ${num(p.greeks.vega)} θ ${num(p.greeks.theta)}`,
          `Fees (est.): ${money(p.estimatedFees.commissionAndRegulatory)} — ${p.estimatedFees.note}`,
          ...(p.warnings || []).map((w) => `! ${w}`),
          `Schwab preview: ${p.schwabPreview?.unavailable ? `unavailable (${p.schwabPreview.unavailable})` : "accepted"}`,
          p.liveTradingEnabled ? "Type CONFIRM (and any extra phrase) within 2 minutes to send." : "Live trading is OFF: this order cannot be sent."].join("\n");
        $("ot-payoff").hidden = false;
        drawPayoff($("ot-payoff"), r.payoff, { spot: p.underlyingPrice, breakevens: r.breakevens });
      }
      const extra = optPreview.extraPhrase || optPreview.shortSale?.extraPhrase || null;
      $("ot-extra-row").hidden = !extra; $("ot-extra-phrase").textContent = extra || ""; $("ot-typed").value = ""; $("ot-extra").value = "";
      $("ot-preview").hidden = false;
      $("ot-confirm").disabled = !optPreview.liveTradingEnabled;
      const mine = optPreview;
      setTimeout(() => { if (optPreview === mine) { $("ot-confirm").disabled = true; $("ot-preview-text").textContent += "\nPreview expired."; } }, 115_000);
    } catch (error) { err(error.message); }
  });
  $("ot-confirm").addEventListener("click", async () => {
    if (!optPreview || optSubmitting) return;
    if ($("ot-typed").value.trim() !== "CONFIRM") return err("Type CONFIRM to send this order.");
    optSubmitting = true; $("ot-confirm").disabled = true;
    try {
      const result = await call("/api/live/confirm", { previewId: optPreview.previewId, idempotencyKey: crypto.randomUUID(),
        typedConfirm: $("ot-typed").value.trim(), extraConfirm: $("ot-extra").value.trim() || undefined });
      $("ot-preview-text").textContent += `\nSent. Schwab order id ${result.orderId ?? "(pending)"} · status ${result.status ?? "submitted"}`;
      optPreview = null;
    } catch (error) { err(error.message); $("ot-confirm").disabled = false; } finally { optSubmitting = false; }
  });
  return d;
}

function addLeg({ instruction = "BUY_TO_OPEN", kind = "put", strike = "", expiration = "", qty = 1 }) {
  const row = $("ot-legs").insertRow();
  const stock = kind === "stock";
  const sel = (f, opts, v) => { const s = el("select", { dataset: { f } }, ...opts.map((o) => el("option", { value: o, textContent: o }))); s.dataset.f = f; s.value = v; return s; };
  const inp = (f, props) => { const i = el("input", props); i.dataset.f = f; return i; };
  row.append(el("td", {}, sel("instruction", stock ? ["BUY", "SELL"] : ["BUY_TO_OPEN", "SELL_TO_OPEN", "BUY_TO_CLOSE", "SELL_TO_CLOSE"], instruction)),
    el("td", {}, stock ? inp("kind", { type: "hidden", value: "stock" }) : sel("kind", ["call", "put"], kind), stock ? "stock" : ""),
    el("td", {}, stock ? "" : inp("strike", { type: "number", step: 0.5, value: strike, style: "width:6em" })),
    el("td", {}, stock ? "" : inp("expiration", { type: "date", value: expiration })),
    el("td", {}, inp("qty", { type: "number", min: 1, step: 1, value: qty, style: "width:5em" })),
    el("td", {}, el("button", { type: "button", textContent: "×", onclick: () => row.remove() })));
}

export async function openOptionTicket(symbol, legs) {
  const d = ticketDialog();
  $("ot-underlying").value = symbol || "";
  $("ot-legs").replaceChildren();
  for (const l of legs || [{}]) {
    if (l.kind === "stock") addLeg({ kind: "stock", instruction: l.qty > 0 ? "BUY" : "SELL", qty: Math.abs(l.qty) });
    else addLeg({ instruction: (l.qty ?? 1) > 0 ? "BUY_TO_OPEN" : "SELL_TO_OPEN", kind: l.kind || "put", strike: l.strike ?? "", expiration: l.expiration ?? "", qty: Math.abs(l.qty ?? 1) });
  }
  const optLegs = (legs || []).filter((l) => l.kind !== "stock");
  $("ot-type").value = (legs || []).length > 1 ? "NET_DEBIT" : "LIMIT";
  if (legs?.length) {
    const net = legs.reduce((s, l) => s + (l.kind === "stock" ? 0 : (l.qty > 0 ? 1 : -1) * (l.price || 0) * Math.abs(l.qty) / Math.max(1, Math.abs(optLegs[0]?.qty || 1))), 0);
    if (legs.length > 1) $("ot-type").value = net >= 0 ? "NET_DEBIT" : "NET_CREDIT";
    $("ot-price").value = Math.abs(net).toFixed(2);
  }
  $("ot-preview").hidden = true; $("ot-confirm").disabled = true; $("ot-error").hidden = true;
  if (!$("ot-account").options.length) {
    try {
      const { accounts } = await call("/api/live/accounts");
      $("ot-account").replaceChildren(...accounts.map((a) => el("option", { value: a.hash, textContent: a.label })));
    } catch (error) { $("ot-error").textContent = `Accounts unavailable: ${error.message}`; $("ot-error").hidden = false; }
  }
  d.showModal();
}

// ------------------------------------------------------------------------------------------------ paper2 panels
async function mountPaperOptions() {
  const host = el("section", { id: "options-paper" }, el("h2", { textContent: "Options, downside & volatility sleeve (paper)" }),
    el("p", { className: "muted", textContent: "VIXY when VIX is calm (low percentile, steep contango); SVIX after a spike fades; every exit has an explicit reason (also in the decision chat). Option prices are MODELED unless a Schwab chain snapshot exists." }));
  const body = el("div", { className: "muted", textContent: "Loading…" });
  const run = el("button", { type: "button", textContent: "Run options paper now" });
  const ticket = el("button", { type: "button", textContent: "Live options / short ticket…" });
  host.append(el("div", { className: "opt-row" }, run, ticket), body);
  const anchor = $("swarm") || document.querySelector("main section.live") || document.querySelector("main");
  anchor.before(host);
  run.addEventListener("click", async () => { try { await call("/api/options/paper/run", {}); run.textContent = "Running… (refresh in a minute)"; } catch (e) { run.textContent = e.message; } });
  ticket.addEventListener("click", () => openOptionTicket("SPY"));
  try {
    const [paper, research] = await Promise.all([call("/api/options/paper"), call("/api/options/research")]);
    body.replaceChildren();
    body.className = "";
    const vs = paper.volSleeve;
    if (vs) {
      const v = vs.vix || {};
      body.append(el("h3", { textContent: `Vol sleeve · equity ${money(vs.equity)} · as of ${vs.asof}` }),
        el("p", { className: vs.vixStale ? "warn" : "muted", textContent: `VIX ${num(v.vix)} (${v.date}) · VIX/VIX3M ${num(v.ratio, 3)} ${v.ratio > 1 ? "(backwardation)" : "(contango)"} · 252d pct ${pct(v.pct252 ?? v.pct)} · z60 ${num(v.z60 ?? v.z)}${vs.vixStale ? " · VIX data is stale: refresh FRED" : ""}` }),
        table(["Position", "Shares", "Entry", "Since", "Held", "Hard stop"], Object.entries(vs.positions || {}).map(([t, p]) => [t, p.shares, money(p.entry), p.entryDate, p.held, money(vs.hardStops?.[t])])),
        table(["Pending (next open)", "Ticker", "Reason"], (vs.pending || []).map((o) => [o.action, o.ticker, o.reason])),
        table(["Ticker", "Entry", "Entry reason", "Exit", "Exit reason", "P&L"], (vs.trades || []).slice(-15).reverse().map((t) => [t.ticker, `${t.entryDate} ${money(t.entry)}`, t.entryReason, t.exitDate ? `${t.exitDate} ${money(t.exit)}` : "open", t.exitReason, money(t.pnl)])));
    } else body.append(el("p", { className: "muted", textContent: "No vol-sleeve paper state yet (run options_paper.py run)." }));
    const book = paper.book;
    if (book?.positions?.length) {
      body.append(el("h3", { textContent: "Options paper positions" }),
        table(["Id", "Strategy", "Underlying", "Legs", "Opened", "Entry cash", "Pricing"], book.positions.map((p) => [p.id, p.strategy, p.underlying,
          p.legs.map((l) => `${l.qty > 0 ? "+" : ""}${l.qty} ${l.kind}${l.strike ? ` ${l.strike}` : ""}${l.expiration ? ` ${l.expiration}` : ""}`).join(", "), p.opened, money(p.entryCash), p.pricing])));
    }
    if (paper.overlay) body.append(el("p", { textContent: `Options overlay (config-gated) since ${paper.overlay.start}: P&L ${money(paper.overlay.overlayPnl)} · ${paper.overlay.pricing}` }));
    const wf = research.volSleeve?.walkForward;
    if (wf?.outOfSample) body.append(el("h3", { textContent: "Vol sleeve walk-forward (out of sample)" }),
      el("p", { textContent: `From ${wf.oosStart}: return ${pct(wf.outOfSample.totalReturn, 2)}, Sharpe ${num(wf.outOfSample.sharpe)}, max DD ${pct(wf.outOfSample.maxDrawdown)}, ${wf.trades} trades · p(mean>0) ${num(research.volSleeve.pOosMeanReturnGt0)} · p(full exits beat time-only) ${num(research.volSleeve.pFullExitsBeatTimeOnly)}` }),
      el("p", { className: "muted small", textContent: research.volSleeve.honesty || "" }));
    if (research.overlays?.results?.length) body.append(el("h3", { textContent: "Options / downside overlays (walk-forward vs core TSMOM, MODELED options)" }),
      table(["Overlay", "Sizing", "Overlay Sharpe", "Base Sharpe", "Overlay CAGR", "Max DD", "p", "Holm p"], research.overlays.results.map((r) => [r.mode, r.sizing.split(" (")[0],
        num(r.overlay?.sharpe), num(r.base?.sharpe), pct(r.overlay?.cagr), pct(r.overlay?.maxDrawdown), num(r.p), num(r.holmP)])),
      el("p", { className: "muted small", textContent: research.overlays.honesty || "" }));
    const td = research.topDetector;
    if (td) body.append(el("h3", { textContent: "Top / adverse-regime detector" }), el("p", { textContent: td.verdict }),
      el("p", { className: "muted small", textContent: `Latest ${td.latest?.date}: ${td.latest?.count ?? "–"} signals on · HMM risk prob ${num(td.latest?.hmm_prob)} · base rates ${Object.entries(td.baseRates || {}).map(([k, v]) => `${k} ${pct(v)}`).join(", ")}` }));
  } catch (error) { body.textContent = `Options panel unavailable: ${error.message}`; }
}

// ------------------------------------------------------------------------------------------------ boot
function boot() {
  const path = location.pathname;
  if (/ticker/.test(path)) {
    const symbol = (new URLSearchParams(location.search).get("symbol") || "SPY").toUpperCase().trim();
    if (/^[A-Z][A-Z0-9.\-]{0,9}$/.test(symbol)) mountTickerOptions(symbol);
  }
  if (/paper2/.test(path)) mountPaperOptions();
  const liveNew = $("live-new");
  if (liveNew && !$("live-new-options")) liveNew.after(" ", el("button", { type: "button", id: "live-new-options", textContent: "New options / short ticket", onclick: () => openOptionTicket("") }));
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
window.openOptionTicket = openOptionTicket;
