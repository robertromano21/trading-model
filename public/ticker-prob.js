// Walk-forward calibrated probabilities on the ticker page. Advisory only; not a trade ticket.
async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}
function pct(v) {
  if (v == null || Number.isNaN(Number(v))) return "–";
  return `${(Number(v) * 100).toFixed(0)}%`;
}
function label(key) {
  return ({
    excess_5: "P(5d excess vs SPY > 0)",
    excess_20: "P(20d excess vs SPY > 0)",
    ret_5_0p0: "P(5d return > 0)",
    ret_20_0p0: "P(20d return > 0)",
    spy_up_5: "P(SPY 5d up)",
    spy_up_20: "P(SPY 20d up)",
  })[key] || key;
}
function render(host, doc, symbol) {
  if (!doc || doc.error || doc.available === false) {
    host.classList.add("muted");
    host.textContent = doc?.reason || doc?.error || "No probability study yet. Run python research.py prob.";
    return;
  }
  const rec = doc.tickers?.[symbol] || doc.tickers?.[symbol?.toUpperCase?.()] || null;
  const spy = doc.spy || {};
  const keys = ["excess_5", "excess_20", "ret_5_0p0", "ret_20_0p0"];
  const rows = keys
    .filter((k) => rec?.[k] != null || spy[k] != null || spy[k.replace("excess", "spy_up").replace("ret_5_0p0", "spy_up_5").replace("ret_20_0p0", "spy_up_20")] != null)
    .map((k) => `<dt>${label(k)}</dt><dd>${pct(rec?.[k])}</dd>`)
    .join("");
  const spyKeys = Object.keys(spy).filter((k) => k.startsWith("spy_up") || k.startsWith("excess") || k.startsWith("ret_"));
  const spyRows = (spyKeys.length ? spyKeys : ["excess_5", "excess_20"]).map((k) => `<dt>SPY ${label(k)}</dt><dd>${pct(spy[k])}</dd>`).join("");
  const recs = doc.verdict?.recommendation ? `<p class="small">${doc.verdict.recommendation}</p>` : "";
  host.classList.remove("muted");
  host.innerHTML = `<p class="small muted">As of ${doc.asof || "–"}. Walk-forward, purged and calibrated. Advisory only — the paper trader is still plain trend-following unless you change it.</p>
    ${rec ? `<dl class="kv">${rows}</dl>` : `<p class="muted">No row for ${symbol} in the latest tape (the study universe is the core ETFs plus the watchlist).</p>`}
    <h3 class="small">Market</h3><dl class="kv">${spyRows || "<dt>SPY</dt><dd>–</dd>"}</dl>
    ${recs}`;
}
async function loadTickerProb() {
  const host = document.getElementById("prob-body");
  if (!host) return;
  const symbol = (new URLSearchParams(location.search).get("symbol") || "").toUpperCase();
  try {
    const doc = await getJson("/api/research/probabilities");
    render(host, doc, symbol);
  } catch (error) {
    host.classList.add("muted");
    host.textContent = error.message.includes("404")
      ? "No probability study yet. Run python research.py prob."
      : `Probabilities unavailable: ${error.message}`;
  }
}
loadTickerProb();
