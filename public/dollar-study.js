const esc = (value) => String(value ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (value) => value == null ? "—" : `${value > 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
const num = (value) => value == null ? "—" : Number(value).toFixed(2);
const res = (point) => point?.n ? `${pct(point.median)} (${point.n})` : "—";
const table = (headers, rows) => `<table><thead><tr>${headers.map((v) => `<th>${esc(v)}</th>`).join("")}</tr></thead>
  <tbody>${rows.map((row) => `<tr>${row.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;

function drawStudy(root, data) {
  const usd = data.dollar;
  const micro = data.intraday;
  const forecast = data.tradeDollarOutlook;
  const assets = data.assets.map((a) => [`${a.label} (${a.symbol})`, num(a.monthCorrelation), num(a.recentMonthCorrelation),
    res(a.atPriceTurns.peak.recent), res(a.atPriceTurns.valley.recent), res(a.afterDollarTurns.peak), res(a.afterDollarTurns.valley)]);
  const trade = data.trade.map((t) => [t.label, t.quarter, pct(t.volumeYoY), pct(t.dollarYoY),
    num(t.correlation), num(t.nextQuarterCorrelation), t.n]);
  const events = Object.entries(micro?.events ?? {}).flatMap(([coin, turns]) => Object.entries(turns).map(([kind, t]) =>
    [coin, kind, t.events, t.inverseMatches, t.sameMatches, res(t.nextHour), res(t.nextHourNetLong)]));
  const replay = micro?.leaderboard?.rows ?? [];
  root.innerHTML = `
    <p>Broad U.S. dollar index ${num(usd.level)} on ${esc(usd.asOf)}. Change over one week ${pct(usd.week)}, one month
      ${pct(usd.month)}, three months ${pct(usd.quarter)}, and one year ${pct(usd.year)}.</p>
    <p class="muted">The market table compares 21-session price changes since 2015. Peak and valley columns show
      median dollar growth at nominal asset price turns over the past year. Forward returns start after a dollar turn was confirmable.</p>
    ${table(["Asset", "Monthly corr.", "Last-year corr.", "USD YoY at price peaks", "USD YoY at price valleys",
      "Asset 14d after USD peak", "Asset 14d after USD valley"], assets)}
    <h3>Real U.S. imports and exports</h3>
    ${trade.length ? table(["Series", "Quarter start", "Real growth YoY", "USD growth YoY",
      "Same-quarter corr.", "Next-quarter trade corr.", "Quarters"], trade) : "<p>Waiting for the official trade-volume series.</p>"}
    ${forecast ? `<h4>Does published trade data improve the dollar outlook?</h4>
      <p>Experimental ${forecast.horizonSessions}-session model as of ${esc(forecast.asOf)}. Adding imports and exports
        projects ${pct(forecast.tradeMove)} versus ${pct(forecast.baselineMove)} from the dollar-only model.
        ${forecast.beatsBaselines ? "The trade model had lower error than both baselines in this sample." :
          "The trade model did not beat both baselines in this sample."}</p>
      ${table(["Model", "Walk-forward RMSE", "Direction hit rate"], [
        ["Flat dollar", pct(forecast.rmse.flat), "—"],
        ["Dollar history only", pct(forecast.rmse.dollar), pct(forecast.directionHitRate.dollar)],
        ["Dollar history plus import/export releases", pct(forecast.rmse.trade), pct(forecast.directionHitRate.trade)],
      ])}<p class="muted">${forecast.n} quarterly forecast origins. ${forecast.vintages ?
        "Uses release vintages where available." : "Uses latest revisions with conservative release lags."}</p>` : ""}
    <p class="muted"><a href="https://www.ismworld.org/supply-management-news-and-reports/reports/ism-pmi-reports/">ISM Manufacturing New Export Orders</a>
       tracks new orders early in the month. Shipment indicators include
       <a href="https://www.ams.usda.gov/market-news/national-grain-reports">weekly USDA grain inspections</a>,
       <a href="https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?f=W&amp;n=PET&amp;s=WCREXUS2">weekly EIA crude export volumes</a>,
       and <a href="https://portoflosangeles.org/business/statistics/container-statistics">loaded export containers at the Port of Los Angeles</a>.
       Each covers a subset of goods exports.</p>
    ${events.length ? `<details><summary>Intraday test using one-minute UUP data</summary>
      <p>${esc(micro.start.slice(0, 10))} through ${esc(micro.end.slice(0, 10))}, ${micro.overlap} crypto bars with fresh UUP observations.
        Matches count turns within 30 minutes. The forward test measures a one-hour BTC or ETH move after confirming the UUP turn.</p>
      ${table(["Coin", "UUP momentum turn", "Events", "Inverse matches", "Same-direction matches", "Next-hour median", "After costs"],
        events)}
      <h4>40-day replay leaderboard</h4>
      ${table(["Strategy", "Net return", "Max drawdown", "Fills"],
        replay.map((r) => [r.label, pct(r.return), pct(r.maxDrawdown), r.fills]))}
      </details>` : ""}
    <p class="muted">Sources <a href="https://fred.stlouisfed.org/series/DTWEXBGS">Federal Reserve broad dollar</a>,
      <a href="https://fred.stlouisfed.org/series/CBBTCUSD">Coinbase BTC via FRED</a>,
      <a href="https://fred.stlouisfed.org/series/IMPGSC1">BEA real imports</a>, and
      <a href="https://fred.stlouisfed.org/series/EXPGSC1">BEA real exports</a>.</p>`;
}

const root = document.getElementById("dollar-study");
if (root) {
  try {
    const response = await fetch("/api/research/dollar-study", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    drawStudy(root, await response.json());
  } catch (error) {
    root.textContent = `Dollar analysis is unavailable (${error.message}).`;
  }
}
