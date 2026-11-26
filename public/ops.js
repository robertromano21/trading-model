// Research & data hub: allow-listed jobs, research report viewer (incl. normality), day-trading research,
// data health (session dates, split repairs, exclusions, dividends, intraday cache), Schwab status, log viewer.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (x, d = 1) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
const num = (x, d = 2) => (x == null || !Number.isFinite(Number(x)) ? "–" : Number(x).toFixed(d));
async function json(url, body) {
  const response = await fetch(url, body === undefined ? { cache: "no-store" } :
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}
function table(el, headers, rows) {
  el.innerHTML = `<thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${
    rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

// ------------------------------------------------------------------ jobs
let watching = null;
async function loadJobs() {
  try {
    const { available, jobs } = await json("/api/ops/jobs");
    if (!$("job-buttons").children.length) {
      for (const j of available) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = `${j.confirm ? "⚠ " : ""}${j.label} (~${j.minutes} min)`;
        b.onclick = async () => {
          if (j.confirm && !confirm(`${j.label}\n\nThis changes saved paper/data state (no real orders). Run it now?`)) return;
          try {
            const job = await json("/api/ops/run", { job: j.name, confirm: j.confirm ? true : undefined });
            watching = job.id;
            loadJobs();
          } catch (error) { alert(error.message); }
        };
        $("job-buttons").appendChild(b);
      }
    }
    table($("job-table"), ["Job", "Status", "Started (ET)", "Finished", "Exit", ""], jobs.map((j) => [esc(j.label), 
      `<span class="${j.status === "complete" ? "ok" : j.status === "running" ? "" : "bad"}">${esc(j.status)}</span>`,
      esc(new Date(j.startedAt).toLocaleString("en-US", { timeZone: "America/New_York" })),
      esc(j.finishedAt ? new Date(j.finishedAt).toLocaleTimeString("en-US", { timeZone: "America/New_York" }) : ""), esc(j.exitCode ?? ""),
      `<button type="button" data-job="${esc(j.id)}">Log</button>`]));
    for (const b of $("job-table").querySelectorAll("button[data-job]")) b.onclick = () => { watching = b.dataset.job; showJob(); };
    if (watching) showJob();
    if (jobs.some((j) => j.status === "running")) setTimeout(loadJobs, 4000);
  } catch (error) {
    $("job-buttons").textContent = error.message;
  }
}
async function showJob() {
  try {
    const j = await json(`/api/ops/jobs/${watching}`);
    $("job-log").hidden = false;
    $("job-log").textContent = `${j.label} — ${j.status}\n\n${j.log}`;
    $("job-log").scrollTop = $("job-log").scrollHeight;
    if (j.status !== "running" && j.name?.startsWith("research")) loadReport();
  } catch { /* gone */ }
}

// ------------------------------------------------------------------ research
function markdown(md) {
  const lines = md.split(/\r?\n/);
  let html = "", i = 0;
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`(.+?)`/g, "<code>$1</code>");
  while (i < lines.length) {
    const line = lines[i];
    if (/^\|/.test(line) && /^\|[\s|:-]+\|$/.test(lines[i + 1] ?? "")) {
      const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      const head = cells(line);
      const id = /h \| Obs/.test(line) ? ' id="normality"' : "";
      html += `<table${id}><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\|/.test(lines[i])) { html += `<tr>${cells(lines[i]).map((c) => `<td>${c}</td>`).join("")}</tr>`; i += 1; }
      html += "</tbody></table>";
      continue;
    }
    const h = /^(#{1,3})\s+(.*)$/.exec(line);
    if (h) {
      const hid = /Return attribution/i.test(h[2]) ? ' id="attribution"' : "";
      html += `<h${h[1].length}${hid}>${inline(h[2])}</h${h[1].length}>`;
    }
    else if (/^- /.test(line)) html += `<li>${inline(line.slice(2))}</li>`;
    else if (line.trim()) html += `<p>${inline(line)}</p>`;
    i += 1;
  }
  return html;
}
async function loadReport() {
  try {
    const response = await fetch("/api/research/report", { cache: "no-store" });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || response.statusText);
    $("report").classList.remove("muted");
    $("report").innerHTML = markdown(await response.text());
    if (location.hash === "#normality") document.getElementById("normality")?.scrollIntoView();
  } catch (error) {
    $("report").textContent = `${error.message} Use “Research: all” above.`;
  }
}
async function loadDaytrade() {
  try {
    const r = await json("/api/daytrade/research");
    const row = (name, x) => x ? [esc(name), esc(x.sessions ?? x.performance?.sessions), esc(x.trades), pct(x.performance?.totalReturn),
      num(x.performance?.sharpe), pct(x.performance?.maxDrawdown), pct(x.winRate, 0), num(x.costPerTradeBps, 1), esc(JSON.stringify(x.skipped ?? {}))] : null;
    table($("dt-table"), ["Run", "Sessions", "Trades", "Return", "Sharpe", "Max DD", "Win", "Cost bps/trade", "Skipped"],
      [row("Defaults, PDT on", r.fullPeriodDefaults), row("Defaults, no PDT", r.fullPeriodDefaultsNoPdt),
        row(`Walk-forward OOS (${r.walkForward?.testSessions?.[0]} → ${r.walkForward?.testSessions?.[1]})`, r.walkForward?.outOfSample)].filter(Boolean));
    $("dt-caveats").innerHTML = (r.caveats ?? []).map((c) => `<li>${esc(c)}</li>`).join("") +
      `<li>Data: ${esc(r.data?.fiveMinuteSessions)} five-minute sessions ${esc(r.data?.first)} → ${esc(r.data?.last)}; generated ${esc(r.generatedAt)}.</li>`;
  } catch (error) {
    $("dt-table").textContent = error.message;
  }
}

// ------------------------------------------------------------------ data health
async function loadHealth(refresh = false) {
  $("health-body").textContent = "Checking…";
  try {
    const h = await json(`/api/data-health${refresh ? "?refresh=1" : ""}`);
    const s = h.sessions ?? {}, a = h.anomalies ?? {}, d = h.dividends ?? {}, f = h.files ?? {}, i = h.intraday ?? {};
    const ok = (b) => (b ? '<span class="ok">OK</span>' : '<span class="bad">CHECK</span>');
    $("health-body").classList.remove("muted");
    $("health-body").innerHTML = `
      <dl class="kv">
        <dt>Daily sessions</dt><dd>${esc(s.completeSessions)} complete, ${esc(s.first)} → ${esc(s.last)} ${ok(s.upToDate)} (expected latest ${esc(s.expectedLatest)})</dd>
        <dt>Missing vs NYSE calendar</dt><dd>${s.missingVsCalendar?.length ? `<span class="bad">${s.missingVsCalendar.map(esc).join(", ")}</span>` : '<span class="ok">none</span>'}</dd>
        <dt>Tickers in latest session</dt><dd>${esc(h.tickersLatestSession)}</dd>
        <dt>Session-date check (newest files)</dt><dd>${(f.checked ?? []).map((c) => `${esc(c.date)} ${c.ok ? '<span class="ok">✓</span>' : `<span class="bad">✗ ${esc(c.error)}</span>`}`).join(" · ") || "no files"}
          ${f.setAside?.length ? `<br>Set aside: ${f.setAside.map(esc).join(", ")}` : ""}</dd>
        <dt>Splits synced</dt><dd>${esc(a.splits?.events)} events, ${esc(a.splits?.tickers)} tickers, latest ${esc(a.splits?.latest)}</dd>
        <dt>Repairs</dt><dd>${esc(a.repairs?.tickers)} tickers repaired (last ${esc(a.repairs?.last)})</dd>
        <dt>Excluded tickers</dt><dd>${esc(a.excludedCount)} with unresolved anomalies (open/persists)</dd>
        <dt>Masked days</dt><dd>${(a.masked ?? []).map((m) => `${esc(m.ticker)} ${esc(m.date)}`).join(", ") || "none"}</dd>
        <dt>Dividends</dt><dd>${d.available ? `${esc(d.first_date)} → ${esc(d.last_date)}, ${esc(d.events)} events (latest ex-date ${esc(d.latestExDate)})` : "not synced"}</dd>
        <dt>Schwab intraday cache</dt><dd>${i.available ? (i.byFrequency ?? []).map((x) => `${x.frequency}-min: ${x.bars.toLocaleString()} bars, ${x.symbols} symbols, ${x.sessions} sessions ${x.first} → ${x.last}`).join("<br>") + `<br>last fetch ${esc(i.lastFetch)}, ${esc(i.quoteRows)} cached quotes` : "empty"}</dd>
        <dt>Output files</dt><dd>${Object.entries(h.outputs ?? {}).map(([k, v]) => `${esc(k)}: ${esc(v ?? "missing")}`).join("<br>")}</dd>
      </dl>
      <h3>Anomalies by kind/status</h3><table id="anom"></table>
      <h3>Excluded tickers (by 60-day dollar volume)</h3><table id="excluded"></table>
      <h3>Largest repairs</h3><table id="repairs"></table>
      <p class="small muted">Checked in ${esc(h.elapsedSeconds)}s at ${esc(h.generatedAt)}.</p>`;
    table($("anom"), ["Kind", "Status", "Records", "Tickers"], (a.byKindStatus ?? []).map((r) => [esc(r.kind), esc(r.status), esc(r.n), esc(r.tickers)]));
    table($("excluded"), ["Ticker", "Anomalies", "Last", "Kinds", "Avg $ volume"], (a.excludedTickers ?? []).slice(0, 60).map((r) =>
      [`<a href="/ticker.html?symbol=${encodeURIComponent(r.ticker)}">${esc(r.ticker)}</a>`, esc(r.anomalies), esc(r.last_date), esc(r.kinds),
        r.adv ? `$${(r.adv / 1e6).toFixed(1)}M` : "–"]));
    table($("repairs"), ["Ticker", "Max close change", "Repaired"], (a.repairs?.largest ?? []).map((r) =>
      [esc(r.ticker), pct(r.max_change, 0), esc(r.repaired_at)]));
    renderSchwab(h.schwabLive, h.schwab);
  } catch (error) {
    $("health-body").textContent = `Data health unavailable: ${error.message}`;
  }
}
function renderSchwab(live, py) {
  if (!live) return;
  $("schwab-body").classList.remove("muted");
  $("schwab-body").innerHTML = `<dl class="kv">
    <dt>Status</dt><dd>${live.usable ? '<span class="ok">usable</span>' : `<span class="bad">${esc(live.reason)}</span>`}</dd>
    <dt>App key / secret</dt><dd>${live.config?.appKeySet ? "saved" : "missing"} / ${live.config?.appSecretSet ? "saved" : "missing"} (callback ${esc(live.config?.callbackUrl)})</dd>
    <dt>Access token</dt><dd>${live.accessTokenValid ? "valid" : "expired"} (expires ${esc(live.accessTokenExpires)})</dd>
    <dt>Refresh token</dt><dd>${live.refreshTokenValid ? "valid" : "expired"} until about ${esc(live.refreshTokenExpiresAbout)} (7-day limit: reauthorize before then)</dd>
    <dt>Rate use</dt><dd>${esc(live.requestsLastMinute)} / ${esc(live.limitPerMinute)} requests in the last minute (dashboard)</dd>
    ${live.lastError ? `<dt>Last error</dt><dd class="bad">${esc(live.lastError.message)} at ${esc(live.lastError.at)}</dd>` : ""}
  </dl>`;
}
async function loadSchwab() {
  try { renderSchwab(await json("/api/schwab/status")); } catch (error) { $("schwab-body").textContent = error.message; }
  try {
    const d = await json("/api/schwab/daily-backfill");
    const s = d.summary || {};
    const cov = s.coverage || {};
    const auth = (s.auth && s.auth.message) || (d.token && (d.token.reason || (!d.token.usable && "Schwab is not authorized yet."))) || "";
    const range = cov.rows ? `${cov.tickers ?? "?"} tickers, ${cov.rows.toLocaleString()} rows, ${cov.first ?? "?"} → ${cov.last ?? "?"}` : "no long daily history stored yet";
    const st = d.running ? "running" : (s.status || "not started");
    $("schwab-daily").classList.remove("muted");
    $("schwab-daily").innerHTML = `<strong>20-year daily backfill:</strong> ${esc(st)}. ${esc(range)}. `
      + `Schwab candles are split-adjusted, not dividend-adjusted; Massive grouped bars are preferred on overlap. `
      + `Schwab only has currently listed symbols (survivorship bias). ${esc(auth || s.message || "")}`;
    if (d.running) setTimeout(loadSchwab, 8000);
  } catch (error) {
    $("schwab-daily").textContent = `Long daily backfill status unavailable: ${error.message}`;
  }
}

// ------------------------------------------------------------------ logs
let follow = null;
async function loadLogs() {
  const p = new URLSearchParams({ stream: $("log-stream").value, limit: "500" });
  for (const [k, id] of [["date", "log-date"], ["level", "log-level"], ["source", "log-source"], ["requestId", "log-request"], ["q", "log-q"]]) {
    if ($(id).value.trim()) p.set(k, $(id).value.trim());
  }
  try {
    const r = await json(`/api/log-viewer?${p}`);
    if (!$("log-date").options.length || $("log-date").dataset.loaded !== r.dates.join()) {
      const current = $("log-date").value;
      $("log-date").replaceChildren(...r.dates.map((d) => Object.assign(document.createElement("option"), { value: d, textContent: d, selected: d === (current || r.date) })));
      $("log-date").dataset.loaded = r.dates.join();
    }
    $("log-count").textContent = `${r.entries.length} shown of ${r.total} matching (${r.scanned} lines scanned)`;
    $("log-table").innerHTML = r.entries.slice().reverse().map((e) => {
      const extra = Object.fromEntries(Object.entries(e).filter(([k]) => !["timestamp", "level", "message", "stream", "source", "requestId"].includes(k)));
      const more = Object.keys(extra).length ? `\n${JSON.stringify(extra)}` : "";
      return `<tr class="${esc(e.level)}"><td>${esc((e.timestamp ?? "").replace("T", " ").slice(0, 23))}</td><td>${esc(e.level)}</td><td>${esc(e.source)}</td>
        <td>${e.requestId ? `<a href="#logs" data-rid="${esc(e.requestId)}">${esc(e.requestId.slice(0, 8))}</a>` : ""}</td><td class="msg">${esc(e.message)}${esc(more.slice(0, 1500))}</td></tr>`;
    }).join("");
    for (const a of $("log-table").querySelectorAll("a[data-rid]")) a.onclick = () => { $("log-request").value = a.dataset.rid; $("log-stream").value = "all"; loadLogs(); };
  } catch (error) {
    $("log-count").textContent = error.message;
  }
}
$("log-go").onclick = loadLogs;
$("log-follow").onchange = () => { clearInterval(follow); if ($("log-follow").checked) follow = setInterval(loadLogs, 5000); };
for (const id of ["log-stream", "log-level", "log-date"]) $(id).onchange = loadLogs;
$("health-refresh").onclick = () => loadHealth(true);
$("schwab-reauth").onclick = () => window.TradingSuite?.openReauth();
$("schwab-daily-run").onclick = async () => {
  if (!confirm("Start the Schwab 20-year daily history backfill now? It is resumable, uses only market-data endpoints, and does not change Massive grouped bars. A valid Schwab token is required.")) return;
  try {
    await json("/api/schwab/daily-backfill", {});
    $("schwab-daily").textContent = "Backfill started. This page will refresh status shortly.";
    setTimeout(loadSchwab, 2000);
  } catch (error) { $("schwab-daily").textContent = error.message; }
};
loadJobs();
loadReport();
loadDaytrade();
loadSchwab();
loadHealth();
loadLogs();
