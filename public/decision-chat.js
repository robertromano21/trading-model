// Paper-trader decision chat: every autopilot/paper decision (signals, orders, fills, skips/blocks with reasons,
// stops, exits, day trades, provisional intraday signals) as chat entries with rationale and the AI verdict used,
// filterable, with follow-up questions to the AI analyst about any decision. Paper only.
// Mounts into #decision-chat if present, otherwise appends a section to <main>/<body>.
(() => {
  if (!document.querySelector('link[href="/decision-chat.css"]')) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/decision-chat.css";
    document.head.appendChild(link);
  }
  const root = document.getElementById("decision-chat") || (() => {
    const section = document.createElement("section");
    section.id = "decision-chat";
    (document.querySelector("main") || document.body).appendChild(section);
    return section;
  })();
  root.classList.add("dchat");
  root.innerHTML = `
    <h2>Paper-trader decisions &amp; chat</h2>
    <div class="bar">
      <strong class="ap-state">Autopilot …</strong>
      <button type="button" class="ap-toggle">…</button>
      <select class="ap-task" aria-label="Run a paper task now">
        <option value="">Run now…</option><option value="swing">Swing paper run</option>
        <option value="daytrade_watch">Day-trader watch (intraday)</option><option value="daytrade_close">Day-trader close run</option>
        <option value="feed">Refresh decision log</option>
      </select>
      <span class="ap-detail muted"></span>
    </div>
    <div class="bar">
      <select class="f-kind" aria-label="Kind">
        <option value="">All entries</option>
        <option value="order,entry,watch">Signals &amp; orders</option><option value="fill">Fills</option>
        <option value="skip,blocked,advisory">Skipped / blocked / advisory</option><option value="stop,exit">Stops &amp; exits</option>
        <option value="run">Runs (regime, breaker, equity)</option><option value="followup_q,followup_a">Follow-ups</option>
      </select>
      <select class="f-account" aria-label="Account"><option value="">All accounts</option><option value="default">Swing (default)</option><option value="daytrade">Day trading</option></select>
      <input class="f-ticker" placeholder="Ticker" size="7" aria-label="Ticker filter">
      <input class="f-q" placeholder="Search text" size="16" aria-label="Search">
      <button type="button" class="f-go">Filter</button>
      <span class="count muted"></span>
    </div>
    <div class="log" role="log" aria-live="polite"></div>
    <div class="ctx" hidden></div>
    <form><input class="ask" placeholder="Ask about the selected decision (or anything about the paper book)…" autocomplete="off"><button type="submit">Ask</button></form>`;
  const q = (s) => root.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  let lastId = 0, selected = null;

  async function json(url, body) {
    const response = await fetch(url, body === undefined ? { cache: "no-store" } :
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || response.statusText);
    return data;
  }
  function filters() {
    const p = new URLSearchParams();
    for (const [k, sel] of [["kind", ".f-kind"], ["account", ".f-account"], ["ticker", ".f-ticker"], ["q", ".f-q"]]) {
      const v = q(sel).value.trim();
      if (v) p.set(k, k === "ticker" ? v.toUpperCase() : v);
    }
    return p;
  }
  function render(m) {
    const div = document.createElement("div");
    div.className = `msg ${m.role}`;
    div.dataset.id = m.id;
    const v = m.verdict;
    const ticker = m.ticker ? `<a href="/ticker.html?symbol=${encodeURIComponent(m.ticker)}">${esc(m.ticker)}</a>` : "";
    div.innerHTML = `<div class="meta"><span class="kind ${esc(m.kind)}">${esc(m.kind.replace("followup_", ""))}</span>
        <span>${esc((m.at || "").replace("T", " ").slice(0, 19))}${m.at?.endsWith("Z") || m.at?.includes("+") ? " UTC" : ""}</span>
        ${m.account ? `<span>${esc(m.account)}</span>` : ""}${ticker}${m.decision_date ? `<span>for ${esc(m.decision_date)}</span>` : ""}
        ${v?.flag ? `<span class="flag ${esc(v.flag)}">AI ${esc(v.flag)}</span>` : ""}
        ${m.role === "trader" ? `<button type="button" class="about">Ask about this</button>` : ""}</div>
      <div>${esc(m.text)}</div>${m.rationale ? `<div class="why">${esc(m.rationale)}</div>` : ""}`;
    div.querySelector(".about")?.addEventListener("click", () => select(m, div));
    return div;
  }
  function select(m, div) {
    root.querySelectorAll(".msg.sel").forEach((el) => el.classList.remove("sel"));
    selected = m;
    div.classList.add("sel");
    q(".ctx").hidden = false;
    q(".ctx").innerHTML = `Asking about #${m.id}: ${esc(m.text.slice(0, 140))} <button type="button" class="clear">clear</button>`;
    q(".ctx .clear").onclick = () => { selected = null; q(".ctx").hidden = true; div.classList.remove("sel"); };
    q(".ask").focus();
  }
  async function load(reset = true) {
    try {
      const p = filters();
      if (!reset && lastId) p.set("afterId", lastId);
      p.set("limit", reset ? "300" : "100");
      const { messages, kinds } = await json(`/api/chat?${p}`);
      const log = q(".log");
      const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
      if (reset) log.replaceChildren();
      for (const m of messages) { log.appendChild(render(m)); lastId = Math.max(lastId, m.id); }
      if (reset && !messages.length) log.innerHTML = "<p class='muted'>No paper-trader decisions logged yet. They appear after the autopilot's first run (or “Run now… → Refresh decision log”).</p>";
      if (reset || atBottom) log.scrollTop = log.scrollHeight;
      q(".count").textContent = kinds.map((k) => `${k.kind} ${k.n}`).join(" · ");
    } catch (error) {
      q(".count").textContent = `Chat log unavailable: ${error.message}`;
    }
  }
  async function autopilot() {
    try {
      const s = await json("/api/autopilot");
      q(".ap-state").textContent = s.forcedOff ? "Autopilot OFF (TRADING_AUTOPILOT=off)" : s.enabled ? "Autopilot ON (paper only)" : "Autopilot OFF";
      q(".ap-toggle").textContent = s.enabled ? "Turn off" : "Turn on";
      q(".ap-toggle").disabled = s.forcedOff;
      q(".ap-toggle").dataset.enabled = String(s.enabled);
      const last = Object.values(s.last ?? {}).sort((a, b) => String(b.startedAt).localeCompare(String(a.startedAt)))[0];
      q(".ap-detail").textContent = [s.running ? `running: ${s.running.name}…` : "",
        last ? `last: ${last.name} ${last.ok ? "ok" : "FAILED"} at ${new Date(last.finishedAt || last.startedAt).toLocaleString("en-US", { timeZone: "America/New_York" })} ET${last.summary ? ` — ${last.summary}` : ""}` : "no runs yet",
        `schedule: swing ${s.schedule.swing}, day-trader watch ${s.schedule.daytradeWatch}, close run ${s.schedule.daytradeClose}`].filter(Boolean).join(" · ");
    } catch (error) {
      q(".ap-state").textContent = `Autopilot unavailable: ${error.message}`;
    }
  }
  q(".ap-toggle").addEventListener("click", async () => {
    try { await json("/api/autopilot/toggle", { enabled: q(".ap-toggle").dataset.enabled !== "true" }); } catch (e) { q(".ap-detail").textContent = e.message; }
    autopilot();
  });
  q(".ap-task").addEventListener("change", async () => {
    const task = q(".ap-task").value;
    q(".ap-task").value = "";
    if (!task) return;
    try { await json("/api/autopilot/run", { task }); } catch (e) { q(".ap-detail").textContent = e.message; }
    autopilot();
  });
  q(".f-go").addEventListener("click", () => { lastId = 0; load(true); });
  for (const sel of [".f-kind", ".f-account"]) q(sel).addEventListener("change", () => { lastId = 0; load(true); });
  q("form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = q(".ask").value.trim();
    if (!question) return;
    q(".ask").value = "";
    q(".ask").disabled = true;
    q(".count").textContent = "Asking the AI analyst…";
    try {
      await json("/api/chat/ask", { question, messageId: selected?.id });
    } catch (error) {
      q(".count").textContent = error.message;
    } finally {
      q(".ask").disabled = false;
      await load(false);
    }
  });
  const params = new URLSearchParams(location.search);
  if (params.get("ticker")) q(".f-ticker").value = params.get("ticker").toUpperCase();
  load(true);
  autopilot();
  setInterval(() => { load(false); autopilot(); }, 20_000);
})();
