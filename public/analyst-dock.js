/* Command console: always-visible command prompt + scrolling log, lower-right on every dashboard page
 * (loaded by nav.js; styles live in /analyst-dock.css because the dashboard's Content-Security-Policy
 * blocks injected <style> tags).
 *
 * One log holds the AI analyst Q&A and the live decision feed (paper trades, swarm, crypto, options, signals,
 * black swan, dollar shock, regime, macro, system), each type in its own color.
 * Analyst questions go to the same /api/ai/command route as /ai.html (advisory only; nothing here can place
 * or preview orders). Alerts arrive through TradingAlerts (SSE /api/alerts/stream, backfilled from
 * /api/alerts/recent). Voice and per-category mute stay in TradingAlerts ("Mute..." opens its settings).
 *
 * A reply of type "choice" ({type:"choice", prompt, options:[{label, value}]}, e.g. a symbol that is both a coin
 * and a stock) shows numbered buttons; click one or press its number with an empty prompt to send its value.
 * Unknown reply types render as plain text.
 *
 * Actions: the server adjudicates each prompt into one dashboard action. Reads run at once and show their numbers
 * (and a small table). Anything that changes state comes back as kind "confirm": an Allow / Cancel prompt with
 * a summary of the action, its arguments and the accounts it affects. Nothing runs until Allow is clicked (or
 * Enter with an empty prompt); Cancel or Esc drops it; it expires at expiresAt. It is never allowed automatically.
 *
 * Keys: Enter sends (or Allows an open change with an empty prompt), Esc cancels an open change, Shift+Enter new line, Up/Down walk the command history, Esc folds the log away,
 * Alt+Shift+A shows the console and focuses the prompt.
 */
(function () {
  "use strict";
  if (window.__tmAnalystDock || typeof document === "undefined") return;
  window.__tmAnalystDock = true;

  const STORE = "tm.console.v1";
  const OLD_STORE = "tm.analystDock.v1";
  const HIST = "tm.console.history";
  const MAX_INPUT = 2000;           // the server builds up to ~9,500 chars of data context around the question
  const MAX_ENTRIES = 200;
  const MAX_BYTES = 150_000;
  const MAX_HIST = 50;
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // Log types: [id, short tag, filter group]. Alert categories match server/alerts/categories.js.
  const TYPES = {
    user: ["YOU", "analyst"], analyst: ["ANALYST", "analyst"], pending: ["ANALYST", "analyst"], err: ["ERROR", "system"],
    trade_main: ["MAIN", "trades"], trade_swarm: ["SWARM", "trades"], trade_crypto: ["CRYPTO", "trades"], trade_options: ["OPTIONS", "trades"],
    signals: ["SIGNAL", "signals"], black_swan: ["SWAN", "risk"], dollar_shock: ["DOLLAR", "risk"],
    regime: ["REGIME", "macro"], macro: ["MACRO", "macro"], system: ["SYSTEM", "system"],
  };
  const GROUPS = [["all", "All"], ["analyst", "Analyst"], ["trades", "Trades"], ["signals", "Signals"], ["risk", "Risk"], ["macro", "Macro"], ["system", "System"]];

  if (!window.TradingVoiceSummary) {
    const s = document.createElement("script");
    s.src = "/voice-summary.js";
    document.head.appendChild(s);
  }

  // ---- state -------------------------------------------------------------------------------------------------
  function load() {
    let v = {};
    try { v = JSON.parse(localStorage.getItem(STORE) || "{}") || {}; } catch {}
    const st = {
      mode: ["full", "prompt", "tab"].includes(v.mode) ? v.mode : "full",
      filter: GROUPS.some((g) => g[0] === v.filter) ? v.filter : "all",
      font: v.font === "plain" ? "plain" : "pixel",
      entries: Array.isArray(v.entries) ? v.entries : [],
      draft: typeof v.draft === "string" ? v.draft : "",
    };
    if (!Array.isArray(v.entries)) { // one-time carry-over of the earlier analyst chat
      try {
        const old = JSON.parse(localStorage.getItem(OLD_STORE) || "{}");
        if (Array.isArray(old.msgs)) st.entries = old.msgs.filter((m) => m && m.role !== "pending").map((m) =>
          m.role === "ai" ? { type: "analyst", v: m.v, ts: m.ts } : { type: m.role === "err" ? "err" : "user", text: m.text, ts: m.ts, retry: m.retry });
      } catch {}
    }
    return st;
  }
  let state = load();
  function save() {
    state.entries = state.entries.slice(-MAX_ENTRIES);
    let json = JSON.stringify(state);
    while (json.length > MAX_BYTES && state.entries.length > 5) { state.entries.shift(); json = JSON.stringify(state); }
    try { localStorage.setItem(STORE, json); } catch {}
  }
  let history = [];
  try { history = JSON.parse(localStorage.getItem(HIST) || "[]").filter((x) => typeof x === "string").slice(-MAX_HIST); } catch {}
  function remember(text) {
    if (history[history.length - 1] !== text) history.push(text);
    history = history.slice(-MAX_HIST);
    try { localStorage.setItem(HIST, JSON.stringify(history)); } catch {}
  }

  // Keep only what the renderer needs (no raw research blobs in storage).
  function compact(v) {
    if (!v || typeof v !== "object") return { kind: "text", text: String(v ?? "") };
    if (v.type === "choice" || v.kind === "choice") {  // e.g. BTC: crypto or the stock? -> clickable options
      const options = (Array.isArray(v.options) ? v.options : []).slice(0, 9).map((o) => (o && typeof o === "object")
        ? { label: String(o.label ?? o.value ?? "").slice(0, 200), value: String(o.value ?? o.label ?? "").slice(0, MAX_INPUT) }
        : { label: String(o).slice(0, 200), value: String(o).slice(0, MAX_INPUT) }).filter((o) => o.value);
      return { kind: "choice", prompt: String(v.prompt || v.text || v.answer || "Which one did you mean?").slice(0, 2000), options };
    }
    if (v.kind === "confirm" && v.token) {
      const sm = v.summary || {};
      const list = (a, n) => (Array.isArray(a) ? a : []).slice(0, n).map((x) => String(x).slice(0, 300));
      return { kind: "confirm", token: String(v.token).slice(0, 64), expiresAt: String(v.expiresAt || ""), status: "open",
        summary: { action: String(sm.action || "").slice(0, 60), title: String(sm.title || "").slice(0, 200), args: list(sm.args, 12), affects: list(sm.affects, 8) } };
    }
    if (v.kind === "action" || v.kind === "clarify" || v.kind === "manual" || v.kind === "cancelled" || (v.kind === "error" && v.text)) {
      const keep = { kind: v.kind, label: String(v.label || "").slice(0, 200), text: String(v.answer || v.text || "").slice(0, 8000) };
      if (v.table && Array.isArray(v.table.columns) && Array.isArray(v.table.rows)) {
        keep.table = { columns: v.table.columns.slice(0, 8).map((c) => String(c).slice(0, 40)), rows: v.table.rows.slice(0, 12).map((r) => (Array.isArray(r) ? r : []).slice(0, 8).map((c) => String(c ?? "").slice(0, 60))) };
      }
      return keep;
    }
    const keep = { kind: v.kind || v.type || "answer" };
    for (const k of ["answer", "text", "message", "prompt", "stance", "conviction", "source", "model", "error"]) if (v[k] != null) keep[k] = String(v[k]).slice(0, 12000);
    if (Array.isArray(v.followUps)) keep.followUps = v.followUps.slice(0, 4).map((x) => String(x).slice(0, 200));
    if (Array.isArray(v.levels)) keep.levels = v.levels.slice(0, 8).map((l) => ({ label: String(l.label ?? ""), price: l.price }));
    if (Array.isArray(v.citations)) keep.citations = v.citations.slice(0, 8).map((c) => ({ source: String(c.source ?? ""), asOf: String(c.asOf ?? ""), detail: String(c.detail ?? "").slice(0, 300) }));
    if (Array.isArray(v.decisions)) keep.decisions = v.decisions.slice(0, 30).map((d) => ({ ticker: d.ticker, usable: !!d.usable, flag: d.flag, action: d.action, sizeScale: d.sizeScale, stop: d.stop, reason: d.reason }));
    if (v.kind === "job" && v.result) keep.text = String(v.result).slice(0, 4000);
    if (!keep.answer && !keep.text && (keep.message || keep.prompt)) keep.text = keep.message || keep.prompt;
    if (!keep.answer && !keep.text && !keep.decisions) keep.text = JSON.stringify(v.research || v.verdicts || v, null, 1).slice(0, 4000);
    return keep;
  }

  function markdown(text) { // small, safe subset: escape first, then format
    const lines = esc(text).split(/\r?\n/), out = [];
    let list = false;
    const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
    for (const line of lines) {
      const h = line.match(/^(#{1,4})\s+(.*)$/), li = line.match(/^\s*[-*]\s+(.*)$/);
      if (li) { if (!list) { out.push("<ul>"); list = true; } out.push("<li>" + inline(li[1]) + "</li>"); continue; }
      if (list) { out.push("</ul>"); list = false; }
      if (h) out.push("<p><strong>" + inline(h[2]) + "</strong></p>");
      else if (line.trim()) out.push("<p>" + inline(line) + "</p>");
    }
    if (list) out.push("</ul>");
    return out.join("");
  }

  function renderReply(v) {
    if (v.kind === "choice") {
      const opts = (v.options || []).map((o, i) => `<button type="button" class="tmc-opt" data-value="${esc(o.value)}" title="Press ${i + 1} or click">${i + 1}. ${esc(o.label)}</button>`).join("");
      return `<p>${esc(v.prompt)}</p>` + (opts ? `<div class="tmc-fus tmc-opts">${opts}</div><div class="tmc-hint">Press 1-${v.options.length} or click an option.</div>` : "");
    }
    if (v.kind === "confirm") {
      const sm = v.summary || {}, open = v.status === "open";
      const done = { allowed: "Allowed", cancelled: "Cancelled — nothing changed", expired: "Expired — nothing changed", failed: "Allowed (it failed; see below)" }[v.status] || "";
      const exp = v.expiresAt && !Number.isNaN(Date.parse(v.expiresAt)) ? new Date(v.expiresAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "";
      return `<div class="tmc-confirm${open ? " open" : " done"}" role="group" aria-label="Allow this change?">` +
        `<div class="tmc-cf-q">Allow this change?</div>` +
        `<div class="tmc-cf-act"><strong>${esc(sm.title)}</strong> <code>${esc(sm.action)}</code></div>` +
        ((sm.affects || []).length ? `<ul class="tmc-cf-aff">${sm.affects.map((a) => `<li>${esc(a)}</li>`).join("")}</ul>` : "") +
        ((sm.args || []).length ? `<div class="tmc-cf-args">${sm.args.map((a) => `<code>${esc(a)}</code>`).join(" ")}</div>` : "") +
        (open ? `<div class="tmc-cf-btns"><button type="button" class="tmc-allow">Allow</button><button type="button" class="tmc-cancel">Cancel</button>` +
          `<span class="tmc-hint">Enter = Allow · Esc = Cancel${exp ? ` · expires ${esc(exp)}` : ""}</span></div>` : `<div class="tmc-cf-done">${esc(done)}</div>`) + `</div>`;
    }
    if (v.kind === "action" || v.kind === "clarify" || v.kind === "manual" || v.kind === "cancelled" || v.kind === "error") {
      const table = v.table && v.table.columns.length ? `<div class="tmc-tablewrap"><table class="tmc-table"><thead><tr>${v.table.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>` +
        v.table.rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("") + `</tbody></table></div>` : "";
      const head = v.kind === "action" && v.label ? `<div class="tmc-meta"><span class="tmc-b">${esc(v.label)}</span></div>` : "";
      return head + `<div class="tmc-act k-${esc(v.kind)}">${markdown(v.text || "")}</div>` + table;
    }
    if (v.kind === "help") return `<pre>${esc(v.text)}</pre>`;
    if (v.kind === "decision" && v.decisions) {
      return "<pre>" + esc(v.decisions.map((d) => `${d.ticker}: ${d.usable ? `${d.flag} ${d.action}, size x${d.sizeScale}, stop ${d.stop}` : `no change (${d.reason})`}`).join("\n")) + "</pre>";
    }
    if (v.answer != null) {
      const meta = [v.stance && `<span class="tmc-b">${esc(v.stance)}</span>`, v.conviction && `conviction ${esc(v.conviction)}`,
        v.source && `<span class="tmc-b ${v.source === "LLM" ? "" : "fb"}">${esc(v.source)}${v.model ? " · " + esc(v.model) : ""}</span>`].filter(Boolean).join(" ");
      const levels = (v.levels || []).map((l) => `<li>${esc(l.label)}: ${esc(l.price)}</li>`).join("");
      const cites = (v.citations || []).map((c) => `<li>${esc(c.source)} (${esc(c.asOf)}): ${esc(c.detail)}</li>`).join("");
      const fu = (v.followUps || []).map((f) => `<button type="button" class="tmc-fu">${esc(f)}</button>`).join("");
      return (meta ? `<div class="tmc-meta">${meta}</div>` : "") + markdown(v.answer) +
        (levels ? `<details><summary>Levels</summary><ul>${levels}</ul></details>` : "") +
        (cites ? `<details><summary>Sources</summary><ul>${cites}</ul></details>` : "") + (fu ? `<div class="tmc-fus">${fu}</div>` : "");
    }
    return `<pre>${esc(v.text || "")}</pre>`;
  }

  // Alert -> log entry (only display fields are kept).
  function alertType(a) {
    const ta = window.TradingAlerts;
    const c = (ta && ta.categoryOf ? ta.categoryOf(a) : null) || a.category || (a.meta && a.meta.category) || "system";
    return TYPES[c] ? c : "system";
  }
  function fromAlert(a) {
    return {
      id: "a:" + a.id, type: alertType(a), ts: a.ts || new Date().toISOString(), severity: a.severity || "info",
      title: String(a.title || "").slice(0, 160), text: String(a.text || a.speech || "").slice(0, 2000),
    };
  }

  // ---- UI ----------------------------------------------------------------------------------------------------
  function build() {
    if (document.getElementById("tm-console")) return;
    if (!document.querySelector('link[href^="/analyst-dock.css"]')) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = "/analyst-dock.css" + (window.__tmAssetVersion ? "?v=" + window.__tmAssetVersion : "");
      document.head.appendChild(link);
    }
    document.documentElement.classList.add("tm-has-console");

    const root = document.createElement("section");
    root.id = "tm-console";
    root.setAttribute("aria-label", "Command console");
    root.innerHTML = `
      <div class="tmc-logwrap" id="tm-console-log">
        <header class="tmc-head">
          <span class="tmc-title">Command log</span>
          <span class="tmc-filters" role="group" aria-label="Show in log">${GROUPS.map(([id, label]) =>
            `<button type="button" class="tmc-chip" data-filter="${id}" aria-pressed="false">${esc(label)}</button>`).join("")}</span>
          <span class="tmc-tools">
            <button type="button" class="tmc-voice" aria-pressed="true" title="Speak analyst replies and trade alerts">Voice</button>
            <button type="button" class="tmc-mute" title="Voice settings and mute by category">Mute…</button>
            <button type="button" class="tmc-font" title="Switch the log between pixel and plain text">Aa</button>
            <button type="button" class="tmc-clear" title="Clear the log">Clear</button>
            <button type="button" class="tmc-fold" title="Hide the log (Esc)" aria-label="Hide the log">▾</button>
          </span>
        </header>
        <div class="tmc-log" role="log" aria-live="polite" aria-relevant="additions"></div>
        <button type="button" class="tmc-jump" hidden>▼ new</button>
      </div>
      <div class="tmc-box">
        <div class="tmc-sub"><span>Ask the analyst · ↑↓ history · /help</span><button type="button" class="tmc-unlock" hidden>Click to enable voice</button></div>
        <textarea class="tmc-input" rows="1" maxlength="${MAX_INPUT}" aria-label="Command or question for the AI analyst" placeholder="Type a command"></textarea>
        <div class="tmc-btns">
          <button type="button" class="tmc-send">Command</button>
          <span class="tmc-status" aria-live="polite"></span>
          <button type="button" class="tmc-logbtn" aria-controls="tm-console-log" aria-expanded="true" title="Show or hide the log">Log ▾</button>
          <button type="button" class="tmc-hide" title="Collapse the console to a tab" aria-label="Collapse console">_</button>
        </div>
      </div>
      <button type="button" class="tmc-tab" title="Command console (Alt+Shift+A)">Command ▴</button>`;
    document.body.appendChild(root);

    const q = (sel) => root.querySelector(sel);
    const log = q(".tmc-log"), input = q(".tmc-input"), status = q(".tmc-status"), jump = q(".tmc-jump"), voiceBtn = q(".tmc-voice"),
      unlockBtn = q(".tmc-unlock"), tab = q(".tmc-tab"), logBtn = q(".tmc-logbtn");
    input.value = state.draft || "";

    // Tell the alert UI how tall the console is, so its hint and captions sit above it.
    const syncHeight = () => document.documentElement.style.setProperty("--tm-console-h", Math.round(root.getBoundingClientRect().height) + "px");
    if (window.ResizeObserver) new ResizeObserver(syncHeight).observe(root);

    function setMode(mode, focus) {
      state.mode = mode;
      root.dataset.mode = mode;
      logBtn.setAttribute("aria-expanded", String(mode === "full"));
      logBtn.textContent = mode === "full" ? "Log ▾" : "Log ▴";
      if (mode === "full") { tab.dataset.unread = "0"; scrollEnd(true); }
      if (mode !== "tab") tab.dataset.unread = "0";
      if (focus && mode !== "tab") setTimeout(() => input.focus(), 0);
      save(); syncHeight();
    }
    function setFilter(f) {
      state.filter = f;
      root.dataset.filter = f;
      root.querySelectorAll(".tmc-chip").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.filter === f)));
      save(); scrollEnd(true);
    }
    function setFont(f) { state.font = f; root.dataset.font = f; save(); }

    const atBottom = () => log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    function scrollEnd(force) {
      if (force || atBottom()) { log.scrollTop = log.scrollHeight; jump.hidden = true; } else jump.hidden = false;
    }
    log.addEventListener("scroll", () => { if (atBottom()) jump.hidden = true; });
    jump.addEventListener("click", () => scrollEnd(true));

    function fmt(ts) {
      try { return new Date(ts).toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" }); } catch { return ""; }
    }
    function full(ts) {
      try { return new Date(ts).toLocaleString("en-US", { timeZone: "America/New_York" }) + " ET"; } catch { return ""; }
    }
    function renderEntry(e) {
      const t = TYPES[e.type] ? e.type : "system";
      const div = document.createElement("div");
      div.className = `tmc-e t-${t} g-${TYPES[t][1]}` + (e.severity && e.severity !== "info" ? " sev-" + esc(e.severity) : "");
      let body;
      if (t === "user") body = `<span class="tmc-cmd">&gt; ${esc(e.text)}</span>`;
      else if (t === "pending") body = `<em class="tmc-work">Working… gathering quotes, history, filings and news<span class="tmc-dots"></span></em>`;
      else if (t === "err") body = `${esc(e.text)}${e.retry ? ` <button type="button" class="tmc-fu tmc-retry">Ask again</button>` : ""}`;
      else if (t === "analyst") body = renderReply(e.v || {}).replace(e.picked ? /class="tmc-opt"/g : /$^/, 'class="tmc-opt" disabled');
      else body = (e.title ? `<strong>${esc(e.title)}</strong>${e.text ? " — " : ""}` : "") + esc(e.text);
      div.innerHTML = `<span class="tmc-ts" title="${esc(full(e.ts))}">${esc(fmt(e.ts))}</span><span class="tmc-tag">${TYPES[t][0]}</span><div class="tmc-body">${body}</div>`;
      div.querySelectorAll(".tmc-fu:not(.tmc-retry)").forEach((b) => b.addEventListener("click", () => { input.value = b.textContent; input.focus(); }));
      div.querySelectorAll(".tmc-opt").forEach((b) => b.addEventListener("click", () => pickOption(b)));
      const allowBtn = div.querySelector(".tmc-allow"), cancelBtn = div.querySelector(".tmc-cancel");
      if (allowBtn) allowBtn.addEventListener("click", () => decide(e, true));
      if (cancelBtn) cancelBtn.addEventListener("click", () => decide(e, false));
      if (allowBtn) div.addEventListener("keydown", (ev) => { if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); decide(e, false); } });
      const retry = div.querySelector(".tmc-retry");
      if (retry) retry.addEventListener("click", () => send(e.retry));
      return div;
    }
    function renderAll() {
      log.replaceChildren();
      if (!state.entries.length) {
        log.innerHTML = `<p class="tmc-empty">Ask about any ticker, the paper accounts, the models or the economy. Paper-trader decisions and alerts show up here as they happen. Replies are advisory; the paper trader's risk engine stays final.</p>`;
      }
      for (const e of state.entries) log.appendChild(renderEntry(e));
      scrollEnd(true);
    }
    function add(entry) {
      if (entry.id && state.entries.some((x) => x.id === entry.id)) return;
      if (!state.entries.length) log.replaceChildren();
      const stick = atBottom();
      state.entries.push(entry);
      if (state.entries.length > MAX_ENTRIES) { state.entries.shift(); if (log.firstChild) log.firstChild.remove(); }
      log.appendChild(renderEntry(entry));
      save();
      if (stick) scrollEnd(true); else jump.hidden = false;
      if (state.mode !== "full") tab.dataset.unread = "1";
    }

    function syncVoice() {
      const ta = window.TradingAlerts;
      const on = ta && ta.voiceOn ? ta.voiceOn() : false;
      voiceBtn.setAttribute("aria-pressed", String(on));
      voiceBtn.textContent = on ? "Voice on" : "Voice off";
      unlockBtn.hidden = !(on && ta && ta.audioLocked && ta.audioLocked());
    }

    // ---- analyst requests ----
    async function api(path, body) {
      const r = await fetch("/api/ai/" + path, body === undefined ? { cache: "no-store" } :
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const v = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(v.error || "Request failed (" + r.status + ")");
      return v;
    }
    async function pollJob(id) {
      for (let i = 0; i < 600; i++) {
        const j = await api("flag/" + encodeURIComponent(id));
        if (j.status !== "running") {
          if (j.status !== "complete") throw new Error("Flag job failed: " + (j.error || j.status));
          const lines = (j.result?.verdicts || []).map((v) => `${v.flag} ${v.ticker} ${v.action || ""} — ${v.summary || ""}`);
          return { kind: "text", text: `Wrote ${j.result?.exported?.count ?? lines.length} verdicts.\n` + lines.join("\n") };
        }
        await new Promise((r) => setTimeout(r, 2000));
      }
      throw new Error("Flag job is still running; check /ai.html.");
    }

    // Choice replies (e.g. "BTC: Bitcoin or the stock?"): only the newest choice is live; picking resends its value.
    function liveChoice() {
      for (let i = state.entries.length - 1; i >= 0; i--) {
        const e = state.entries[i];
        if (e.type === "user" || e.type === "pending") return null;
        if (e.type === "analyst" && e.v && e.v.kind === "choice" && !e.picked) return e;
        if (e.type === "analyst") return null;
      }
      return null;
    }
    function pickOption(btn) {
      const choice = liveChoice();
      const entryEl = btn.closest(".tmc-e");
      if (!choice || busy || !entryEl || entryEl !== [...log.querySelectorAll(".tmc-e.t-analyst")].pop()) return;
      choice.picked = true; save();
      entryEl.querySelectorAll(".tmc-opt").forEach((b) => { b.disabled = true; b.classList.toggle("on", b === btn); });
      send(btn.dataset.value);
    }

    // Allow / Cancel prompts for state-changing actions. Only the newest open one is live; it expires on its own.
    let confirmShownAt = 0;
    function liveConfirm() {
      for (let i = state.entries.length - 1; i >= 0; i--) {
        const e = state.entries[i];
        if (e.type === "analyst" && e.v && e.v.kind === "confirm" && e.v.status === "open") {
          if (e.v.expiresAt && Date.parse(e.v.expiresAt) <= Date.now()) { e.v.status = "expired"; save(); renderAll(); return null; }
          return e;
        }
        if (e.type === "user" || e.type === "pending") return null;
      }
      return null;
    }
    function expireConfirms() {
      let changed = false;
      for (const e of state.entries) {
        if (e.type === "analyst" && e.v && e.v.kind === "confirm" && e.v.status === "open" && e.v.expiresAt && Date.parse(e.v.expiresAt) <= Date.now()) { e.v.status = "expired"; changed = true; }
      }
      if (changed) { save(); renderAll(); }
    }
    setInterval(expireConfirms, 5000);
    async function decide(entry, allow) {
      if (busy || !entry || !entry.v || entry.v.status !== "open") return;
      if (entry.v.expiresAt && Date.parse(entry.v.expiresAt) <= Date.now()) { entry.v.status = "expired"; save(); renderAll(); return; }
      const token = entry.v.token;
      entry.v.status = allow ? "allowed" : "cancelled";
      save(); renderAll();
      if (!allow) {
        api("actions/cancel", { token }).catch(() => {});
        add({ type: "analyst", v: { kind: "cancelled", text: "Cancelled. Nothing was changed." }, ts: new Date().toISOString() });
        return;
      }
      busy = true; root.dataset.busy = "1"; status.textContent = "working…";
      let result;
      try { result = compact(await api("actions/confirm", { token })); } catch (err) { result = { kind: "error", text: err.message || String(err) }; }
      finally { busy = false; delete root.dataset.busy; status.textContent = ""; }
      if (result.kind === "error") { entry.v.status = /expired/i.test(result.text) ? "expired" : "failed"; save(); renderAll(); }
      add({ type: "analyst", v: result, ts: new Date().toISOString() });
      speak(result.kind === "action" ? `${result.label || "Done"}: done.` : result.text);
      input.focus();
    }
    function speak(text) {
      if (text && window.TradingAlerts && window.TradingAlerts.say) window.TradingAlerts.say(String(text).slice(0, 400), { category: "analyst", title: "AI analyst", pop: false, text: String(text).slice(0, 400) });
    }

    let busy = false, histAt = -1;
    async function send(text) {
      text = String(text || "").trim().slice(0, MAX_INPUT);
      if (!text || busy) return;
      if (text === "/clear") { input.value = ""; state.entries = []; state.draft = ""; save(); renderAll(); return; }
      busy = true; histAt = -1; remember(text);
      root.dataset.busy = "1"; status.textContent = "working…";
      const id = "p:" + Date.now();
      add({ type: "user", text, ts: new Date().toISOString() });
      add({ type: "pending", id, text, ts: new Date().toISOString() });
      state.draft = ""; input.value = ""; autosize(); save();
      if (state.mode === "tab") setMode("prompt");
      let entry;
      try {
        let v = await api("command", { text });
        if (v.kind === "job" && v.job?.id) v = await pollJob(v.job.id);
        entry = { type: "analyst", v: compact(v), ts: new Date().toISOString() };
        if (entry.v.kind === "confirm") confirmShownAt = Date.now();
        const sum = entry.v.kind === "choice" ? entry.v.prompt
          : entry.v.kind === "confirm" ? `Allow ${entry.v.summary.title}? Press Enter to allow or Escape to cancel.`
          : ["action", "clarify", "manual", "error"].includes(entry.v.kind) ? (entry.v.label ? entry.v.label + ". " : "") + String(entry.v.text || "").replace(/^- /gm, "").split("\n").slice(0, 3).join(". ")
          : window.TradingVoiceSummary && window.TradingVoiceSummary.summarizeReply(v);
        if (sum && window.TradingAlerts && window.TradingAlerts.say) {
          window.TradingAlerts.say(sum, { category: "analyst", title: "AI analyst", pop: false, text: sum });
        }
      } catch (e) {
        entry = { type: "err", text: e.message || String(e), retry: text, ts: new Date().toISOString() };
      } finally {
        busy = false; delete root.dataset.busy; status.textContent = "";
      }
      state = { ...load(), mode: state.mode, filter: state.filter, font: state.font };  // another tab may have added entries
      const at = state.entries.findIndex((m) => m.id === id);
      if (at >= 0) state.entries.splice(at, 1, entry); else state.entries.push(entry);
      save(); renderAll();
      if (state.mode !== "full") tab.dataset.unread = "1";
    }

    // A request interrupted by navigating away leaves a "pending" entry: offer to ask again.
    let touched = false;
    state.entries = state.entries.map((m) => {
      if (m.type === "pending" && Date.now() - Date.parse(m.ts) > 3 * 60 * 1000) { touched = true; return { type: "err", text: "That answer was interrupted (the page changed).", retry: m.text, ts: m.ts }; }
      return m;
    });
    if (touched) save();

    function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight + 2, 120) + "px"; }
    q(".tmc-send").addEventListener("click", () => send(input.value));
    input.addEventListener("keydown", (e) => {
      if (!input.value.trim() && !e.shiftKey && !e.altKey && !e.ctrlKey && !e.metaKey && (e.key === "Enter" || e.key === "Escape")) {
        const pend = liveConfirm();
        if (pend) {
          e.preventDefault();
          if (e.key === "Escape") decide(pend, false);
          else if (Date.now() - confirmShownAt > 700) decide(pend, true);   // ignore a double-tapped Enter from the send
          return;
        }
      }
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(input.value); return; }
      if (!input.value && /^[1-9]$/.test(e.key) && !e.altKey && !e.ctrlKey && !e.metaKey && liveChoice()) {
        const btn = [...log.querySelectorAll(".tmc-e.t-analyst")].pop()?.querySelectorAll(".tmc-opt")[Number(e.key) - 1];
        if (btn) { e.preventDefault(); pickOption(btn); return; }
      }
      const v = input.value, start = input.selectionStart, end = input.selectionEnd;
      if (e.key === "ArrowUp" && history.length && (histAt >= 0 || !v.slice(0, start).includes("\n")) && start === end) {
        e.preventDefault();
        histAt = histAt < 0 ? history.length - 1 : Math.max(0, histAt - 1);
        input.value = history[histAt]; autosize();
      } else if (e.key === "ArrowDown" && histAt >= 0 && start === end) {
        e.preventDefault();
        histAt += 1;
        input.value = histAt >= history.length ? (histAt = -1, state.draft || "") : history[histAt];
        autosize();
      } else if (e.key === "Escape") {
        e.preventDefault(); setMode(state.mode === "full" ? "prompt" : "tab");
        if (state.mode === "tab") tab.focus();
      }
    });
    input.addEventListener("input", () => { histAt = -1; state.draft = input.value; autosize(); save(); });
    root.querySelectorAll(".tmc-chip").forEach((b) => b.addEventListener("click", () => setFilter(b.dataset.filter)));
    q(".tmc-fold").addEventListener("click", () => setMode("prompt"));
    logBtn.addEventListener("click", () => setMode(state.mode === "full" ? "prompt" : "full"));
    q(".tmc-hide").addEventListener("click", () => { setMode("tab"); tab.focus(); });
    tab.addEventListener("click", () => setMode("full", true));
    q(".tmc-clear").addEventListener("click", () => { state.entries = []; save(); renderAll(); input.focus(); });
    q(".tmc-font").addEventListener("click", () => setFont(state.font === "pixel" ? "plain" : "pixel"));
    q(".tmc-mute").addEventListener("click", () => window.TradingAlerts && window.TradingAlerts.toggleSettings && window.TradingAlerts.toggleSettings());
    unlockBtn.addEventListener("click", () => setTimeout(syncVoice, 50)); // the click itself unlocks audio
    voiceBtn.addEventListener("click", () => {
      const ta = window.TradingAlerts;
      if (!ta || !ta.setMasterMute) return;
      ta.setMasterMute(ta.voiceOn());
      syncVoice();
    });
    document.addEventListener("keydown", (e) => {
      if (e.altKey && e.shiftKey && (e.key === "A" || e.key === "a" || e.code === "KeyA")) {
        e.preventDefault();
        if (state.mode === "tab" || document.activeElement === input) setMode(state.mode === "tab" ? "full" : "tab", true);
        else setMode(state.mode, true);
      }
    });
    document.addEventListener("pointerdown", () => setTimeout(syncVoice, 50), { capture: true });
    document.addEventListener("keydown", () => setTimeout(syncVoice, 50), { capture: true });
    window.addEventListener("storage", (e) => {
      if (e.key !== STORE || busy) return;
      state = { ...load(), mode: state.mode, filter: state.filter, font: state.font };
      renderAll();
    });

    // ---- decision feed ----
    function wantAlert(a, opts) {
      if (!a || !a.id) return false;
      if (opts && opts.local && alertType(a) === "analyst") return false; // spoken summary of a reply already in the log
      return true;
    }
    const hook = () => {
      const ta = window.TradingAlerts;
      if (!ta || !ta.onChange) return false;
      ta.onChange(syncVoice); syncVoice();
      if (ta.onAlert) ta.onAlert((a, opts) => { if (wantAlert(a, opts)) add(fromAlert(a)); });
      return true;
    };
    if (!hook()) { let tries = 0; const t = setInterval(() => { if (hook() || ++tries > 40) clearInterval(t); }, 250); }
    fetch("/api/alerts/recent?limit=60", { cache: "no-store" }).then((r) => r.ok ? r.json() : { alerts: [] }).then((j) => {
      const known = new Set(state.entries.map((e) => e.id).filter(Boolean));
      const fresh = (j.alerts || []).filter((a) => a && a.id && !known.has("a:" + a.id) && !(a.category === "analyst" && String(a.id).startsWith("local-"))).map(fromAlert);
      if (!fresh.length) return;
      if (!state.entries.length) log.replaceChildren();
      state.entries = state.entries.concat(fresh).sort((x, y) => Date.parse(x.ts) - Date.parse(y.ts));
      save(); renderAll();
    }).catch(() => {});

    root.dataset.filter = state.filter; root.dataset.font = state.font;
    root.querySelectorAll(".tmc-chip").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.filter === state.filter)));
    renderAll();
    autosize();
    setMode(state.mode, false);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
})();
