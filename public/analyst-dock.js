/* Command console: centered command prompt + lower-right scrolling log on every dashboard page
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

  const STORE = "tm.console.v2";
  const PRIOR_STORE = "tm.console.v1";
  const OLD_STORE = "tm.analystDock.v1";
  const CHAT_PREFIX = "tm.console.chat.";
  const FEED = "tm.console.alerts";
  const HIST = "tm.console.history";
  const LOG_LAYOUT = "tm.console.logLayout";
  const LOG_ZOOMS = [100, 125, 150, 200, 300];
  const LOG_MARGIN = 8;
  const LOG_MIN_WIDTH = 240;
  const LOG_MIN_HEIGHT = 126;
  const MAX_INPUT = 2000;           // the server builds up to ~9,500 chars of data context around the question
  const MAX_ENTRIES = 200;
  const MAX_HIST = 50;
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // Log types: [id, short tag, filter group]. Alert categories match server/alerts/categories.js.
  const TYPES = {
    user: ["YOU", "analyst"], analyst: ["ANALYST", "analyst"], pending: ["ANALYST", "analyst"], err: ["ERROR", "analyst"],
    trade_main: ["MAIN", "trades"], trade_swarm: ["SWARM", "trades"], trade_crypto: ["CRYPTO", "trades"], trade_options: ["OPTIONS", "trades"],
    signals: ["SIGNAL", "signals"], black_swan: ["SWAN", "risk"], dollar_shock: ["DOLLAR", "risk"],
    regime: ["REGIME", "macro"], macro: ["MACRO", "macro"], system: ["SYSTEM", "system"],
  };
  const GROUPS = [["all", "All"], ["analyst", "Analyst"], ["trades", "Trades"], ["signals", "Signals"], ["risk", "Risk"], ["macro", "Macro"], ["system", "System"]];

  if (!window.TradingVoiceSummary) {
    const s = document.createElement("script");
    s.src = "/voice-summary.js" + (window.__tmAssetVersion ? "?v=" + window.__tmAssetVersion : "");
    document.head.appendChild(s);
  }

  // ---- state -------------------------------------------------------------------------------------------------
  function readLocal(key) {
    try { return JSON.parse(localStorage.getItem(key)); } catch { return null; }
  }
  function writeLocal(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch { return false; }
  }
  const prefs = readLocal(STORE) || readLocal(PRIOR_STORE) || {};
  const state = {
    mode: ["full", "prompt", "tab"].includes(prefs.mode) ? prefs.mode : "full",
    filter: GROUPS.some(([id]) => id === prefs.filter) ? prefs.filter : "all",
  };
  const chatStorage = new Map();
  const freshChat = (id = crypto.randomUUID()) => ({ id, title: "", updatedAt: new Date().toISOString(), entries: [], draft: "" });
  function readChat(id) {
    const saved = chatStorage.get(id) || readLocal(CHAT_PREFIX + id);
    return saved?.id === id && Array.isArray(saved.entries) ? saved : null;
  }
  function saveSettings() { writeLocal(STORE, { ...state, chatId: chat.id }); }
  function legacyEntries() {
    if (Array.isArray(prefs.entries)) return prefs.entries;
    if (prefs.chatId) return [];
    const old = readLocal(OLD_STORE);
    return Array.isArray(old?.msgs) ? old.msgs.filter((m) => m && m.role !== "pending").map((m) =>
      m.role === "ai" ? { type: "analyst", v: m.v, ts: m.ts } : { type: m.role === "err" ? "err" : "user", text: m.text, ts: m.ts, retry: m.retry }) : [];
  }
  const isAlert = (entry) => String(entry?.id || "").startsWith("a:");
  const previous = legacyEntries().filter(Boolean);
  const savedAlerts = readLocal(FEED);
  let alerts = Array.isArray(savedAlerts) ? savedAlerts : previous.filter(isAlert);
  const storedChat = readChat(prefs.chatId || "previous");
  let chat = storedChat ?? freshChat(prefs.chatId || "previous");
  if (!storedChat && !prefs.chatId && previous.length) {
    chat.entries = previous.filter((entry) => !isAlert(entry));
    chat.draft = typeof prefs.draft === "string" ? prefs.draft : "";
    chat.title = chat.entries.find((entry) => entry.type === "user")?.text?.trim().slice(0, 56) || "";
    writeLocal(FEED, alerts);
  } else if (!storedChat && !prefs.chatId && typeof prefs.draft === "string") chat.draft = prefs.draft;

  function savedChats() {
    const chats = new Map([[chat.id, chat], ...chatStorage]);
    try {
      for (let index = 0; index < localStorage.length; index++) {
        const key = localStorage.key(index);
        if (!key?.startsWith(CHAT_PREFIX)) continue;
        const id = key.slice(CHAT_PREFIX.length);
        if (!chats.has(id)) {
          const saved = readChat(id);
          if (saved) chats.set(id, saved);
        }
      }
    } catch {}
    return [...chats.values()].filter((saved) => saved.id === chat.id || saved.entries.length || saved.draft)
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }

  function conversation() {
    return chat.entries.filter((entry) => entry.type === "user" || (entry.type === "analyst" && entry.v?.kind !== "confirm"))
      .slice(-8).map((entry) => entry.type === "user" ? { role: "user", content: entry.text }
        : { role: "assistant", content: [entry.v?.explanation, entry.v?.answer || entry.v?.text || entry.v?.prompt].filter(Boolean).join("\n").slice(0, 2000) })
      .filter((message) => message.content);
  }
  function loadLogLayout() {
    let saved;
    try { saved = JSON.parse(localStorage.getItem(LOG_LAYOUT)); } catch {}
    const layout = { zoom: LOG_ZOOMS.includes(saved?.zoom) ? saved.zoom : 100 };
    for (const key of ["width", "height", "right", "bottom"]) {
      if (Number.isFinite(saved?.[key])) layout[key] = saved[key];
    }
    return layout;
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
      const keep = { kind: v.kind, label: String(v.label || "").slice(0, 200), text: String(v.answer || v.text || "").slice(0, 8000),
        explanation: String(v.explanation || "").slice(0, 4000) };
      if (v.table && Array.isArray(v.table.columns) && Array.isArray(v.table.rows)) {
        keep.table = { columns: v.table.columns.slice(0, 8).map((c) => String(c).slice(0, 40)), rows: v.table.rows.slice(0, 12).map((r) => (Array.isArray(r) ? r : []).slice(0, 8).map((c) => String(c ?? "").slice(0, 60))) };
      }
      return keep;
    }
    const keep = { kind: v.kind || v.type || "answer" };
    for (const k of ["answer", "text", "explanation", "message", "prompt", "stance", "conviction", "source", "model", "error"]) if (v[k] != null) keep[k] = String(v[k]).slice(0, 12000);
    if (Array.isArray(v.followUps)) keep.followUps = v.followUps.slice(0, 4).map((x) => String(x).slice(0, 200));
    if (Array.isArray(v.levels)) keep.levels = v.levels.slice(0, 8).map((l) => ({ label: String(l.label ?? ""), price: l.price }));
    if (Array.isArray(v.citations)) keep.citations = v.citations.slice(0, 8).map((c) => ({ source: String(c.source ?? ""), asOf: String(c.asOf ?? ""), detail: String(c.detail ?? "").slice(0, 300) }));
    if (Array.isArray(v.decisions)) keep.decisions = v.decisions.slice(0, 30).map((d) => ({ ticker: d.ticker, usable: !!d.usable, flag: d.flag, action: d.action, sizeScale: d.sizeScale, stop: d.stop, reason: d.reason }));
    if (v.kind === "job" && v.result) keep.text = String(v.result).slice(0, 4000);
    if (!keep.answer && !keep.text && (keep.message || keep.prompt)) keep.text = keep.message || keep.prompt;
    if (!keep.answer && !keep.text && !keep.decisions) keep.text = JSON.stringify(v.research || v.verdicts || v.run || v, null, 1).slice(0, 4000);
    return keep;
  }

  function renderTable(columns, rows, format = esc) {
    const headers = columns.map((cell) => `<th scope="col">${format(cell)}</th>`).join("");
    const body = rows.map((row) => `<tr>${columns.map((_, index) => `<td>${format(row[index] ?? "")}</td>`).join("")}</tr>`).join("");
    return `<div class="tmc-tablewrap" tabindex="0" role="region" aria-label="Analyst results table"><table class="tmc-table">` +
      `<thead><tr>${headers}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  const tablePipe = /(?<!\\)\|/;
  const tableCells = (line) => line.trim().replace(/^\||(?<!\\)\|$/g, "").split(tablePipe)
    .map((cell) => cell.trim().replace(/\\\|/g, "|"));

  function markdown(text) { // small, safe subset: escape first, then format
    const lines = esc(text).split(/\r?\n/), out = [];
    let list = false;
    const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
    for (let index = 0; index < lines.length; index++) {
      const line = lines[index];
      const h = line.match(/^(#{1,4})\s+(.*)$/), li = line.match(/^\s*[-*]\s+(.*)$/);
      if (li) { if (!list) { out.push("<ul>"); list = true; } out.push("<li>" + inline(li[1]) + "</li>"); continue; }
      if (list) { out.push("</ul>"); list = false; }
      const divider = lines[index + 1] ?? "";
      if (tablePipe.test(line) && tablePipe.test(divider)) {
        const columns = tableCells(line), separators = tableCells(divider);
        if (columns.length === separators.length && separators.every((cell) => /^:?-{3,}:?$/.test(cell))) {
          const rows = [];
          index++;
          while (tablePipe.test(lines[index + 1] ?? "")) {
            const row = tableCells(lines[index + 1]);
            if (row.length !== columns.length) break;
            rows.push(row);
            index++;
          }
          out.push(renderTable(columns, rows, inline));
          continue;
        }
      }
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
      const table = v.table?.columns?.length ? renderTable(v.table.columns, v.table.rows ?? []) : "";
      const head = v.kind === "action" && v.label ? `<div class="tmc-meta"><span class="tmc-b">${esc(v.label)}</span></div>` : "";
      const details = `<div class="tmc-act k-${esc(v.kind)}">${markdown(v.text || "")}</div>` + table;
      return head + (v.explanation ? markdown(v.explanation) + `<details><summary>Data and figures</summary>${details}</details>` : details);
    }
    if (v.kind === "help") return `<pre>${esc(v.text)}</pre>`;
    if (v.kind === "decision" && v.decisions) {
      const details = "<pre>" + esc(v.decisions.map((d) => `${d.ticker}: ${d.usable ? `${d.flag} ${d.action}, size x${d.sizeScale}, stop ${d.stop}` : `no change (${d.reason})`}`).join("\n")) + "</pre>";
      return v.explanation ? markdown(v.explanation) + `<details><summary>Data and figures</summary>${details}</details>` : details;
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
    const data = `<pre>${esc(v.text || "")}</pre>`;
    return v.explanation ? markdown(v.explanation) + `<details><summary>Data and figures</summary>${data}</details>` : data;
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
          <div class="tmc-layout">
            <button type="button" class="tmc-resize" title="Drag or use arrow keys to resize" aria-label="Resize command log">⤡</button>
            <button type="button" class="tmc-title" title="Drag or use arrow keys to move" aria-label="Move command log">Command log</button>
            <select class="tmc-zoom" aria-label="Command log zoom" title="Command log zoom">${LOG_ZOOMS.map((zoom) => `<option value="${zoom}">${zoom}%</option>`).join("")}</select>
            <button type="button" class="tmc-reset-layout" title="Restore the default size, position, and zoom">Reset</button>
          </div>
          <div class="tmc-chats">
            <select class="tmc-chat-select" aria-label="Saved chats"></select>
            <button type="button" class="tmc-new-chat">New chat</button>
          </div>
          <p class="tmc-chat-warning" role="status" hidden>Browser storage is full or unavailable. This chat could not be saved.</p>
          <span class="tmc-filters" role="group" aria-label="Show in log">${GROUPS.map(([id, label]) =>
            `<button type="button" class="tmc-chip" data-filter="${id}" aria-pressed="false">${esc(label)}</button>`).join("")}</span>
          <span class="tmc-tools">
            <button type="button" class="tmc-voice" aria-pressed="true" title="Speak analyst replies and trade alerts">Voice</button>
            <select class="tmc-narration" aria-label="Analyst narration length" title="How much of each reply to read aloud">
              <option value="verbose">Verbose</option><option value="light">Light</option>
            </select>
            <button type="button" class="tmc-mute" title="Voice settings and mute by category">Mute…</button>
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
    const panel = q(".tmc-logwrap"), zoomSelect = q(".tmc-zoom"), narrationSelect = q(".tmc-narration");
    const page = document.documentElement;
    const chatSelect = q(".tmc-chat-select"), newChat = q(".tmc-new-chat"), chatWarning = q(".tmc-chat-warning");
    input.value = chat.draft || "";

    function save() {
      if (!chat.entries.length && !chat.draft && !readChat(chat.id)) { chatWarning.hidden = true; return; }
      const saved = writeLocal(CHAT_PREFIX + chat.id, chat);
      if (saved) chatStorage.delete(chat.id);
      else chatStorage.set(chat.id, chat);
      chatWarning.hidden = saved;
    }
    function renderChats() {
      chatSelect.replaceChildren(...savedChats().map((saved) => {
        const date = new Date(saved.updatedAt).toLocaleDateString();
        return new Option(saved.title ? `${saved.title} · ${date}` : (saved.entries.length ? "Previous conversation" : "New chat"), saved.id);
      }));
      chatSelect.value = chat.id;
    }
    function openChat(next) {
      if (busy || !next || next.id === chat.id) return;
      chat.entries = readChat(chat.id)?.entries ?? chat.entries;
      for (const entry of chat.entries) {
        if (entry.v?.kind !== "confirm" || entry.v.status !== "open") continue;
        entry.v.status = "cancelled";
        api("actions/cancel", { token: entry.v.token }).catch(() => {});
      }
      save();
      chat = next;
      recoverPending();
      expireConfirms(true);
      histAt = -1;
      input.value = chat.draft || "";
      autosize();
      save();
      setFilter("analyst");
      renderChats();
      setMode("full", true);
    }

    let layout = loadLogLayout();
    let drag;
    const clamp = (value, min, max) => Math.round(Math.min(Math.max(value, min), Math.max(min, max)));
    function saveLayout() {
      try { localStorage.setItem(LOG_LAYOUT, JSON.stringify(layout)); } catch {}
    }
    function applyLayout(stick) {
      const { width, height, right, bottom } = layout;
      const isFloating = [width, height, right, bottom].every(Number.isFinite);
      panel.toggleAttribute("data-floating", isFloating);
      if (isFloating) {
        const { clientWidth, clientHeight } = page;
        const maxWidth = Math.max(1, clientWidth - 2 * LOG_MARGIN);
        const maxHeight = Math.max(1, clientHeight - 2 * LOG_MARGIN);
        const fitWidth = clamp(width, Math.min(LOG_MIN_WIDTH, maxWidth), maxWidth);
        const fitHeight = clamp(height, Math.min(LOG_MIN_HEIGHT, maxHeight), maxHeight);
        panel.style.setProperty("--tmc-log-width", `${fitWidth}px`);
        panel.style.setProperty("--tmc-log-height", `${fitHeight}px`);
        panel.style.setProperty("--tmc-log-right", `${clamp(right, LOG_MARGIN, clientWidth - fitWidth - LOG_MARGIN)}px`);
        panel.style.setProperty("--tmc-log-bottom", `${clamp(bottom, LOG_MARGIN, clientHeight - fitHeight - LOG_MARGIN)}px`);
      } else {
        panel.style.removeProperty("--tmc-log-width");
        panel.style.removeProperty("--tmc-log-height");
      }
      panel.style.setProperty("--tmc-zoom", String(layout.zoom / 100));
      zoomSelect.value = String(layout.zoom);
      if (stick) scrollEnd(true);
    }
    function adjustPanel(action, bounds, dx, dy, stick) {
      const moving = action === "move";
      layout.right = page.clientWidth - bounds.right - (moving ? dx : 0);
      layout.bottom = page.clientHeight - bounds.bottom - (moving ? dy : 0);
      layout.width = moving ? bounds.width : clamp(bounds.width - dx, LOG_MIN_WIDTH, bounds.right - LOG_MARGIN);
      layout.height = moving ? bounds.height : clamp(bounds.height - dy, LOG_MIN_HEIGHT, bounds.bottom - LOG_MARGIN);
      applyLayout(stick);
    }
    for (const [selector, action] of [[".tmc-title", "move"], [".tmc-resize", "resize"]]) {
      const handle = q(selector);
      handle.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || drag) return;
        drag = { id: event.pointerId, bounds: panel.getBoundingClientRect(), x: event.clientX, y: event.clientY, stick: atBottom() };
        handle.setPointerCapture(event.pointerId);
      });
      handle.addEventListener("pointermove", (event) => {
        if (drag?.id !== event.pointerId) return;
        const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
        if (!drag.changed && dx === 0 && dy === 0) return;
        drag.changed = true;
        adjustPanel(action, drag.bounds, dx, dy, drag.stick);
      });
      handle.addEventListener("lostpointercapture", (event) => {
        if (drag?.id !== event.pointerId) return;
        if (drag.changed) saveLayout();
        drag = undefined;
      });
      handle.addEventListener("keydown", (event) => {
        if (event.altKey || event.ctrlKey || event.metaKey || drag) return;
        const step = event.shiftKey ? 40 : 12;
        const offset = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[event.key];
        if (!Array.isArray(offset)) return;
        event.preventDefault();
        adjustPanel(action, panel.getBoundingClientRect(), ...offset, atBottom());
        saveLayout();
      });
    }
    zoomSelect.addEventListener("change", () => {
      layout.zoom = Number(zoomSelect.value);
      applyLayout(atBottom());
      saveLayout();
    });
    q(".tmc-reset-layout").addEventListener("click", () => {
      layout = { zoom: 100 };
      applyLayout(atBottom());
      saveLayout();
    });
    window.addEventListener("resize", () => applyLayout(atBottom()));

    const prompt = q(".tmc-box");
    // Centered captions clear the prompt on wide screens and the entire stacked console on narrow screens.
    const syncHeight = () => {
      const styles = document.documentElement.style;
      styles.setProperty("--tm-console-h", root.offsetHeight + "px");
      styles.setProperty("--tm-console-prompt-h", Math.max(prompt.offsetHeight, tab.offsetHeight) + "px");
    };
    if (window.ResizeObserver) {
      const observer = new ResizeObserver(syncHeight);
      observer.observe(root);
      observer.observe(prompt);
    }

    function setMode(mode, focus) {
      state.mode = mode;
      root.dataset.mode = mode;
      logBtn.setAttribute("aria-expanded", String(mode === "full"));
      logBtn.textContent = mode === "full" ? "Log ▾" : "Log ▴";
      if (mode === "full") { tab.dataset.unread = "0"; scrollEnd(true); }
      if (mode !== "tab") tab.dataset.unread = "0";
      if (focus && mode !== "tab") setTimeout(() => input.focus(), 0);
      saveSettings(); syncHeight();
    }
    function setFilter(f) {
      state.filter = f;
      root.dataset.filter = f;
      root.querySelectorAll(".tmc-chip").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.filter === f)));
      saveSettings(); renderAll();
    }

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
      if (isAlert(e)) div.dataset.alert = "";
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
      if (!chat.entries.length && (state.filter === "analyst" || !alerts.length)) {
        log.innerHTML = `<p class="tmc-empty">Ask the analyst a question to start this chat.</p>`;
      }
      const visible = state.filter === "analyst" ? chat.entries : [...chat.entries, ...alerts].sort((a, b) => Date.parse(a.ts) - Date.parse(b.ts));
      for (const e of visible) log.appendChild(renderEntry(e));
      scrollEnd(true);
    }
    function add(entry) {
      if (entry.id && chat.entries.some((x) => x.id === entry.id)) return;
      const stick = atBottom();
      log.querySelector(".tmc-empty")?.remove();
      chat.entries.push(entry);
      chat.updatedAt = entry.ts;
      if (!chat.title && entry.type === "user") {
        chat.title = entry.text.replace(/\s+/g, " ").slice(0, 56);
        renderChats();
      }
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
      narrationSelect.value = ta?.settings?.()?.analystNarration === "light" ? "light" : "verbose";
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
      for (let i = chat.entries.length - 1; i >= 0; i--) {
        const e = chat.entries[i];
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
      for (let i = chat.entries.length - 1; i >= 0; i--) {
        const e = chat.entries[i];
        if (e.type === "analyst" && e.v && e.v.kind === "confirm" && e.v.status === "open") {
          if (e.v.expiresAt && Date.parse(e.v.expiresAt) <= Date.now()) { e.v.status = "expired"; save(); renderAll(); return null; }
          return e;
        }
        if (e.type === "user" || e.type === "pending") return null;
      }
      return null;
    }
    function expireConfirms(quiet = false) {
      let changed = false;
      for (const e of chat.entries) {
        if (e.type === "analyst" && e.v && e.v.kind === "confirm" && e.v.status === "open" && e.v.expiresAt && Date.parse(e.v.expiresAt) <= Date.now()) { e.v.status = "expired"; changed = true; }
      }
      if (changed && !quiet) { save(); renderAll(); }
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
      setBusy(true);
      let result;
      try { result = compact(await api("actions/confirm", { token })); } catch (err) { result = { kind: "error", text: err.message || String(err) }; }
      finally { setBusy(false); }
      if (result.kind === "error") { entry.v.status = /expired/i.test(result.text) ? "expired" : "failed"; save(); renderAll(); }
      add({ type: "analyst", v: result, ts: new Date().toISOString() });
      speak(result.kind === "action" ? `${result.label || "Done"}: done.` : result.text);
      input.focus();
    }
    function speak(text) {
      if (text && window.TradingAlerts && window.TradingAlerts.say) window.TradingAlerts.say(String(text).slice(0, 400), { category: "analyst", title: "AI analyst", pop: false, text: String(text).slice(0, 400) });
    }

    let busy = false, histAt = -1;
    function setBusy(value) {
      busy = value;
      chatSelect.disabled = newChat.disabled = value;
      if (value) root.dataset.busy = "1"; else delete root.dataset.busy;
      status.textContent = value ? "working…" : "";
    }
    async function send(text, model) {
      text = String(text || "").trim().slice(0, MAX_INPUT);
      if (!text || busy) return;
      if (text === "/new" || text === "/clear") { chat.draft = ""; openChat(freshChat()); return; }
      chat.entries = readChat(chat.id)?.entries ?? chat.entries;
      const history = conversation();
      setBusy(true); histAt = -1; remember(text);
      const id = "p:" + crypto.randomUUID();
      add({ type: "user", text, ts: new Date().toISOString() });
      add({ type: "pending", id, text, ts: new Date().toISOString() });
      chat.draft = ""; input.value = ""; autosize(); save();
      if (state.mode === "tab") setMode("prompt");
      let entry;
      try {
        let v = await api("command", { text, history, model });
        if (v.kind === "job" && v.job?.id) v = await pollJob(v.job.id);
        entry = { type: "analyst", v: compact(v), ts: new Date().toISOString() };
        if (entry.v.kind === "confirm") confirmShownAt = Date.now();
        const voice = window.TradingVoiceSummary;
        const sum = entry.v.kind === "choice" ? entry.v.prompt
          : entry.v.kind === "confirm" ? `Allow ${entry.v.summary.title}? Press Enter to allow or Escape to cancel.`
          : voice?.narrateReply?.(entry.v, narrationSelect.value) ?? voice?.summarizeReply?.(entry.v);
        if (sum && window.TradingAlerts && window.TradingAlerts.say) {
          window.TradingAlerts.say(sum, { category: "analyst", title: "AI analyst", pop: false, narrate: true });
        }
      } catch (e) {
        entry = { type: "err", text: e.message || String(e), retry: text, ts: new Date().toISOString() };
      } finally {
        setBusy(false);
      }
      chat.entries = readChat(chat.id)?.entries ?? chat.entries;
      const at = chat.entries.findIndex((m) => m.id === id);
      if (at >= 0) chat.entries.splice(at, 1, entry); else chat.entries.push(entry);
      chat.updatedAt = entry.ts;
      save(); renderAll();
      if (state.mode !== "full") tab.dataset.unread = "1";
    }

    // A request interrupted by navigating away leaves a "pending" entry: offer to ask again.
    function recoverPending() {
      chat.entries = chat.entries.map((m) => m.type === "pending" && Date.now() - Date.parse(m.ts) > 3 * 60 * 1000
        ? { type: "err", text: "That answer was interrupted (the page changed).", retry: m.text, ts: m.ts } : m);
    }

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
        input.value = histAt >= history.length ? (histAt = -1, chat.draft || "") : history[histAt];
        autosize();
      } else if (e.key === "Escape") {
        e.preventDefault(); setMode(state.mode === "full" ? "prompt" : "tab");
        if (state.mode === "tab") tab.focus();
      }
    });
    input.addEventListener("input", () => {
      histAt = -1;
      chat.entries = readChat(chat.id)?.entries ?? chat.entries;
      chat.draft = input.value;
      autosize(); save();
    });
    root.querySelectorAll(".tmc-chip").forEach((b) => b.addEventListener("click", () => setFilter(b.dataset.filter)));
    q(".tmc-fold").addEventListener("click", () => setMode("prompt"));
    logBtn.addEventListener("click", () => setMode(state.mode === "full" ? "prompt" : "full"));
    q(".tmc-hide").addEventListener("click", () => { setMode("tab"); tab.focus(); });
    tab.addEventListener("click", () => setMode("full", true));
    newChat.addEventListener("click", () => openChat(freshChat()));
    chatSelect.addEventListener("change", () => openChat(readChat(chatSelect.value)));
    q(".tmc-mute").addEventListener("click", () => window.TradingAlerts && window.TradingAlerts.toggleSettings && window.TradingAlerts.toggleSettings());
    narrationSelect.addEventListener("change", () => window.TradingAlerts?.setAnalystNarration?.(narrationSelect.value));
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
      if (e.key === LOG_LAYOUT && !drag) {
        layout = loadLogLayout();
        applyLayout(atBottom());
        return;
      }
      if (!e.key?.startsWith(CHAT_PREFIX)) return;
      if (!busy && e.key === CHAT_PREFIX + chat.id) {
        const saved = readChat(chat.id);
        if (saved) chat = document.activeElement === input ? { ...saved, draft: chat.draft } : saved;
        input.value = chat.draft || "";
        autosize();
        renderAll();
      }
      renderChats();
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
      if (ta.onAlert) ta.onAlert((a, opts) => {
        if (!wantAlert(a, opts) || alerts.some((entry) => entry.id === "a:" + a.id)) return;
        const entry = fromAlert(a), full = alerts.length >= MAX_ENTRIES;
        alerts = [...alerts, entry].slice(-MAX_ENTRIES);
        writeLocal(FEED, alerts);
        if (state.filter !== "analyst") {
          const stick = atBottom();
          log.querySelector(".tmc-empty")?.remove();
          if (full) log.querySelector(".tmc-e[data-alert]")?.remove();
          log.appendChild(renderEntry(entry));
          scrollEnd(stick);
        }
      });
      return true;
    };
    if (!hook()) { let tries = 0; const t = setInterval(() => { if (hook() || ++tries > 40) clearInterval(t); }, 250); }
    fetch("/api/alerts/recent?limit=60", { cache: "no-store" }).then((r) => r.ok ? r.json() : { alerts: [] }).then((j) => {
      const known = new Set(alerts.map((e) => e.id).filter(Boolean));
      const fresh = (j.alerts || []).filter((a) => a && a.id && !known.has("a:" + a.id) && !(a.category === "analyst" && String(a.id).startsWith("local-"))).map(fromAlert);
      if (!fresh.length) return;
      alerts = alerts.concat(fresh).sort((x, y) => Date.parse(x.ts) - Date.parse(y.ts)).slice(-MAX_ENTRIES);
      writeLocal(FEED, alerts);
      if (state.filter !== "analyst") renderAll();
    }).catch(() => {});

    root.dataset.filter = state.filter;
    root.querySelectorAll(".tmc-chip").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.filter === state.filter)));
    recoverPending();
    expireConfirms(true);
    save();
    renderChats();
    renderAll();
    autosize();
    setMode(state.mode, false);
    applyLayout(true);
    window.TradingAnalystChat = { submit(text, model) {
      if (busy) return false;
      setMode("full", true);
      void send(text, model);
      return true;
    } };
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
})();
