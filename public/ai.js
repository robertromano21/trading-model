// AI Analyst page: command prompt, model picker, provider-key entry and verdicts.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const consoleEl = $("console"), history = [];
  let historyIndex = 0, models = null;

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function markdown(text) { // small, safe Markdown subset (escape first, then format)
    const lines = esc(text).split(/\r?\n/), out = [];
    let list = false;
    for (const line of lines) {
      const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
      const h = line.match(/^(#{1,4})\s+(.*)$/), li = line.match(/^\s*[-*]\s+(.*)$/);
      if (li) { if (!list) { out.push("<ul>"); list = true; } out.push("<li>" + inline(li[1]) + "</li>"); continue; }
      if (list) { out.push("</ul>"); list = false; }
      if (h) out.push(`<h${Math.min(4, h[1].length + 2)}>${inline(h[2])}</h${Math.min(4, h[1].length + 2)}>`);
      else if (line.trim()) out.push("<p>" + inline(line) + "</p>");
    }
    if (list) out.push("</ul>");
    return out.join("");
  }
  async function api(path, body) {
    const r = await fetch("/api/ai/" + path, body === undefined ? { cache: "no-store" } :
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const v = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(v.error || "Request failed (" + r.status + ")");
    return v;
  }
  function entry(question) {
    const div = document.createElement("div");
    div.className = "entry";
    div.innerHTML = `<div class="q">&gt; ${esc(question)}</div><div class="body"><em>Working… gathering live quote, history, filings and news.</em></div>`;
    consoleEl.append(div); consoleEl.scrollTop = consoleEl.scrollHeight;
    return div.querySelector(".body");
  }
  function renderAnswer(v) {
    const badge = `<span class="badge ${esc(v.stance)}">${esc(v.stance)}</span> <span class="badge ${v.source === "LLM" ? "" : "FALLBACK"}">${esc(v.source)}${v.model ? " · " + esc(v.model) : ""}</span>`;
    const levels = (v.levels || []).map((l) => `<li>${esc(l.label)}: ${esc(l.price)}</li>`).join("");
    const cites = (v.citations || []).map((c) => `<li>${esc(c.source)} (${esc(c.asOf)}): ${esc(c.detail)}</li>`).join("");
    const asOf = (v.dataAsOf || []).map((d) => `${esc(d.ticker)}: bar ${esc(d.lastBar)}, price ${esc(d.price)}${d.gaps?.length ? ", gaps: " + esc(d.gaps.join("; ")) : ""}`).join("<br>");
    return `<div class="meta">${badge} conviction ${esc(v.conviction)}</div>${markdown(v.answer)}` +
      (levels ? `<h4>Levels</h4><ul>${levels}</ul>` : "") + (cites ? `<details><summary>Sources</summary><ul>${cites}</ul></details>` : "") +
      (v.followUps?.length ? `<div class="meta">Try: ${v.followUps.map(esc).join(" · ")}</div>` : "") + `<div class="meta">${asOf}</div>`;
  }
  function renderDecisions(list) {
    return "<pre>" + esc(list.map((d) => `${d.ticker}: ${d.usable ? d.flag + " " + d.action + ", size x" + d.sizeScale + ", band " + d.entryBand?.join("–") + ", stop " + d.stop : "no change (" + d.reason + ")"}`).join("\n")) + "</pre>";
  }
  async function pollJob(id, body) {
    for (;;) {
      const j = await api("flag/" + id);
      body.innerHTML = `<em>Flagging… ${j.progress ? j.progress.done + "/" + j.progress.total + " (" + esc(j.progress.ticker) + ")" : ""}</em>`;
      if (j.status !== "running") {
        body.innerHTML = j.status === "complete"
          ? `<div class="meta">Wrote ${j.result.exported.count} model verdicts to ${esc(j.result.exported.path)}</div>` +
            j.result.verdicts.map((v) => `<div><span class="badge ${esc(v.flag)}">${esc(v.flag)}</span> <strong>${esc(v.ticker)}</strong> ${esc(v.action || "")} — ${esc(v.summary || "")}</div>`).join("")
          : `<em>Failed: ${esc(j.error)}</em>`;
        loadVerdicts(); return;
      }
      await new Promise((r) => setTimeout(r, 2000));
    }
  }
  async function send(text) {
    const body = entry(text);
    try {
      const v = await api("command", { text, model: $("model-select").value || undefined });
      if (v.kind === "answer") body.innerHTML = renderAnswer(v);
      else if (v.kind === "help") body.innerHTML = `<pre>${esc(v.text)}</pre>`;
      else if (v.kind === "decision") body.innerHTML = renderDecisions(v.decisions);
      else if (v.kind === "job") await pollJob(v.job.id, body);
      else body.innerHTML = `<pre>${esc(JSON.stringify(v.research || v.verdicts || v, null, 2))}</pre>`;
    } catch (e) { body.innerHTML = `<em>${esc(e.message)}</em>`; }
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }
  async function loadModels() {
    models = await api("models");
    const sel = $("model-select");
    sel.replaceChildren(...models.all.map((m) => Object.assign(document.createElement("option"),
      { value: m.id, textContent: m.name + (m.status === "ready" ? "" : " (needs key)"), selected: m.id === models.current.id })));
    $("model-status").textContent = models.current.ready ? "Ready: " + models.current.name : "No key for " + models.current.name + " — answers fall back to a data summary.";
    $("key-state").innerHTML = Object.entries(models.keys).map(([p, s]) => `<li>${esc(p)}: ${s === "env" ? "environment key" : s === "file" ? "saved key" : "no key"}</li>`).join("");
  }
  async function loadVerdicts() {
    const { verdicts } = await api("verdicts");
    $("verdicts").querySelector("tbody").innerHTML = verdicts.map((v) => `<tr><td>${esc(v.ticker)}</td><td><span class="badge ${esc(v.flag)}">${esc(v.flag)}</span></td><td>${esc(v.action)}</td><td>${esc(v.conviction)}</td><td>${esc(v.overextension)}</td><td>${v.entryLow ?? "–"}–${v.entryHigh ?? "–"}</td><td>${v.stop ?? "–"}</td><td>${esc(v.diversification)}</td><td><span class="badge ${v.source === "LLM" ? "" : "FALLBACK"}">${esc(v.source)}</span></td><td>${esc(new Date(v.createdAt).toLocaleString())}</td></tr>`).join("");
  }
  $("prompt-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("prompt").value.trim(); if (!text) return;
    history.push(text); historyIndex = history.length; $("prompt").value = ""; send(text);
  });
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "ArrowUp" && historyIndex > 0) { $("prompt").value = history[--historyIndex]; e.preventDefault(); }
    if (e.key === "ArrowDown") { historyIndex = Math.min(history.length, historyIndex + 1); $("prompt").value = history[historyIndex] || ""; e.preventDefault(); }
  });
  $("model-select").addEventListener("change", async (e) => {
    const m = models.all.find((x) => x.id === e.target.value);
    if (m && m.status !== "ready") { $("key-provider").value = m.provider; $("key-dialog").showModal(); }
    try { await api("models", { id: e.target.value }); } catch (err) { $("model-status").textContent = err.message; }
    loadModels();
  });
  $("key-button").addEventListener("click", () => $("key-dialog").showModal());
  $("key-save").addEventListener("click", async () => {
    const input = $("key-input"), msg = $("key-msg"), provider = $("key-provider").value;
    $("key-save").disabled = true; msg.textContent = "Saving…";
    try {
      const persistOnly = models?.keys?.[provider] === "env";
      await api("provider-key", { provider, key: input.value, persistOnly });
      msg.textContent = "Saved."; await loadModels();
    } catch (e) { msg.textContent = e.message; }
    finally { input.value = ""; $("key-save").disabled = false; }
  });
  $("flag-button").addEventListener("click", async () => {
    const body = entry("/flag (paper-trader targets)");
    try { const j = await api("flag", { tickers: [] }); await pollJob(j.id, body); } catch (e) { body.innerHTML = esc(e.message); }
  });
  loadModels().catch((e) => { $("model-status").textContent = e.message; });
  loadVerdicts().catch(() => {});
  $("prompt").focus();
})();
