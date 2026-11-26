// FRED API key card: masked input, saved server-side to .env.local, only "set / not set" ever comes back.
// Mounts into [data-fred-key] elements and adds itself to the Schwab setup dialog (nav.js) when that opens.
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function call(url, body) {
  const r = await fetch(url, body === undefined ? { cache: "no-store" } : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

export function mountFredKey(el) {
  if (!el || el.dataset.fredMounted) return;
  el.dataset.fredMounted = "1";
  el.innerHTML = `<fieldset class="fred-key"><legend>FRED API key (macro data &amp; nowcasts)</legend>
    <p class="muted small">Free from <a href="https://fredaccount.stlouisfed.org/apikeys" target="_blank" rel="noopener">fredaccount.stlouisfed.org</a>. Saved to <code>.env.local</code> on this computer as <code>FRED_API_KEY</code>; it is never sent back to the browser.</p>
    <p>Status: <b class="fred-status">checking…</b></p>
    <label>New key <input type="password" class="fred-input" autocomplete="off" spellcheck="false" placeholder="32 letters and digits" size="36"></label>
    <button type="button" class="fred-save">Save key</button>
    <button type="button" class="fred-test">Test saved key</button>
    <p class="fred-msg small muted" aria-live="polite"></p></fieldset>`;
  const q = (s) => el.querySelector(s);
  const show = (st) => { q(".fred-status").textContent = st.set ? "set" : "not set"; q(".fred-status").className = `fred-status ${st.set ? "good" : "warn"}`; };
  const msg = (t, bad) => { q(".fred-msg").textContent = t; q(".fred-msg").className = `fred-msg small ${bad ? "bad" : "muted"}`; };
  call("/api/keys/fred").then(show).catch((e) => { q(".fred-status").textContent = `unknown (${e.message}; restart the dashboard once to load this route)`; });
  q(".fred-save").addEventListener("click", async () => {
    const key = q(".fred-input").value.trim();
    if (!key) return msg("Paste a key first.", true);
    msg("Checking with FRED…");
    try { const r = await call("/api/keys/fred", { key }); q(".fred-input").value = ""; show(r); msg(r.message || "Saved."); }
    catch (e) { q(".fred-input").value = ""; msg(e.message, true); }
  });
  q(".fred-test").addEventListener("click", async () => {
    msg("Testing…");
    try { const r = await call("/api/keys/fred/test", {}); show(r); msg(r.message, !r.ok); } catch (e) { msg(e.message, true); }
  });
}

function scan() {
  document.querySelectorAll("[data-fred-key]").forEach(mountFredKey);
  // Schwab setup dialog from nav.js: append a FRED section without touching nav.js
  document.querySelectorAll("dialog.suite-dialog").forEach((d) => {
    if (d.querySelector(".fred-key-slot")) return;
    const slot = document.createElement("div");
    slot.className = "fred-key-slot";
    const anchor = d.querySelector("form") || d;
    anchor.appendChild(slot);
    mountFredKey(slot);
  });
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", scan); else scan();
new MutationObserver(() => scan()).observe(document.documentElement, { childList: true, subtree: true });
export { esc };
