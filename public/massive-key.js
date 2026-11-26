// Massive API key card (market data: daily bars, news, dividends). Masked input, saved server-side to .env.local as
// MASSIVE_API_KEY; only "set / not set" ever comes back. Mounts into [data-massive-key] elements and the setup dialog.
async function call(url, body) {
  const r = await fetch(url, body === undefined ? { cache: "no-store" } : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

export function mountMassiveKey(el) {
  if (!el || el.dataset.massiveMounted) return;
  el.dataset.massiveMounted = "1";
  el.innerHTML = `<fieldset class="massive-key"><legend>Massive API key (market data)</legend>
    <p class="muted small">From your account at <a href="https://massive.com/dashboard" target="_blank" rel="noopener">massive.com</a>. Saved to <code>.env.local</code> on this computer as <code>MASSIVE_API_KEY</code>; it is never sent back to the browser.</p>
    <p>Status: <b class="massive-status">checking…</b></p>
    <label>New key <input type="password" class="massive-input" autocomplete="off" spellcheck="false" placeholder="paste key" size="36" aria-label="Massive API key"></label>
    <button type="button" class="massive-save">Save key</button>
    <button type="button" class="massive-test">Test saved key</button>
    <p class="massive-msg small muted" aria-live="polite"></p></fieldset>`;
  const q = (s) => el.querySelector(s);
  const show = (st) => { q(".massive-status").textContent = st.set ? "set" : "not set"; q(".massive-status").className = `massive-status ${st.set ? "good" : "warn"}`; };
  const msg = (t, bad) => { q(".massive-msg").textContent = t; q(".massive-msg").className = `massive-msg small ${bad ? "bad" : "muted"}`; };
  call("/api/keys/massive").then(show).catch((e) => { q(".massive-status").textContent = `unknown (${e.message}; restart the dashboard once to load this route)`; });
  q(".massive-save").addEventListener("click", async () => {
    const key = q(".massive-input").value.trim();
    if (!key) return msg("Paste a key first.", true);
    msg("Checking with Massive…");
    try { const r = await call("/api/keys/massive", { key }); q(".massive-input").value = ""; show(r); msg(r.message || "Saved."); }
    catch (e) { q(".massive-input").value = ""; msg(e.message, true); }
  });
  q(".massive-test").addEventListener("click", async () => {
    msg("Testing…");
    try { const r = await call("/api/keys/massive/test", {}); show(r); msg(r.message, !r.ok); } catch (e) { msg(e.message, true); }
  });
}

function scan() {
  document.querySelectorAll("[data-massive-key]").forEach(mountMassiveKey);
  document.querySelectorAll("dialog.suite-dialog").forEach((d) => {
    if (d.querySelector(".massive-key-slot")) return;
    const slot = document.createElement("div");
    slot.className = "massive-key-slot";
    (d.querySelector("form") || d).appendChild(slot);
    mountMassiveKey(slot);
  });
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", scan); else scan();
new MutationObserver(() => scan()).observe(document.documentElement, { childList: true, subtree: true });
