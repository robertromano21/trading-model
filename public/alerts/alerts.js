/* Trading-model voice alerts. Include on any page:
 *   <script src="/alerts/alerts.js" defer></script>
 * Popups + SID-chip speech (male/female) with per-category mute. Settings persist in localStorage and are
 * mirrored to the server (/api/alerts/prefs). Voice is ON by default; browsers block audio until the first
 * click or key press, so speech waits (queued) until then and a small "click to enable voice" hint shows.
 * Speech falls back SID score -> server WAV -> browser speechSynthesis; popups always work.
 * Only one tab speaks a given alert (shared spoken-id list in localStorage).
 */
(function (root) {
  "use strict";
  if (root.TradingAlerts && root.TradingAlerts.__booted) return;

  const STORAGE = "tm.alerts.settings";
  const SPOKEN_KEY = "tm.alerts.spoken";
  const SEEN_KEY = "tm.alerts.seen";
  const TYPE_LABEL = {
    decision: "Paper decision", fill: "Fill", stop_hit: "Stop hit",
    circuit_breaker: "Circuit breaker", flag_change: "AI flag", overextension: "Overextension",
    price_threshold: "Price threshold", schwab_auth: "Schwab auth", error: "Error",
    drift: "Backtest drift", macro_release: "Macro release", black_swan: "Black swan",
    dilution: "Dilution", regime: "Regime change", macro_nowcast: "Macro nowcast", analyst: "AI analyst", system: "System",
  };
  // Same ids/order as server/alerts/categories.js.
  const CATS = [
    ["analyst", "AI analyst replies"],
    ["trade_main", "Trades: main paper"],
    ["trade_swarm", "Trades: swarm shadow"],
    ["trade_crypto", "Trades: crypto paper"],
    ["trade_options", "Trades: options paper"],
    ["signals", "AI flags, overextension, dilution, drift"],
    ["black_swan", "Black swan"],
    ["dollar_shock", "Dollar shock"],
    ["regime", "Regime changes"],
    ["macro", "Macro releases + nowcast"],
    ["system", "System / server"],
  ];
  const CAT_IDS = CATS.map((c) => c[0]);
  const DEFAULTS = {
    gender: "female",
    voiceMode: "speech",   // legacy; masterMute is the switch shown in the panel
    masterMute: false,     // voice ON by default
    analystNarration: "verbose",
    engine: "sid",         // "sid" (SID chip voice) or "browser" (speechSynthesis)
    volume: 0.85,
    rate: 1,
    clock: "2x",
    popups: true,          // visual alerts master
    speakCats: Object.fromEntries(CAT_IDS.map((c) => [c, true])),
    popCats: Object.fromEntries(CAT_IDS.map((c) => [c, true])),
    quietHoursEnabled: false,
    quietStart: "21:00",
    quietEnd: "07:00",
  };

  function normalizeSettings(raw) {
    raw = raw && typeof raw === "object" ? raw : {};
    const s = { ...DEFAULTS, ...raw };
    s.speakCats = { ...DEFAULTS.speakCats, ...(raw.speakCats || {}) };
    s.popCats = { ...DEFAULTS.popCats, ...(raw.popCats || {}) };
    s.volume = Math.max(0, Math.min(1.5, Number.isFinite(Number(s.volume)) ? Number(s.volume) : DEFAULTS.volume));
    s.rate = Math.max(0.7, Math.min(1.4, Number(s.rate) || 1));
    if (!["male", "female"].includes(s.gender)) s.gender = "female";
    if (!["speech", "off"].includes(s.voiceMode)) s.voiceMode = "speech";
    if (!["verbose", "light"].includes(s.analystNarration)) s.analystNarration = "verbose";
    if (!["sid", "browser"].includes(s.engine)) s.engine = "sid";
    if (!["1x", "2x", "4x"].includes(s.clock)) s.clock = "2x";
    s.masterMute = s.masterMute === true;
    s.popups = s.popups !== false;
    delete s.speakTypes; delete s.popTypes; // superseded by per-category switches
    return s;
  }
  function loadSettings() {
    try {
      const stored = localStorage.getItem(STORAGE);
      const s = normalizeSettings(stored ? JSON.parse(stored) : {});
      s.__fresh = !stored;
      return s;
    } catch { return normalizeSettings({}); }
  }
  let prefsTimer = null;
  const listeners = new Set();
  const alertListeners = new Set(); // every new alert (e.g. the command console log)
  function saveSettings() {
    try { const { __fresh, ...keep } = settings; localStorage.setItem(STORAGE, JSON.stringify(keep)); } catch {}
    settings.__fresh = false;
    clearTimeout(prefsTimer);
    prefsTimer = setTimeout(() => {
      const { __fresh, ...prefs } = settings;
      fetch("/api/alerts/prefs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prefs }) }).catch(() => {});
    }, 600);
    listeners.forEach((fn) => { try { fn(publicSettings()); } catch {} });
  }
  function publicSettings() {
    return { ...settings, speakCats: { ...settings.speakCats }, popCats: { ...settings.popCats } };
  }

  let settings = loadSettings();
  // Another tab changed settings: follow it.
  root.addEventListener && root.addEventListener("storage", (e) => {
    if (e.key !== STORAGE || !e.newValue) return;
    try { settings = normalizeSettings(JSON.parse(e.newValue)); renderPanel(); listeners.forEach((fn) => fn(publicSettings())); } catch {}
  });

  function categoryOf(alert) {
    const c = String(alert.category || (alert.meta && alert.meta.category) || "");
    if (CAT_IDS.includes(c)) return c;
    const t = alert.type, m = alert.meta || {};
    if (t === "black_swan") return /dollar/i.test(String(m.eventType || "")) ? "dollar_shock" : "black_swan";
    if (t === "analyst") return "analyst";
    if (t === "regime") return "regime";
    if (t === "macro_release" || t === "macro_nowcast") return "macro";
    if (["flag_change", "overextension", "price_threshold", "dilution", "drift"].includes(t)) return "signals";
    if (["schwab_auth", "error", "system"].includes(t)) return "system";
    const a = String(m.account || "").toLowerCase();
    if (/crypto/.test(a) || /^2\d\d$/.test(a)) return "trade_crypto";
    if (/option/.test(a)) return "trade_options";
    if (/swarm|shadow/.test(a) || /^1\d\d$/.test(a)) return "trade_swarm";
    return "trade_main";
  }

  function parseHm(value, fallback) {
    const m = String(value ?? "").trim().match(/^(\d{1,2}):(\d{2})$/);
    if (!m) return fallback;
    const h = Number(m[1]), min = Number(m[2]);
    if (h > 23 || min > 59) return fallback;
    return h * 60 + min;
  }
  function inQuietHours(now) {
    if (!settings.quietHoursEnabled) return false;
    const start = parseHm(settings.quietStart, 21 * 60);
    const end = parseHm(settings.quietEnd, 7 * 60);
    const cur = now.getHours() * 60 + now.getMinutes();
    if (start === end) return true;
    return start < end ? cur >= start && cur < end : cur >= start || cur < end;
  }
  function voiceOn() { return !settings.masterMute && (settings.voiceMode || "speech") !== "off"; }
  function shouldSpeak(alert, now) {
    if (!voiceOn()) return false;
    if (inQuietHours(now || new Date())) return false;
    if (settings.speakCats[categoryOf(alert)] === false) return false;
    // Drift: popup always (if shown); speech only for severe/critical.
    if (alert.type === "drift") return alert.severity === "critical";
    // Scheduled macro prints: speak only large surprises; nowcast alerts only fire on meaningful moves.
    if (alert.type === "macro_release") return alert.severity === "critical" || !!(alert.meta && alert.meta.big);
    return true;
  }
  function shouldPop(alert) {
    if (settings.popups === false) return false;
    return settings.popCats[categoryOf(alert)] !== false;
  }

  function createQueue(speak, maxPending) {
    const pending = [];
    let current = null, running = false;
    maxPending = maxPending || 20;
    async function pump() {
      if (running) return;
      running = true;
      try {
        while (pending.length) {
          const job = pending.shift();
          current = job;
          try { if (job.speak && speak) await speak(job.alert, job.card); }
          catch (e) { console.warn("[alerts] speech skipped", e && e.message); }
          finally { current = null; job.resolve && job.resolve(); }
        }
      } finally { running = false; }
    }
    return {
      enqueue(alert, opts) {
        return new Promise((resolve) => {
          if (pending.length >= maxPending) pending.shift();
          pending.push({ alert, speak: !!(opts && opts.speak), card: opts && opts.card, resolve });
          pump();
        });
      },
      get busy() { return running; },
    };
  }


  function inject() {
    if (document.getElementById("tm-alerts-root")) return;
    if (!document.querySelector('link[href^="/alerts/alerts.css"]')) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.id = "tm-alerts-style";
      link.href = "/alerts/alerts.css" + (root.__tmAssetVersion ? "?v=" + root.__tmAssetVersion : "");
      document.head.appendChild(link);
    }
    const rootEl = document.createElement("div");
    rootEl.id = "tm-alerts-root";
    rootEl.innerHTML = `
      <div id="tm-alerts-stack" aria-live="polite"></div>
      <div id="tm-alerts-sub" role="status"><div class="cap"></div><div class="words"></div></div>
      <div id="tm-alerts-panel" hidden role="dialog" aria-label="Alert and voice settings"></div>
      <button type="button" id="tm-alerts-unlock" title="Browsers block sound until you interact with the page">Click anywhere to enable voice</button>
      <button type="button" id="tm-alerts-fab" title="Alert and voice settings (mute per category)" aria-expanded="false">Alerts</button>
    `;
    document.body.appendChild(rootEl);
    renderPanel();
    document.getElementById("tm-alerts-fab").addEventListener("click", () => { unlockAudio(); togglePanel(); });
    document.getElementById("tm-alerts-unlock").addEventListener("click", () => unlockAudio());
    document.getElementById("tm-alerts-panel").addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.stopPropagation(); togglePanel(false); document.getElementById("tm-alerts-fab").focus(); }
    });
  }

  function togglePanel(force) {
    const panel = document.getElementById("tm-alerts-panel");
    if (!panel) return;
    const open = typeof force === "boolean" ? force : !panel.classList.contains("open");
    panel.classList.toggle("open", open);
    panel.hidden = !open;
    const fab = document.getElementById("tm-alerts-fab");
    if (fab) fab.setAttribute("aria-expanded", String(open));
    if (open) {
      unread = 0; bumpFab();
      const first = panel.querySelector("input,select,button");
      if (first) setTimeout(() => first.focus(), 0);
    }
  }

  function renderPanel() {
    const panel = document.getElementById("tm-alerts-panel");
    if (!panel) return;
    const catRows = CATS.map(([id, label]) => `
      <tr><td>${esc(label)}</td>
        <td class="c"><input type="checkbox" data-speak-cat="${id}" aria-label="Speak ${esc(label)}" ${settings.speakCats[id] !== false ? "checked" : ""}></td>
        <td class="c"><input type="checkbox" data-pop-cat="${id}" aria-label="Show ${esc(label)}" ${settings.popCats[id] !== false ? "checked" : ""}></td></tr>`).join("");
    panel.innerHTML = `
      <button type="button" class="tm-close" aria-label="Close settings">X</button>
      <h2>Voice and alerts</h2>
      <label>Voice (master) <input type="checkbox" data-k="voiceOn" ${voiceOn() ? "checked" : ""}></label>
      <label>Visual alerts (popups) <input type="checkbox" data-k="popups" ${settings.popups !== false ? "checked" : ""}></label>
      <label>Voice engine
        <select data-k="engine" title="SID chip voice uses the local speech models; browser voice is the fallback">
          <option value="sid"${settings.engine === "sid" ? " selected" : ""}>SID chip voice</option>
          <option value="browser"${settings.engine === "browser" ? " selected" : ""}>Browser voice</option>
        </select>
      </label>
      <label>Analyst voice
        <select data-k="gender">
          <option value="female"${settings.gender === "female" ? " selected" : ""}>Female</option>
          <option value="male"${settings.gender === "male" ? " selected" : ""}>Male</option>
        </select>
      </label>
      <label>Volume <input type="range" min="0" max="1.5" step="0.05" data-k="volume" value="${settings.volume}"></label>
      <label>Rate <input type="range" min="0.7" max="1.4" step="0.05" data-k="rate" value="${settings.rate}"></label>
      <label>SID clock
        <select data-k="clock" title="1x is a stock SID clock; 2x is the default (clearest speech); 4x is the highest clock.">
          <option value="1x"${settings.clock === "1x" ? " selected" : ""}>1x</option>
          <option value="2x"${settings.clock === "2x" ? " selected" : ""}>2x</option>
          <option value="4x"${settings.clock === "4x" ? " selected" : ""}>4x</option>
        </select>
      </label>
      <label>Quiet hours (speech only) <input type="checkbox" data-k="quietHoursEnabled" ${settings.quietHoursEnabled ? "checked" : ""}></label>
      <label>Quiet from <input type="time" data-k="quietStart" value="${settings.quietStart}"></label>
      <label>Quiet until <input type="time" data-k="quietEnd" value="${settings.quietEnd}"></label>
      <h2 class="tm-gap">Mute by category</h2>
      <table><thead><tr><th>Category</th><th>Speak</th><th>Show</th></tr></thead><tbody>${catRows}</tbody></table>
      <div class="tm-actions">
        <button type="button" data-all="on">Unmute all</button>
        <button type="button" data-all="off">Mute all speech</button>
        <button type="button" data-demo="female">Preview female</button>
        <button type="button" data-demo="male">Preview male</button>
      </div>
      <div id="tm-alerts-status"></div>
    `;
    panel.querySelector(".tm-close").addEventListener("click", () => { togglePanel(false); document.getElementById("tm-alerts-fab").focus(); });
    panel.querySelectorAll("[data-k]").forEach((el) => {
      el.addEventListener("change", () => {
        const k = el.getAttribute("data-k");
        if (k === "voiceOn") {
          settings.masterMute = !el.checked;
          if (el.checked) settings.voiceMode = "speech";
          if (!el.checked) stopSpeaking();
        } else {
          settings[k] = el.type === "checkbox" ? el.checked : (el.type === "range" ? Number(el.value) : el.value);
        }
        saveSettings();
        unlockAudio();
      });
    });
    panel.querySelectorAll("[data-speak-cat]").forEach((el) => el.addEventListener("change", () => {
      settings.speakCats[el.getAttribute("data-speak-cat")] = el.checked; saveSettings();
    }));
    panel.querySelectorAll("[data-pop-cat]").forEach((el) => el.addEventListener("change", () => {
      settings.popCats[el.getAttribute("data-pop-cat")] = el.checked; saveSettings();
    }));
    panel.querySelectorAll("[data-all]").forEach((el) => el.addEventListener("click", () => {
      const on = el.getAttribute("data-all") === "on";
      CAT_IDS.forEach((c) => { settings.speakCats[c] = on; if (on) settings.popCats[c] = true; });
      if (on) { settings.masterMute = false; settings.voiceMode = "speech"; settings.popups = true; }
      saveSettings(); renderPanel();
    }));
    panel.querySelectorAll("[data-demo]").forEach((el) => el.addEventListener("click", () => {
      settings.gender = el.getAttribute("data-demo"); saveSettings();
      const gender = settings.gender;
      ingest({
        id: "demo-" + gender + "-" + Date.now(),
        ts: new Date().toISOString(),
        type: "flag_change", severity: "warning", ticker: "NVDA", category: "signals",
        title: "AI flag",
        text: "Nvidia is on a yellow flag today. The downtrend is intact, so I'd wait for a better entry.",
        speech: "Nvidia is on a yellow flag today. The downtrend is intact, so I'd wait for a better entry.",
        href: "/ticker.html?symbol=NVDA",
        meta: { flag: "YELLOW", demo: true, gender },
      }, { forceSpeak: true, local: true });
    }));
    refreshStatus();
  }

  async function refreshStatus() {
    const el = document.getElementById("tm-alerts-status");
    if (!el) return;
    try {
      const s = await fetch("/api/alerts/status").then((r) => r.json());
      const v = s.voice || {};
      const models = v.models || {};
      el.textContent = (v.available
        ? `SID voice ready (${models.female ? "female" : "no female"}${models.male ? ", male" : ", no male"}; clock ${v.clock || settings.clock}).`
        : "SID voice models are not loaded, so speech uses the browser voice instead. Popups still work.")
        + (audioLocked() ? " Sound is blocked until you click or press a key on the page." : "");
      sidAvailable = !!v.available;
    } catch {
      el.textContent = "Alert server is not reachable.";
    }
  }

  function fmtTime(iso) {
    try {
      return new Date(iso).toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" }) + " ET";
    } catch { return ""; }
  }

  function showCard(alert) {
    const stack = document.getElementById("tm-alerts-stack");
    const card = document.createElement("div");
    card.className = "tm-alert-card";
    card.dataset.sev = alert.severity || "info";
    card.dataset.id = alert.id;
    card.setAttribute("role", "status");
    const ticker = alert.ticker ? `<span class="tm-alert-ticker">${esc(alert.ticker)}</span>` : "";
    card.innerHTML = `
      <button type="button" class="tm-alert-close" aria-label="Dismiss">X</button>
      <div class="tm-alert-meta">${ticker}<span>${esc(alert.title || TYPE_LABEL[alert.type] || alert.type)}</span><span class="tm-alert-time">${esc(fmtTime(alert.ts))}</span></div>
      <div class="tm-alert-words tm-pending">\u2026</div>
    `;
    card.querySelector(".tm-alert-close").addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); dismiss(card); });
    card.addEventListener("click", (e) => {
      if (e.target.closest(".tm-alert-close")) return;
      const href = alert.href || (alert.ticker ? "/ticker.html?symbol=" + encodeURIComponent(alert.ticker) : "");
      if (href) location.href = href;
    });
    stack.prepend(card);
    while (stack.children.length > 6) stack.lastElementChild.remove();
    return card;
  }

  function revealCard(card, text) {
    if (!card) return;
    const words = card.querySelector(".tm-alert-words");
    if (words) { words.textContent = text; words.classList.remove("tm-pending"); }
  }

  function dismiss(card) {
    if (!card || !card.parentNode) return;
    card.remove();
  }

  function esc(s) {
    return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  const sub = {
    show(speaker, pending) {
      const box = document.getElementById("tm-alerts-sub");
      if (!box) return;
      box.querySelector(".cap").textContent = speaker;
      box.querySelector(".words").textContent = pending ? "\u2026" : "";
      box.dataset.open = "1";
    },
    reveal(text) {
      const box = document.getElementById("tm-alerts-sub");
      if (!box || box.dataset.open !== "1") return;
      box.querySelector(".words").textContent = text;
    },
    hide() {
      const box = document.getElementById("tm-alerts-sub");
      if (box) box.dataset.open = "0";
    },
  };

  let audioCtx = null, voiceReady = null, workerSourceText = null, sidAvailable = true;
  let gestureSeen = false, unlockWaiters = [];

  function audioLocked() {
    if (!gestureSeen) return true;
    return !!(audioCtx && audioCtx.state !== "running");
  }
  function setUnlockHint(show) {
    const el = document.getElementById("tm-alerts-unlock");
    if (el) el.dataset.open = show && voiceOn() ? "1" : "0";
  }
  function unlockAudio() {
    try {
      if (!audioCtx) audioCtx = new (root.AudioContext || root.webkitAudioContext)();
      if (audioCtx.state === "suspended") audioCtx.resume().then(flushUnlock, () => {});
    } catch { /* popup-only fallback */ }
    return audioCtx;
  }
  function flushUnlock() {
    if (audioCtx && audioCtx.state !== "running") return;
    setUnlockHint(false);
    const waiters = unlockWaiters.splice(0);
    waiters.forEach((fn) => fn());
  }
  // First click / key press / touch anywhere unlocks audio (browsers require a user gesture).
  function onGesture() {
    gestureSeen = true;
    unlockAudio();
    if (!audioCtx || audioCtx.state === "running") flushUnlock();
    if (audioCtx && audioCtx.state === "running") {
      ["pointerdown", "keydown", "touchend"].forEach((t) => document.removeEventListener(t, onGesture, true));
    }
  }
  ["pointerdown", "keydown", "touchend"].forEach((t) => document.addEventListener(t, onGesture, { capture: true }));
  function waitForAudio() {
    if (!audioLocked()) return Promise.resolve();
    setUnlockHint(true);
    return new Promise((resolve) => unlockWaiters.push(resolve));
  }

  let currentStop = null;
  function stopSpeaking() {
    try { currentStop && currentStop(); } catch {}
    try { root.speechSynthesis && root.speechSynthesis.cancel(); } catch {}
  }

  function speakBrowser(text) {
    return new Promise((resolve, reject) => {
      const synth = root.speechSynthesis;
      if (!synth || typeof root.SpeechSynthesisUtterance !== "function") return reject(new Error("speechSynthesis unavailable"));
      const u = new root.SpeechSynthesisUtterance(text);
      u.volume = Math.min(1, settings.volume);
      u.rate = settings.rate;
      const voices = synth.getVoices ? synth.getVoices() : [];
      const want = settings.gender === "male" ? /male|david|guy|mark|ryan/i : /female|zira|aria|jenny|samantha/i;
      const v = voices.find((x) => /^en/i.test(x.lang) && want.test(x.name)) || voices.find((x) => /^en/i.test(x.lang));
      if (v) u.voice = v;
      const done = setTimeout(resolve, Math.min(30000, 3000 + text.length * 90));
      u.onend = () => { clearTimeout(done); resolve(); };
      u.onerror = (e) => { clearTimeout(done); (e && e.error === "interrupted" ? resolve : reject)(new Error("speech " + (e && e.error))); };
      currentStop = () => { synth.cancel(); resolve(); };
      synth.speak(u);
    });
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      if (root.TradingSidPlayer) return resolve();
      const s = document.createElement("script");
      s.src = src;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("sidPlayer missing"));
      document.head.appendChild(s);
    });
  }

  async function ensureVoice() {
    if (voiceReady) return voiceReady;
    voiceReady = (async () => {
      await loadScript("/voice/sidPlayer.js");
      const src = await fetch("/voice/sid-worker.js").then((r) => {
        if (!r.ok) throw new Error("SID worker " + r.status);
        return r.text();
      });
      workerSourceText = src;
      if (!root.TradingSidPlayer) throw new Error("TradingSidPlayer missing");
      return true;
    })().catch((error) => { voiceReady = null; throw error; });
    return voiceReady;
  }

  function speakerName() {
    return settings.gender === "male" ? "Analyst (male)" : "Analyst (female)";
  }
  function analystProfile() {
    return settings.gender === "male"
      ? { name: "Analyst (male)", kind: "humanoid", f0: 118, fs: 0.98, rate: 4.6 * settings.rate, range: 3.2, breath: 0.08, rasp: 0.05, warm: 0.15, bright: 1.0, vibAmt: 0.15, vibRate: 5.0, pause: 1.0 }
      : { name: "Analyst (female)", kind: "humanoid", f0: 205, fs: 1.04, rate: 4.6 * settings.rate, range: 3, breath: 0.06, rasp: 0, warm: 0.22, bright: 1.08, vibAmt: 0.2, vibRate: 5.2, pause: 0.95 };
  }

  const prefetchCache = new Map();
  function sentenceSplit(text) {
    return String(text || "").replace(/\s+/g, " ").trim().split(/(?<=[.!?]['\u2019)\]]*)\s+(?=\S)/).map((s) => s.trim()).filter(Boolean);
  }

  function captionChunks(text) {
    const max = 200;
    const chunks = [];
    for (let sentence of sentenceSplit(text)) {
      while (sentence.length > max) {
        const end = sentence.lastIndexOf(" ", max);
        const split = end > 0 ? end : max;
        chunks.push(sentence.slice(0, split));
        sentence = sentence.slice(split).trimStart();
      }
      if (sentence) chunks.push(sentence);
    }
    return chunks;
  }

  function fetchScore(text) {
    const key = settings.clock + "|" + settings.gender + "|" + settings.rate + "|" + text;
    if (prefetchCache.has(key)) return prefetchCache.get(key);
    const p = fetch("/voice/line.sidvoice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        speaker: speakerName(),
        text,
        profile: analystProfile(),
        clock: settings.clock,
        chip: "10582",
        timbre: "soft",
      }),
    }).then(async (r) => {
      if (!r.ok) {
        const detail = await r.json().catch(() => null);
        throw new Error("voice " + r.status + (detail && detail.error ? ": " + detail.error : ""));
      }
      const duration = Number(r.headers.get("X-Voice-Dur"));
      if (!Number.isFinite(duration) || duration < 0.05 || duration > 120) throw new Error("Invalid SID voice duration");
      const bytes = await r.arrayBuffer();
      return { bytes, duration };
    });
    p.catch(() => prefetchCache.delete(key));
    prefetchCache.set(key, p);
    while (prefetchCache.size > 16) prefetchCache.delete(prefetchCache.keys().next().value);
    return p;
  }

  function prefetch(text) {
    try { for (const s of sentenceSplit(text).slice(0, 6)) fetchScore(s).catch(() => {}); } catch {}
  }

  async function playWavFallback(text) {
    const ctx = unlockAudio();
    const r = await fetch("/api/alerts/speak.wav", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, gender: settings.gender, clock: settings.clock }),
    });
    if (!r.ok) throw new Error("wav " + r.status);
    const buf = await r.arrayBuffer();
    if (ctx) {
      const audio = await ctx.decodeAudioData(buf.slice(0));
      const src = ctx.createBufferSource();
      const g = ctx.createGain();
      g.gain.value = settings.volume;
      src.buffer = audio;
      src.connect(g); g.connect(ctx.destination);
      await new Promise((resolve, reject) => {
        src.onended = resolve;
        src.onerror = reject;
        src.start();
        setTimeout(resolve, (audio.duration + 0.3) * 1000);
      });
      return;
    }
    const blob = new Blob([buf], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    await new Promise((resolve, reject) => {
      const a = new Audio(url);
      a.volume = Math.min(1, settings.volume);
      a.onended = () => { URL.revokeObjectURL(url); resolve(); };
      a.onerror = (e) => { URL.revokeObjectURL(url); reject(e); };
      a.play().catch(reject);
    });
  }

  async function speakSid(alert, card) {
    const text = alert.speech || alert.text;
    const sents = sentenceSplit(text);
    if (!sents.length) return;
    if (audioLocked()) {
      revealCard(card, text);
      await waitForAudio();
      // Muted meanwhile, or too old to be worth saying now: skip.
      if (!voiceOn() || (!alert.__force && !shouldSpeak(alert, new Date()))) return;
      if (alert.ts && Date.now() - Date.parse(alert.ts) > 10 * 60 * 1000) return;
    }
    if (settings.engine === "browser" || !sidAvailable) {
      sub.show(alert.__speaker || ("AI Analyst (" + (settings.gender === "male" ? "male" : "female") + ")"), false);
      sub.reveal(text); revealCard(card, text);
      try { await speakBrowser(text); } finally { sub.hide(); currentStop = null; }
      return;
    }
    const speaker = alert.__speaker || ("AI Analyst (" + (settings.gender === "male" ? "male" : "female") + ")");
    sub.show(speaker, true);
    const reveal = () => { sub.reveal(text); revealCard(card, text); };
    try {
      await ensureVoice();
      const ctx = unlockAudio();
      if (!ctx) throw new Error("audio locked");
      if (ctx.state !== "running") await ctx.resume();
      const scores = sents.map((s) => fetchScore(s));
      scores.forEach((p) => p.catch(() => {}));
      const first = await scores[0];
      const g = ctx.createGain();
      g.gain.value = settings.volume;
      g.connect(ctx.destination);
      let end = ctx.currentTime + 0.03;
      let stopped = false;
      const play = async (score) => {
        const player = await root.TradingSidPlayer.open(score.bytes, {
          context: ctx, destination: g, source: workerSourceText, loop: false,
        });
        player.volume = Math.min(1, settings.volume);
        end = Math.max(end, ctx.currentTime + 0.03);
        if (alert.__narrate && !shouldSpeak(alert, new Date())) { player.dispose(); return; }
        let finish;
        const done = new Promise((resolve) => {
          const timer = setTimeout(resolve, (score.duration + 0.4) * 1000);
          finish = () => { clearTimeout(timer); resolve(); };
          player.onended = finish;
        });
        currentStop = () => { stopped = true; try { player.stop(); } finally { finish(); } };
        try { await player.play(end); end += player.duration; await done; }
        finally { finish(); currentStop = null; try { player.dispose(); } catch {} }
      };
      reveal();
      await play(first);
      for (let i = 1; i < scores.length && !stopped; i++) {
        const score = await scores[i];
        end = Math.max(end + 0.12, ctx.currentTime + 0.03);
        await play(score);
      }
    } catch (error) {
      try {
        reveal();
        await playWavFallback(text);
      } catch (err) {
        reveal();
        try { await speakBrowser(text); }
        catch {
          await new Promise((r) => setTimeout(r, Math.min(8000, 2000 + text.split(/\s+/).length * 280)));
          throw error;
        }
      }
    } finally {
      sub.hide();
      currentStop = null;
    }
  }

  async function speakAlert(alert, card) {
    if (!alert.__narrate) return speakSid(alert, card);
    const parts = captionChunks(alert.speech || alert.text);
    for (let index = 0; index < parts.length; index++) {
      if (!shouldSpeak(alert, new Date())) return;
      if (index + 1 < parts.length && !audioLocked() && settings.engine === "sid") prefetch(parts[index + 1]);
      await speakSid({ ...alert, speech: parts[index] }, index === 0 ? card : null);
    }
  }

  const queue = createQueue(speakAlert, 20);
  const seen = new Set(readList(sessionStorageSafe(), SEEN_KEY));
  let unread = 0;

  function sessionStorageSafe() { try { return root.sessionStorage; } catch { return null; } }
  function localStorageSafe() { try { return root.localStorage; } catch { return null; } }
  function readList(store, key) {
    try { const v = JSON.parse((store && store.getItem(key)) || "[]"); return Array.isArray(v) ? v : []; } catch { return []; }
  }
  function writeList(store, key, list, max) {
    try { store && store.setItem(key, JSON.stringify(list.slice(-max))); } catch {}
  }
  // Claim an alert for speech so only one open tab says it. Hidden tabs wait a moment so the visible tab wins.
  function claimSpeech(id) {
    const store = localStorageSafe();
    const list = readList(store, SPOKEN_KEY);
    if (list.includes(id)) return false;
    list.push(id);
    writeList(store, SPOKEN_KEY, list, 300);
    return true;
  }

  function bumpFab() {
    const fab = document.getElementById("tm-alerts-fab");
    if (fab) fab.dataset.count = String(unread);
  }

  function ingest(alert, opts) {
    if (!alert || !alert.id || seen.has(alert.id)) return;
    seen.add(alert.id);
    if (seen.size > 300) seen.delete(seen.values().next().value);
    writeList(sessionStorageSafe(), SEEN_KEY, [...seen], 300);
    for (const fn of alertListeners) { try { fn(alert, opts || {}); } catch (e) { console.warn("[alerts] listener", e); } }
    const now = new Date();
    const ageMs = alert.ts ? now - Date.parse(alert.ts) : 0;
    // The stream replays recent alerts on every page load: only show ones from the last 10 minutes,
    // and only speak ones from the last 3 minutes.
    if (!(opts && opts.local) && ageMs > 10 * 60 * 1000) return;
    const force = !!(opts && opts.forceSpeak);
    const pop = !(opts && opts.noPop) && shouldPop(alert);
    let speak = force || (shouldSpeak(alert, now) && (!!(opts && opts.local) || ageMs < 3 * 60 * 1000));
    let card = null;
    if (pop) {
      card = showCard(alert);
      if (!speak) revealCard(card, alert.speech || alert.text);
      unread += 1; bumpFab();
    }
    const finish = (doSpeak) => {
      if (doSpeak) {
        if (force) alert.__force = true;
        if (opts && opts.speaker) alert.__speaker = opts.speaker;
        if (!alert.__narrate && !audioLocked() && settings.engine === "sid") prefetch(alert.speech || alert.text);
        queue.enqueue(alert, { speak: true, card }).then(() => { if (card) setTimeout(() => dismiss(card), 8000); });
      } else if (pop && card) {
        revealCard(card, alert.speech || alert.text);
        setTimeout(() => dismiss(card), 12000);
      }
    };
    if (!speak) return finish(false);
    if (force || (opts && opts.local)) return finish(claimSpeech(alert.id) || force);
    const visible = typeof document === "undefined" || document.visibilityState === "visible";
    setTimeout(() => finish(claimSpeech(alert.id)), visible ? 0 : 1500);
  }

  let es = null;
  function connect() {
    if (es) try { es.close(); } catch {}
    es = new EventSource("/api/alerts/stream");
    es.addEventListener("alert", (ev) => {
      try { ingest(JSON.parse(ev.data)); } catch (e) { console.warn("[alerts] bad event", e); }
    });
    es.onerror = () => { /* browser will retry */ };
  }

  function boot() {
    inject();
    connect();
    refreshStatus();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  function boot2() {
    // Fresh browser profile: pick up the server-side copy of the settings if there is one.
    if (settings.__fresh) {
      fetch("/api/alerts/prefs").then((r) => r.json()).then((j) => {
        if (j && j.prefs && settings.__fresh) { settings = normalizeSettings({ ...settings, ...j.prefs }); settings.__fresh = false; try { localStorage.setItem(STORAGE, JSON.stringify(settings)); } catch {} renderPanel(); }
      }).catch(() => {});
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot2);
  else boot2();

  root.TradingAlerts = {
    __booted: true,
    categories: CATS.map(([id, label]) => ({ id, label })),
    settings: publicSettings,
    emit(alert, opts) { ingest(alert, opts); },
    // Speak (and optionally show) a local message, e.g. an AI analyst reply. Respects the category mute.
    say(text, opts) {
      opts = opts || {};
      const alert = {
        id: opts.id || "local-" + Date.now() + "-" + Math.random().toString(36).slice(2, 8),
        ts: new Date().toISOString(),
        type: opts.type || "analyst", category: opts.category || "analyst", severity: opts.severity || "info",
        title: opts.title || "AI analyst", text: opts.text || text, speech: text, meta: { category: opts.category || "analyst" },
        __narrate: opts.narrate === true,
      };
      ingest(alert, { local: true, noPop: opts.pop === false, speaker: opts.speaker });
      return alert.id;
    },
    openSettings() { unlockAudio(); togglePanel(true); },
    toggleSettings() { unlockAudio(); togglePanel(); },
    setMasterMute(muted) { settings.masterMute = !!muted; if (!muted) settings.voiceMode = "speech"; if (muted) stopSpeaking(); saveSettings(); renderPanel(); },
    setAnalystNarration(mode) { if (mode === "verbose" || mode === "light") { settings.analystNarration = mode; saveSettings(); } },
    voiceOn,
    audioLocked,
    stop: stopSpeaking,
    onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    onAlert(fn) { alertListeners.add(fn); return () => alertListeners.delete(fn); },
    categoryOf,
    prefetch,
  };
})(typeof window === "object" ? window : globalThis);
