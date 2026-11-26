// Voice alerts for the trading model: emit API + SSE stream + SID speech routes.
// Import and mount with one line from server.js. Voice deps are optional; emit still works without them.
import { createRequire } from "node:module";
import { randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ALERT_TYPES, SEVERITIES, normalizeType, phraseAlert, templateSpeech } from "./phrase.js";
import { CATEGORIES, CATEGORY_IDS, categorize } from "./categories.js";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const rootDir = join(here, "..", "..");

const ANALYST_PROFILES = {
  female: {
    name: "Analyst (female)", kind: "humanoid",
    f0: 205, fs: 1.04, rate: 4.6, range: 3, breath: 0.06, rasp: 0,
    warm: 0.22, bright: 1.08, vibAmt: 0.2, vibRate: 5.2, pause: 0.95,
  },
  male: {
    name: "Analyst (male)", kind: "humanoid",
    f0: 118, fs: 0.98, rate: 4.6, range: 3.2, breath: 0.08, rasp: 0.05,
    warm: 0.15, bright: 1.0, vibAmt: 0.15, vibRate: 5.0, pause: 1.0,
  },
};

const clients = new Set();
const recent = [];
const MAX_RECENT = 200;
let seq = 0;
let logFn = (level, message, meta) => console[level === "error" ? "error" : "info"](`[alerts] ${message}`, meta || "");

function loadVoice() {
  try {
    return require("../voice/sidVoice");
  } catch (error) {
    logFn("warn", "SID voice module not loaded", { error: error.message });
    return null;
  }
}

function loadBundle() {
  try {
    return require("../voice/workerBundle.js").getWorkerBundle();
  } catch (error) {
    logFn("warn", "SID worker bundle unavailable", { error: error.message });
    return null;
  }
}

function sanitize(raw = {}) {
  const type = normalizeType(raw.type);
  if (!type) return { error: "Unknown alert type." };
  const severity = SEVERITIES.includes(raw.severity) ? raw.severity : ALERT_TYPES[type].severity;
  const ticker = raw.ticker ? String(raw.ticker).toUpperCase().replace(/[^A-Z0-9.\-]/g, "").slice(0, 10) : "";
  const text = String(raw.text || raw.message || raw.title || "").replace(/\s+/g, " ").trim().slice(0, 1000);
  const title = String(raw.title || ALERT_TYPES[type].label).replace(/\s+/g, " ").trim().slice(0, 120);
  const href = raw.href ? String(raw.href).slice(0, 300) : (ticker ? `/ticker.html?symbol=${encodeURIComponent(ticker)}` : "");
  if (href && !href.startsWith("/") && !href.startsWith("http://localhost") && !href.startsWith("http://127.0.0.1")) {
    return { error: "href must be a local path or localhost URL." };
  }
  const meta = raw.meta && typeof raw.meta === "object" && !Array.isArray(raw.meta)
    ? JSON.parse(JSON.stringify(raw.meta)) : {};
  if (raw.flag && !meta.flag) meta.flag = String(raw.flag).toUpperCase();
  if (raw.account != null && meta.account == null) meta.account = String(raw.account).slice(0, 40);
  if (raw.category && !meta.category) meta.category = String(raw.category).toLowerCase().slice(0, 30);
  if (type === "black_swan" && !meta.eventType && /dollar/i.test(String(raw.type || ""))) meta.eventType = "dollar_shock";
  return { type, severity, ticker, title, text, href, meta, speech: raw.speech ? String(raw.speech).slice(0, 400) : "" };
}

async function buildAlert(raw) {
  const fields = sanitize(raw);
  if (fields.error) return fields;
  const speech = await phraseAlert(fields);
  const alert = {
    id: `alt_${Date.now().toString(36)}_${(++seq).toString(36)}_${randomUUID().slice(0, 8)}`,
    ts: new Date().toISOString(),
    type: fields.type,
    severity: fields.severity,
    category: categorize({ type: fields.type, meta: fields.meta }),
    ticker: fields.ticker,
    title: fields.title,
    text: fields.text || speech,
    speech,
    href: fields.href,
    meta: fields.meta,
  };
  recent.push(alert);
  if (recent.length > MAX_RECENT) recent.shift();
  const payload = `id: ${alert.id}\nevent: alert\ndata: ${JSON.stringify(alert)}\n\n`;
  for (const client of [...clients]) {
    try { client.write(payload); }
    catch { clients.delete(client); }
  }
  logFn("info", "alert emitted", { id: alert.id, type: alert.type, ticker: alert.ticker, severity: alert.severity });
  return { alert };
}

export async function emitAlert(raw) {
  try {
    const result = await buildAlert(raw);
    if (result.error) return { ok: false, error: result.error };
    return { ok: true, alert: result.alert };
  } catch (error) {
    logFn("error", "emit failed", { error: error.message });
    return { ok: false, error: error.message };
  }
}

export function cleanPrefs(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const out = {};
  const bool = (k) => { if (typeof raw[k] === "boolean") out[k] = raw[k]; };
  ["masterMute", "popups", "spoken", "quietHoursEnabled"].forEach(bool);
  if (["speech", "off"].includes(raw.voiceMode)) out.voiceMode = raw.voiceMode;
  if (["sid", "browser"].includes(raw.engine)) out.engine = raw.engine;
  if (["male", "female"].includes(raw.gender)) out.gender = raw.gender;
  if (["1x", "2x", "4x"].includes(raw.clock)) out.clock = raw.clock;
  if (Number.isFinite(Number(raw.volume))) out.volume = Math.max(0, Math.min(1.5, Number(raw.volume)));
  if (Number.isFinite(Number(raw.rate))) out.rate = Math.max(0.7, Math.min(1.4, Number(raw.rate)));
  for (const k of ["quietStart", "quietEnd"]) if (/^\d{1,2}:\d{2}$/.test(String(raw[k] || ""))) out[k] = raw[k];
  for (const k of ["speakCats", "popCats"]) {
    if (raw[k] && typeof raw[k] === "object") {
      out[k] = {};
      for (const id of CATEGORY_IDS) if (typeof raw[k][id] === "boolean") out[k][id] = raw[k][id];
    }
  }
  return out;
}

function voiceStatus(SidVoice) {
  const piper = SidVoice ? (() => {
    try { return require("../voice/sidVoice/piper.js"); } catch { return null; }
  })() : null;
  const models = piper ? {
    female: piper.hasVoice?.("lj") || false,
    male: piper.hasVoice?.("ryan") || false,
    femalePath: piper.VOICES?.lj || piper.MODEL || "",
    malePath: piper.VOICES?.ryan || "",
  } : { female: false, male: false };
  return {
    available: Boolean(SidVoice) && (models.female || models.male),
    sidVoice: Boolean(SidVoice),
    models,
    clock: (() => { try { return require("../voice/sidMode.js").get(); } catch { return "2x"; } })(),
    profiles: Object.keys(ANALYST_PROFILES),
  };
}

export function mountAlerts(app, { root = rootDir, log, voice = process.env.TM_ALERTS_NO_VOICE !== "1" } = {}) {
  if (typeof log === "function") logFn = log;
  else if (log && typeof log.info === "function") {
    logFn = (level, message, meta) => (log[level] || log.info).call(log, message, { source: "alerts", ...meta });
  }
  mkdirSync(join(root, "logs", "voice"), { recursive: true });

  const SidVoice = voice ? loadVoice() : null;
  if (SidVoice?.setProfiles) {
    try { SidVoice.setProfiles(Object.values(ANALYST_PROFILES)); } catch (error) {
      logFn("warn", "could not register analyst profiles", { error: error.message });
    }
  }
  let voiceMounted = false;
  if (SidVoice?.install) {
    try {
      const expressLib = require("express");
      SidVoice.install(app, expressLib);
      voiceMounted = true;
      logFn("info", "SID voice routes mounted at /voice/*");
    } catch (error) {
      logFn("warn", "SID voice install failed; alerts still work as popups", { error: error.message });
    }
  }

  let bundle = null;
  try { bundle = loadBundle(); } catch { bundle = null; }
  app.get("/voice/sid-worker.js", (_req, res) => {
    if (!bundle) {
      try { bundle = loadBundle(); } catch { /* keep null */ }
    }
    if (!bundle) return res.status(503).type("text/plain").send("SID worker bundle unavailable");
    res.type("application/javascript").send(bundle);
  });

  app.get("/api/alerts/status", (_req, res) => {
    res.json({ ok: true, voice: voiceStatus(SidVoice), streamClients: clients.size, recent: recent.length, voiceMounted });
  });

  app.get("/api/alerts/types", (_req, res) => {
    res.json({ types: ALERT_TYPES, severities: SEVERITIES, categories: CATEGORIES });
  });

  app.get("/api/alerts/recent", (req, res) => {
    const limit = Math.min(200, Math.max(1, Number(req.query.limit) || 50));
    res.json({ alerts: recent.slice(-limit) });
  });

  app.get("/api/alerts/stream", (req, res) => {
    res.set({
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    });
    if (typeof res.flushHeaders === "function") res.flushHeaders();
    res.write(`event: hello\ndata: ${JSON.stringify({ ok: true, ts: new Date().toISOString() })}\n\n`);
    for (const alert of recent.slice(-20)) {
      res.write(`id: ${alert.id}\nevent: alert\ndata: ${JSON.stringify(alert)}\n\n`);
    }
    clients.add(res);
    const beat = setInterval(() => {
      try { res.write(`event: ping\ndata: ${Date.now()}\n\n`); }
      catch { clearInterval(beat); clients.delete(res); }
    }, 25000);
    req.on("close", () => { clearInterval(beat); clients.delete(res); });
  });

  // Server-side copy of the browser's mute/voice settings (the browser keeps localStorage as the source of truth
  // and syncs here so a fresh browser profile picks the same choices up). Only known keys/types are stored.
  const prefsPath = join(root, "logs", "voice", "alert-prefs.json");
  app.get("/api/alerts/prefs", (_req, res) => {
    try { res.json({ ok: true, prefs: JSON.parse(readFileSync(prefsPath, "utf8")) }); }
    catch { res.json({ ok: true, prefs: null }); }
  });
  app.post("/api/alerts/prefs", (req, res) => {
    const clean = cleanPrefs(req.body?.prefs ?? req.body);
    if (!clean) return res.status(400).json({ ok: false, error: "Send a prefs object." });
    try {
      const tmp = `${prefsPath}.tmp`;
      writeFileSync(tmp, JSON.stringify(clean, null, 1));
      renameSync(tmp, prefsPath);
      res.json({ ok: true, prefs: clean });
    } catch (error) {
      res.status(500).json({ ok: false, error: error.message });
    }
  });

  const handleEmit = async (req, res) => {
    const result = await emitAlert(req.body || {});
    if (!result.ok) return res.status(400).json(result);
    res.json(result);
  };
  app.post("/api/alerts", handleEmit);
  app.post("/api/alerts/emit", handleEmit);

  app.post("/api/alerts/demo", async (req, res) => {
    const gender = String(req.body?.gender || req.query.gender || "female").toLowerCase() === "male" ? "male" : "female";
    const result = await emitAlert({
      type: "flag_change",
      ticker: "NVDA",
      flag: "YELLOW",
      severity: "warning",
      text: "Nvidia is trending down, 24 percent below its 50-day average, with strong cash and a yellow flag.",
      speech: gender === "male"
        ? "Nvidia is on a yellow flag today. The downtrend is intact, so I'd wait for a better entry."
        : "Nvidia is on a yellow flag today. The downtrend is intact, so I'd wait for a better entry.",
      meta: { flag: "YELLOW", demo: true, gender },
    });
    res.json({ ...result, gender, speaker: ANALYST_PROFILES[gender].name });
  });

  // Convenience WAV render for demos and fallback playback (uses the SID vocoder when models are present).
  app.post("/api/alerts/speak.wav", async (req, res) => {
    if (!SidVoice?.wavFor) return res.status(503).json({ error: "SID voice is not available." });
    const gender = String(req.body?.gender || "female").toLowerCase() === "male" ? "male" : "female";
    const text = String(req.body?.text || req.body?.speech || "").slice(0, 400);
    if (!text) return res.status(400).json({ error: "text is required" });
    const clock = String(req.body?.clock || "").toLowerCase();
    const profile = { ...ANALYST_PROFILES[gender] };
    if (req.body?.rate) profile.rate = Math.max(2.4, Math.min(7, Number(req.body.rate) || profile.rate));
    try {
      const r = await SidVoice.wavFor(profile.name, text, true, clock || undefined, undefined, undefined, undefined, { profile });
      res.set({
        "Content-Type": "audio/wav",
        "Cache-Control": "no-store",
        "X-Voice-Dur": String(r.dur || ""),
        "X-Voice-Gender": gender,
      });
      res.end(r.wav);
    } catch (error) {
      res.status(503).json({ error: error.message });
    }
  });

  return { emitAlert, voiceStatus: () => voiceStatus(SidVoice) };
}

export { ALERT_TYPES, ANALYST_PROFILES, CATEGORIES, templateSpeech };
