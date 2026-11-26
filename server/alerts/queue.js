// Shared speech/popup queue. Voices never overlap. Quiet hours mute speech only.
// Used by tests and kept in lockstep with the algorithm in public/alerts/alerts.js.

function parseHm(value, fallback) {
  const m = String(value ?? "").trim().match(/^(\d{1,2}):(\d{2})$/);
  if (!m) return fallback;
  const h = Number(m[1]), min = Number(m[2]);
  if (h > 23 || min > 59) return fallback;
  return h * 60 + min;
}

export function inQuietHours(settings, now = new Date()) {
  if (!settings?.quietHoursEnabled) return false;
  const start = parseHm(settings.quietStart, 21 * 60);
  const end = parseHm(settings.quietEnd, 7 * 60);
  const cur = now.getHours() * 60 + now.getMinutes();
  if (start === end) return true;
  return start < end ? cur >= start && cur < end : cur >= start || cur < end;
}

export function shouldSpeak(alert, settings, now = new Date()) {
  if (!alert) return false;
  if ((settings?.voiceMode || "speech") === "off") return false;
  if (inQuietHours(settings, now)) return false;
  const types = settings?.speakTypes;
  if (types && typeof types === "object" && Object.hasOwn(types, alert.type)) return !!types[alert.type];
  return true;
}

export function shouldPop(alert, settings) {
  if (!alert) return false;
  if (settings?.popups === false) return false;
  const types = settings?.popTypes;
  if (types && typeof types === "object" && Object.hasOwn(types, alert.type)) return !!types[alert.type];
  return true;
}

export function createAlertQueue({ speak, maxPending = 20 } = {}) {
  const pending = [];
  let current = null;
  let running = false;

  async function pump() {
    if (running) return;
    running = true;
    try {
      while (pending.length) {
        const job = pending.shift();
        current = job;
        try { if (job.speak && speak) await speak(job.alert); }
        catch { /* next item still plays */ }
        finally { current = null; job.resolve?.(); }
      }
    } finally {
      running = false;
    }
  }

  function enqueue(alert, { speak: doSpeak } = {}) {
    return new Promise((resolve) => {
      if (pending.length >= maxPending) pending.shift();
      pending.push({ alert, speak: !!doSpeak, resolve });
      pump();
    });
  }

  function clear() {
    const leftover = pending.splice(0);
    leftover.forEach((job) => job.resolve?.());
    current = null;
  }

  return {
    enqueue,
    clear,
    get length() { return pending.length + (current ? 1 : 0); },
    get busy() { return running; },
    get current() { return current?.alert || null; },
  };
}
