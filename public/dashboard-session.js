// Dashboard presence: keep the Prices server alive while a tab
// is open, and let it shut down after the last tab or window closes.
// EventSource presence plus a heartbeat and sendBeacon on pagehide.
(() => {
  if (window.TradingDashboardSession || typeof window === "undefined") return;
  const clientId = (globalThis.crypto?.randomUUID?.() || `tab-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`);
  let presence;
  let beats;
  function ping(path) {
    const body = JSON.stringify({ clientId });
    if (path.endsWith("/leave") && typeof navigator !== "undefined" && typeof navigator.sendBeacon === "function") {
      try {
        return navigator.sendBeacon(path, new Blob([body], { type: "application/json" }));
      } catch {
        // Fall through to fetch.
      }
    }
    try {
      return fetch(path, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
        keepalive: true,
      }).catch(() => {});
    } catch {
      return false;
    }
  }
  function connect() {
    if (typeof EventSource === "function") {
      if (!presence || presence.readyState === EventSource.CLOSED) {
        presence = new EventSource(`/api/lifecycle/presence?clientId=${encodeURIComponent(clientId)}`);
      }
    }
    if (!beats) {
      ping("/api/lifecycle/heartbeat");
      beats = setInterval(() => ping("/api/lifecycle/heartbeat"), 12_000);
    }
  }
  function leave() {
    try { presence?.close(); } catch { /* already closed */ }
    presence = null;
    if (beats) {
      clearInterval(beats);
      beats = null;
    }
    ping("/api/lifecycle/leave");
  }
  window.TradingDashboardSession = { clientId, connect, leave };
  connect();
  window.addEventListener("pagehide", leave);
  window.addEventListener("pageshow", connect);
})();
