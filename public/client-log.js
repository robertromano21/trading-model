// Browser log capture (loaded on every page by nav.js): console errors/warnings, uncaught errors, unhandled
// promise rejections, failed fetches (with the X-Request-Id that ties them to server log lines) and UI actions.
// Entries are batched to POST /api/client-logs -> logs/browser.YYYY-MM-DD.log. Input values are never recorded.
(() => {
  if (window.__tmClientLog) return;
  const originalFetch = window.fetch.bind(window);
  const session = (crypto.randomUUID?.() ?? String(Math.random()).slice(2)).slice(0, 12);
  const page = location.pathname || "/";
  const queue = [];
  let sentThisMinute = 0;
  setInterval(() => { sentThisMinute = 0; }, 60_000);
  const text = (v) => {
    try { return v instanceof Error ? `${v.name}: ${v.message}` : typeof v === "string" ? v : JSON.stringify(v); } catch { return String(v); }
  };
  function push(level, kind, message, detail, requestId) {
    if (sentThisMinute > 200) return;
    sentThisMinute += 1;
    queue.push({ level, kind, message: String(message).slice(0, 2000), detail, requestId, page, session,
      timestamp: new Date().toISOString() });
    if (queue.length >= 20 || level === "error") flush();
  }
  function flush(beacon = false) {
    if (!queue.length) return;
    const entries = queue.splice(0, 50);
    const body = JSON.stringify({ entries });
    if (beacon && navigator.sendBeacon) {
      navigator.sendBeacon("/api/client-logs", new Blob([body], { type: "application/json" }));
      return;
    }
    originalFetch("/api/client-logs", { method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: true })
      .catch(() => { /* the server may be restarting; drop */ });
  }
  setInterval(flush, 5000);
  addEventListener("pagehide", () => flush(true));

  for (const [method, level] of [["error", "error"], ["warn", "warn"]]) {
    const original = console[method].bind(console);
    console[method] = (...args) => { original(...args); push(level, "console", args.map(text).join(" ")); };
  }
  addEventListener("error", (event) => {
    if (event.target && event.target !== window && event.target.tagName) {
      push("error", "error", `Failed to load ${event.target.tagName.toLowerCase()} ${event.target.src || event.target.href || ""}`);
      return;
    }
    push("error", "error", event.message || "Script error", { file: event.filename, line: event.lineno, column: event.colno,
      stack: event.error?.stack?.slice(0, 2000) });
  }, true);
  document.addEventListener("securitypolicyviolation", (event) => push("error", "error",
    `CSP blocked ${event.violatedDirective}: ${event.blockedURI || "inline"} (${event.sourceFile || ""}:${event.lineNumber || ""})`));
  addEventListener("unhandledrejection", (event) => push("error", "rejection", `Unhandled promise rejection: ${text(event.reason)}`,
    { stack: event.reason?.stack?.slice(0, 2000) }));

  window.fetch = async (input, init = {}) => {
    const url = typeof input === "string" ? input : input?.url ?? String(input);
    const sameOrigin = url.startsWith("/") || url.startsWith(location.origin);
    const path = sameOrigin ? new URL(url, location.origin).pathname : "(external)";
    if (path === "/api/client-logs" || !sameOrigin) return originalFetch(input, init);
    const requestId = crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`.replace(".", "");
    const headers = new Headers(init.headers ?? (input instanceof Request ? input.headers : undefined));
    if (!headers.has("X-Request-Id")) headers.set("X-Request-Id", requestId);
    const started = performance.now();
    try {
      const response = await originalFetch(input, { ...init, headers });
      if (!response.ok) {
        let error = "";
        try { error = (await response.clone().json()).error ?? ""; } catch { /* not json */ }
        push(response.status >= 500 ? "error" : "warn", "fetch", `${init.method ?? "GET"} ${path} -> HTTP ${response.status} ${error}`.trim(),
          { status: response.status, ms: Math.round(performance.now() - started) }, response.headers.get("X-Request-Id") ?? requestId);
      }
      return response;
    } catch (error) {
      push("error", "fetch", `${init.method ?? "GET"} ${path} failed: ${error.message}`, { ms: Math.round(performance.now() - started) }, requestId);
      throw error;
    }
  };

  const label = (el) => (el.getAttribute("aria-label") || el.textContent || el.title || el.value && el.type === "submit" && el.value || "")
    .trim().replace(/\s+/g, " ").slice(0, 60);
  addEventListener("click", (event) => {
    const el = event.target.closest?.("button, a, summary, [role=button], input[type=checkbox], input[type=radio]");
    if (!el) return;
    const kind = el.tagName.toLowerCase();
    push("info", "action", `click ${kind}${el.id ? `#${el.id}` : ""} "${label(el)}"`,
      kind === "a" ? { href: el.getAttribute("href") } : el.type === "checkbox" ? { checked: el.checked } : undefined);
  }, true);
  addEventListener("change", (event) => {
    const el = event.target;
    if (el.tagName === "SELECT") push("info", "action", `select ${el.id || el.name || ""} = ${el.value}`.slice(0, 200));
  }, true);
  push("info", "navigation", `open ${page}${location.search}`, { referrer: document.referrer ? new URL(document.referrer).pathname : "" });
  window.__tmClientLog = { push, flush };
})();
