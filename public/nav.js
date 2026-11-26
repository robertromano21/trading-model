// Shared top navigation + Schwab token banner ("Click here to reauthorize") for all Trading Model pages.
// Plain script (no module) so it works on every page; it adds its own stylesheet.
(() => {
  if (document.querySelector(".suite-nav")) return;
  // Bump on UI changes: injected scripts/styles get ?v=<this> so browsers fetch the new copy.
  const V = (window.__tmAssetVersion = "202611260846");
  if (!window.__tmClientLog) {
    const logger = document.createElement("script");
    logger.src = "/client-log.js?v=" + V;
    document.head.appendChild(logger);
  }
  if (![...document.querySelectorAll("script")].some((s) => (s.getAttribute("src") || "").includes("/dashboard-session.js"))) {
    const session = document.createElement("script");
    session.src = "/dashboard-session.js?v=" + V;
    document.head.appendChild(session);
  }
  if (![...document.querySelectorAll("script")].some((s) => (s.getAttribute("src") || "").includes("/alerts/alerts.js"))) {
    const alerts = document.createElement("script");
    alerts.src = "/alerts/alerts.js?v=" + V;
    alerts.defer = true;
    document.head.appendChild(alerts);
  }
  // Data-key cards (FRED, Massive): add themselves to the Schwab setup dialog and any [data-*-key] slot.
  for (const src of ["/fred-key.js", "/massive-key.js"]) {
    if ([...document.querySelectorAll("script")].some((s) => (s.getAttribute("src") || "").split("?")[0] === src)) continue;
    const mod = document.createElement("script");
    mod.type = "module";
    mod.src = src + "?v=" + V;
    document.head.appendChild(mod);
  }
  // Command console (prompt + log of analyst Q&A and trade/alert decisions), lower right on every page.
  if (![...document.querySelectorAll("script")].some((s) => (s.getAttribute("src") || "").includes("/analyst-dock.js"))) {
    const dock = document.createElement("script");
    dock.src = "/analyst-dock.js?v=" + V;
    dock.defer = true;
    document.head.appendChild(dock);
  }
  const css = document.createElement("link");
  css.rel = "stylesheet";
  css.href = "/suite.css?v=" + V;
  document.head.appendChild(css);
  const pickerCss = document.createElement("link");
  pickerCss.rel = "stylesheet";
  pickerCss.href = "/chart-picker.css?v=" + V;
  document.head.appendChild(pickerCss);

  const pages = [
    ["/", "Dashboard", "Market data, history, ADX, old papertrade, model, backtester, logs"],
    ["/ticker.html", "Charts", "Per-ticker chart, indicators, AI verdict, fundamentals/news, Buy/Sell"],
    ["/paper2.html", "Paper trading", "Swing + day-trading paper accounts, risk state, live trading panel"],
    ["/ops.html", "Research & data", "Backtests, walk-forward, normality, run jobs, data health, Schwab"],
    ["/ai.html", "AI analyst", "Command prompt, flags, verdicts, model and API keys, paper-trader decision chat"],
    ["/ops.html#logs", "Logs", "Server, browser and engine logs with filters"],
  ];
  const here = location.pathname === "/index.html" ? "/" : location.pathname;
  const nav = document.createElement("nav");
  nav.className = "suite-nav";
  nav.setAttribute("aria-label", "Trading Model");
  const brand = document.createElement("span");
  brand.className = "brand";
  brand.textContent = "Trading Model";
  nav.appendChild(brand);
  for (const [href, label, title] of pages) {
    const a = document.createElement("a");
    a.href = href;
    a.textContent = label;
    a.title = title;
    if (href === here) a.className = "active";
    nav.appendChild(a);
  }
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = "Schwab …";
  chip.title = "Schwab market-data token";
  nav.appendChild(chip);
  // Chart picker (top right). Pure local lookup (/api/symbols/lookup: fixed crypto list + local price database, no AI):
  // one match opens directly; several (e.g. BTC the coin vs BTC the ETF) show a dropdown to choose from.
  const form = document.createElement("form");
  form.className = "chart-picker";
  form.innerHTML = '<input name="symbol" placeholder="Ticker or coin" aria-label="Open chart: ticker or coin" autocomplete="off" maxlength="24" aria-controls="chart-picker-menu" aria-expanded="false"><button type="submit">Chart</button>';
  const pickerMenu = document.createElement("div");
  pickerMenu.className = "chart-picker-menu";
  pickerMenu.id = "chart-picker-menu";
  pickerMenu.setAttribute("role", "listbox");
  pickerMenu.hidden = true;
  form.appendChild(pickerMenu);
  const pickerCache = new Map();
  const TICKER_RE = /^[A-Z][A-Z0-9.\-]{0,9}$/;
  const stockHref = (s) => `/ticker.html?symbol=${encodeURIComponent(s)}`;
  const closePicker = () => { pickerMenu.hidden = true; pickerMenu.replaceChildren(); form.symbol.setAttribute("aria-expanded", "false"); };
  // Symbols not in the local data are looked up in Schwab's instrument search (read-only, cached server-side);
  // their history is fetched only when one is picked (ticker.html?fetch=1).
  async function pickerLookup(raw) {
    const q = raw.trim().toUpperCase().replace(/^\$/, "");
    if (pickerCache.has(q)) return pickerCache.get(q);
    let matches = null;
    try { matches = JSON.parse(sessionStorage.getItem("chartPicker2." + q) || "null"); } catch { matches = null; }
    if (!matches) {
      try {
        const r = await fetch(`/api/symbols/lookup?q=${encodeURIComponent(q)}&remote=1`);
        if (r.ok) {
          matches = (await r.json()).matches || [];
          try { sessionStorage.setItem("chartPicker2." + q, JSON.stringify(matches)); } catch { /* storage full */ }
        }
      } catch { matches = null; }
    }
    if (!matches) matches = TICKER_RE.test(q) ? [{ kind: "stock", symbol: q, name: "", detail: "Stock / ETF", href: stockHref(q) }] : [];
    pickerCache.set(q, matches);
    return matches;
  }
  function showPicker(matches, focus = true) {
    pickerMenu.replaceChildren(...matches.map((m) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = `chart-picker-option is-${m.kind === "crypto" ? "crypto" : m.kind === "schwab" ? "schwab" : "stock"}`;
      b.setAttribute("role", "option");
      const sym = document.createElement("span");
      sym.className = "cp-symbol";
      sym.textContent = m.symbol;
      const name = document.createElement("span");
      name.className = "cp-name";
      name.textContent = m.name || (m.kind === "crypto" ? "Crypto" : "Stock / ETF");
      const detail = document.createElement("span");
      detail.className = "cp-detail";
      detail.textContent = m.detail || "";
      b.append(sym, name, detail);
      b.addEventListener("click", () => { location.href = m.href; });
      return b;
    }));
    pickerMenu.hidden = false;
    form.symbol.setAttribute("aria-expanded", "true");
    if (focus) pickerMenu.querySelector("button")?.focus();
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const q = form.symbol.value.trim();
    if (!q) return;
    const matches = await pickerLookup(q);
    if (matches.length === 1) location.href = matches[0].href;
    else if (matches.length > 1) showPicker(matches);
  });
  let pickerTimer = null;
  form.symbol.addEventListener("input", () => {
    closePicker();
    clearTimeout(pickerTimer);
    const q = form.symbol.value.trim();
    if (q.length < 2) return;
    pickerTimer = setTimeout(async () => {   // debounced as-you-type suggestions
      const matches = await pickerLookup(q);
      if (form.symbol.value.trim() !== q) return;
      if (matches.length > 1 || matches.some((m) => m.kind === "schwab")) showPicker(matches, false);
    }, 350);
  });
  form.addEventListener("keydown", (event) => {
    const options = [...pickerMenu.querySelectorAll("button")];
    if (event.key === "Escape") { closePicker(); form.symbol.focus(); return; }
    if (pickerMenu.hidden || !options.length || !["ArrowDown", "ArrowUp"].includes(event.key)) return;
    event.preventDefault();
    const i = options.indexOf(document.activeElement);
    const next = event.key === "ArrowDown" ? (i + 1) % options.length : (i <= 0 ? options.length - 1 : i - 1);
    options[next].focus();
  });
  document.addEventListener("click", (event) => { if (!form.contains(event.target)) closePicker(); });
  nav.appendChild(form);

  const banner = document.createElement("div");
  banner.className = "suite-banner";
  banner.hidden = true;
  banner.setAttribute("role", "status");
  const bannerText = document.createElement("span");
  const bannerButton = document.createElement("button");
  bannerButton.type = "button";
  bannerButton.textContent = "Click here to reauthorize";
  banner.append(bannerText, bannerButton);

  const dialog = document.createElement("dialog");
  dialog.className = "suite-dialog";
  dialog.innerHTML = `
    <h3 class="title">Reauthorize Schwab</h3>
    <div class="setup" hidden>
      <p>Connect your Schwab account for market data (quotes, intraday bars) and manual order tickets:</p>
      <ol>
        <li>Open the <a href="https://developer.schwab.com" target="_blank" rel="noopener noreferrer">Schwab Developer Portal</a> and sign in.</li>
        <li><b>Dashboard → Apps → Create App</b>. Choose the <b>Accounts and Trading Production</b> + <b>Market Data Production</b> (Trader API) products.</li>
        <li>Set the <b>Callback URL</b> to <code>https://127.0.0.1:8182</code> (must match the field below exactly).</li>
        <li>Wait until the app status is <b>Ready For Use</b> (can take a few days), then copy its <b>App Key</b> (client ID) and <b>Secret</b> here.</li>
      </ol>
    </div>
    <p class="intro muted">Runs the schwab-py login (<code>engine/schwab_reauth.py</code>, the same <code>easy_client</code> call you used).
      A browser window opens on this PC: log in to Schwab, approve the app, then accept the 127.0.0.1 certificate warning if shown.</p>
    <ul class="state"></ul>
    <button type="button" class="edit">Change app key / secret / callback…</button>
    <div class="keys" hidden>
      <p class="muted">Saved on this PC in <code>.env.local</code> (gitignored); never logged and never sent back to the browser. Leave a field blank to keep the saved value.</p>
      <label>App key (client ID) <input type="password" name="appKey" autocomplete="off" spellcheck="false"></label>
      <label>App secret (client secret) <input type="password" name="appSecret" autocomplete="off" spellcheck="false"></label>
      <label>Callback URL <input type="text" name="callbackUrl" autocomplete="off" spellcheck="false" placeholder="https://127.0.0.1:8182"></label>
      <div class="row"><button type="button" class="save">Save</button></div>
    </div>
    <pre class="log" hidden></pre>
    <p class="msg" role="status"></p>
    <div class="row"><button type="button" class="go">Authorize</button><button type="button" class="close">Close</button></div>`;
  const q = (sel) => dialog.querySelector(sel);

  async function json(url, body) {
    const response = await fetch(url, body === undefined ? { cache: "no-store" } :
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || response.statusText);
    return data;
  }

  let status = null;
  async function refresh() {
    try {
      status = await json("/api/schwab/status");
      chip.textContent = status.setupNeeded && !status.tokenFile ? "Schwab: set up" : status.usable ? (status.accessTokenValid ? "Schwab live" : "Schwab (refreshable)") : "Schwab offline";
      chip.className = `chip ${status.usable ? "ok" : "bad"}`;
      chip.title = `Access token expires ${status.accessTokenExpires ?? "?"}; refresh token about ${status.refreshTokenExpiresAbout ?? "?"}`;
      banner.hidden = status.usable;
      bannerText.textContent = `Schwab data unavailable: ${status.reason ?? "token problem"} Charts and quotes fall back to the local database.`;
      bannerButton.textContent = status.setupNeeded ? "Click here to set up Schwab" : "Click here to reauthorize";
      window.dispatchEvent(new CustomEvent("schwab-status", { detail: status }));
    } catch {
      chip.textContent = "Schwab ?";
    }
    return status;
  }

  let poll = null;
  async function watch() {
    clearInterval(poll);
    poll = setInterval(async () => {
      try {
        const view = await json("/api/schwab/reauth");
        const job = view.job;
        if (!job) return;
        q(".log").hidden = false;
        q(".log").textContent = job.lines.join("\n");
        q(".log").scrollTop = q(".log").scrollHeight;
        if (job.status !== "running") {
          clearInterval(poll);
          q(".go").disabled = false;
          const token = await refresh();
          if (job.status === "complete" && token?.usable) {
            try { await json("/api/schwab/daily-backfill", {}); } catch { /* already running or no token yet */ }
            q(".msg").textContent = "Schwab reauthorized. Starting the 20-year daily history backfill (see Research & data)…";
            setTimeout(() => location.reload(), 1500);
          } else {
            q(".msg").textContent = `Reauthorization ${job.status} (exit ${job.exitCode}). ${token?.reason ?? ""}`;
          }
        }
      } catch (error) {
        q(".msg").textContent = error.message;
      }
    }, 1500);
  }

  function renderState(current) {
    const c = current?.config ?? {};
    const items = [
      ["App key (client ID)", c.appKeySet ? "saved" : "missing", c.appKeySet],
      ["App secret", c.appSecretSet ? "saved" : "missing", c.appSecretSet],
      ["Callback URL", c.callbackUrl ?? "https://127.0.0.1:8182", true],
      ["Token", current?.tokenFile ? (current.usable ? `valid (refresh token until about ${current.refreshTokenExpiresAbout})` : "expired") : "not created yet", current?.usable],
    ];
    q(".state").replaceChildren(...items.map(([label, value, ok]) => {
      const li = document.createElement("li");
      li.textContent = `${ok ? "✓" : "✗"} ${label}: ${value}`;
      return li;
    }));
    const setup = Boolean(current?.setupNeeded);
    q(".title").textContent = setup ? "Set up Schwab" : "Reauthorize Schwab";
    q(".setup").hidden = !setup;
    q(".keys").hidden = !setup;
    q(".edit").hidden = setup;
    q("[name=callbackUrl]").value = c.callbackUrl ?? "https://127.0.0.1:8182";
  }

  async function saveKeys() {
    const body = { appKey: q("[name=appKey]").value.trim(), appSecret: q("[name=appSecret]").value.trim(),
      callbackUrl: q("[name=callbackUrl]").value.trim() };
    if (!body.appKey && !body.appSecret && body.callbackUrl === (status?.config?.callbackUrl ?? "")) return status;
    const saved = await json("/api/schwab/app-keys", body);
    q("[name=appKey]").value = "";
    q("[name=appSecret]").value = "";
    status = saved;
    renderState(saved);
    q(".msg").textContent = "Saved.";
    return saved;
  }

  async function openDialog() {
    const current = await refresh();
    renderState(current);
    q(".msg").textContent = "";
    q(".log").hidden = true;
    const view = await json("/api/schwab/reauth").catch(() => ({}));
    if (view.job?.status === "running") { q(".go").disabled = true; watch(); }
    dialog.showModal();
  }
  q(".edit").addEventListener("click", () => { q(".keys").hidden = !q(".keys").hidden; });
  q(".save").addEventListener("click", async () => {
    try { await saveKeys(); } catch (error) { q(".msg").textContent = error.message; }
  });
  q(".close").addEventListener("click", () => dialog.close());
  q(".go").addEventListener("click", async () => {
    q(".go").disabled = true;
    q(".msg").textContent = "Starting…";
    try {
      const current = await saveKeys();
      if (!current?.config?.appKeySet || !current?.config?.appSecretSet) throw new Error("Enter the app key and secret first.");
      await json("/api/schwab/reauth", {});
      q(".msg").textContent = "Waiting for the Schwab login in your browser (up to 5 minutes)…";
      watch();
    } catch (error) {
      q(".msg").textContent = error.message;
      q(".go").disabled = false;
    }
  });
  bannerButton.addEventListener("click", openDialog);
  chip.addEventListener("click", openDialog);
  chip.style.cursor = "pointer";

  const mount = () => {
    document.body.prepend(nav, banner);
    document.body.appendChild(dialog);
    refresh();
    setInterval(refresh, 60_000);
  };
  if (document.body) mount(); else document.addEventListener("DOMContentLoaded", mount);
  window.TradingSuite = { refreshSchwab: refresh, openReauth: openDialog, getSchwabStatus: () => status };
})();
