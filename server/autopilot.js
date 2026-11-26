// Autonomous PAPER trading and decision chat log. Schedule (US/Eastern):
//   * on launch (+3 min): swing catch-up run (processes any sessions not yet booked: fills at the next open, stops)
//   * 21:05 price sync (dashboard sync job), 21:30 swing paper run (today's close -> orders for tomorrow's open)
//   * every 5 min during NYSE sessions: Schwab refresh and provisional stock day-trade and swing target checks
//   * every 5 min, 24/7: BTC and ETH paper accounts advance on completed Coinbase 15-minute candles
//   * 16:10: day-trader paper run (books today's session: entries, stops, targets, flat by close, PDT, breaker)
//   * swing run starts with analyst_scoring.py run (scores matured AI flags, refreshes the advisory gate; always exits 0)
//   * after every run: decision_feed.py turns all new decisions into chat entries (Massive/chat_log.sqlite3)
// The toggle (Massive/Research/autopilot.json, default ON) and TRADING_AUTOPILOT=off (forced off) control it.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { easternClock } from "./workbench.js";

const CHAT_SCHEMA = `
CREATE TABLE IF NOT EXISTS chat_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, role TEXT NOT NULL, kind TEXT NOT NULL,
  account TEXT, ticker TEXT, side TEXT, qty REAL, price REAL, decision_date TEXT, text TEXT NOT NULL,
  rationale TEXT, verdict TEXT, data TEXT, source_key TEXT UNIQUE, parent_id INTEGER);
CREATE INDEX IF NOT EXISTS chat_by_ticker ON chat_messages (ticker, id);
CREATE INDEX IF NOT EXISTS chat_by_kind ON chat_messages (kind, id);`;

function chatValue(value) {
  if (!value || typeof value !== "string") return value ?? null;
  try { return JSON.parse(value); } catch { return value; }
}

export const TASKS = {
  swing: { label: "Swing paper run (fills at the open, stops, exits, new orders)", steps: [["analyst_scoring.py", ["run"]], ["paper_trader.py", ["run"]]] },
  daytrade_watch: { label: "Stock intraday watch (Schwab bars, paper signals, swing targets)", steps: [["daytrader.py", ["watch"]]] },
  crypto_intraday: { label: "Crypto intraday paper run (BTC and ETH)", steps: [["crypto_intraday.py", []]] },
  daytrade_close: { label: "Day-trader paper run after the close (books today's session)", steps: [["daytrader.py", ["run", "--fetch"]]] },
  feed: { label: "Refresh the decision chat log", steps: [] },
  schwab_daily: { label: "Schwab long daily backfill (incremental, 20 minutes)",
    steps: [["schwab_daily.py", ["backfill", "--budget-minutes", "20", "--ok-if-auth-missing"]]] },
};

export function createAutopilot({ root, dataDir, python = "python", port = 3003, log = () => {}, env = process.env,
  spawnImpl = spawn, now = () => Date.now(), syncPrices = null, fetchImpl = globalThis.fetch, sqlite = null, memoryIngest = null } = {}) {
  const engineDir = join(root, "engine");
  const settingsPath = join(dataDir, "Research", "autopilot.json");
  const chatPath = join(dataDir, "chat_log.sqlite3");
  const forcedOff = String(env.TRADING_AUTOPILOT ?? "").toLowerCase() === "off";
  const state = { running: null, stockWatch: null, cryptoIntraday: null, last: {}, timer: null };
  const cryptoEnabled = () => !["TRADING_CRYPTO_PAPER", "TRADING_CRYPTO_INTRADAY"]
    .some((key) => String(env[key] ?? "").toLowerCase() === "off");

  function settings() {
    try { return { enabled: true, ...JSON.parse(readFileSync(settingsPath, "utf8")) }; } catch { return { enabled: true }; }
  }
  function saveSettings(value) {
    mkdirSync(join(dataDir, "Research"), { recursive: true });
    writeFileSync(`${settingsPath}.tmp`, JSON.stringify(value, null, 2));
    renameSync(`${settingsPath}.tmp`, settingsPath);
  }
  const enabled = () => !forcedOff && settings().enabled !== false;
  if (existsSync(settingsPath)) {
    try { state.last = settings().last ?? {}; } catch { /* ignore */ }
  }

  function runStep(script, args) {
    return new Promise((resolve) => {
      const child = spawnImpl(python, ["-W", "ignore", script, ...args], {
        cwd: engineDir, env: { ...env, PYTHONUTF8: "1", TM_JOB_ID: `autopilot-${script}` }, windowsHide: true });
      let out = "", err = "";
      child.stdout?.on("data", (d) => { out += d; if (out.length > 2e6) out = out.slice(-1e6); });
      child.stderr?.on("data", (d) => { err += d; if (err.length > 2e5) err = err.slice(-1e5); });
      child.on("error", (e) => resolve({ code: -1, out, err: e.message }));
      child.on("close", (code) => resolve({ code, out, err }));
    });
  }

  async function feed(watchJson = null) {
    let file = null;
    if (watchJson) {
      file = join(tmpdir(), `tm_watch_${randomUUID()}.json`);
      writeFileSync(file, watchJson);
    }
    const result = await runStep("decision_feed.py", file ? ["--watch-json", file] : []);
    if (file) try { unlinkSync(file); } catch { /* ignore */ }
    return result;
  }

  const slot = (name) => name === "daytrade_watch" ? "stockWatch" : name === "crypto_intraday" ? "cryptoIntraday" : "running";
  const busy = (name) => state[slot(name)] ||
    (name === "daytrade_watch" && state.running?.name === "daytrade_close" ? state.running : null) ||
    (name === "daytrade_close" ? state.stockWatch : null);

  async function runTask(name, reason = "schedule") {
    const task = TASKS[name];
    if (!task) throw Object.assign(new Error("Unknown autopilot task."), { status: 400 });
    const active = busy(name);
    if (active) throw Object.assign(new Error(`Busy: ${active.name} is running.`), { status: 409 });
    const record = { name, reason, startedAt: new Date(now()).toISOString(), finishedAt: null, ok: null, summary: "" };
    state[slot(name)] = record;
    log("info", "Autopilot task started", { task: name, reason });
    try {
      if (slot(name) !== "running") {
        const [[script, args]] = task.steps;
        const r = await runStep(script, args);
        if (r.code !== 0) throw new Error(`${script} failed: ${(r.err || r.out).trim().split(/\r?\n/).at(-1)}`.slice(0, 500));
        const result = JSON.parse(r.out.slice(r.out.indexOf("{")));
        record.summary = result.message;
        record.status = result.status;
        if (name === "daytrade_watch") {
          record.session = result.session;
          record.closeMinute = result.closeMinute;
          record.holiday = result.holiday;
          record.freshCount = result.freshCount;
          record.targetCount = result.targetCount;
          record.swingTargets = result.swingTargets;
          if (result.warning) log("warn", "Stock watch used cached bars", { error: result.warning.split(" To fix:")[0].slice(0, 300) });
          if (result.tradesThisRun?.length || result.swingTargets?.some((t) => t.signal === "STOP WATCH")) {
            const f = await feed(JSON.stringify(result));
            if (f.code !== 0) throw new Error("Could not update the stock decision log.");
          }
        } else {
          const fillKey = result.recentFills?.slice(0, 2).map((f) => `${f.account}:${f.at}`).join(",");
          if (fillKey && fillKey !== state.last.crypto_intraday?.loggedFills) {
            const f = await feed(JSON.stringify({ recentFills: result.recentFills }));
            if (f.code !== 0) throw new Error("Could not update the crypto decision log.");
          }
          record.loggedFills = fillKey;
        }
        record.ok = true;
        return record;
      }
      if (name === "swing") {
        const dil = await runStep("dilution.py", ["scan"]);
        try { record.dilution = JSON.parse(dil.out.slice(dil.out.indexOf('{'))); }
        catch { record.dilution = { error: (dil.err || dil.out || "").trim().slice(-300), code: dil.code }; }
        if (dil.code !== 0) log("warn", "Dilution scan failed (paper run kept)", { task: name, code: dil.code });
      }
      for (const [script, args] of task.steps) {
        const scheduledClose = name === "daytrade_close" && reason === "schedule";
        const r = await runStep(script, scheduledClose ? [...args, "--scheduled-close"] : args);
        if (r.code !== 0) {
          const tail = r.err.trim().split(/\r?\n/).slice(-1)[0] ?? "";
          record.ok = false;
          record.summary = `${script} ${args.join(" ")} failed: ${tail}`.slice(0, 500);
          break;
        }
        try {
          const parsed = JSON.parse(r.out.slice(r.out.indexOf("{")));
          record.summary = parsed.lastRun?.message ?? parsed.message ?? record.summary;
          if (scheduledClose && parsed.status === "closed") {
            record.ok = true;
            return record;
          }
        } catch { /* not JSON */ }
      }
      const f = await feed();
      try { record.feed = JSON.parse(f.out); } catch { record.feed = { error: f.err.trim().slice(-300) }; }
      if (name === "swing" || name === "daytrade_close") {
        const d = await runStep("drift_monitor.py", ["run"]);
        try { record.drift = JSON.parse(d.out.slice(d.out.indexOf("{"))); }
        catch { record.drift = { error: (d.err || d.out || "").trim().slice(-300), code: d.code }; }
        if (d.code !== 0) log("warn", "Drift monitor failed (paper run kept)", { task: name, code: d.code });
      }
      record.memory = await ingestMemory();
      // Swarm forward mode (engine/swarm.py): advance the simulated $10k shadow accounts; never fails the paper run.
      if (name === "swing" && String(env.TRADING_SWARM ?? "").toLowerCase() !== "off") {
        const s = await runStep("swarm.py", ["forward"]);
        try { record.swarm = JSON.parse(s.out.slice(s.out.lastIndexOf("{"))); }
        catch { record.swarm = { error: (s.err || s.out || "").trim().slice(-300), code: s.code }; }
        if (s.code !== 0) log("warn", "Swarm forward failed (paper run kept)", { task: name, code: s.code });
      }
      // Crypto PAPER shadow accounts (engine/crypto_paper.py, ids 200-219; no orders). Never fails the paper run.
      if (name === "swing" && String(env.TRADING_CRYPTO_PAPER ?? "").toLowerCase() !== "off") {
        const c = await runStep("crypto_paper.py", ["run", "--fetch"]);
        try { record.crypto = JSON.parse(c.out.slice(c.out.lastIndexOf("{"))); }
        catch { record.crypto = { error: (c.err || c.out || "").trim().slice(-300), code: c.code }; }
        if (c.code !== 0) log("warn", "Crypto paper accounts failed (paper run kept)", { task: name, code: c.code });
      }
      if (name === "swing") {
        const dollar = await runStep("dollar_study.py", ["--refresh-trade"]);
        if (dollar.code !== 0) log("warn", "Dollar study refresh failed", { task: name, code: dollar.code });
        const m = await runStep("macro_service.py", ["refresh"]);
        try { record.macroCalendar = JSON.parse(m.out.slice(m.out.indexOf("{"))); }
        catch { record.macroCalendar = { error: (m.err || m.out || "").trim().slice(-300), code: m.code }; }
        if (m.code !== 0) log("warn", "Macro calendar refresh failed (paper run kept)", { task: name, code: m.code });
      }
      // Black swan / geopolitical poll (feeds + confirmation + paper playbook state). Never fails the paper run.
      if ((name === "swing" || name === "daytrade_close")
          && String(env.TRADING_BLACKSWAN ?? "").toLowerCase() !== "off") {
        const b = await runStep("black_swan.py", ["poll"]);
        try { record.blackSwan = JSON.parse(b.out.slice(b.out.indexOf("{"))); }
        catch { record.blackSwan = { error: (b.err || b.out || "").trim().slice(-300), code: b.code }; }
        if (b.code !== 0) log("warn", "Black swan poll failed (paper run kept)", { task: name, code: b.code });
        state.lastBlackSwanAt = new Date(now()).toISOString();
      }
      // Options: chain snapshot (Schwab market data; skipped cleanly without auth) + vol-sleeve / overlay PAPER runner
      // (engine/options_paper.py writes its own decision-chat entries). Never fails the paper run. TRADING_OPTIONS_PAPER=off skips.
      if (name === "swing" && String(env.TRADING_OPTIONS_PAPER ?? "").toLowerCase() !== "off") {
        record.options = {};
        for (const [key, script, args] of [["snapshot", "options_data.py", ["snapshot"]], ["paper", "options_paper.py", ["run", "--refresh-vix"]]]) {
          const o = await runStep(script, args);
          try { record.options[key] = { code: o.code, ...JSON.parse(o.out.slice(o.out.indexOf("{"))) }; }
          catch { record.options[key] = { error: (o.err || o.out || "").trim().slice(-300), code: o.code }; }
          if (o.code !== 0) log("warn", `Options ${key} failed (paper run kept)`, { task: name, code: o.code });
        }
      }
      if (record.ok === null) record.ok = true;
    } catch (error) {
      record.ok = false;
      record.summary = error.message;
    } finally {
      record.finishedAt = new Date(now()).toISOString();
      state.last[name] = record;
      state[slot(name)] = null;
      try { saveSettings({ ...settings(), last: state.last }); } catch { /* ignore */ }
      log(record.ok ? "info" : "warn", "Autopilot task finished", { task: name, ok: record.ok, summary: record.summary });
    }
    return record;
  }

  // ------------------------------------------------------------------ schedule
  const didToday = (name, clock) => Boolean(state.last[name]?.startedAt) &&
    easternClock(Date.parse(state.last[name].startedAt)).date === clock.date;
  const elapsed = (name) => now() - (Date.parse(state.last[name]?.startedAt ?? "") || 0);
  function stockDue(clock) {
    const last = state.last.daytrade_watch;
    return clock.weekday && clock.minute >= 9 * 60 + 35 && clock.minute < 16 * 60 &&
      !(last?.status === "closed" && last.session === clock.date) && elapsed("daytrade_watch") >= 5 * 60_000;
  }
  function due(clock, includeWatch = true) {
    if (!clock.weekday) return null;
    const m = clock.minute;
    const watch = state.last.daytrade_watch;
    const todayWatch = watch?.session === clock.date ? watch : null;
    const close = todayWatch?.closeMinute ?? 16 * 60;
    if (!todayWatch?.holiday && m >= close + 10 && !didToday("daytrade_close", clock)) return "daytrade_close";
    if (m >= 21 * 60 + 5 && !state.syncedOn?.startsWith(clock.date) && syncPrices) {
      state.syncedOn = clock.date;
      try { syncPrices(); log("info", "Autopilot started the evening price sync"); } catch (e) { log("warn", "Price sync failed", { error: e.message }); }
      return null;
    }
    const swing = state.last.swing;
    const eveningSwing = didToday("swing", clock) && (swing.reason !== "launch catch-up" ||
      easternClock(Date.parse(swing.startedAt)).minute >= 21 * 60 + 30);
    if (m >= 21 * 60 + 30 && !eveningSwing) return "swing";
    if (m >= 22 * 60 + 15 && !didToday("schwab_daily", clock)) return "schwab_daily";
    if (includeWatch && stockDue(clock)) return "daytrade_watch";
    return null;
  }
  function nextTimes(clock) {
    return { swingCatchUpOnLaunch: "3 minutes after launch", priceSync: "21:05 ET weekdays", swing: "21:30 ET weekdays",
      schwabDaily: "22:15 ET weekdays (incremental, 20-minute budget)", daytradeWatch: "every 5 min during NYSE sessions",
      cryptoIntraday: cryptoEnabled() ? "every 5 min, 24/7" : "disabled", daytradeClose: "10 min after NYSE close", now: `${clock.date} ${String(Math.floor(clock.minute / 60)).padStart(2, "0")}:${String(clock.minute % 60).padStart(2, "0")} ET` };
  }
  async function tick() {
    if (!enabled()) return;
    const clock = easternClock(now());
    const tasks = [];
    if (!busy("crypto_intraday") && cryptoEnabled() && elapsed("crypto_intraday") >= 5 * 60_000) tasks.push("crypto_intraday");
    if (!busy("daytrade_watch") && stockDue(clock)) tasks.push("daytrade_watch");
    const name = !state.running && due(clock, false);
    if (name && !busy(name)) tasks.push(name);
    await Promise.all(tasks.map((task) => runTask(task).catch((e) => log("warn", "Autopilot task error", { error: e.message }))));
    if (!state.running && String(env.TRADING_BLACKSWAN ?? "").toLowerCase() !== "off") {
      const last = Date.parse(state.lastBlackSwanAt ?? 0);
      if (now() - last >= 15 * 60_000) {
        state.lastBlackSwanAt = new Date(now()).toISOString();
        const b = await runStep("black_swan.py", ["poll"]).catch((e) => ({ code: 1, out: "", err: e.message }));
        if (b.code !== 0) log("warn", "Black swan poll failed", { code: b.code, error: (b.err || b.out || "").trim().slice(-300) });
      }
    }
  }
  function start() {
    if (state.timer) return;
    log("info", "Autopilot ready", { enabled: enabled(), forcedOff });
    state.launchTimer = setTimeout(() => {
      if (enabled()) runTask("swing", "launch catch-up").catch(() => {});
    }, Number(env.TRADING_AUTOPILOT_LAUNCH_DELAY_MS ?? 180_000));
    state.timer = setInterval(() => { tick(); }, 60_000);
    state.timer.unref?.();
    state.launchTimer.unref?.();
    setImmediate(() => { if (state.timer) tick(); });
  }
  function stop() { clearInterval(state.timer); clearTimeout(state.launchTimer); state.timer = null; }

  function status() {
    const active = [state.running, state.stockWatch, state.cryptoIntraday].filter(Boolean);
    return { enabled: enabled(), forcedOff, paperOnly: true, running: active[0] ?? null, active, last: state.last,
      schedule: nextTimes(easternClock(now())), market: easternClock(now()), tasks: Object.fromEntries(Object.entries(TASKS).map(([k, t]) => [k, t.label])) };
  }

  // ------------------------------------------------------------------ chat log
  async function chatDb(write = false) {
    const { DatabaseSync } = sqlite ?? await import("node:sqlite");
    if (!write && !existsSync(chatPath)) return null;
    mkdirSync(dataDir, { recursive: true });
    const db = new DatabaseSync(chatPath, write ? {} : { readOnly: true });
    db.exec("PRAGMA busy_timeout = 5000");
    if (write) db.exec(CHAT_SCHEMA);
    return db;
  }
  async function listChat({ kind, ticker, account, q, beforeId, afterId, limit = 200, role }) {
    const db = await chatDb();
    if (!db) return { messages: [], kinds: [] };
    try {
      const where = [], args = [];
      if (kind) { where.push(`kind IN (${kind.split(",").map(() => "?").join(",")})`); args.push(...kind.split(",")); }
      if (role) { where.push("role = ?"); args.push(role); }
      if (ticker) { where.push("ticker = ?"); args.push(ticker.toUpperCase()); }
      if (account) { where.push("account = ?"); args.push(account); }
      if (q) { where.push("(text LIKE ? OR rationale LIKE ?)"); args.push(`%${q}%`, `%${q}%`); }
      if (beforeId) { where.push("id < ?"); args.push(Number(beforeId)); }
      if (afterId) { where.push("id > ?"); args.push(Number(afterId)); }
      const sql = `SELECT * FROM chat_messages ${where.length ? `WHERE ${where.join(" AND ")}` : ""} ORDER BY id DESC LIMIT ?`;
      const rows = db.prepare(sql).all(...args, Math.max(1, Math.min(1000, Number(limit) || 200))).map((r) => ({ ...r,
        verdict: chatValue(r.verdict), data: chatValue(r.data) }));
      const kinds = db.prepare("SELECT kind, COUNT(*) AS n FROM chat_messages GROUP BY kind ORDER BY n DESC").all().map((r) => ({ ...r }));
      return { messages: rows.reverse(), kinds };
    } finally {
      db.close();
    }
  }
  async function addMessage(message) {
    const db = await chatDb(true);
    try {
      const cols = Object.keys(message);
      const r = db.prepare(`INSERT INTO chat_messages (${cols.join(",")}) VALUES (${cols.map(() => "?").join(",")})`)
        .run(...cols.map((c) => (typeof message[c] === "object" && message[c] !== null ? JSON.stringify(message[c]) : message[c])));
      return Number(r.lastInsertRowid);
    } finally {
      db.close();
    }
  }
  async function getMessage(id) {
    const db = await chatDb();
    if (!db) return null;
    try {
      const row = db.prepare("SELECT * FROM chat_messages WHERE id = ?").get(Number(id));
      return row ? { ...row } : null;
    } finally { db.close(); }
  }
  async function ask({ question, messageId }) {
    question = String(question ?? "").trim().slice(0, 2000);
    if (!question) throw Object.assign(new Error("Type a question."), { status: 400 });
    const ref = messageId ? await getMessage(messageId) : null;
    if (messageId && !ref) throw Object.assign(new Error("Unknown decision."), { status: 404 });
    const at = new Date(now()).toISOString();
    const qid = await addMessage({ at, role: "user", kind: "followup_q", ticker: ref?.ticker ?? null, account: ref?.account ?? null,
      text: question, parent_id: ref?.id ?? null });
    const context = ref ? [
      "This question is about a decision made by the paper trader (simulation only, no real orders).",
      `Decision: ${ref.text}`, `Rationale: ${ref.rationale ?? ""}`, `Decision date: ${ref.decision_date ?? ""}`,
      ref.verdict ? `AI verdict in force: ${ref.verdict}` : "AI verdict in force: none",
      ref.data ? `Raw record: ${String(ref.data).slice(0, 3000)}` : "",
      "Explain using the trading model's rules (TSMOM 50/100-day core, regime-filtered mean-reversion overlay, ORB day trading, risk caps, PDT, circuit breaker, advisory verdicts that can only reduce risk).",
      `Question: ${question}`].join("\n") : question;
    let answer, source = "AI", stance = null;
    try {
      const response = await fetchImpl(`http://127.0.0.1:${port}/api/ai/ask`, { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: context, tickers: ref?.ticker ? [ref.ticker] : [] }) });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
      answer = [body.answer ?? body.text ?? "", ...(body.keyPoints ?? []).map((k) => `• ${k}`)].filter(Boolean).join("\n");
      source = body.source ?? "AI";
      stance = body.stance ?? null;
    } catch (error) {
      answer = `The AI analyst is unavailable (${error.message}). Decision details:\n${ref ? `${ref.text}\n${ref.rationale ?? ""}` : ""}`;
      source = "UNAVAILABLE";
    }
    const aid = await addMessage({ at: new Date(now()).toISOString(), role: "ai", kind: "followup_a", ticker: ref?.ticker ?? null,
      account: ref?.account ?? null, text: answer || "(empty answer)", rationale: `source: ${source}${stance ? `, stance ${stance}` : ""}`,
      parent_id: qid });
    return { questionId: qid, answerId: aid, answer, source };
  }

  // ------------------------------------------------------------------ AI analyst memory hook (server/ai, optional)
  // Pushes new paper-trader chat entries into the AI layer's analyst memory as kind DECISION, via
  // getMemory({dataDir}).ingest(...) from server/ai/memory/memory.js (same in-process singleton as /api/ai/*).
  // Guarded: a missing or failing memory never affects paper trading or the chat log. Deduped by chat source key.
  async function resolveIngest() {
    try {
      const fn = memoryIngest?.();
      if (typeof fn === "function") return fn;
    } catch { /* injected hook unavailable */ }
    try {
      const mod = await import("./ai/memory/memory.js");
      const memory = mod.getMemory?.({ dataDir });
      return typeof memory?.ingest === "function" ? memory.ingest.bind(memory) : null;
    } catch { return null; }
  }
  function memoryItem(m) {
    const verdict = chatValue(m.verdict);
    const text = [m.text, m.rationale && `Why: ${m.rationale}`,
      verdict && `AI verdict used: ${verdict.verdict ?? verdict.light ?? ""} ${verdict.summary ?? verdict.reason ?? ""}`.trim()]
      .filter(Boolean).join("\n");
    return { kind: "DECISION", ticker: m.ticker || null, text, ts: m.at,
      meta: { refPrice: m.price ?? undefined, action: m.side ? String(m.side).toUpperCase() : undefined, shares: m.qty ?? undefined,
        account: m.account ?? undefined, chatKind: m.kind, decisionDate: m.decision_date ?? undefined,
        dedupeKey: `chat:${m.source_key || m.id}`, source: "paper_trader" } };
  }
  async function ingestMemory() {
    const ingest = await resolveIngest();
    if (!ingest) return { status: "no memory layer" };
    let sent = 0, failed = 0, duplicate = 0;
    try {
      let since = Number(settings().memoryIngestedId ?? 0);
      for (let page = 0; page < 10; page += 1) {
        const db = await chatDb();
        if (!db) break;
        let rows;
        try { rows = db.prepare("SELECT * FROM chat_messages WHERE id > ? ORDER BY id LIMIT 500").all(since).map((r) => ({ ...r })); }
        finally { db.close(); }
        if (!rows.length) break;
        for (const m of rows) {
          if (m.role === "trader" && m.kind !== "run" && m.kind !== "drift") {
            try {
              let timer;
              const out = await Promise.race([Promise.resolve().then(() => ingest(memoryItem(m))),
                new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("timeout")), 15_000); })])
                .finally(() => clearTimeout(timer));
              if (out?.duplicate) duplicate += 1; else sent += 1;
            } catch { failed += 1; }
          }
          since = m.id;
        }
        saveSettings({ ...settings(), memoryIngestedId: since, last: state.last });
      }
    } catch (error) {
      log("warn", "AI memory ingest skipped", { error: error.message });
      return { status: "error", error: error.message, sent, failed };
    }
    if (failed) log("warn", "AI memory ingest had failures", { sent, failed });
    return { status: "ok", sent, duplicate, failed };
  }

  // ------------------------------------------------------------------ routes
  function mount(app) {
    const fail = (response, error) => response.status(error.status || 500).json({ error: error.message });
    app.get("/api/autopilot", (_request, response) => response.json(status()));
    app.post("/api/autopilot/toggle", (request, response) => {
      if (forcedOff) return response.status(409).json({ error: "TRADING_AUTOPILOT=off is set for this server." });
      saveSettings({ ...settings(), enabled: request.body?.enabled === true, last: state.last });
      log("info", "Autopilot toggled", { enabled: request.body?.enabled === true });
      response.json(status());
    });
    app.post("/api/autopilot/run", (request, response) => {
      const name = String(request.body?.task ?? "");
      if (!TASKS[name]) return response.status(400).json({ error: "Unknown autopilot task." });
      const active = busy(name);
      if (active) return response.status(409).json({ error: `Busy: ${active.name} is running.` });
      if (name === "crypto_intraday" && !cryptoEnabled()) return response.status(409).json({ error: "Crypto paper trading is disabled." });
      runTask(name, "manual").catch(() => {});
      response.status(202).json(status());
    });
    app.get("/api/chat", async (request, response) => {
      try {
        const qs = request.query;
        const text = (v, n) => (typeof v === "string" ? v.slice(0, n) : "");
        response.json(await listChat({ kind: text(qs.kind, 120).replace(/[^\w,]/g, ""), ticker: text(qs.ticker, 10).replace(/[^\w.\-]/g, ""),
          account: text(qs.account, 40), q: text(qs.q, 200), beforeId: qs.beforeId, afterId: qs.afterId, limit: qs.limit,
          role: text(qs.role, 10) }));
      } catch (error) { fail(response, error); }
    });
    app.post("/api/chat/ask", async (request, response) => {
      try { response.json(await ask({ question: request.body?.question, messageId: request.body?.messageId })); } catch (error) { fail(response, error); }
    });
  }

  return { mount, start, stop, status, runTask, tick, due, listChat, ask, addMessage, ingestMemory, _state: state };
}
