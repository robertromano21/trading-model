#!/usr/bin/env node
// Command-line entry for the AI analyst (same code path as the dashboard routes).
//   node server/ai/cli.js flag [TICKERS...]      verdicts for the targets -> Massive/Research/advisory_verdicts.json
//   node server/ai/cli.js ask "What do you think of Nvidia today?"
//   node server/ai/cli.js decide NVDA AMD
//   node server/ai/cli.js research NVDA          data bundle only (no model call)
//   node server/ai/cli.js export                 rewrite advisory_verdicts.json from stored verdicts
//   node server/ai/cli.js models                 model list and key status (never prints keys)
//   node server/ai/cli.js memory NVDA <query>     MiniLM recall; memory-status; reindex (embed pending rows)
//   node server/ai/cli.js tool STATS/NORMALITY NVDA horizons=[1,5,20]   run one quant tool (no model call)
//   node server/ai/cli.js prompt [--refresh]      print the master system prompt (generated from engine code + research)
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createAi } from "./routes.js";
import { classify } from "./command.js";
import { parseToolCommand } from "./tools.js";

const root = fileURLToPath(new URL("../../", import.meta.url));
if (existsSync(join(root, ".env.local"))) process.loadEnvFile(join(root, ".env.local"));
const dataDir = resolve(process.env.PRICES_DATA_DIR || join(root, "Massive"));
const ai = createAi({ root, dataDir, python: process.env.PRICES_PYTHON || (process.platform === "win32" ? "python" : "python3"),
  log: (level, message, meta) => console.error(`[${level}] ${message}`, meta ? JSON.stringify(meta) : "") });
const [command = "models", ...args] = process.argv.slice(2);
const out = (v) => console.log(JSON.stringify(v, null, 2));
try {
  if (command === "flag") out(await ai.advisor.flagTargets(args, { onProgress: (p) => console.error(`  ${p.done}/${p.total} ${p.ticker}`) }));
  else if (command === "ask") {
    const question = args.join(" ");
    const c = await classify(question, ai);
    const result = await ai.advisor.ask(question, c.tickers);
    console.log(result.answer); console.error("\n---"); out({ ...result, answer: undefined, intent: c });
  } else if (command === "decide") out(await ai.advisor.decision(args.length ? args : ai.advisor.targets()));
  else if (command === "memory") { await ai.memory?.embedPending(5000); out(await ai.memory.recall({ query: args.slice(1).join(" ") || args[0] || "", tickers: args[0] ? [args[0].toUpperCase()] : [] })); }
  else if (command === "memory-status") out(ai.memory?.stats());
  else if (command === "reindex") out(await ai.memory?.embedPending(100000));
  else if (command === "tool") { const r = await ai.advisor.runTool(parseToolCommand(args.join(" "))); out({ ...r, summary: undefined }); }
  else if (command === "prompt") { if (args.includes("--refresh")) await ai.card.get({ refresh: true }); console.log(await ai.advisor.system()); }
  else if (command === "research") out(await ai.advisor.research(args[0] || "SPY"));
  else if (command === "export") out(ai.advisor.exportAdvisory());
  else out(ai.models.view());
} finally { ai.store.close(); ai.prices?.close(); await ai.memory?.close(); }
