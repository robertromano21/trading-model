// Read-only status for the black swan monitor. Never places or previews orders.
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

function readJson(path) {
  try { return JSON.parse(readFileSync(path, "utf8")); } catch { return null; }
}

function readJsonl(path, last = 30) {
  if (!existsSync(path)) return [];
  try {
    const lines = readFileSync(path, "utf8").split(/\r?\n/).filter(Boolean);
    const rows = [];
    for (const line of lines.slice(-last)) {
      try { rows.push(JSON.parse(line)); } catch { /* skip */ }
    }
    return rows;
  } catch { return []; }
}

export function attachBlackSwanRoutes(app, { dataDir }) {
  const dir = join(dataDir, "Research", "black_swan");
  app.get("/api/black-swan", (_request, response) => {
    const state = readJson(join(dir, "state.json")) || { state: "NORMAL", headline: "Monitor has not run yet." };
    const feeds = readJson(join(dir, "feeds_status.json")) || {};
    const thresholds = readJson(join(dir, "thresholds.json")) || {};
    const study = readJson(join(dir, "study.json")) || {};
    const dollarStudy = readJson(join(dir, "dollar_shock.json")) || {};
    const events = readJsonl(join(dir, "events.jsonl"), 20);
    const disabled = String(process.env.TRADING_BLACKSWAN || "").toLowerCase() === "off";
    response.json({
      disabled,
      state,
      feeds,
      thresholds,
      recentEvents: events.slice(-12).reverse(),
      study: {
        generatedAt: study.generatedAt || null,
        nEvents: study.nEvents || 0,
        honesty: study.honesty || [],
        coverage: study.coverage || null,
        summary: study.summary || null,
        walkForward: study.walkForward ? {
          ok: study.walkForward.ok,
          baseline: study.walkForward.baseline,
          playbook: study.walkForward.playbook,
          nApplied: study.walkForward.nApplied,
          nConsidered: study.walkForward.nConsidered,
          eventWindowExcessMean: study.walkForward.eventWindowExcessMean,
          eventWindowExcessSE: study.walkForward.eventWindowExcessSE,
          honesty: study.walkForward.honesty,
        } : null,
        calibratedThresholds: study.calibratedThresholds || null,
      },
      dollarShock: state.dollarShock || null,
      dollarStudy: dollarStudy.generatedAt ? {
        generatedAt: dollarStudy.generatedAt,
        events: dollarStudy.events ? {
          n: dollarStudy.events.n,
          nSurge: dollarStudy.events.nSurge,
          nWeaken: dollarStudy.events.nWeaken,
          nPre: dollarStudy.events.nPre,
          nHoldout: dollarStudy.events.nHoldout,
        } : null,
        verdict: dollarStudy.verdict || null,
        honesty: dollarStudy.honesty || [],
        holdoutStart: dollarStudy.holdoutStart || null,
        latest: dollarStudy.latest || null,
      } : null,
    });
  });
}
