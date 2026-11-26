#!/usr/bin/env node
// Render two short SID-voice sample alerts (female LJ-guided, male Ryan-guided) to WAV.
// Usage from server/: node alerts/render-demo.mjs [outdir]
import { createRequire } from "node:module";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ANALYST_PROFILES } from "./index.js";

const require = createRequire(import.meta.url);
const SidVoice = require("../voice/sidVoice");
const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const outDir = process.argv[2] || join(root, "logs", "voice");
mkdirSync(outDir, { recursive: true });

const TEXT = "Nvidia is on a yellow flag today. The downtrend is intact, so I'd wait for a better entry.";
SidVoice.setProfiles(Object.values(ANALYST_PROFILES));

async function render(gender) {
  const profile = ANALYST_PROFILES[gender];
  const t0 = Date.now();
  const r = await SidVoice.wavFor(profile.name, TEXT, true, "2x", undefined, undefined, undefined, { profile });
  const file = join(outDir, `analyst-${gender}.wav`);
  writeFileSync(file, r.wav);
  console.log(JSON.stringify({ gender, file, dur: r.dur, ms: Date.now() - t0, bytes: r.wav.length }));
  return file;
}

const files = [];
files.push(await render("female"));
files.push(await render("male"));
await SidVoice.close();
console.log("wrote", files.join(" , "));
