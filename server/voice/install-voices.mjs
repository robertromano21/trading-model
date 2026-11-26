#!/usr/bin/env node
// Check the Piper voice models the SID analyst voice uses. Never downloads anything.
// Models are large and gitignored under /models/piper (see server/voice/README.md for where to get them).
//
// Usage, from server/:
//   node voice/install-voices.mjs
//
// Resolution for each voice:
//   1. TM_PIPER_MODEL (female) / TM_PIPER_MALE (male): full path to the .onnx (its .onnx.json must sit beside it)
//   2. TM_PIPER_DIR/<file>, default <repo>/models/piper/<file>
//
// Required files (public Piper voices, rhasspy/piper-voices):
//   en_US-ljspeech-high.onnx + en_US-ljspeech-high.onnx.json   (female)
//   en_US-ryan-high.onnx     + en_US-ryan-high.onnx.json       (male)
// Optional (better consonant timing; build with server/voice/tools/make-dur-models.py):
//   en_US-ljspeech-high-dur.onnx, en_US-ryan-high-dur.onnx
//
// npm packages (in server/package.json): onnxruntime-node, phonemizer.
// Without the models, alerts still pop and the browser's own speech voice is used instead.

import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");
const modelDir = process.env.TM_PIPER_DIR ? resolve(process.env.TM_PIPER_DIR) : join(repo, "models", "piper");
const files = [
  "en_US-ljspeech-high.onnx",
  "en_US-ljspeech-high.onnx.json",
  "en_US-ljspeech-high-dur.onnx",
  "en_US-ryan-high.onnx",
  "en_US-ryan-high.onnx.json",
  "en_US-ryan-high-dur.onnx",
];
const required = files.filter((f) => !f.includes("-dur."));

function locate(file) {
  const named = process.env[file.startsWith("en_US-ryan") ? "TM_PIPER_MALE" : "TM_PIPER_MODEL"];
  const list = [];
  if (named && named.endsWith(".onnx")) {
    if (file.endsWith(".onnx.json")) list.push(named + ".json");
    else if (file.includes("-dur.")) list.push(named.replace(/\.onnx$/, "-dur.onnx"));
    else list.push(named);
  }
  list.push(join(modelDir, file));
  return list.find((p) => existsSync(p)) || null;
}

function havePkg(name) {
  try { require.resolve(name, { paths: [join(repo, "server")] }); return true; }
  catch { return false; }
}

console.log("Trading model Piper / SID voice check");
console.log("  models:", modelDir);
console.log("");
let missingRequired = 0;
for (const file of files) {
  const path = locate(file);
  const req = required.includes(file);
  if (req && !path) missingRequired += 1;
  console.log(`  [${path ? "ok" : req ? "MISSING" : "optional-missing"}] ${file}${path ? "  ->  " + path : ""}`);
}
console.log("");
console.log("  npm  onnxruntime-node:", havePkg("onnxruntime-node") ? "ok" : "MISSING (npm install in server/)");
console.log("  npm  phonemizer:      ", havePkg("phonemizer") ? "ok" : "MISSING (npm install in server/)");

if (missingRequired) {
  console.log("");
  console.log("SID speech is off until the required .onnx + .onnx.json files are in models/piper.");
  console.log("Download them from https://huggingface.co/rhasspy/piper-voices (en/en_US/ljspeech/high and en/en_US/ryan/high).");
  console.log("Until then, alerts still pop and the browser's built-in voice speaks instead.");
  process.exit(1);
}
console.log("\nPiper voices are ready. Alerts can speak through the SID voice.");
