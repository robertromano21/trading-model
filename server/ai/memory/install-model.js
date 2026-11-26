#!/usr/bin/env node
// Install the local MiniLM memory model: downloads
// sentence-transformers/all-MiniLM-L6-v2 (quantized ONNX + vocab) at an immutable revision, verifies SHA-256 and
// writes models/memory/manifest.json. No API key needed. Usage: node server/ai/memory/install-model.js [targetDir]
import { createHash } from "node:crypto";
import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const model = "sentence-transformers/all-MiniLM-L6-v2", root = "https://huggingface.co/";
const directory = process.argv[2] || process.env.TRADING_AI_MEMORY_MODEL_DIR || fileURLToPath(new URL("../../../models/memory/", import.meta.url));
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const get = async (url) => { const r = await fetch(url, { signal: AbortSignal.timeout(180000) }); if (!r.ok) throw Error(`HTTP ${r.status}: ${url}`); return r; };
try {
  const metadata = await (await get(`${root}api/models/${model}?blobs=true`)).json();
  if (!/^[a-f0-9]{40}$/.test(metadata.sha)) throw Error("Missing immutable model revision.");
  const files = {};
  mkdirSync(directory, { recursive: true });
  for (const [source, name] of [["onnx/model_quint8_avx2.onnx", "model.onnx"], ["vocab.txt", "vocab.txt"]]) {
    const bytes = Buffer.from(await (await get(`${root}${model}/resolve/${metadata.sha}/${source}`)).arrayBuffer());
    const expected = metadata.siblings.find((row) => row.rfilename === source)?.lfs?.sha256;
    if (expected && digest(bytes) !== expected) throw Error(`Downloaded checksum mismatch: ${source}`);
    writeFileSync(join(directory, name + ".partial"), bytes);
    renameSync(join(directory, name + ".partial"), join(directory, name));
    files[name] = digest(bytes);
    console.log(`${name}: ${bytes.length} bytes; SHA-256 verified`);
  }
  writeFileSync(join(directory, "manifest.json"), JSON.stringify({ model, revision: metadata.sha, files }, null, 2));
  console.log("Local MiniLM memory model installed in " + directory);
} catch (error) { console.error(error.message); process.exitCode = 1; }
