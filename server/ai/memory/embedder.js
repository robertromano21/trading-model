// Local MiniLM embedder (all-MiniLM-L6-v2, 384-d, ONNX on CPU in a worker thread).
// Semantic memory embedder: manifest SHA-256 verification of model.onnx + vocab.txt, worker
// startup timeout, per-request timeouts, serialized work chain, batches of 8, finite/384-d checks, normalisation,
// worker unref so it never keeps the process alive. Tests inject `embed` instead of loading the model.
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";

export const DIMS = 384;
const digest = (v) => createHash("sha256").update(v).digest("hex");

export function createEmbedder({ modelDir, embed = null, logger = () => {}, startupMs = 30000, requestMs = 30000 } = {}) {
  const manifestPath = modelDir ? join(modelDir, "manifest.json") : null;
  const configured = Boolean(embed) || Boolean(manifestPath && existsSync(manifestPath));
  let worker = null, ready = null, sequence = 0, closed = false, status = configured ? "idle" : "not-installed", lastError = null, modelKey = null, info = null;
  const pending = new Map();
  let chain = Promise.resolve();

  function start() {
    if (closed) return Promise.reject(Error("Memory embedder stopped."));
    if (ready) return ready;
    ready = (async () => {
      if (!configured) throw Object.assign(Error("MiniLM model not installed (models/memory)."), { code: "MEMORY_MODEL_MISSING" });
      if (embed) { modelKey = "test-encoder"; status = "ready"; return; }
      const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
      for (const file of ["model.onnx", "vocab.txt"])
        if (digest(readFileSync(join(modelDir, file))) !== manifest.files?.[file]) throw Error(`Memory model checksum mismatch: ${file}`);
      modelKey = digest(JSON.stringify(manifest)).slice(0, 16);
      info = { model: manifest.model, revision: manifest.revision };
      worker = new Worker(fileURLToPath(new URL("./embeddingWorker.cjs", import.meta.url)), { workerData: { modelDir } });
      worker.unref();
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(Error("Local memory model startup timed out.")), startupMs);
        worker.on("message", (message) => {
          if (message.ready) { clearTimeout(timer); resolve(); return; }
          if (message.fatal) { clearTimeout(timer); reject(Error(message.fatal)); return; }
          const task = pending.get(message.id);
          if (!task) return;
          pending.delete(message.id); clearTimeout(task.timer);
          message.error ? task.reject(Error(message.error)) : task.resolve(message.vectors);
          if (!pending.size) worker.unref();
        });
        const failAll = (error) => { clearTimeout(timer); reject(error); for (const t of pending.values()) { clearTimeout(t.timer); t.reject(error); } pending.clear(); };
        worker.on("error", failAll);
        worker.on("exit", (code) => { failAll(Error(`Local memory worker exited (${code}).`)); ready = null; worker = null; status = "stopped"; });
      });
      status = "ready";
      logger("info", "AI memory embedder ready", info);
    })().catch((error) => { status = "error"; lastError = error.message; ready = null; throw error; });
    return ready;
  }

  function encodeBatch(texts) {
    if (embed) return Promise.resolve(embed(texts));
    return new Promise((resolve, reject) => {
      const id = ++sequence;
      const timer = setTimeout(() => { pending.delete(id); reject(Error("Local memory encoding timed out.")); }, requestMs);
      pending.set(id, { resolve, reject, timer }); worker.ref(); worker.postMessage({ id, texts });
    });
  }

  // Embed texts -> normalized Float32Array[384] each. Serialized so the CPU worker is never flooded.
  function vectorize(texts) {
    const work = async () => {
      await start();
      const out = [];
      for (let i = 0; i < texts.length; i += 8) {
        const batch = texts.slice(i, i + 8), encoded = await encodeBatch(batch);
        if (encoded.length !== batch.length) throw Error("Incomplete local memory embeddings.");
        for (const v of encoded) {
          if (v.length !== DIMS || !Array.from(v).every(Number.isFinite)) throw Error("Invalid local memory embedding.");
          const norm = Math.hypot(...v) || 1;
          out.push(Float32Array.from(v, (x) => x / norm));
        }
      }
      return out;
    };
    const result = chain.then(work);
    chain = result.catch(() => {});
    return result;
  }

  return {
    vectorize, start, configured,
    key: () => modelKey,
    status: () => ({ status, configured, error: lastError, ...(info || {}) }),
    close: async () => { closed = true; for (const t of pending.values()) { clearTimeout(t.timer); t.reject(Error("Memory embedder stopped.")); } pending.clear(); if (worker) await worker.terminate(); },
  };
}
