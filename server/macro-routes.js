// Local macro snapshot and DXY-proxy series for the paper UI and ticker chart.
// Reads gitignored files under Massive/Macro/ and Massive/Research/. No secrets, no broker calls.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

function readJson(path) {
  if (!existsSync(path)) return null;
  try { return JSON.parse(readFileSync(path, "utf8")); } catch { return null; }
}

function readCsv(path) {
  if (!existsSync(path)) return null;
  const text = readFileSync(path, "utf8").trim();
  const lines = text.split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  const dateIdx = headers.findIndex((h) => /^date$/i.test(h.trim()));
  const dxyIdx = headers.findIndex((h) => /^dxy$/i.test(h.trim()));
  if (dateIdx < 0 || dxyIdx < 0) return [];
  const rows = [];
  for (const line of lines.slice(1)) {
    const parts = line.split(",");
    const day = parts[dateIdx];
    const value = Number(parts[dxyIdx]);
    if (!day || !Number.isFinite(value)) continue;
    rows.push({ time: day, value });
  }
  return rows;
}

export function attachMacroRoutes(app, { dataDir }) {
  const snapshotPaths = [
    join(dataDir, "Macro", "snapshot.json"),
    join(dataDir, "Research", "macro_snapshot.json"),
  ];
  app.get("/api/macro/snapshot", (_request, response) => {
    const outlook = readJson(join(dataDir, "Macro", "dxy_outlook.json"))
      || readJson(join(dataDir, "Research", "dxy_forecast_outlook.json"));
    for (const path of snapshotPaths) {
      const payload = readJson(path);
      if (payload) {
        if (outlook && !payload.dxyOutlook) payload.dxyOutlook = outlook;
        return response.json(payload);
      }
    }
    response.json({ available: false, reason: "Macro snapshot not built yet. Run python macro_data.py update." });
  });
  app.get("/api/macro/dxy", (_request, response) => {
    const rows = readCsv(join(dataDir, "Macro", "dxy_proxy.csv")) || [];
    response.json({ source: "DXY-style proxy (ICE weights; not ICE futures)", bars: rows, n: rows.length });
  });
}
