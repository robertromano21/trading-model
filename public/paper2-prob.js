// Probability-model study panel on the paper-trading page. Recommendation only.
async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}
const num = (v, d = 2) => (v == null || Number.isNaN(Number(v)) ? "–" : Number(v).toFixed(d));
const pct = (v, d = 1) => (v == null || Number.isNaN(Number(v)) ? "–" : `${(Number(v) * 100).toFixed(d)}%`);
function row(label, value) { return `<tr><th>${label}</th><td>${value}</td></tr>`; }

async function loadProbStudy() {
  const host = document.getElementById("prob-study-body");
  const note = document.getElementById("prob-study-note");
  if (!host) return;
  try {
    const [latest, study] = await Promise.all([
      getJson("/api/research/probabilities").catch((e) => ({ error: e.message })),
      getJson("/api/research/prob-walkforward").catch((e) => ({ error: e.message })),
    ]);
    if (latest?.error && study?.error) {
      host.textContent = "No probability study yet. Run python research.py prob (or the Research: probability walk-forward job).";
      return;
    }
    const v = study.verdict || latest.verdict || {};
    const refs = study.references || {};
    const champ = (study.leaderboard || []).find((r) => r.id === v.champion);
    const cal = (study.calibrationSummary || []).slice(0, 6).map((c) =>
      `${c.model} ${c.target}: Brier ${num(c.brier, 3)}, AUC ${num(c.auc, 2)}, IC ${num(c.ic, 2)}`).join("<br>");
    const spyP = latest.spy || {};
    host.classList.remove("muted");
    host.innerHTML = `
      <table class="kv">
        ${row("Champion (pre-holdout)", v.championLabel || "–")}
        ${row("Pre Sharpe", num(champ?.pre?.sharpe))}
        ${row("Holdout Sharpe", num(champ?.holdout?.sharpe))}
        ${row("TSMOM holdout Sharpe", num(refs.tsmom_baseline?.holdout?.sharpe))}
        ${row("SPY holdout Sharpe", num(refs.spy?.holdout?.sharpe))}
        ${row("PBO (CSCV)", num(study.stats?.pboPre?.pbo))}
        ${row("Deflated Sharpe prob (cumulative)", num(study.stats?.deflatedSharpeCumulative?.deflatedSharpeProbability))}
        ${row("SPA p vs TSMOM (holdout)", num(study.stats?.tests?.["tsmom_baseline:holdout"]?.spaP))}
        ${row("Holdout looks", study.holdoutLedger?.views ?? "–")}
        ${row("SPY P(20d excess > 0)", pct(spyP.excess_20, 0))}
      </table>
      <p>${v.recommendation || ""}</p>
      <p class="muted small">Calibration (pre-holdout fold means)</p>
      <p class="small">${cal || "–"}</p>`;
    if (note) {
      note.textContent = `Generated ${study.generatedAt || latest.asof || ""}. ${((study.caveats || latest.caveats || [])[0]) || ""} Live paper account is unchanged.`;
    }
  } catch (error) {
    host.textContent = `Probability study unavailable: ${error.message}`;
  }
}
loadProbStudy();
