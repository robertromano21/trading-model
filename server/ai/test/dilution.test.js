// node --test server/ai/test/dilution.test.js
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  classifyGrowth, yoyShareChange, shareGrowthAsof, parseSubmissions, isOfferingFiling,
  offeringsAsof, evaluate, applyDilutionToVerdict, extractShareSeries, pointInTime,
} from "../dilution.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixtures = join(here, "../../../engine/test/fixtures/dilution");
const submissions = JSON.parse(readFileSync(join(fixtures, "submissions_offering.json"), "utf8"));
const atmText = readFileSync(join(fixtures, "atm_supplement.txt"), "utf8");
const share = (end, val, filed) => ({ end, val, filed });

test("10% YoY is a standalone red flag; 5-10% is a warning; threshold is configurable", () => {
  assert.equal(classifyGrowth(0.10), "DILUTION");
  assert.equal(classifyGrowth(0.12), "DILUTION");
  assert.equal(classifyGrowth(0.099), "DILUTION_WARNING");
  assert.equal(classifyGrowth(0.05), "DILUTION_WARNING");
  assert.equal(classifyGrowth(0.049), null);
  assert.equal(classifyGrowth(0.12, { redThreshold: 0.15, warnThreshold: 0.08 }), "DILUTION_WARNING");
});

test("known splits and exact 2x/3x/4x/10x jumps are ignored; 2.2x after a 2-for-1 is 10% dilution", () => {
  const splits = [{ date: "2026-03-15", from: 1, to: 2 }];
  assert.equal(yoyShareChange(200, 100, splits, "2026-06-30", "2025-06-30"), 0);
  for (const ratio of [2, 3, 4, 10]) {
    assert.equal(yoyShareChange(100 * ratio, 100, null, "2026-06-30", "2025-06-30"), 0);
  }
  assert.equal(Number(yoyShareChange(220, 100, splits, "2026-06-30", "2025-06-30").toFixed(6)), 0.1);
});

test("basic shares outstanding are the fallback; preferred in a split window", () => {
  const basic = [share("2025-06-30", 100, "2025-08-01"), share("2026-06-30", 112, "2026-08-01")];
  const growth = shareGrowthAsof([], null, "2026-11-01", basic);
  assert.equal(growth.source, "basic");
  assert.equal(Number(growth.shareChangeYoY.toFixed(6)), 0.12);

  const diluted = [share("2025-06-30", 100, "2025-08-01"), share("2026-06-30", 150, "2026-08-01")];
  const basicSplit = [share("2025-06-30", 100, "2025-08-01"), share("2026-06-30", 200, "2026-08-01")];
  const after = shareGrowthAsof(diluted, [{ date: "2026-01-15", from: 1, to: 2 }], "2026-11-01", basicSplit);
  assert.equal(after.source, "basic");
  assert.equal(after.shareChangeYoY, 0);
});

test("point-in-time growth uses the filing date, not the period-end date", () => {
  const series = [share("2024-12-31", 100, "2025-02-10"), share("2025-12-31", 130, "2026-02-15")];
  assert.equal(shareGrowthAsof(series, null, "2026-01-31"), null);
  assert.equal(pointInTime(series, "2026-01-31").length, 1);
  const later = shareGrowthAsof(series, null, "2026-02-15");
  assert.equal(Number(later.shareChangeYoY.toFixed(6)), 0.3);
  assert.equal(later.current.filed, "2026-02-15");
});

test("offering detection: S-3ASR, 424B5, 8-K item 3.02; not earnings 8-K or S-8; 30-day window is as-of", () => {
  const filings = parseSubmissions(submissions);
  const s3 = filings.find((f) => f.form === "S-3ASR");
  const b5 = filings.find((f) => f.form === "424B5");
  const item302 = filings.find((f) => f.form === "8-K" && String(f.items).includes("3.02"));
  const earnings = filings.find((f) => f.form === "8-K" && String(f.items).includes("2.02"));
  assert.ok(isOfferingFiling(s3));
  assert.equal(isOfferingFiling({ form: "424B5", items: "", description: "424B5" }), false);
  assert.equal(isOfferingFiling({ form: "424B5", items: "", description: "424B5 FORWARD PROSUPP" }), true);
  assert.ok(isOfferingFiling(item302));
  assert.equal(isOfferingFiling(earnings), false);
  assert.equal(isOfferingFiling({ form: "S-8", items: "", description: "Employee plan" }), false);
  assert.equal(isOfferingFiling({ form: "424B2", items: "", description: "424B2" }), false);
  assert.equal(isOfferingFiling({ form: "424B2", items: "", description: "Prospectus supplement for senior notes" }), false);
  assert.equal(isOfferingFiling({ form: "S-3ASR", items: "", description: "S-3ASR" }), false);
  assert.equal(isOfferingFiling({ form: "424B2", items: "", description: "Prospectus for common stock" }), true);
  assert.equal(isOfferingFiling({ form: "S-3ASR", items: "", description: "Automatic shelf registration of common stock" }), true);
  const hits = offeringsAsof(submissions, "2026-11-25", { offeringDays: 30 });
  assert.ok(hits.some((h) => h.form === "S-3ASR" && h.filingDate === "2026-11-10"));
  assert.ok(hits.some((h) => h.form === "8-K" && h.filingDate === "2026-11-05"));
  assert.equal(hits.some((h) => h.form === "424B5"), false);
  assert.deepEqual(offeringsAsof(submissions, "2026-08-20", { offeringDays: 30 }), []);
});

test("ATM language in cheaply provided prospectus text flags an otherwise ordinary 8-K", () => {
  const eightk = { form: "8-K", filingDate: "2026-11-20", items: "1.01", description: "Entry into agreement", accession: "0002" };
  assert.equal(isOfferingFiling(eightk), false);
  assert.equal(isOfferingFiling(eightk, atmText), true);
  const hits = offeringsAsof(
    [{ form: "424B5", filingDate: "2026-11-20", items: "", description: "Prospectus supplement", accession: "0001" }],
    "2026-11-25", { offeringDays: 30 }, { "0001": atmText },
  );
  assert.equal(hits.length, 1);
  assert.equal(hits[0].atm, true);
});

test("evaluate + applyDilutionToVerdict: 12% growth forces RED, WEAK, DILUTION risk with the number, and strips a buy", () => {
  const result = evaluate({
    ticker: "XYZ", asof: "2026-11-25",
    shareSeries: [share("2025-09-30", 100, "2025-11-01"), share("2026-09-30", 112, "2026-11-01")],
  });
  assert.ok(result.flags.includes("DILUTION"));
  assert.equal(Number(result.shareChangeYoY.toFixed(6)), 0.12);
  const { verdict } = applyDilutionToVerdict({
    flag: "GREEN", action: "BUY", fundamentalHealth: "STRONG",
    entry: { orderType: "LIMIT", bandLow: 10, bandHigh: 11, limitPrice: 10.5 }, risks: [],
  }, result);
  assert.equal(verdict.flag, "RED");
  assert.equal(verdict.action, "WATCH");
  assert.equal(verdict.fundamentalHealth, "WEAK");
  assert.equal(verdict.entry.orderType, "NONE");
  const risk = verdict.risks.find((r) => r.kind === "DILUTION");
  assert.ok(risk);
  assert.match(risk.detail, /12\.0%/);
});

test("offering-only flag forces at least YELLOW and blocks a buy; warning alone does not force RED", () => {
  const offer = evaluate({
    ticker: "XYZ", asof: "2026-11-25",
    shareSeries: [share("2025-09-30", 100, "2025-11-01"), share("2026-09-30", 101, "2026-11-01")],
    filings: submissions,
  });
  assert.ok(offer.flags.includes("DILUTION_OFFERING"));
  const { verdict } = applyDilutionToVerdict({ flag: "GREEN", action: "BUY", fundamentalHealth: "STRONG", risks: [] }, offer);
  assert.equal(verdict.flag, "YELLOW");
  assert.equal(verdict.action, "WATCH");
  const warn = evaluate({
    ticker: "XYZ", asof: "2026-11-25",
    shareSeries: [share("2025-09-30", 100, "2025-11-01"), share("2026-09-30", 107, "2026-11-01")],
  });
  assert.deepEqual(warn.flags, ["DILUTION_WARNING"]);
  const mild = applyDilutionToVerdict({ flag: "GREEN", action: "BUY", fundamentalHealth: "STRONG", risks: [] }, warn);
  assert.equal(mild.verdict.flag, "GREEN");
  assert.equal(mild.verdict.action, "BUY");
  assert.ok(mild.verdict.risks.some((r) => r.kind === "DILUTION"));
});

test("extractShareSeries reads diluted us-gaap and dei EntityCommonStockSharesOutstanding with filed dates", () => {
  const facts = {
    facts: {
      "us-gaap": {
        WeightedAverageNumberOfDilutedSharesOutstanding: {
          units: { shares: [
            { form: "10-Q", start: "2025-07-01", end: "2025-09-30", filed: "2025-11-01", val: 100 },
            { form: "10-Q", start: "2026-07-01", end: "2026-09-30", filed: "2026-11-01", val: 112 },
          ] },
        },
      },
      dei: {
        EntityCommonStockSharesOutstanding: {
          units: { shares: [
            { form: "10-Q", end: "2025-09-30", filed: "2025-11-01", val: 100 },
            { form: "10-Q", end: "2026-09-30", filed: "2026-11-01", val: 112 },
          ] },
        },
      },
    },
  };
  const { dilutedShares, basicShares } = extractShareSeries(facts);
  assert.equal(dilutedShares.length, 2);
  assert.equal(basicShares.length, 2);
  assert.equal(basicShares[1].filed, "2026-11-01");
  assert.equal(dilutedShares[0].period, "Q");
});
