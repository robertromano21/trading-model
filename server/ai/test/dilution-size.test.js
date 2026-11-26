// Size-relative offering flag (mirrors engine/test_dilution_size.py). Offline.
import test from "node:test";
import assert from "node:assert/strict";
import { evaluate, offeringSeverity, applyDilutionToVerdict, riskEntry, annotateFinancials } from "../dilution.js";

const hit = (tier, pct) => ({ form: "424B5", filingDate: "2026-11-10", accession: "a1", description: "common stock offering",
  primaryDocument: "d.htm", items: "", sizeTier: tier, sizePct: pct });
const run = (offers, settings) => evaluate({ ticker: "T", asof: "2026-11-20", filings: offers, settings });

test("severity tiers", () => {
  assert.equal(offeringSeverity([]), null);
  assert.equal(offeringSeverity([hit("SMALL", 0.02)]), "LOW");
  assert.equal(offeringSeverity([hit("SMALL", 0.02), hit("MEDIUM", 0.08)]), "MEDIUM");
  assert.equal(offeringSeverity([hit("LARGE", 0.3)]), "HIGH");
  assert.equal(offeringSeverity([{ form: "S-1" }]), "UNKNOWN");
});

test("small offering keeps the binary flag but does not block", () => {
  const r = run([hit("SMALL", 0.02)]);
  assert.ok(r.offering);
  assert.ok(r.flags.includes("DILUTION_OFFERING"));
  assert.equal(r.offeringSeverity, "LOW");
  assert.equal(r.blocksBuy, false);
  assert.equal(riskEntry(r).severity, "LOW");
  const { verdict } = applyDilutionToVerdict({ flag: "GREEN", action: "BUY" }, r);
  assert.equal(verdict.flag, "GREEN");
  assert.equal(verdict.action, "BUY");
});

test("large offering forces RED and WATCH", () => {
  const r = run([hit("LARGE", 0.25)]);
  assert.equal(r.blocksBuy, true);
  const { verdict } = applyDilutionToVerdict({ flag: "GREEN", action: "BUY", entry: { orderType: "LIMIT", bandLow: 1 } }, r);
  assert.equal(verdict.flag, "RED");
  assert.equal(verdict.action, "WATCH");
  assert.equal(verdict.entry.orderType, "NONE");
});

test("unsized offering stays conservative; switch off restores binary", () => {
  assert.equal(run([{ ...hit(undefined, undefined), sizeTier: undefined }]).blocksBuy, true);
  assert.equal(run([hit("SMALL", 0.02)], { offeringSizeRelative: false }).blocksBuy, true);
});

test("python snapshot severity is honoured", () => {
  const flagsFile = { tickers: { T: { flags: ["DILUTION_OFFERING"], offerings: [hit("SMALL", 0.01)], offeringSeverity: "LOW", offeringSizePct: 0.01 } } };
  const { dilution, derived } = annotateFinancials({ ticker: "T", derived: {}, quarterly: {}, history: {} }, { flagsFile, asof: "2026-11-20" });
  assert.equal(dilution.offeringSeverity, "LOW");
  assert.equal(dilution.blocksBuy, false);
  assert.equal(derived.dilutionOfferingSeverity, "LOW");
});
