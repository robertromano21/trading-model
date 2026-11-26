/* Spoken summary of an AI analyst reply: a short first-person gist, never the tables or long text verbatim.
 * Works in the browser (window.TradingVoiceSummary) and in Node tests (module.exports). */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TradingVoiceSummary = api;
})(typeof window === "object" ? window : globalThis, function () {
  "use strict";
  const MAX = 260;

  function plain(markdown) {
    return String(markdown || "")
      .replace(/```[\s\S]*?```/g, " ")              // code blocks
      .split(/\r?\n/)
      .filter((line) => !/^\s*\|/.test(line))         // table rows
      .filter((line) => !/^\s*[-=:|\s]{3,}$/.test(line))
      .map((line) => line.replace(/^\s*#{1,6}\s+/, "").replace(/^\s*[-*+]\s+/, "").replace(/^\s*\d+[.)]\s+/, ""))
      .join(" ")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")         // links -> text
      .replace(/https?:\/\/\S+/g, "")
      .replace(/[*_`>#]+/g, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function sentences(text) {
    return String(text || "").split(/(?<=[.!?])\s+(?=[A-Z0-9"'(])/).map((s) => s.trim()).filter(Boolean);
  }

  function clip(text, max) {
    const s = String(text || "").trim();
    if (s.length <= max) return s;
    const cut = s.slice(0, max - 1);
    const at = Math.max(cut.lastIndexOf(". "), cut.lastIndexOf("; "), cut.lastIndexOf(", "), cut.lastIndexOf(" "));
    return (at > max * 0.5 ? cut.slice(0, at) : cut).replace(/[,;:\s]+$/, "") + ".";
  }

  // Numbers that read badly through a speech chip: $1,234.5 -> 1,234.5 dollars, 12.3% -> 12.3 percent.
  function speakable(s) {
    return String(s || "")
      .replace(/\$\s?([\d,]+(?:\.\d+)?)/g, "$1 dollars")
      .replace(/(\d)\s?%/g, "$1 percent")
      .replace(/\s+[\u2013\u2014-]\s+/g, ", ")
      .replace(/\s+/g, " ")
      .trim();
  }

  /** v: reply from /api/ai/command (kind answer|help|decision|job|...). Returns "" when nothing should be spoken. */
  function summarizeReply(v, opts) {
    const max = (opts && opts.max) || MAX;
    if (!v || typeof v !== "object") return "";
    if (v.kind === "help" || v.kind === "job") return "";
    if (v.kind === "decision" && Array.isArray(v.decisions)) {
      const usable = v.decisions.filter((d) => d && d.usable);
      if (!usable.length) return "No decision changes from the analyst.";
      const parts = usable.slice(0, 3).map((d) => `${d.ticker} ${String(d.flag || "").toLowerCase()} ${String(d.action || "").toLowerCase()}`.trim());
      return clip(speakable(`Decision update: ${parts.join("; ")}${usable.length > 3 ? `, and ${usable.length - 3} more` : ""}.`), max);
    }
    const body = plain(v.answer || v.text || "");
    if (!body) return "";
    const lead = [];
    if (v.stance && v.kind === "answer" && !/^(n\/a|none|neutral_info)$/i.test(String(v.stance))) {
      lead.push(`${String(v.stance).toLowerCase().replace(/_/g, " ")}${v.conviction ? `, ${String(v.conviction).toLowerCase()} conviction` : ""}.`);
    }
    const ss = sentences(body);
    let out = lead.join(" ");
    for (const s of ss) {
      if ((out + " " + s).trim().length > max) break;
      out = (out + " " + s).trim();
      if (out.length > max * 0.6) break;
    }
    if (!out || out === lead.join(" ")) out = (out + " " + clip(ss[0] || body, max - out.length - 1)).trim();
    return clip(speakable(out), max);
  }

  return { summarizeReply, plain, sentences, speakable, clip, MAX };
});
