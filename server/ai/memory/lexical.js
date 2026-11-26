// Lexical fallback ranking + passage chunking (lexical, words)
// and semantic passages ( ~420-char sentence-packed passages). Used whenever the MiniLM embedder is
// missing, errors or is slower than the recall timeout, so recall never blocks a model call.
const STOP = new Set(("the and for with that this from your you are was not but any all its into has have will can may must only " +
  "each their them they who what when where how about also than then there these those our out one per stock stocks share shares " +
  "today think company inc corp").split(" "));
export const words = (s) => (String(s).toLowerCase().match(/[a-z0-9][a-z0-9.'-]{1,}/g) || []).filter((w) => !STOP.has(w));

export function lexical(query, rows) {
  const q = new Set(words(query)), df = new Map();
  for (const r of rows) for (const w of new Set(words(r.text))) df.set(w, (df.get(w) || 0) + 1);
  const scores = new Map();
  for (const r of rows) {
    let s = 0;
    for (const w of new Set(words(r.text))) if (q.has(w)) s += Math.log(1 + rows.length / df.get(w));
    scores.set(r.id, s / Math.sqrt(1 + r.text.length / 400));
  }
  return scores;
}

export function passages(text, size = 420) {
  const sentences = String(text).match(/[^.!?\n]+(?:[.!?]+["'\u201d\u2019]*|$)/gu) || [String(text)];
  const parts = []; let buf = "";
  for (const sentence of sentences) {
    if (buf && buf.length + sentence.length > size) { parts.push(buf); buf = ""; }
    for (let n = 0; n < sentence.length; n += size) {
      const part = sentence.slice(n, n + size).trim();
      if (!part) continue;
      if (buf && buf.length + part.length > size) { parts.push(buf); buf = ""; }
      buf += (buf ? " " : "") + part;
    }
  }
  if (buf) parts.push(buf);
  return parts.length ? parts : [String(text).slice(0, size)];
}
