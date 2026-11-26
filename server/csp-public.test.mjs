// Every dashboard response carries Content-Security-Policy "default-src 'self'" (server.js). The browser then drops
// injected <style> elements, <style> blocks, style="" attributes, inline <script> blocks, on*="" handlers and
// javascript: URLs. Styles must live in .css files loaded with <link>; dynamic sizes go through element.style.*.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pub = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "public");
const skipDir = new Set(["vendor", "node_modules"]);
function walk(dir, out = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) { if (!skipDir.has(e.name)) walk(p, out); }
    else if (/\.(m?js|html)$/.test(e.name) && !/\.bak/.test(e.name)) out.push(p);
  }
  return out;
}
const isComment = (line) => /^\s*(\/\/|\/?\*)/.test(line);

const RULES = [
  ["injects a <style> element", /createElement\(\s*["'`]style["'`]\s*\)/],
  ["sets a style attribute", /setAttribute\(\s*["'`]style["'`]/],
  ["has a style=\"\" attribute", /<[a-z][^>]*\sstyle\s*=/i],
  ["has a style=\"\" attribute (template)", /\sstyle\s*=\s*\\?["']/],
  ["has a <style> block", /<style[\s>]/i],
  ["has an inline on*= handler", /<[a-z][^>]*\son[a-z]+\s*=\s*\\?["']/i],
  ["uses a javascript: URL", /["'`]\s*javascript:/i],
];

test("public pages and scripts contain nothing the strict CSP blocks", () => {
  const problems = [];
  for (const file of walk(pub)) {
    const rel = path.relative(pub, file).split(path.sep).join("/");
    const lines = fs.readFileSync(file, "utf8").split(/\r?\n/);
    lines.forEach((line, i) => {
      if (isComment(line)) return;
      for (const [what, re] of RULES) if (re.test(line)) problems.push(`${rel}:${i + 1} ${what}`);
    });
    if (rel.endsWith(".html")) {
      const html = lines.join("\n");
      for (const m of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)) {
        if (!/\bsrc\s*=/.test(m[1]) && m[2].trim()) problems.push(`${rel} has an inline <script> block`);
      }
    }
  }
  assert.deepEqual(problems, [], "Move these into .css/.js files:\n" + problems.join("\n"));
});

test("the stylesheets the scripts link to exist", () => {
  for (const f of ["analyst-dock.css", "alerts/alerts.css", "analyst-scorecard.css", "paper2-blackswan.css", "macro-nowcast.css", "swarm-panel.css"]) {
    assert.ok(fs.existsSync(path.join(pub, f)), f);
  }
});
