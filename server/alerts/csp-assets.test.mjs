// The dashboard sends Content-Security-Policy "default-src 'self'", which blocks injected <style> tags and
// style="" attributes. The command console and alert UI must therefore ship their styles as files.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pub = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "public");
const read = (f) => fs.readFileSync(path.join(pub, f), "utf8");

for (const f of ["analyst-dock.js", "alerts/alerts.js"]) {
  test(`${f} does not inject inline styles`, () => {
    const s = read(f);
    assert.doesNotMatch(s, /createElement\(\s*["']style["']\s*\)/);
    assert.doesNotMatch(s, /\sstyle=\\?["']/);
  });
}

test("console and alert stylesheets and the pixel font are present", () => {
  for (const f of ["analyst-dock.css", "alerts/alerts.css", "fonts/PressStart2P.ttf", "fonts/PressStart2P-OFL.txt"]) {
    assert.ok(fs.statSync(path.join(pub, f)).size > 100, f);
  }
  assert.match(read("analyst-dock.js"), /\/analyst-dock\.css/);
  assert.match(read("alerts/alerts.js"), /\/alerts\/alerts\.css/);
  assert.match(read("alerts/alerts.js"), /onAlert\(fn\)/);
});

test("nav.js loads the console with a cache-busting version", () => {
  const s = read("nav.js");
  assert.match(s, /__tmAssetVersion = "\d{12}"/);
  assert.match(s, /"\/analyst-dock\.js\?v=" \+ V/);
});

test("console renders choice replies as option buttons and unknown types as text", () => {
  const s = read("analyst-dock.js");
  assert.match(s, /v\.type === "choice" \|\| v\.kind === "choice"/);
  assert.match(s, /class="tmc-opt"/);
  assert.match(s, /\/\^\[1-9\]\$\/\.test\(e\.key\)/);
});
