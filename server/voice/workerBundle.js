'use strict';
// Bundles the SID live worker (jsid + 10582 + register score + sidToWav + sidLiveWorker)
// so the browser AudioWorklet player can render SID register scores.
const fs = require('node:fs');
const path = require('node:path');
const VOICE = __dirname;
const SERVER = path.join(__dirname, '..');
const PUBLIC_VOICE = path.join(__dirname, '..', '..', 'public', 'voice');

function moduleSource(file, imports = '') {
  return '(()=>{const module={exports:{}};' + imports + fs.readFileSync(file, 'utf8') + ';return module.exports;})()';
}

function workerBundle() {
  const analyzer = fs.readFileSync(path.join(SERVER, 'tools', 'analyze-sid-reference.cjs'), 'utf8');
  const a = analyzer.indexOf('const hex ='), b = analyzer.indexOf('function trace(');
  if (a < 0 || b <= a) throw Error('SID tracer source boundaries changed; converter bundle not built');
  const core = '(()=>{' + analyzer.slice(a, b) + ';return {header,Cpu};})()';
  return 'const __bundleCore=' + core
    + ';const __bundleSid=' + moduleSource(path.join(VOICE, 'sidVoice', 'jsid.js'))
    + ';const __bundleExtended=' + moduleSource(path.join(VOICE, 'sid10582.js'),
      "const require=id=>{if(id==='./sidVoice/jsid')return __bundleSid;throw Error('Unknown bundled import');};")
    + ';const __bundleFormat=' + moduleSource(path.join(VOICE, 'sidExportFormat.js'))
    + ';const __bundleTiming=' + moduleSource(path.join(VOICE, 'sidMusicTiming.js'))
    + ';const __bundleMix=' + moduleSource(path.join(VOICE, 'sidMusicMix.js'))
    + ';const __bundleVoice=' + moduleSource(path.join(VOICE, 'sidVoice', 'registerScore.js'),
      "const require=id=>{if(id==='./jsid')return __bundleSid;throw Error('Unknown bundled import');};")
    + ';const Converter=' + moduleSource(path.join(VOICE, 'sidToWav.js'),
      "const require=id=>{if(id==='../tools/analyze-sid-reference.cjs')return __bundleCore;if(id==='./sidVoice/jsid')return __bundleSid;if(id==='./sidVoice/registerScore')return __bundleVoice;if(id==='./sid10582')return __bundleExtended;if(id==='./sidExportFormat')return __bundleFormat;if(id==='./sidMusicTiming')return __bundleTiming;if(id==='./sidMusicMix')return __bundleMix;throw Error('Unknown bundled import');};")
    + ';' + fs.readFileSync(path.join(PUBLIC_VOICE, 'sidLiveWorker.js'), 'utf8');
}

let cached = null;
function getWorkerBundle() {
  if (!cached) cached = workerBundle();
  return cached;
}

module.exports = { workerBundle, getWorkerBundle };
