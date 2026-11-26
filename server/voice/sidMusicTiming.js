'use strict';

const FRAMES_PER_SECOND = 50;
const UNITS_PER_BAR = 16;
const framesPerBar = bpm => Math.round(240 * FRAMES_PER_SECOND / bpm);
const framesAtUnit = (frames, unit) => Math.round(frames * unit / UNITS_PER_BAR);

function timeline(tempo, bars) {
  if (!Number.isInteger(bars) || bars < 1 || bars > 4000 ||
      !Array.isArray(tempo) || !tempo.length || tempo[0]?.[0] !== 0) throw Error('Invalid SID tempo map.');
  let startFrame = 0;
  return tempo.map((entry, i) => {
    if (!Array.isArray(entry) || entry.length !== 2) throw Error('Invalid SID tempo span.');
    const [startBar, frames] = entry, endBar = tempo[i + 1]?.[0] ?? bars;
    if (!Number.isInteger(startBar) || !Number.isInteger(endBar) || endBar <= startBar || endBar > bars ||
        !Number.isInteger(frames) || frames < 60 || frames > 200) throw Error('Invalid SID tempo span.');
    const span = { startBar, endBar, startFrame, frames };
    startFrame += (endBar - startBar) * frames;
    return span;
  });
}

function atUnit(spans, unit) {
  const bar = Math.floor(unit / UNITS_PER_BAR);
  const span = spans.findLast(item => item.startBar <= bar);
  return span.startFrame + (bar - span.startBar) * span.frames +
    framesAtUnit(span.frames, unit % UNITS_PER_BAR);
}

function barAtFrame(span, frame) {
  return span.startBar + (frame - span.startFrame) / span.frames;
}

module.exports = { FRAMES_PER_SECOND, UNITS_PER_BAR, framesPerBar, framesAtUnit, timeline, atUnit, barAtFrame };
