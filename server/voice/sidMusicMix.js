'use strict';

// Room tracker order is lead, bass, harmony/percussion.
const VOICE_GAINS = Object.freeze([0.55, 1.06, 1.25]);

function isRoomMusic({ author, released }) {
  return released.startsWith('TradingModel;sec=') || /\bTradingModel (?:SID|arranger)\b/.test(author);
}

module.exports = { VOICE_GAINS, isRoomMusic };
