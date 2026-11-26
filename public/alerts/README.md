# Voice alerts

SID-chip voice popups for the trading model. Alerts appear as a stacked dialogue popup (severity color, ticker, text, timestamp). The AI analyst can speak each one, with a male or female voice, subtitles synced to the line, and a queue so voices never overlap. If Piper models or phonemizer are missing, popups still work.

## Include on a page

```html
<script src="/alerts/alerts.js" defer></script>
```

That single script injects the popup stack, the subtitle banner, and the Alerts settings button (bottom right). Settings are stored in `localStorage` under `tm.alerts.settings`.

Click a popup to open the ticker detail view (`/ticker.html?symbol=NVDA`).

## Emit an alert

### HTTP

`POST /api/alerts` (alias: `POST /api/alerts/emit`)

```json
{
  "type": "fill",
  "ticker": "IEF",
  "title": "Fill",
  "text": "Bought 18 IEF at 96.20",
  "speech": "IEF: the buy filled for 18 at 96.2.",
  "severity": "success",
  "href": "/ticker.html?symbol=IEF",
  "meta": { "side": "buy", "qty": 18, "price": 96.2 }
}
```

`speech` is optional. If omitted, the server writes a short first-person line in the analyst's voice. If `server/ai/phrase.js` later exports `phraseAlert` / `summarizeAlert` / `summarize`, that helper is used; otherwise a template is used. Emit never fails because phrasing failed.

### Event types

| type | When to send |
| --- | --- |
| `decision` | Paper-trader decision (aliases: `paper_trade`, `paper_decision`) |
| `fill` | Order fill (aliases: `filled`, `order_fill`) |
| `stop_hit` | Stop loss hit |
| `circuit_breaker` | Daily loss halt |
| `flag_change` | AI flag changed to GREEN / YELLOW / RED (`meta.flag` or `flag`) |
| `overextension` | Price stretched from its average (`meta.pct`) |
| `price_threshold` | A watched level was crossed (`meta.threshold`) |
| `schwab_auth` | Schwab token expired or setup is needed |
| `error` | Operational or data error |

Severities: `info`, `success`, `warning`, `critical`. If omitted, a default is chosen from the type.

### SSE

`GET /api/alerts/stream` — `event: hello`, then `event: alert` (JSON in `data:`), plus `event: ping` keep-alives. The last 20 alerts are replayed to new subscribers.

`GET /api/alerts/recent?limit=50`  
`GET /api/alerts/status` — whether SID voice and Piper models are available  
`GET /api/alerts/types`

### From Node

```js
import { emitAlert } from "./alerts/index.js";
await emitAlert({ type: "stop_hit", ticker: "NVDA", meta: { price: 121.49 } });
```

### From Python (never raises)

```python
from alerts_emit import emit
emit({"type": "fill", "ticker": "IEF", "meta": {"side": "buy", "qty": 18, "price": 96.2}})
```

`engine/alerts_emit.py` POSTs to `http://127.0.0.1:$PRICES_PORT/api/alerts` (default port 3003) (or `TM_ALERTS_URL`). On any failure it logs to stderr and returns `False`.

## Settings (in the Alerts panel)

- Voice: **Speech** or **Off**
- Analyst: **Female** or **Male**
- Volume slider
- Rate
- SID clock: **1x / 2x / 4x** (default 2x)
- Per-type: speak vs just pop
- Quiet hours (speech muted, popups still show)

Preview buttons speak a sample Nvidia yellow-flag line in each voice.

## Voice pipeline and install

Speech uses a Piper-analysis -> 64-SID vocoder pipeline (`server/voice/` and `public/voice/`). The browser plays live SID register scores with `public/voice/sidPlayer.js` and `/voice/sid-worker.js`. If that fails, the server falls back to a SID-rendered WAV (`POST /api/alerts/speak.wav`), then the browser's built-in speech voice. Popups always work.

The Piper voice models are large, gitignored under `/models/piper`, and must be downloaded once on a fresh clone. See **`server/voice/README.md`** for the file names (`en_US-ljspeech-high`, `en_US-ryan-high`), where to get them, the optional `-dur` variants, and license notes. Check with:

```
cd server
node voice/install-voices.mjs
```

## Tests

From `server/`, on a spare port (does not touch the running dashboard):

```
node --test alerts/alerts.test.mjs
node --check alerts/index.js
```
