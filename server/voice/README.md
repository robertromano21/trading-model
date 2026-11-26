# SID analyst voice

The dashboard's spoken alerts and AI analyst replies use a SID-chip voice. Text is turned into phonemes,
a Piper neural voice model is run locally **only as an analysis guide** (its audio is never played),
and the result is re-synthesised through an emulated SID sound chip (`server/voice/sidVoice/`,
browser player in `public/voice/`). Everything runs locally; nothing is sent to an outside service.

If the models below are missing, nothing breaks: alerts still pop up and the browser's built-in
speech voice reads them instead (Alerts settings → Voice engine).

## Getting the voice models (fresh clone)

The models are large and are **not** in the repository (`/models/` is gitignored). Download these
public Piper voices from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) and put
them in `models/piper/` at the repo root:

| File | Where on Hugging Face | Voice |
|---|---|---|
| `en_US-ljspeech-high.onnx` + `en_US-ljspeech-high.onnx.json` | `en/en_US/ljspeech/high/` | female |
| `en_US-ryan-high.onnx` + `en_US-ryan-high.onnx.json` | `en/en_US/ryan/high/` | male |

Then check from `server/`:

```
node voice/install-voices.mjs
```

To keep the models somewhere else, set `TM_PIPER_DIR` (a folder), or `TM_PIPER_MODEL` / `TM_PIPER_MALE`
(full paths to each `.onnx`, with the `.onnx.json` next to it).

### Optional `-dur` models (sharper consonants)

`en_US-ljspeech-high-dur.onnx` and `en_US-ryan-high-dur.onnx` are **not** published Piper files. Each is
the same voice with one extra ONNX output: the duration predictor's rounded per-phoneme durations
(`/Ceil_output_0`, frames of 256 samples at 22,050 Hz). No weights change. The voice uses them for
phone timings when present and falls back to the plain model when not (you'll see a one-time
"no phone timings" warning in the server log, and consonants are a little softer).

To build them (needs the `onnx` Python package, which this project does not install: `pip install onnx`):

```
python server/voice/tools/make-dur-models.py
```

## Licenses and attribution

- **Piper voices** (rhasspy/piper-voices, MIT-licensed project). Each voice inherits its dataset's terms:
  - `en_US-ljspeech-high`: LJ Speech dataset, **public domain**.
  - `en_US-ryan-high`: RyanSpeech dataset, **CC BY-NC-SA 4.0** (non-commercial, attribution, share-alike).
    Fine for personal use; don't use the male voice in a commercial product. Attribution: RyanSpeech,
    https://www.kaggle.com/datasets/roholazandie/ryanspeech.
- **npm packages**: `onnxruntime-node` (MIT), `phonemizer` (Apache-2.0; bundles espeak-ng compiled to
  WASM; espeak-ng itself is GPL-3.0, so check that before redistributing a bundled build), `express` (MIT).
- **SID voice / player code** in `server/voice/` and `public/voice/` was written by the repository owner
  for an earlier personal project and is included here under the repository's own license.
