# AccentSense: UK & Ireland Accent Detection with Temporal Evidence

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow CPU](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![Vite 7](https://img.shields.io/badge/Vite-7.3-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev)

---

## What is AccentSense?

AccentSense is a full-stack web application that classifies a speaker's
**UK & Ireland English accent** from an audio recording and shows
**where in the recording** the evidence came from.

Upload a short recording (1–30 seconds) and receive one of seven
classes:

```
Irish · Midlands · Northern · Scottish · Southern · Welsh · Not a speech
```

along with **temporal evidence** — a per-frame score curve and a list of
the time regions that contributed most to the decision — rendered as an
interactive waveform timeline in the UI.

It also includes a **downstream ASR demonstration** panel that shows how
accent-aware Whisper prompt conditioning (regional pronunciation hints)
could reduce accent-driven transcription errors. This panel is an
**honest static illustration** — it never executes an ASR model, and
every response says so via `is_static_example: true`.

> **A self-contained, local-first demo.** Everything runs from this
> repository — there is no training pipeline, and no mock or fabricated
> predictions anywhere in the codebase.

### How it works

```
   audio upload (any common format)
        │
        ▼
 ┌─ 1. DECODE & VALIDATE ────────────────────────────────┐
 │  content sniffing (extensions never trusted)          │
 │  reject, never truncate: <1 s · >30 s · >15 MB        │
 │  FFmpeg via argument arrays + timeout · no shell      │
 └───────────────────────┬───────────────────────────────┘
                         ▼
              mono float32 @ 16 kHz
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 ┌─ 2. YAMNet ────────────┐   ┌─ 3. ACCENT CLASSIFIER ──────────┐
 │ frame encoder          │   │ 7-class accent model            │
 │ → N×1024 embeddings    │──▶│ → N×7 per-frame probabilities   │
 │ → AudioSet scores      │   │ (trust_remote_code = False)     │
 └────────────────────────┘   └────────────────┬────────────────┘
        │                                      │
        │ speech ratio                         ▼ mean-over-frames + argmax
        ▼                             predicted class + model_score
 evidence_curve / regions  ◀─────────── per-frame scores of that class
                         │
                         ▼
        JSON response → React UI (waveform, score timeline,
        evidence regions, insufficient-speech banner)
```

1. **Decode & validate** — untrusted upload is content-sniffed
   (`soundfile` first, FFmpeg fallback for compressed formats), then
   explicitly rejected if empty, too short (<1 s), too long (>30 s) or
   oversized (>15 MB). Output is always mono float32 at 16 kHz.
2. **Frame encoding** — YAMNet (local TF-Hub copy) turns the signal
   into ~21 frames/s of 1024-dim embeddings plus AudioSet event scores;
   frames YAMNet labels "Speech" give `speech_frame_ratio`.
3. **Classification** — the accent model scores every frame across the
   seven classes; probabilities are averaged over frames and argmax'd
   to give the predicted class and `model_score`.
4. **Evidence** — the per-frame score of the predicted class becomes
   the `evidence_curve`; contiguous high-scoring stretches become
   `evidence_regions`, drawn on the waveform in the UI.

The frame hop (0.4762 s), file checksums, pinned revisions and
provenance are all recorded in a committed
[`model_manifest.json`](Backend/models/model_manifest.json) and
verified before the service will load.

### Honest by design

- **No mock predictions.** If the model can't run, `/api/predict`
  returns a generic **503** — never a guess, never a stack trace to
  the client (details go to server logs only).
- **Offline inference.** All artifacts live in `Backend/models/`;
  `HF_HUB_OFFLINE=1` is forced before any TensorFlow/HF import. The
  only network-allowed step is `bootstrap_models.py`, and its output
  is checksum-verified on every start.
- **"Not a speech" is a real class.** When it wins, the API sets
  `is_sufficient_speech: false` and the UI shows an insufficient-speech
  state instead of a fake accent.
- **Honest scoring.** `model_score` is the model's mean softmax — **not
  a calibrated confidence**. Validation accuracy is **~51 %** on a
  7-way, non speaker-disjoint split, and every response says so in
  `model_score_note`.
- **Evidence, not explanations.** Regions come from real per-frame
  probabilities — no gradients, no phonological pseudo-explanations.

Design decisions and trade-offs: see
[`Backend/docs/ARCHITECTURE.md`](Backend/docs/ARCHITECTURE.md).

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12 · FastAPI · uvicorn · slowapi (rate limiting) |
| ML runtime | `tensorflow-cpu` 2.21 (CPU-only) · `tensorflow-hub` · `huggingface_hub` |
| Model | YAMNet frame encoder (TF-Hub) + 7-class accent classifier (pinned, checksum-verified) |
| Audio | `soundfile`/libsndfile · FFmpeg fallback · `scipy` resampling |
| Frontend | React 18 · TypeScript 5.5 · Vite 7 · Tailwind CSS 3 · framer-motion |
| Quality | pytest · ruff · bandit · pip-audit · ESLint 9 · `tsc` |

---

## Prerequisites

| Requirement | Version | Why |
|---|---|---|
| Python | **3.10 – 3.13** (3.12 recommended) | TensorFlow 2.21 has no wheels for 3.14+ |
| FFmpeg | any recent (`ffmpeg` + `ffprobe` **on PATH**) | decoding compressed uploads (WAV goes through `soundfile` directly) |
| Node.js | **20.19+ or 22.12+** | Vite 7 requirement |
| RAM | ~1 GB free | model load peaks around 830 MB RSS |
| uv *(optional)* | latest | faster environment/dependency management |

---

## Running the project

### Step 1 — Backend environment & dependencies

```bash
cd Backend

# create a virtualenv (either tool works)
uv venv --python 3.12 .venv
# or:  python3.12 -m venv .venv

# install runtime dependencies
.venv/bin/python -m pip install -r requirements.txt

# optional: dev extras (pytest, ruff, bandit, pip-audit)
.venv/bin/python -m pip install -r requirements-dev.txt
```

Windows (PowerShell) uses `.venv\Scripts\python` instead of
`.venv/bin/python`.

### Step 2 — Fetch the model artifacts *(one-time, needs network)*

```bash
.venv/bin/python bootstrap_models.py
```

This is the **only step that may touch the network**. It fetches the
classifier and frame-encoder artifacts at pinned revisions, probes both
models, and writes `models/model_manifest.json` (SHA-256 per file +
provenance). All caches are kept inside `Backend/models/`.

Re-verify at any time — runs fully offline:

```bash
.venv/bin/python bootstrap_models.py --check
# [check] All artifacts verified; models load and validate offline.
```

### Step 3 — Start the API

```bash
.venv/bin/python run_api.py
```

- API base: <http://localhost:8000>
- Interactive docs: <http://localhost:8000/docs>
- Readiness: <http://localhost:8000/health>

Configuration via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | set `0.0.0.0` to expose beyond localhost |
| `PORT` | `8000` | API port |
| `LOG_LEVEL` | `INFO` | server log verbosity |
| `CORS_ORIGINS` | `localhost:5173, 5174, 4173, 3000` | comma-separated allowed origins for deploys |

### Step 4 — Start the frontend *(second terminal)*

```bash
cd Frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The Vite dev server proxies `/api` and
`/health` to `localhost:8000`, so no extra configuration is needed
locally.

### Step 5 — Verify

```bash
# readiness (200 once models are loaded; 503 otherwise)
curl http://localhost:8000/health

# classify a recording: 1–30 s, ≤15 MB
curl -F "file=@/path/to/recording.wav" http://localhost:8000/api/predict

# static downstream-ASR illustration
curl -F "detected_accent=Northern" http://localhost:8000/api/downstream-asr
```

First model load takes ~8 s (warm-up runs at startup); inference itself
is ~30 ms per request on CPU.

---

## API reference

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | **Real readiness**: verifies manifest checksums, loads models, reports `inference_ready`. 503 when not ready. |
| `/api/predict` | POST | Real inference on the uploaded audio (`file` field). 4xx for invalid audio, **503 for model failures** with a generic message. |
| `/api/downstream-asr` | POST | Static Whisper prompt-conditioning illustration (`is_static_example: true` + `disclaimer`). Rejects unknown classes and the "Not a speech" state. |

### `/api/predict` response fields

| Field | Meaning |
|---|---|
| `predicted_influence` | exact class from the fixed taxonomy — never remapped |
| `language_family` | human-readable description of that class |
| `model_score` | mean softmax of the predicted class across frames — **not a calibrated confidence** |
| `all_scores` | all seven class scores in canonical order (sum ≈ 1) |
| `evidence_regions` | contiguous high-scoring stretches: `start_time_sec`, `end_time_sec`, `duration_sec`, `model_score`, `label`, `detail` |
| `timestamps`, `evidence_curve` | per-frame (0.4762 s hop) score of the predicted class |
| `speech_frame_ratio` | fraction of frames YAMNet labelled "Speech" |
| `is_sufficient_speech` | `false` ⇔ predicted class is "Not a speech" |
| `model_score_note` | honesty note stating the ~51 % validation accuracy |

### Supported input

* Duration **1–30 s** · size **≤ 15 MB** · sample rate **8–192 kHz** ·
  any format FFmpeg/libsndfile understand (WAV, MP3, M4A, OGG, FLAC…)
* Too-short/too-long/too-large input is **rejected with a clear error**
  — never silently truncated.

---

## Repository structure

```
Accent Sense/
├── README.md                     # this file
├── Backend/                      # Python inference service (FastAPI)
│   ├── bootstrap_models.py       # one-time artifact download (only network step)
│   ├── run_api.py                # API launcher (HOST/PORT env, port 8000)
│   ├── requirements{,-dev}.txt   # runtime / dev dependencies
│   ├── models/                   # model artifacts + committed model_manifest.json
│   ├── src/
│   │   ├── config.py             # class taxonomy, cache isolation, CORS
│   │   ├── audio/io.py           # secure audio decoding/validation
│   │   ├── models/service.py     # load-once inference service
│   │   ├── asr/adaptation.py     # static prompt-conditioning examples
│   │   └── api/main.py           # /health, /api/predict, /api/downstream-asr
│   ├── tests/                    # offline + end-to-end suite
│   ├── docs/ARCHITECTURE.md      # decision record (start here)
│   ├── docs/archive/             # superseded research documents
│   └── AGENTS.md · SKILLS.md     # engineering roles & competencies
└── Frontend/                     # Vite + React + TypeScript + Tailwind UI
    └── src/
        ├── components/           # upload, player, results dossier, evidence panels
        ├── hooks/                # audio player, waveform, prediction state
        └── lib/                  # API client, types, formatting
```

---

## Quality gates

```bash
# Backend
cd Backend
.venv/bin/python -m pytest                          # full suite (runs the real model)
.venv/bin/python -m ruff check .                    # lint
.venv/bin/python -m bandit -r src bootstrap_models.py run_api.py   # security
.venv/bin/python -m pip_audit -r requirements.txt   # dependency CVEs

# Frontend
cd ../Frontend
npm run typecheck && npm run lint && npm run build
```

---

## Known limitations

* **~51 % validation accuracy** on a 7-way, non speaker-disjoint split
  — treat output as a research demo, not a production decision-maker;
  `model_score` is uncalibrated. Accuracy also varies sharply by class:

  | Class | Precision | Recall |
  |---|---|---|
  | Irish | 17.2 % | 63.4 % |
  | Midlands | 13.4 % | 51.7 % |
  | Northern | 30.2 % | 50.6 % |
  | Scottish | 28.9 % | 32.6 % |
  | Southern | 76.3 % | 28.1 % |
  | Welsh | 74.3 % | 83.3 % |
  | Not a speech | 98.8 % | 99.9 % |

  *Precision* = when the verdict is X, how often it is actually right.
  *Recall* = how often a true X is detected at all. A genuine Scottish
  utterance is labelled correctly under a third of the time, so
  confident wrong answers on the weaker classes are expected — the UI
  surfaces every class score alongside the verdict so the uncertainty
  stays visible.
* The downstream-ASR panel is **static by design** — it illustrates
  prompt-conditioning strategies, it does not transcribe audio.
* Audio files under `Backend/data/audio/` in some checkouts are
  placeholder tones, not speech — the system correctly answers
  "Not a speech" for them. Use your own real speech recordings for
  meaningful accent predictions.

---

## Deployment

The repository is configured for Vercel via the root `package.json` and
`vercel.json` (build: `npm --prefix Frontend run build`, output:
`Frontend/dist`). Point `VITE_API_URL` at your backend deployment and
set `CORS_ORIGINS` on the backend to your frontend origin. For a
non-localhost bind, run with `HOST=0.0.0.0`.

## Further reading

* [`Backend/docs/ARCHITECTURE.md`](Backend/docs/ARCHITECTURE.md) —
  decision record: why this pipeline, what was removed and why.
* [`Backend/README.md`](Backend/README.md) — backend-specific setup,
  layout and operating notes.
* [`Backend/AGENTS.md`](Backend/AGENTS.md) — roles and standing rules
  for maintaining this repository.
