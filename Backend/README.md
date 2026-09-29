# AccentSense Backend

Inference-only service for UK/Ireland accent classification with temporal
evidence. Serves a pretrained model — **there is no training pipeline and
no mock predictions** in this codebase.

**Read `docs/ARCHITECTURE.md` first** — it is the decision record for
everything below (why YAMNet + a SavedModel, why the old WavLM/Captum
stack was removed, how honesty guarantees are enforced).

---

## Setup

```bash
# Python 3.10–3.13 required (TensorFlow 2.21 has no cp314 wheels)
uv venv --python 3.12 .venv
.venv/bin/python -m pip install -r requirements.txt        # runtime
.venv/bin/python -m pip install -r requirements-dev.txt    # + tests/lint/security
```

### 1. Bootstrap model artifacts (the only step that may use the network)

```bash
.venv/bin/python bootstrap_models.py          # download + probe + manifest
.venv/bin/python bootstrap_models.py --check  # offline re-verification
```

This fills `Backend/models/` with:

| Path | Content |
|---|---|
| `models/uk_ireland_accent_classification/` | HF SavedModel @ pinned revision `ebe681a4…` |
| `models/yamnet/` | TF-Hub `google/yamnet/1` copy (frame encoder) |
| `models/model_manifest.json` | SHA-256 per file, provenance, measured frame hop (committed) |

Everything else under `models/` is git-ignored. All caches
(`HF_HOME`, `HF_HUB_CACHE`, `TFHUB_CACHE_DIR`) point inside `models/`
and `HF_HUB_OFFLINE=1` is forced before any TF/HF import — normal
inference never touches the network (enforced by `tests/test_offline.py`).

### 2. Run the API

```bash
.venv/bin/python run_api.py    # http://localhost:8000  (docs at /docs)
```

## Endpoints

| Endpoint | Semantics |
|---|---|
| `GET /health` | **Real readiness**: verifies manifest + checksums, loads the models, reports `inference_ready`. 503 when not ready. |
| `POST /api/predict` | Real inference on the uploaded audio. 4xx for invalid audio (explicit rejection — never truncation), **503 for any model failure** with a generic message; details go to the server log only. |
| `POST /api/downstream-asr` | Static illustration of Whisper prompt conditioning. `is_static_example: true` + `disclaimer` — no ASR model is executed. Rejects unknown classes and the "Not a speech" state. |

### `POST /api/predict` response (abridged)

```jsonc
{
  "predicted_influence": "Northern",        // exact class, never remapped
  "language_family": "Northern English (Northern England)",
  "model_score": 0.61,                      // mean softmax over frames — NOT a calibrated confidence
  "all_scores": { "Irish": …, "Midlands": …, "Northern": …, "Scottish": …,
                  "Southern": …, "Welsh": …, "Not a speech": … },
  "evidence_regions": [ { "start_time_sec":…, "end_time_sec":…, "duration_sec":…,
                          "model_score":…, "label":…, "detail":… } ],
  "timestamps": […], "evidence_curve": […],  // per-frame score of the predicted class
  "speech_frame_ratio": 0.93,               // frames YAMNet labelled Speech
  "is_sufficient_speech": true,             // false ⇔ predicted class is "Not a speech"
  "model_score_note": "…~51% published validation accuracy… not calibrated…"
}
```

Class order is fixed: `Irish, Midlands, Northern, Scottish, Southern,
Welsh, Not a speech` — enforced in `src/config.py`, in the manifest, and
against the model output shape at load time.

## Layout

```
Backend/
├── bootstrap_models.py      # the ONLY network-allowed component
├── run_api.py               # uvicorn launcher (port 8000)
├── requirements.txt         # runtime deps (no torch/captum/transformers)
├── requirements-dev.txt     # pytest, ruff, bandit, pip-audit
├── models/                  # artifacts + committed model_manifest.json
├── src/
│   ├── config.py            # class taxonomy, paths, cache isolation (imported first, everywhere)
│   ├── audio/io.py          # secure decode: content sniffing, FFmpeg arg-arrays, no truncation
│   ├── models/service.py    # load-once singleton: checksums, probes, infer, evidence
│   ├── asr/adaptation.py    # static prompt-conditioning benchmark data (no ASR executed)
│   └── api/main.py          # FastAPI endpoints
├── tests/                   # offline + e2e suite (missing artifacts = FAIL, not skip)
├── docs/ARCHITECTURE.md     # decision record — read this first
├── docs/archive/            # superseded research docs (history, not instructions)
└── AGENTS.md                # roles & standing rules for this repo
```

## Quality gates

```bash
.venv/bin/python -m pytest                # full suite (runs the real model)
.venv/bin/python -m ruff check .          # lint
.venv/bin/python -m bandit -r src bootstrap_models.py   # security
.venv/bin/python -m pip_audit -r requirements.txt       # CVE scan
```

## Operating notes

* CPU-only (`tensorflow-cpu`); no CUDA anywhere.
* Model failure → 503, never a fabricated answer. If `/health` says
  `degraded`, run `bootstrap_models.py` and restart the server.
* The served model's published validation accuracy is ~51 % on a 7-way,
  non speaker-disjoint split — every response says so via
  `model_score_note`. Do not present `model_score` as confidence.
