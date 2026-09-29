# AccentSense Multi-Agent Engineering Framework (`AGENTS.md`)

This document defines the agent roles, communication protocols, and
execution workflows for maintaining the **AccentSense** inference service.

The system is **inference-only**: a pretrained accent classifier served
by a FastAPI backend with a React frontend. There is no training
pipeline, no checkpointing, and no explainability-gradient stack — see
`docs/ARCHITECTURE.md` for the decision record.

---

## 👥 Agent Roster & Specializations

```
                               ┌─────────────────────────────┐
                               │   Lead / Review Agent       │
                               │   (Honesty & Architecture)  │
                               └──────────────┬──────────────┘
                                              │
        ┌───────────────────┬─────────────────┼───────────────────┐
        ▼                   ▼                 ▼                   ▼
┌───────────────┐   ┌───────────────┐ ┌───────────────┐   ┌───────────────┐
│   Model &     │ │    Audio      │ │  API & Demo   │   │   Quality &   │
│  Artifacts    │   │   Ingest      │ │   Service     │   │   Security    │
│  Steward      │   │   Engineer    │ │   Engineer    │   │   Engineer    │
└───────────────┘   └───────────────┘ └───────────────┘   └───────────────┘
```

### 1. Lead / Review Agent (`agent_director`)
- **Role**: Architecture review and honesty enforcement.
- **Responsibilities**:
  - Keep claims aligned with evidence: `model_score` is never called a
    confidence; the published validation accuracy (~51 %, non
    speaker-disjoint split) stays visible in responses and UI.
  - Enforce the class taxonomy: `Irish, Midlands, Northern, Scottish,
    Southern, Welsh, Not a speech` — exact order, never remapped.
  - Reject any change that reintroduces mock/fabricated predictions,
    silent error swallowing, or network access during inference.

### 2. Model & Artifacts Steward (`agent_model_steward`)
- **Role**: Supply chain for `Backend/models/`.
- **Responsibilities**:
  - Run `python bootstrap_models.py` (the ONLY network-allowed step):
    pinned HF revision, YAMNet copy, `model_manifest.json` with SHA-256
    checksums and provenance.
  - Run `python bootstrap_models.py --check` to verify artifacts offline.
  - Keep `src/config.py` cache isolation intact (`HF_HUB_OFFLINE=1`,
    project-local `HF_HOME` / `TFHUB_CACHE_DIR`) and TensorFlow pinned
    to CPU wheels.
  - Treat checksum/class-order/frame-hop mismatches as stop-the-line
    failures — the service refuses to load tampered artifacts.

### 3. Audio Ingest Engineer (`agent_audio_ingest`)
- **Role**: Upload validation and decoding (`src/audio/io.py`).
- **Responsibilities**:
  - Content sniffing — file extensions are never trusted; both lying
    extensions and mislabelled files must behave correctly.
  - Explicit rejection (never truncation) of too-short, too-long and
    oversized input; mono float32 16 kHz output.
  - FFmpeg only via argument arrays with `-nostdin`, a timeout, and
    guaranteed temp-file cleanup. No shell strings, ever.

### 4. API & Demo Service Engineer (`agent_fullstack`)
- **Role**: FastAPI backend (`src/api/main.py`) and its frontend contract.
- **Responsibilities**:
  - `/health` = real readiness (actual artifact verification + model
    load state; 503 when not ready).
  - `/api/predict` = real inference; model failures return a generic
    503 — stack traces and internal paths go to logs only.
  - `/api/downstream-asr` = static prompt-conditioning illustration,
    labelled `is_static_example: true` with its disclaimer; rejects
    unknown classes and the "Not a speech" state.
  - CORS limited to `GET`/`POST` and the required headers; rate limits
    stay on write endpoints.
  - Frontend consumes `model_score`, `evidence_regions`, `evidence_curve`
    and `is_sufficient_speech` (no `confidence`/`salient_regions`/
    `is_mock_prototype` fields).

### 5. Quality & Security Engineer (`agent_quality`)
- **Role**: Verification gates.
- **Responsibilities**:
  - `pytest` (offline tests disable sockets entirely; missing artifacts
    are a FAIL, not a skip).
  - `ruff check .`, `bandit -r src bootstrap_models.py`, `pip-audit -r requirements.txt`.
  - End-to-end API exercise with `curl` against a live server, plus
    latency/memory measurement.
  - Frontend gates: `npm run typecheck && npm run lint && npm run build`.

---

## 🔄 Inter-Agent Workflow

1. **`agent_model_steward`** bootstraps and verifies artifacts
   (`bootstrap_models.py`, then `--check`).
2. **`agent_audio_ingest`** validates the decode path via
   `tests/test_audio_io.py` (malformed, oversized, too-short, too-long,
   stereo, non-16 kHz, MP3-via-FFmpeg).
3. **`agent_fullstack`** runs the API and the endpoint suite
   (`tests/test_predict.py`, `tests/test_health_and_asr.py`).
4. **`agent_quality`** runs the offline suite (`tests/test_offline.py`),
   lint, and security scans.
5. **`agent_director`** reviews wording and schema against
   `docs/ARCHITECTURE.md` before anything ships.

## 🧪 Standing rules

- Network access is confined to `bootstrap_models.py`. Inference,
  `/health` and every test run offline (`HF_HUB_OFFLINE=1`).
- Every model file served must match `model_manifest.json` checksums.
- If the model cannot run, the API says so (503) — it never guesses.
- "Not a speech" is a first-class outcome surfaced as insufficient
  speech, both in the API (`is_sufficient_speech`) and the UI.
