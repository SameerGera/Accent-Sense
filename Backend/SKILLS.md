# AccentSense Engineering & Domain Skills (`SKILLS.md`)

Competencies this codebase actually exercises today. The system is
inference-only; training/XAI/ASR-research skills from earlier phases are
retired (see `docs/archive/` and `docs/ARCHITECTURE.md` §4).

---

## 🛠️ Core Skill Competencies

### Skill 1: Speech Audio Ingestion (`speech_ingest`)
- **Domain**: Secure decoding and validation of untrusted audio.
- **Tooling**: `soundfile` (libsndfile), `ffmpeg`/`ffprobe` (subprocess), `scipy.signal.resample_poly`, `numpy`.
- **Key operations**:
  - Content sniffing — extensions are never trusted; decoders decide.
  - Mono float32 @ 16 kHz normalization (downmix + polyphase resampling).
  - Explicit rejection of empty / too-short (<1 s) / too-long (>30 s) /
    oversized (>15 MB) input — **never silent truncation**.
  - FFmpeg only as an argument array with `-nostdin`, a hard timeout,
    neutral temp-file names and unconditional cleanup.

### Skill 2: Pretrained Model Serving (`model_serving`)
- **Domain**: Loading large pretrained artifacts safely and exactly once.
- **Tooling**: `tensorflow-cpu`, `tf.saved_model.load`, `tensorflow_hub.load`, `threading` double-checked locking.
- **Key operations**:
  - Load-once singleton (`AccentModelService`) with a warm-up thread at
    API startup; CPU-only, no CUDA assumptions.
  - Load-time validation probes: output shape `(N, 7)`, rows sum to 1,
    finite values, YAMNet `(N, 1024)` embeddings + `(N, 521)` scores.
  - Reference aggregation: mean over YAMNet frames, then argmax.
  - Speech rule: frame is speech iff YAMNet top-1 AudioSet event index
    is 0 ("Speech") — the classifier's own training labelling rule.

### Skill 3: Model Supply-Chain Integrity (`artifact_integrity`)
- **Domain**: Reproducible, tamper-evident model distribution.
- **Tooling**: `huggingface_hub.snapshot_download` (pinned revision), SHA-256 manifests, TF-Hub local copies.
- **Key operations**:
  - Single network-allowed entrypoint: `bootstrap_models.py`.
  - `model_manifest.json`: per-file checksums, provenance, pinned HF
    revision `ebe681a49e9506acdc4cf5312cb79b291275ab67`, measured frame hop.
  - Verification **before** load; checksum / class-order / frame-hop
    tampering makes the service refuse to start serving.
  - Offline verification: `bootstrap_models.py --check`, plus a test
    suite that disables sockets at interpreter start.

### Skill 4: Cache & Network Isolation (`offline_runtime`)
- **Domain**: Guaranteeing air-gapped inference.
- **Tooling**: environment pinning in `src/config.py`, `HF_HUB_OFFLINE`.
- **Key operations**:
  - `HF_HOME`, `HF_HUB_CACHE`, `TFHUB_CACHE_DIR` all forced inside
    `Backend/models/` **before** any TF/HF import (import order enforced
    by convention: `src.config` first, everywhere).
  - No `trust_remote_code`; no downloaded Python is ever executed.

### Skill 5: Honest API Design (`honest_api`)
- **Domain**: Serving ML output without overstating it.
- **Tooling**: FastAPI, Pydantic, `slowapi`, CORS allow-lists.
- **Key operations**:
  - Real readiness (`/health` verifies + loads; 503 when not ready).
  - Model failure → generic 503; internals only in server logs.
  - `model_score` + `model_score_note` (≈51 % published val accuracy,
    not calibrated) instead of "confidence".
  - `evidence_regions` / `evidence_curve` computed from real per-frame
    probabilities — no gradients, no phonological heuristics.
  - `is_sufficient_speech` for the real "Not a speech" class.
  - Static ASR demo self-labels: `is_static_example` + `disclaimer`.
  - CORS restricted to `GET`/`POST` + required headers; rate-limited
    write endpoints.

### Skill 6: Verification Discipline (`quality_gates`)
- **Domain**: Proving behavior instead of asserting it.
- **Tooling**: `pytest`, `ruff`, `bandit`, `pip-audit`, `httpx`/`TestClient`.
- **Key operations**:
  - Tests run the **real model**; missing artifacts are a FAIL, not a skip.
  - Endpoints tested for schema, class order, determinism, 4xx/503
    semantics, and absence of stack traces in responses.
  - Security scans on every change; frontend gates via
    `npm run typecheck && npm run lint && npm run build`.

---

## 📚 Knowledge Base

- **Taxonomy served** (exact, never remapped): `Irish, Midlands,
  Northern, Scottish, Southern, Welsh, Not a speech`.
- **Model provenance**: keras.io example *English speaker accent
  recognition using transfer learning* (Fadi Badine), trained on
  OpenSLR-83 British Isles English Accents (120 speakers); YAMNet
  (AudioSet) as frame encoder. Full details in `models/model_manifest.json`.
- **Known limits**: ~51 % validation accuracy (utterance-level,
  non speaker-disjoint split), uncalibrated scores; the local demo
  corpus under `data/audio/` contains placeholder tone files, not speech.
