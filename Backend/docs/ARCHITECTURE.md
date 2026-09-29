# AccentSense Architecture Decision Record

**Status:** accepted · **Date:** 2026-09
**Supersedes:** the WavLM + PyTorch + Captum training/inference pipeline
(archived in `docs/archive/`, see the migration notes at the end of this file).

---

## 1. Context

The previous backend could not serve a single real prediction:

* `/api/predict` required a trained checkpoint
  (`checkpoints/best_wavlm_accentsense.pt`) that does not exist in the
  repository, so it silently fell through to a **hardcoded prototype
  response** (always "Northern", fabricated saliency curve).
* It hard-required CUDA on a machine with no NVIDIA GPU.
* The training pipeline (WavLM, Captum, VCTK curation) was research-only
  code with no artifacts, dragging in `torch`, `torchaudio`, `captum`,
  `transformers`, `datasets` and `scikit-learn`.

We needed an inference-only service that (a) runs a *real* pretrained
model on CPU, (b) never fabricates output, and (c) is verifiable offline.

## 2. Decision

Serve inference exclusively through the pretrained Hugging Face model
[`fbadine/uk_ireland_accent_classification`](https://huggingface.co/fbadine/uk_ireland_accent_classification)
(artifact of the keras.io example *English speaker accent recognition
using transfer learning*), with **YAMNet** as the frame encoder:

```
upload bytes
  -> secure decode/validate (src/audio/io.py): content sniffing, 16 kHz mono float32,
     explicit rejection of too-short/too-long/too-large input (never truncation)
  -> YAMNet (TF Hub google/yamnet/1, local copy in Backend/models/yamnet/)
       per-frame embeddings: N x 1024, frame hop measured at bootstrap (~0.48 s)
  -> accent classifier (TF SavedModel, Backend/models/uk_ireland_accent_classification/)
       per-frame probabilities: N x 7
  -> mean over frames, then argmax   (the reference implementation's aggregation)
  -> response: class, model score, per-class scores, temporal evidence
```

Key properties of this choice:

| Requirement | How it is met |
|---|---|
| Real predictions, no mocks | The mock fallback was deleted; model failure → HTTP 503 with a generic message, details only in server logs. |
| CPU-only | `tensorflow-cpu`; no CUDA check anywhere. |
| Exact class order | `Irish, Midlands, Northern, Scottish, Southern, Welsh, Not a speech` is pinned in `src/config.py`, cross-checked against the manifest at load time and against the model output shape at probe time. Never remapped. |
| "Not a speech" is real | It is an ordinary 7th class; the API exposes `is_sufficient_speech` so the UI can present an insufficient-speech state instead of an accent. |
| Offline inference | `src/config.py` sets `HF_HUB_OFFLINE=1`, `HF_HOME`, `HF_HUB_CACHE`, `TFHUB_CACHE_DIR` to `Backend/models/…` **at import time, before any TF/HF import**. Only `bootstrap_models.py` may touch the network. Proven by `tests/test_offline.py` (sockets disabled at interpreter start). |
| Supply-chain integrity | Bootstrap pins HF revision `ebe681a49e9506acdc4cf5312cb79b291275ab67`, writes `Backend/models/model_manifest.json` with SHA-256 per file + provenance; the service verifies every checksum **before** loading anything, and refuses class-order or frame-hop tampering. |
| No remote code execution | Loading is `tf.saved_model.load` + `hub.load` on local directories. `trust_remote_code` is never used; no downloaded Python is executed. |
| Duration bugs | `src/audio/io.py` validates the *decoded* duration and rejects out-of-range input; the old truncate-then-measure bug is gone. |
| FFmpeg safety | Argument arrays only (no shell), `-nostdin`, timeout, temp files with neutral names, unconditional cleanup. |
| Honest terminology | "confidence" → `model_score` (published val accuracy ≈ 51 %, non speaker-disjoint, not calibrated — stated verbatim in every response via `model_score_note`); "saliency/IG" → `evidence_curve` / `evidence_regions` derived from real per-frame probabilities. |

### Class-index ↔ speech rule

YAMNet's AudioSet class-map row 0 is "Speech". A frame counts as speech
iff `argmax(scores) == 0` — the identical labelling rule the classifier's
training data used (frames whose top event was not Speech were labelled
"Not a speech"). `speech_frame_ratio` surfaces this per prediction.

### Aggregation

`probabilities.mean(axis=0).argmax()` over **all** frames — exactly the
reference implementation in the keras.io example. Evidence regions are
contiguous runs (≥2 frames) whose per-frame probability for the predicted
class is ≥ max(p75, 0.50); a uniformly uncertain model therefore produces
no "high evidence" claims.

### Deliberately out of scope

* **No live ASR.** `/api/downstream-asr` serves a static prompt-conditioning
  illustration and says so (`is_static_example: true` + `disclaimer`).
  Running Whisper would reintroduce torch; if wanted later, add it as an
  isolated optional service.
* **No gradients/IG/Captum.** Evidence comes from forward-pass outputs only.
* **No phonological "explanation" cards.** The old rules were heuristic
  text not causally produced by the model; regions now carry neutral
  `label`/`detail` strings computed from real scores.
* **No training code.** The WavLM/Captum/VCTK pipeline was archived.

## 3. Consequences

* `Backend/models/` (git-ignored except the manifest) must be populated
  once with `python bootstrap_models.py`; `/health` reports readiness and
  answers 503 until then.
* Python must be 3.10–3.13 (TensorFlow 2.21 wheels); the project venv is
  created with `uv venv --python 3.12`.
* Published model quality bounds the product: ~51 % validation accuracy
  (utterance-level, non speaker-disjoint split). The API and UI say this
  out loud rather than implying calibrated confidence.

## 4. Migration notes (what was removed and why)

| Removed | Replaced by |
|---|---|
| `src/models/wavlm_classifier.py`, `train_wavlm.py`, `train_local_directml.py`, `checkpoints/` | pretrained SavedModel in `Backend/models/` |
| `src/explainability/` (Captum IG, phonetic mapping) | `AccentModelService.temporal_evidence` / `evidence_regions` |
| `download_data.py`, `src/data/`, `curate_data.py`, splits/reports | nothing (inference-only service) |
| `predict_accent.py`, `explain_speech.py`, `downstream_asr.py`, notebooks | `bootstrap_models.py`, `run_api.py`, tests |
| `torch`, `torchaudio`, `captum`, `transformers`, `datasets`, `scikit-learn` | `tensorflow-cpu`, `tensorflow-hub`, `huggingface_hub` |

Old research documents live in `docs/archive/` as history, not as
instructions — `AGENTS.md`, `SKILLS.md` and the READMEs describe the
current system only.
