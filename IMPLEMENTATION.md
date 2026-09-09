# AccentSense Implementation Roadmap (`IMPLEMENTATION.md`)

This document outlines the sequential phases, concrete deliverables, and milestone checkpoints for **AccentSense**.

---

## 🗺️ Phased Implementation Plan

```
[Phase 1: Baselines & Environment] ──► [Phase 2: Data Pipeline & 4-Class Curation]
                │                                            │
                ▼                                            ▼
[Phase 3: WavLM + ASP Model Training] ──► [Phase 4: Explainability & Saliency]
                │                                            │
                ▼                                            ▼
[Phase 5: Downstream Whisper ASR]     ──► [Phase 6: Fullstack FastAPI & React Demo]
```

---

### Phase 1: Environment & Classical Baselines (Completed & Verified)
- **Status**: ✅ Done
- **Deliverables**:
  - Python 3.11 virtual environment initialized via `uv`.
  - `.gitignore` configured to exclude audio binaries and virtualenv.
  - Classical Acoustic Baseline ([`src/models/baseline_mfcc.py`](file:///d:/Docs_Back/Projects/Accent%20Sense/src/models/baseline_mfcc.py)): 39-dim MFCCs + $F_0$ + Energy $\to$ SVM (RBF) / Random Forest.
  - Verified with speaker-disjoint splitting script ([`train_baseline.py`](file:///d:/Docs_Back/Projects/Accent%20Sense/train_baseline.py)).

### Phase 2: Svarah Curation & 4-Class Regional Taxonomy (Completed & Verified)
- **Status**: ✅ Done
- **Target Classes**:
  1. `Northern_Hindi` (Delhi / UP)
  2. `Central_MP` (Madhya Pradesh / Malwa / Bhopal)
  3. `Western_Gujarati` (Gujarat)
  4. `Southern_Tamil` (Tamil Nadu)
- **Deliverables**:
  - Implemented curation runner ([`curate_data.py`](file:///d:/Docs_Back/Projects/Accent%20Sense/curate_data.py) & [`src/data/curate_dataset.py`](file:///d:/Docs_Back/Projects/Accent%20Sense/src/data/curate_dataset.py)).
  - Executed 5-fold `StratifiedGroupKFold` grouped strictly on `speaker_id`.
  - Exported verified splits:
    - Train: 288 samples (24 speakers)
    - Val: 96 samples (8 speakers)
    - Test: 96 samples (8 speakers)
  - Zero speaker leakage mathematically proven in [`data/splits/split_audit_report.json`](file:///d:/Docs_Back/Projects/Accent%20Sense/data/splits/split_audit_report.json).

### Phase 3: Deep Speech Representation (WavLM Base+ with ASP) (Completed & Verified)
- **Status**: ✅ Pipeline Implemented & Sanity-Checked
- **Deliverables**:
  - Architecture ([`src/models/wavlm_classifier.py`](file:///d:/Docs_Back/Projects/Accent%20Sense/src/models/wavlm_classifier.py)):
    `microsoft/wavlm-base-plus` frozen backbone + Attentive Statistics Pooling (ASP) + MLP Head ($\sim 495\text{k}$ trainable params).
  - Training loop ([`train_wavlm.py`](file:///d:/Docs_Back/Projects/Accent%20Sense/train_wavlm.py)):
    Cross-Entropy Loss with class weights, AdamW optimizer, Cosine warmup scheduler, and best checkpoint saving (`checkpoints/best_wavlm_accentsense.pt`).
  - Verified with `--dry_run` sanity check on the curated splits.

### Phase 4: Explainability & Phonetic Grounding
- **Status**: 🟡 Initialized
- **Deliverables**:
  - Saliency engine ([`src/explainability/saliency.py`](file:///d:/Docs_Back/Projects/Accent%20Sense/src/explainability/saliency.py)):
    Frame-level Layer Integrated Gradients downsampled to 20ms steps.
  - Top-region extraction: identifies contiguous high-saliency segments ($\ge 100\text{ ms}$).
  - SLA transfer mapping: maps segments to Central MP, Gujarati, Northern Hindi, and Tamil phonetic markers.
  - Faithfulness test: Area Under Deletion Curve (AUDC) verification.

### Phase 5: Downstream ASR Adaptation (Whisper Conditioning)
- **Status**: ⚪ Planned (Post Review 1)
- **Deliverables**:
  - Run unadapted `whisper-small` on test split $\to$ calculate baseline $\text{WER}_{\text{base}}$.
  - Inject accent conditioning prompt: `"The following is English spoken with a [Regional_Accent] accent:"` $\to$ calculate $\text{WER}_{\text{adapted}}$.
  - Compute Relative Word Error Rate Reduction ($\text{WERR}$).

### Phase 6: Fullstack Demo & API Integration
- **Status**: 🟡 Prototype Ready
- **Deliverables**:
  - FastAPI server ([`src/api/main.py`](file:///d:/Docs_Back/Projects/Accent%20Sense/src/api/main.py)) running on `http://localhost:8000`.
  - Routes:
    - `POST /api/predict`: Accepts audio $\to$ returns confidence, saliency curve, phonetic cards.
    - `POST /api/downstream-asr`: Returns transcript adaptation comparison.
  - React frontend dashboard integration:
    Audio recording, live spectrogram heatmap, attribution cards, and ASR before/after view.
