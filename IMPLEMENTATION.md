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

### Phase 2: Svarah Curation & 4-Class Regional Taxonomy (Ready for Execution)
- **Status**: 🟡 In Progress / Configured
- **Target Classes**:
  1. `Northern_Hindi` (Delhi / UP)
  2. `Central_MP` (Madhya Pradesh / Malwa / Bhopal)
  3. `Western_Gujarati` (Gujarat)
  4. `Southern_Tamil` (Tamil Nadu)
- **Deliverables**:
  - Download metadata from `ai4bharat/Svarah`.
  - Apply `label_regional_4class` mapping.
  - Implement 5-fold `StratifiedGroupKFold` on `speaker_id` ensuring zero speaker overlap.
  - Verify speaker counts ($\ge 20$ speakers per class).

### Phase 3: Deep Speech Representation (WavLM Base+ with ASP)
- **Status**: 🟡 Initialized
- **Deliverables**:
  - Model architecture ([`src/models/wavlm_classifier.py`](file:///d:/Docs_Back/Projects/Accent%20Sense/src/models/wavlm_classifier.py)):
    `microsoft/wavlm-base-plus` backbone (frozen) + Attentive Statistics Pooling (ASP) + MLP Head.
  - Training loop ([`train_wavlm.py`](file:///d:/Docs_Back/Projects/Accent%20Sense/train_wavlm.py)):
    Cross-Entropy Loss with balanced class weights, AdamW ($\text{LR}=10^{-4}$ for head, $\text{LR}=10^{-5}$ for backbone), Cosine Annealing scheduler, early stopping on validation Macro-F1.
  - Execution targets: Train on Google Colab / Kaggle T4 GPU or local GPU machine.

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
