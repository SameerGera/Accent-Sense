# AccentSense Multi-Agent Engineering Framework (`AGENTS.md`)

This document defines the specialized agent roles, communication protocols, and execution workflows for the **AccentSense** research and engineering team.

---

## 👥 Agent Roster & Specializations

```
                               ┌─────────────────────────────┐
                               │   Lead Research Director    │
                               │   (Architecture & Review)   │
                               └──────────────┬──────────────┘
                                              │
        ┌───────────────────┬─────────────────┼───────────────────┬───────────────────┐
        ▼                   ▼                 ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐ ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Speech Audio  │   │  Model Engine │ │Explainability │   │ Downstream ASR│   │ Fullstack API │
│   Curator     │   │   Trainer     │ │  (XAI) Agent  │   │ Adaptor Agent │   │ & Demo Agent  │
└───────────────┘   └───────────────┘ └───────────────┘   └───────────────┘   └───────────────┘
```

---

### 1. Lead Research Director (`agent_director`)
- **Role**: Coordinates overarching system design, academic rigor, and Review defense.
- **Responsibilities**:
  - Enforce zero-leakage protocols (speaker-disjoint splitting).
  - Define evaluation standards (Macro-F1, Balanced Accuracy, ECE, WERR).
  - Ensure ethical boundaries (phonological transfer vs. identity proof).

### 2. Speech Audio Curator (`agent_audio_curator`)
- **Role**: Data pipeline engineer & acoustic preprocessing specialist.
- **Responsibilities**:
  - Ingest AI4Bharat Svarah (`ai4bharat/Svarah`) and AccentDB datasets.
  - Standardize audio: 16 kHz, single-channel mono, 32-bit floating point, silence trimming.
  - Implement `StratifiedGroupKFold` on `speaker_id` to guarantee zero speaker leakage.
  - Map metadata to the 4 Regional Anchors: `Northern_Hindi`, `Central_MP`, `Western_Gujarati`, `Southern_Tamil`.

### 3. Model Engine Trainer (`agent_model_trainer`)
- **Role**: Deep speech representation & training specialist.
- **Responsibilities**:
  - Maintain classical acoustic baselines (39-dim MFCCs + $F_0$ + Energy $\to$ SVM/Random Forest).
  - Implement and train `WavLM Base+` (`microsoft/wavlm-base-plus`) with Attentive Statistics Pooling (ASP).
  - Execute Phase 2 (frozen backbone) and Phase 3 (top-2 layer fine-tuning).
  - Implement class-weighted cross-entropy loss and cosine annealing schedulers.

### 4. Explainability & Phonetics Agent (`agent_xai_phonetics`)
- **Role**: Interpretability & second-language acquisition (SLA) speech scientist.
- **Responsibilities**:
  - Compute frame-level temporal attributions via Layer Integrated Gradients (Captum).
  - Downsample attribution gradients to 20ms frames matching WavLM strides.
  - Correlate salient temporal regions with documented phonological transfer phenomena.
  - Measure explanation faithfulness via Area Under Deletion Curve (AUDC).

### 5. Downstream ASR Adaptor Agent (`agent_asr_adaptor`)
- **Role**: Speech recognition adaptation & evaluation engineer.
- **Responsibilities**:
  - Build prompt-conditioning wrappers for OpenAI `whisper-small` / `whisper-base`.
  - Format dynamic prompt prefixes: `"The following is English spoken with a [Regional_Accent] accent:"`.
  - Benchmark Word Error Rate ($\text{WER}$) and Character Error Rate ($\text{CER}$) before vs. after adaptation.

### 6. Fullstack API & Demo Agent (`agent_fullstack`)
- **Role**: Production API developer & interactive UI integrator.
- **Responsibilities**:
  - Maintain the FastAPI backend service (`src/api/main.py`).
  - Expose `/health`, `/api/predict`, and `/api/downstream-asr` with CORS.
  - Ensure real-time response times ($<500\text{ ms}$ for feature extraction and inference).
  - Bridge JSON responses to the React frontend dashboard.

---

## 🔄 Inter-Agent Workflow

1. **`agent_audio_curator`** prepares speaker-disjoint folds $\to$ passes dataset splits to **`agent_model_trainer`**.
2. **`agent_model_trainer`** trains the baseline and WavLM models $\to$ outputs `checkpoints/best_wavlm.pt`.
3. **`agent_xai_phonetics`** hooks into `best_wavlm.pt` $\to$ generates temporal attribution curves and phonetic cards.
4. **`agent_asr_adaptor`** uses prediction outputs to prompt Whisper $\to$ logs $\Delta \text{WER}$ improvements.
5. **`agent_fullstack`** packages the inference and explainability pipeline into FastAPI $\to$ connects to React.
