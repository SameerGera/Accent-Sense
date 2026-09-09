# AccentSense Engineering & Domain Skills (`SKILLS.md`)

This document defines the specialized skillsets, computational tools, and domain knowledge bases utilized across the **AccentSense** pipeline.

---

## 🛠️ Core Skill Competencies

### Skill 1: Speech Preprocessing & Feature Extraction (`speech_dsp`)
- **Domain**: Digital Signal Processing (DSP) for audio.
- **Tooling**: `librosa`, `torchaudio`, `soundfile`, `scipy`.
- **Key Operations**:
  - Sample-rate normalization to 16 kHz mono.
  - 13/39-dimensional MFCC computation with first ($\Delta$) and second ($\Delta\Delta$) order derivatives.
  - $F_0$ Fundamental frequency tracking via Yin algorithm ($50\text{--}400\text{ Hz}$).
  - Energy/RMS temporal envelope extraction.
  - Spectrogram and Mel-filterbank representation.

### Skill 2: Self-Supervised Speech Modeling (`ssl_speech_architectures`)
- **Domain**: Transformer foundation models for acoustic representation.
- **Tooling**: `transformers`, `torch`, `torchaudio`.
- **Key Operations**:
  - Utilizing `microsoft/wavlm-base-plus` and `ai4bharat/indicwav2vec_v1_bilingual`.
  - Attentive Statistics Pooling (ASP) layer implementation:
    $$\alpha_t = \text{softmax}(w^T \tanh(W h_t + b))$$
    $$\mu = \sum_t \alpha_t h_t, \quad \sigma = \sqrt{\sum_t \alpha_t (h_t - \mu)^2}$$
  - Layer-freezing and top-$k$ fine-tuning strategies to prevent catastrophic forgetting.

### Skill 3: Acoustic-Phonetic SLA Transfer Analysis (`phonetic_transfer`)
- **Domain**: Applied Linguistics and Second-Language Acquisition (SLA).
- **Knowledge Base**:
  - **Northern Hindi**: Alveolar retroflexion ($/t, d/ \to [\ʈ, \ɖ]$), monophthongization ($/e\text{ɪ}/ \to [eː]$), glide merger ($/v/ \leftrightarrow /w/ \to [\ʋ]$).
  - **Central MP**: Moraic vowel lengthening on phrase finals, rising-falling melodic pitch contours, softened retroflex flaps ($[ɽ]$).
  - **Western Gujarati**: Breathy/murmured phonation transfer, $/z/ \to [dʒ] / [s]$ de-voicing, retroflex lateral flap ($[\ɭ]$).
  - **Southern Tamil**: Syllable-timed prosody, terminal vowel epenthesis ($[u]$), intervocalic stop voicing.

### Skill 4: Model Interpretability & Attribution (`speech_xai`)
- **Domain**: Axiomatic Explainable AI (XAI) for audio time-series.
- **Tooling**: `captum`, `scipy.ndimage`.
- **Key Operations**:
  - Layer Integrated Gradients computation on frame representations:
    $$\text{IG}_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i} d\alpha$$
  - Temporal smoothing via 1D Gaussian filters ($\sigma = 3.0 \approx 60\text{ ms}$).
  - Extraction of contiguous high-attribution regions (75th percentile threshold).
  - Explanation Faithfulness verification via Area Under Deletion Curves (AUDC).

### Skill 5: Downstream ASR Adaptation (`asr_adaptation`)
- **Domain**: Speech Recognition conditioning and evaluation.
- **Tooling**: `openai-whisper`, `jiwer`, `huggingface`.
- **Key Operations**:
  - Zero-shot / few-shot prompt injection on `whisper-small`:
    `prompt = "The following is English spoken with a Central Madhya Pradesh accent:"`
  - Word Error Rate (WER) and Character Error Rate (CER) calculation:
    $$\text{WER} = \frac{S + D + I}{N}$$
  - Relative Word Error Rate Reduction (WERR) benchmarking.

### Skill 6: Robust Evaluation & Leakage Control (`ml_eval_rigor`)
- **Domain**: Statistical Machine Learning and Experimental Validation.
- **Tooling**: `scikit-learn`.
- **Key Operations**:
  - `StratifiedGroupKFold` partitioning grouped strictly by `speaker_id`.
  - Class-weighted Cross-Entropy loss computation: $w_c = \frac{N}{C \times N_c}$.
  - Macro-averaged F1 score and Balanced Accuracy reporting.
  - Confusion matrix generation for intra-family vs cross-family error analysis.
