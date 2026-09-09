# AccentSense Architecture & Tensor Contracts (`ARCHITECTURE.md`)

This document details the complete technical architecture, tensor transformations, pooling mathematics, and API communication contracts.

---

## 🏗️ End-to-End System Diagram

```
[Raw Audio Input] 
  │ 16 kHz Mono PCM, ~10 seconds
  │ Shape: [Batch, Time = 160,000]
  ▼
┌────────────────────────────────────────────────────────┐
│ Pretrained WavLM Base+ (Microsoft)                     │
│ 12 Transformer Layers | Hidden Dim = 768               │
│ Temporal Stride = 320 samples (20ms frames @ 50 Hz)    │
│ Output: Last Hidden State [Batch, Frames = 500, 768]   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Attentive Statistics Pooling (ASP)                     │
│ 1. Raw Attention Score: e_t = w^T tanh(W h_t + b)      │
│ 2. Normalized Weights:  α_t = softmax(e_t)             │
│ 3. Weighted Mean:       μ   = Σ α_t h_t                │
│ 4. Weighted Std Dev:    σ   = sqrt(Σ α_t (h_t - μ)^2)  │
│ Concatenation: [μ, σ] ➔ Shape: [Batch, 1536]           │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ Classification Head (MLP) │   │ Explainability Engine     │
│ LayerNorm(1536)           │   │ Layer Integrated Gradients│
│ Dropout(0.3)              │   │ Frame-Level Gradients     │
│ Linear(1536 ➔ 256)        │   │ Saliency Curve [500 frames│
│ GELU Activation           │   │ Top Segments (>= 100ms)   │
│ Dropout(0.15)             │   └─────────────┬─────────────┘
│ Linear(256 ➔ 4 Classes)   │                 │
│ Output Logits [Batch, 4]  │                 ▼
└────────────┬──────────────┘   ┌───────────────────────────┐
             │                  │ Phonetic Rule Matcher     │
             ▼                  │ Maps time to SLA markers  │
┌───────────────────────────┐   │ [MP, Gujarati, Hindi, ...]│
│ Softmax Probabilities     │   └─────────────┬─────────────┘
│ Northern_Hindi:    P1     │                 │
│ Central_MP:        P2     │                 │
│ Western_Gujarati:  P3     │                 │
│ Southern_Tamil:    P4     │                 │
└────────────┬──────────────┘                 │
             │                                │
             └────────────────┬───────────────┘
                              ▼
        ┌───────────────────────────────────────────┐
        │ JSON Payload to React Frontend & ASR Loop │
        └───────────────────────────────────────────┘
```

---

## 📐 Tensor Dimension Contracts

| Stage | Input Shape | Transformation | Output Shape | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Input Waveform** | - | Audio Loading & 16kHz Mono Resampling | `[B, 160000]` | None |
| **WavLM Backbone** | `[B, 160000]` | 7-layer CNN Feature Extractor + 12-layer Transformer | `[B, 500, 768]` | $\sim 94.4\text{M}$ (Frozen) |
| **Attention Scores** | `[B, 500, 768]` | `Linear(768, 128)` $\to$ `Tanh` $\to$ `Linear(128, 1)` | `[B, 500, 1]` | $\sim 98.4\text{k}$ (Trainable) |
| **ASP Pooling** | `[B, 500, 768]`, `[B, 500, 1]` | Attention-Weighted Mean $\oplus$ Weighted Std Dev | `[B, 1536]` | Parameter-free |
| **MLP Projection** | `[B, 1536]` | `LayerNorm(1536)` $\to$ `Dropout(0.3)` $\to$ `Linear(1536, 256)` $\to$ `GELU` | `[B, 256]` | $\sim 396\text{k}$ (Trainable) |
| **Logits Output** | `[B, 256]` | `Dropout(0.15)` $\to$ `Linear(256, 4)` | `[B, 4]` | $\sim 1.0\text{k}$ (Trainable) |
| **Softmax** | `[B, 4]` | `Softmax(dim=-1)` | `[B, 4]` | Parameter-free |

**Total Trainable Parameters in Frozen Mode**: $\sim 495,000$ ($\approx 0.5\text{M}$ params), ensuring rapid convergence without overfitting small datasets.

---

## 🌐 FastAPI JSON Communication Contract

### Endpoint: `POST /api/predict`
```json
{
  "is_mock_prototype": false,
  "predicted_influence": "Central_MP",
  "language_family": "Indo-Aryan (Madhya Pradesh / Malwa / Bhopal)",
  "confidence": 0.784,
  "all_scores": {
    "Central_MP": 0.784,
    "Northern_Hindi": 0.121,
    "Western_Gujarati": 0.058,
    "Southern_Tamil": 0.037
  },
  "timestamps": [0.0, 0.02, 0.04, 0.06, "..."],
  "saliency_curve": [0.12, 0.14, 0.89, 0.94, "..."],
  "salient_regions": [
    {
      "start_time_sec": 1.4,
      "end_time_sec": 2.1,
      "duration_sec": 0.7,
      "salience_score": 0.91,
      "linguistic_phenomenon": "Moraic Vowel Lengthening",
      "phonetic_explanation": "Elongated vowel duration on phrase-final syllables characteristic of Malwa/Central Hindi English."
    },
    {
      "start_time_sec": 3.2,
      "end_time_sec": 3.8,
      "duration_sec": 0.6,
      "salience_score": 0.85,
      "linguistic_phenomenon": "Intonation Pitch Modulation",
      "phonetic_explanation": "Rising-falling melodic pitch contour at clause ending (Central Indian intonation)."
    }
  ]
}
```
