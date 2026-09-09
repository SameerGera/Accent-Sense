# AccentSense: Explainable Native Language Influence Detection in Indian English Speech

AccentSense is an explainable speech-processing research framework designed to analyze phonological transfer and prosodic characteristics (native-language influence) in Indian English speech.

---

## 📌 Review 1 Current Status
- **Frontend Prototype**: Interactive dashboard (Audio recording/upload, attribution heatmap, L1 influence breakdown).
- **Backend Architecture**: FastAPI REST service with endpoints for inference, temporal saliency extraction, and downstream ASR comparison.
- **ML Pipeline**: Modular training pipeline supporting Classical Baselines (MFCC + SVM/RF) and Deep Speech Representations (WavLM Base+ with Attentive Statistics Pooling).
- **Academic Rigor**: Strictly enforced **Speaker-Disjoint Splitting** to eliminate speaker memorization and acoustic leakage.

---

## 📁 Repository Structure
```
Accent Sense/
├── README.md
├── requirements.txt
├── train_baseline.py         # Phase 1: Acoustic MFCC + SVM / Random Forest baseline
├── train_wavlm.py            # Phase 2 & 3: WavLM Base+ fine-tuning with ASP
└── src/
    ├── api/
    │   └── main.py           # FastAPI service (predict, explain, downstream-asr)
    ├── data/
    │   └── svarah_dataset.py # Svarah dataset loader with speaker-disjoint splitting
    ├── explainability/
    │   └── saliency.py       # Frame-level gradient attribution & phonetic transfer mapping
    └── models/
        ├── baseline_mfcc.py  # MFCC feature extraction + Scikit-Learn classifiers
        └── wavlm_classifier.py # WavLM Base+ with Attentive Statistics Pooling
```

---

## 🚀 Quickstart

### 1. Environment Setup (Python 3.11 recommended)
```powershell
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. Run Baseline Experiment (Phase 1)
```powershell
python train_baseline.py
```

### 3. Launch FastAPI Backend
```powershell
uvicorn src.api.main:app --reload --port 8000
```
Interactive API docs will be available at: `http://localhost:8000/docs`.

### 4. Downstream ASR Demonstration
AccentSense connects L1 influence predictions to downstream speech recognition (e.g. OpenAI Whisper prompt conditioning) to evaluate reductions in Word Error Rate (WER).
