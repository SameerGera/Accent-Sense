# AccentSense: Explainable UK Regional Accent Detection in Speech

AccentSense is an explainable speech-processing research framework designed to analyze phonological and prosodic patterns in UK regional accent speech.

---

## 📌 Review 1 Current Status
- **Frontend Prototype**: Interactive dashboard (Audio recording/upload, attribution heatmap, regional accent breakdown).
- **Backend Architecture**: FastAPI REST service with endpoints for inference, temporal saliency extraction, and downstream ASR comparison.
- **ML Pipeline**: Modular training pipeline supporting Classical Baselines (MFCC + SVM/RF) and Deep Speech Representations (WavLM Base+ with Attentive Statistics Pooling).
- **Academic Rigor**: Strictly enforced **Speaker-Disjoint Splitting** to eliminate speaker memorization and acoustic leakage.

---

## 📁 Repository Structure
```
Accent Sense/
├── README.md
├── requirements.txt
├── curate_data.py             # Phase 2: Dataset curation & speaker-disjoint splitting
├── train_baseline.py          # Phase 1: Acoustic MFCC + SVM / Random Forest baseline
├── train_wavlm.py             # Phase 3: WavLM Base+ fine-tuning with ASP
├── explain_speech.py          # Phase 4: XAI attribution & AUDC verification
├── downstream_asr.py          # Phase 5: Downstream Whisper adaptation
├── run_api.py                 # FastAPI service launcher (port 8000)
├── checkpoints/               # Trained model weights (.pt)
├── data/splits/               # Speaker-disjoint CSV splits (train, val, test)
├── notebooks/                 # Google Colab GPU training notebook
├── reports/                   # Audit reports, XAI faithfulness, ASR benchmarks
└── src/
    ├── api/
    │   └── main.py            # FastAPI service (predict, explain, downstream-asr)
    ├── data/
    │   └── dataset.py         # VCTK + Mozilla Common Voice loader with speaker-disjoint splitting
    ├── explainability/
    │   └── saliency.py        # Frame-level gradient attribution & phonological mapping
    └── models/
        ├── baseline_mfcc.py   # MFCC feature extraction + Scikit-Learn classifiers
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
python run_api.py
```
Interactive API docs will be available at: `http://localhost:8000/docs`.

### 4. Downstream ASR Demonstration
AccentSense connects regional accent predictions to downstream speech recognition (e.g. OpenAI Whisper prompt conditioning) to evaluate reductions in Word Error Rate (WER).
