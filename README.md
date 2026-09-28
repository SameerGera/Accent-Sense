# AccentSense: Explainable UK Regional Accent Detection in Speech

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.1.0-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev)
[![Vercel Ready](https://img.shields.io/badge/Vercel-Deployable-000000?style=flat&logo=vercel&logoColor=white)](https://vercel.com)
[![Zero Speaker Leakage](https://img.shields.io/badge/Data%20Audit-0.0%25%20Leakage-success)](Backend/data/splits/split_audit_report.json)

> **Research Capstone Project | Department of Computer Science & Engineering (AI/ML), VIT Bhopal**  
> An explainable speech AI system that identifies acoustic-phonetic patterns in UK regional accent speech, grounds predictions in phonological analysis via axiomatic attribution, and adapts downstream automatic speech recognition (ASR).

---

## 🎯 Project Overview & Review 1 Highlights

UK regional accents are characterized by systematic phonological and prosodic variation across diverse speaker populations. Conventional accent detection models operate as uninterpretable black boxes and frequently suffer from **speaker identity leakage** (evaluating on unseen utterances from the same speakers).

---

## 📁 Repository Structure

```
Accent Sense/
├── package.json                   # Root package pointing to Frontend build scripts
├── vercel.json                    # 1-Click zero-config Vercel deployment configuration
├── .gitignore                     # Comprehensive git ignore for Python and Node
├── README.md                      # Primary project documentation (this file)
│
├── Backend/                       # Python Speech AI & REST API Service
│   ├── src/
│   │   ├── api/main.py            # FastAPI endpoints (/health, /api/predict, /api/downstream-asr)
│   │   ├── asr/adaptation.py      # Whisper prompt prefixing & WERR evaluation
│   │   ├── data/                  # VCTK + Mozilla Common Voice dataset loader & speaker-disjoint curator
│   │   ├── explainability/        # Captum Integrated Gradients & phonological transfer rules
│   │   └── models/                # WavLM Base+ with Attentive Statistics Pooling (ASP)
│   ├── checkpoints/               # Trained neural network weights (.pt)
│   ├── data/splits/               # Verified speaker-disjoint CSV splits (train, val, test)
│   ├── notebooks/                 # Google Colab 1-Click GPU training notebook
│   ├── reports/                   # Audit reports, XAI faithfulness, ASR benchmarks
│   ├── curate_data.py             # Phase 2 dataset curation script
│   ├── train_baseline.py          # Phase 1 classical baseline trainer
│   ├── train_wavlm.py             # Phase 3 deep WavLM training pipeline
│   ├── explain_speech.py          # Phase 4 XAI attribution & AUDC verification runner
│   ├── downstream_asr.py          # Phase 5 downstream Whisper adaptation runner
│   └── run_api.py                 # FastAPI service launcher (port 8000)
│
└── Frontend/                      # Vite + React + TypeScript + Tailwind CSS UI
    ├── src/
    │   ├── components/
    │   │   ├── Hero3D.tsx         # Interactive 3D hero visualization
    │   │   └── LandingPage.tsx    # Live audio upload, waveform, saliency cards, ASR demo
    │   ├── App.tsx                # Application root
    │   └── index.css              # Custom styling & animations
    ├── .env.example               # Environment variables template (VITE_API_URL)
    ├── package.json               # Frontend dependencies (lucide-react, tailwindcss, vite)
    └── vite.config.ts             # Vite configuration with @/ path alias
```

---

## ⚡ Quickstart Guide

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r Backend/requirements.txt
```

### 2. Download Data & Build Manifests
```bash
huggingface-cli login
cd Backend
python download_data.py
```
This downloads UK accent audio data from Hugging Face (`jspaulsen/vctk`) and creates speaker-disjoint train/val/test splits at `Backend/data/manifests/`.

### 3. Train the Model (CUDA GPU required)
```bash
python train_wavlm.py --epochs 20 --batch_size 8
```
Output: `Backend/checkpoints/best_wavlm_accentsense.pt`

### 4. Launch the Backend API (Terminal 1)
```bash
python run_api.py
```
- API Server runs at: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 5. Launch the Frontend UI (Terminal 2)
```bash
cd Frontend
npm install
npm run dev
```
- Web Application runs at: `http://localhost:5173`

---

## ☁️ Deploying to Vercel (1-Click Deployment)

The repository is configured with a root `package.json` and `vercel.json` for deployment on Vercel:

1. Push your repository to GitHub.
2. In [Vercel](https://vercel.com), click **Add New Project** and import your `Accent-Sense` repository.
3. Vercel will automatically detect `vercel.json` and build the frontend:
   - **Framework Preset**: Vite
   - **Build Command**: `npm --prefix Frontend run build`
   - **Output Directory**: `Frontend/dist`
4. In **Project Settings → Environment Variables**, optionally set:
   ```text
   VITE_API_URL = https://your-accentsense-backend.onrender.com
   ```
   *(If not set, it defaults to `http://localhost:8000` for local development).*
5. Click **Deploy**!

---

## 🚀 Training on Google Colab (Alternative)

To train on a free Google Colab NVIDIA T4 GPU instead of locally:
1. Open [`Backend/notebooks/train_accentsense_colab_final_2.ipynb`](Backend/notebooks/train_accentsense_colab_final_2.ipynb) on [Google Colab](https://colab.research.google.com).
2. Set runtime to **GPU** (`Runtime -> Change runtime type -> T4 GPU`).
3. Click **Run All**.
4. The notebook downloads data, builds manifests, and trains for 20 epochs.
5. Download `best_wavlm_accentsense.pt` from Colab and place it in `Backend/checkpoints/`.

**For local CUDA training, use the Quickstart Guide above (`python train_wavlm.py`).**

---

## 📜 Academic Integrity & Citation
All experimental results reported in this repository adhere to strict academic honesty standards:
- The classical baseline results (**Macro-F1: 0.3207**) were empirically computed on local speaker-disjoint splits.
- Zero speaker leakage is proven in [`split_audit_report.json`](Backend/data/splits/split_audit_report.json).
- The prototype API clearly distinguishes calibrated phonological benchmarks from GPU-trained model weights.
