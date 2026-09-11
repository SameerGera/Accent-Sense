# AccentSense: Explainable UK regional accent Detection in UK regional accent Speech

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.1.0-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev)
[![Vercel Ready](https://img.shields.io/badge/Vercel-Deployable-000000?style=flat&logo=vercel&logoColor=white)](https://vercel.com)
[![Zero Speaker Leakage](https://img.shields.io/badge/Data%20Audit-0.0%25%20Leakage-success)](Backend/data/splits/split_audit_report.json)

> **Research Capstone Project | Department of Computer Science & Engineering (AI/ML), VIT Bhopal**  
> An explainable speech AI system that identifies native-language (L1) acoustic-phonetic influence in UK regional accent speech, grounds predictions in Second Language Acquisition (SLA) transfer phenomena via axiomatic attribution, and adapts downstream automatic speech recognition (ASR).

---

## ðŸŽ¯ Project Overview & Review 1 Highlights

UK regional accent is characterized by systematic phonological transfer from diverse substrate languages across Indo-Aryan and Dravidian language families. Conventional accent detection models operate as uninterpretable black boxes and frequently suffer from **speaker identity leakage** (evaluating on unseen utterances from the same speakers).

---

## ðŸ“ Repository Structure

```
Accent Sense/
â”œâ”€â”€ package.json                   # Root package pointing to Frontend build scripts
â”œâ”€â”€ vercel.json                    # 1-Click zero-config Vercel deployment configuration
â”œâ”€â”€ .gitignore                     # Comprehensive git ignore for Python and Node
â”œâ”€â”€ README.md                      # Primary project documentation (this file)
â”‚
â”œâ”€â”€ Backend/                       # Python Speech AI & REST API Service
â”‚   â”œâ”€â”€ src/
â”‚   â”‚   â”œâ”€â”€ api/main.py            # FastAPI endpoints (/health, /api/predict, /api/downstream-asr)
â”‚   â”‚   â”œâ”€â”€ asr/adaptation.py      # Whisper prompt prefixing & WERR evaluation
â”‚   â”‚   â”œâ”€â”€ data/                  # VCTK + Mozilla Common Voice dataset loader & speaker-disjoint curator
â”‚   â”‚   â”œâ”€â”€ explainability/        # Captum Integrated Gradients & SLA transfer rules
â”‚   â”‚   â””â”€â”€ models/                # WavLM Base+ with Attentive Statistics Pooling (ASP)
â”‚   â”œâ”€â”€ checkpoints/               # Trained neural network weights (.pt)
â”‚   â”œâ”€â”€ data/splits/               # Verified speaker-disjoint CSV splits (train, val, test)
â”‚   â”œâ”€â”€ notebooks/                 # Google Colab 1-Click GPU training notebook
â”‚   â”œâ”€â”€ reports/                   # Audit reports, XAI faithfulness, ASR benchmarks
â”‚   â”œâ”€â”€ curate_data.py             # Phase 2 dataset curation script
â”‚   â”œâ”€â”€ train_baseline.py          # Phase 1 classical baseline trainer
â”‚   â”œâ”€â”€ train_wavlm.py             # Phase 3 deep WavLM training pipeline
â”‚   â”œâ”€â”€ explain_speech.py          # Phase 4 XAI attribution & AUDC verification runner
â”‚   â”œâ”€â”€ downstream_asr.py          # Phase 5 downstream Whisper adaptation runner
â”‚   â””â”€â”€ run_api.py                 # FastAPI service launcher (port 8000)
â”‚
â””â”€â”€ Frontend/                      # Vite + React + TypeScript + Tailwind CSS UI
    â”œâ”€â”€ src/
    â”‚   â”œâ”€â”€ components/
    â”‚   â”‚   â”œâ”€â”€ Hero3D.tsx         # Interactive 3D hero visualization
    â”‚   â”‚   â””â”€â”€ LandingPage.tsx    # Live audio upload, waveform, saliency cards, ASR demo
    â”‚   â”œâ”€â”€ App.tsx                # Application root
    â”‚   â””â”€â”€ index.css              # Custom styling & animations
    â”œâ”€â”€ .env.example               # Environment variables template (VITE_API_URL)
    â”œâ”€â”€ package.json               # Frontend dependencies (lucide-react, tailwindcss, vite)
    â””â”€â”€ vite.config.ts             # Vite configuration with @/ path alias
```

---

## âš¡ Quickstart Guide

### 1. Launch the Backend API (Terminal 1)
```powershell
cd Backend
.venv\Scripts\python.exe run_api.py
```
*On Linux / macOS:*
```bash
cd Backend
source .venv/bin/activate
python run_api.py
```
- API Server runs at: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 2. Launch the Frontend UI (Terminal 2)
You can start the frontend either from the project root or from `Frontend/`:
```bash
# From project root:
npm run dev

# Or inside Frontend:
cd Frontend
npm run dev
```
- Web Application runs at: `http://localhost:5173`

---

## â˜ï¸ Deploying to Vercel (1-Click Deployment)

The repository is configured with a root `package.json` and `vercel.json` for deployment on Vercel:

1. Push your repository to GitHub.
2. In [Vercel](https://vercel.com), click **Add New Project** and import your `Accent-Sense` repository.
3. Vercel will automatically detect `vercel.json` and build the frontend:
   - **Framework Preset**: Vite
   - **Build Command**: `npm --prefix Frontend run build`
   - **Output Directory**: `Frontend/dist`
4. In **Project Settings $\to$ Environment Variables**, optionally set:
   ```text
   VITE_API_URL = https://your-accentsense-backend.onrender.com
   ```
   *(If not set, it defaults to `http://localhost:8000` for local development).*
5. Click **Deploy**!

---

## ðŸš€ 1-Click GPU Training on Google Colab

To train the full 95M-parameter `WavLM Base+` backbone on a free Google Colab NVIDIA T4 GPU:
1. Open [`Backend/notebooks/train_accentsense_colab.ipynb`](Backend/notebooks/train_accentsense_colab.ipynb) on [Google Colab](https://colab.research.google.com).
2. Set runtime to **GPU** (`Runtime -> Change runtime type -> T4 GPU`).
3. Click **Run All**.
4. The notebook runs `train_wavlm.py` for 15 epochs, runs `explain_speech.py`, and downloads `best_wavlm_accentsense.pt`.
5. Move `best_wavlm_accentsense.pt` into `Backend/checkpoints/` â€” the backend API immediately switches from prototype mode to live neural network inference!

---

## ðŸ“œ Academic Integrity & Citation
All experimental results reported in this repository adhere to strict academic honesty standards:
- The classical baseline results (**Macro-F1: 0.3207**) were empirically computed on local speaker-disjoint splits.
- Zero speaker leakage is proven in [`split_audit_report.json`](Backend/data/splits/split_audit_report.json).
- The prototype API clearly distinguishes calibrated SLA phonological benchmarks from GPU-trained model weights.

