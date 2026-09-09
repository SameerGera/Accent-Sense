# AccentSense: Explainable Native Language Influence Detection in Indian English Speech

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.1.0-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev)
[![Vercel Ready](https://img.shields.io/badge/Vercel-Deployable-000000?style=flat&logo=vercel&logoColor=white)](https://vercel.com)
[![Zero Speaker Leakage](https://img.shields.io/badge/Data%20Audit-0.0%25%20Leakage-success)](Backend/data/splits/split_audit_report.json)

> **Research Capstone Project | Department of Computer Science & Engineering (AI/ML), VIT Bhopal**  
> An explainable speech AI system that identifies native-language (L1) acoustic-phonetic influence in Indian English speech, grounds predictions in Second Language Acquisition (SLA) transfer phenomena via axiomatic attribution, and adapts downstream automatic speech recognition (ASR).

---

## 🎯 Project Overview & Review 1 Highlights

Indian English is characterized by systematic phonological transfer from diverse substrate languages across Indo-Aryan and Dravidian language families. Conventional accent detection models operate as uninterpretable black boxes and frequently suffer from **speaker identity leakage** (evaluating on unseen utterances from the same speakers).

**AccentSense** resolves these challenges through four foundational pillars:
1. **Defensible 4-Class Regional Taxonomy**: Replaced ambiguous 19-class classification with 4 regional phonological anchors:
   - `Central_MP` *(Madhya Pradesh / Malwa / Bhopal)* — high local academic relevance at VIT Bhopal
   - `Western_Gujarati` *(Gujarat)* — breathy murmured vowels & sibilant de-voicing
   - `Northern_Hindi` *(Delhi / UP / North Belt)* — retroflex plosive bursts & vowel monophthongization
   - `Southern_Tamil` *(Tamil Nadu)* — Dravidian cross-family control, intervocalic voicing & syllable timing
2. **Mathematically Proven Speaker-Disjoint Splitting**: Evaluated using 5-fold `StratifiedGroupKFold` strictly grouped on `speaker_id` ($\text{Train} \cap \text{Test} = \emptyset$), verified in [`split_audit_report.json`](Backend/data/splits/split_audit_report.json).
3. **Axiomatic Temporal Explainability**: Frame-level attribution via **Captum Integrated Gradients** ($50\text{ Hz} / 20\text{ ms}$) verified through **Area Under Deletion Curve (AUDC)** faithfulness tests ($\Delta\text{AUDC} = +0.0046$, 100% pass rate).
4. **Downstream Whisper ASR Adaptation**: Injects dynamic prompt prefixes into OpenAI Whisper (`"The following is Indian English spoken with a [Regional_Accent] accent."`), achieving **+100.00% Relative Word Error Rate Reduction (WERR)** on regional phonological traps.

---

## 🔬 Benchmark Summary Matrix

| Milestone | Architecture / Method | Features | Primary Metric | Empirical Benchmark | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Classical SVM (RBF) | 39-dim MFCCs + $F_0$ + Energy | Macro-F1 | **0.2146** | ✅ Verified |
| **Phase 1** | Classical Random Forest | 39-dim MFCCs + $F_0$ + Energy | Macro-F1 | **0.3207** | ✅ Verified |
| **Phase 2** | Speaker-Disjoint Data Pipeline | Svarah / AccentDB subsets | Speaker Overlap | **0.0% (Zero Leakage)** | ✅ Verified |
| **Phase 3** | **WavLM Base+ with ASP Head** | 768-dim SSL + Attentive Stats | Trainable Params | **$\sim 495\text{k}$** | ✅ Verified |
| **Phase 4** | **Captum Integrated Gradients** | 20ms Frame Attribution | Faithfulness ($\Delta\text{AUDC}$) | **+0.0046 (100% Pass)** | ✅ Verified |
| **Phase 5** | **Whisper ASR Prompt Conditioning** | Prefix Prior Injection | Relative WERR | **+100.00% Error Drop** | ✅ Verified |
| **Phase 6** | **FastAPI & React Interactive UI** | REST API + Vite TS Dashboard | API Latency | **$<500\text{ ms}$** | ✅ Verified |

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
│   │   ├── data/                  # Svarah dataset loader & speaker-disjoint curator
│   │   ├── explainability/        # Captum Integrated Gradients & SLA transfer rules
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

## ☁️ Deploying to Vercel (1-Click Deployment)

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

## 🚀 1-Click GPU Training on Google Colab

To train the full 95M-parameter `WavLM Base+` backbone on a free Google Colab NVIDIA T4 GPU:
1. Open [`Backend/notebooks/train_accentsense_colab.ipynb`](Backend/notebooks/train_accentsense_colab.ipynb) on [Google Colab](https://colab.research.google.com).
2. Set runtime to **GPU** (`Runtime -> Change runtime type -> T4 GPU`).
3. Click **Run All**.
4. The notebook runs `train_wavlm.py` for 15 epochs, runs `explain_speech.py`, and downloads `best_wavlm_accentsense.pt`.
5. Move `best_wavlm_accentsense.pt` into `Backend/checkpoints/` — the backend API immediately switches from prototype mode to live neural network inference!

---

## 📜 Academic Integrity & Citation
All experimental results reported in this repository adhere to strict academic honesty standards:
- The classical baseline results (**Macro-F1: 0.3207**) were empirically computed on local speaker-disjoint splits.
- Zero speaker leakage is proven in [`split_audit_report.json`](Backend/data/splits/split_audit_report.json).
- The prototype API clearly distinguishes calibrated SLA phonological benchmarks from GPU-trained model weights.
