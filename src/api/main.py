"""
AccentSense FastAPI Backend Service
Provides endpoints for audio upload, L1 influence prediction, explainability saliency, and downstream ASR comparison.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import io
import os
import torch
import torchaudio
import numpy as np

app = FastAPI(
    title="AccentSense API",
    description="Explainable Native Language Influence Detection and ASR Adaptation Service",
    version="0.1.0",
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["Northern_Hindi", "Central_MP", "Western_Gujarati", "Southern_Tamil"]
FAMILY_MAP = {
    "Northern_Hindi": "Indo-Aryan (Delhi / UP)",
    "Central_MP": "Indo-Aryan (Madhya Pradesh / Malwa / Bhopal)",
    "Western_Gujarati": "Indo-Aryan (Gujarat)",
    "Southern_Tamil": "Dravidian (Tamil Nadu)",
}

# In Review 1, models are in training/prototype phase.
MODEL_LOADED = False
device = "cuda" if torch.cuda.is_available() else "cpu"


class SalientRegion(BaseModel):
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    salience_score: float
    linguistic_phenomenon: str
    phonetic_explanation: str


class PredictionResponse(BaseModel):
    is_mock_prototype: bool
    predicted_influence: str
    language_family: str
    confidence: float
    all_scores: Dict[str, float]
    salient_regions: List[SalientRegion]
    timestamps: List[float]
    saliency_curve: List[float]


class DownstreamASRResponse(BaseModel):
    baseline_transcript: str
    accent_adapted_transcript: str
    adaptation_strategy: str
    detected_accent_profile: str
    phonetic_corrections_noted: List[str]


from src.models.wavlm_classifier import WavLMForL1Influence
from src.explainability.saliency import compute_temporal_saliency, map_explanations_to_phonetics

CHECKPOINT_PATH = os.path.join("checkpoints", "best_wavlm_accentsense.pt")
wavlm_model: Optional[WavLMForL1Influence] = None


def get_live_model() -> Optional[WavLMForL1Influence]:
    global wavlm_model
    if wavlm_model is not None:
        return wavlm_model
    if os.path.exists(CHECKPOINT_PATH):
        try:
            print(f"[API] Loading trained checkpoint from: {CHECKPOINT_PATH}")
            model = WavLMForL1Influence(num_classes=len(CLASSES), freeze_encoder=True)
            state = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(state, strict=False)
            model.eval()
            model.to(device)
            wavlm_model = model
            return wavlm_model
        except Exception as e:
            print(f"[API Warning] Failed to load checkpoint: {e}")
            return None
    return None


@app.get("/health")
def health_check():
    live_model = get_live_model()
    return {
        "status": "healthy",
        "service": "AccentSense ML Engine",
        "device": device,
        "model_loaded": live_model is not None,
        "supported_classes": CLASSES,
        "phase": "Review 1 - Prototype & Inference Service",
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_speech(file: UploadFile = File(...)):
    """
    Receives an audio file (10-20 sec speech) and returns the predicted
    native-language influence profile with temporal saliency explanations.
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a")):
        raise HTTPException(status_code=400, detail="Invalid audio file format. Please upload WAV/MP3/OGG/FLAC.")

    audio_bytes = await file.read()
    
    try:
        import soundfile as sf
        audio_io = io.BytesIO(audio_bytes)
        try:
            data, sr = sf.read(audio_io)
            if data.ndim > 1:
                data = np.mean(data, axis=-1)
            waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        except Exception:
            audio_io.seek(0)
            waveform, sr = torchaudio.load(audio_io)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        duration_sec = waveform.shape[1] / 16000.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing error: {str(e)}")

    live_model = get_live_model()
    if live_model is not None:
        try:
            saliency_result = compute_temporal_saliency(
                model=live_model,
                waveform=waveform.squeeze(0),
                device=device,
                method="gradient",  # fast for web latency (<500ms)
            )
            pred_idx = saliency_result["predicted_class_index"]
            pred_lang = CLASSES[pred_idx]
            pred_confidence = float(saliency_result["probabilities"][pred_idx])
            all_scores = {CLASSES[i]: round(float(p), 4) for i, p in enumerate(saliency_result["probabilities"])}
            annotated_regions = map_explanations_to_phonetics(
                language_name=pred_lang,
                salient_regions=saliency_result["salient_regions"],
            )
            regions_pydantic = [
                SalientRegion(
                    start_time_sec=r["start_time_sec"],
                    end_time_sec=r["end_time_sec"],
                    duration_sec=r["duration_sec"],
                    salience_score=r["salience_score"],
                    linguistic_phenomenon=r["linguistic_phenomenon"],
                    phonetic_explanation=r["phonetic_explanation"],
                )
                for r in annotated_regions
            ]
            return PredictionResponse(
                is_mock_prototype=False,
                predicted_influence=pred_lang,
                language_family=FAMILY_MAP.get(pred_lang, "Indian English"),
                confidence=round(pred_confidence, 4),
                all_scores=all_scores,
                salient_regions=regions_pydantic,
                timestamps=saliency_result["timestamps"],
                saliency_curve=saliency_result["saliency_curve"],
            )
        except Exception as e:
            print(f"[API Inference Error] Fallback to calibrated prototype: {e}")

    # Fallback to calibrated prototype when trained checkpoint is not yet present
    num_frames = int(duration_sec * 50)
    timestamps = [round(i * 0.02, 2) for i in range(num_frames)]
    
    # Generate realistic saliency pattern
    base_saliency = np.sin(np.linspace(0, 3 * np.pi, num_frames)) ** 2
    noise = np.random.normal(0, 0.05, num_frames)
    saliency = np.clip(base_saliency + noise, 0.0, 1.0)
    saliency_list = [round(float(s), 3) for s in saliency]

    pred_lang = "Central_MP"
    salient_regions = [
        SalientRegion(
            start_time_sec=1.4,
            end_time_sec=2.1,
            duration_sec=0.7,
            salience_score=0.91,
            linguistic_phenomenon="Moraic Vowel Lengthening",
            phonetic_explanation="Elongated vowel duration on phrase-final syllables characteristic of Malwa/Central Hindi English.",
        ),
        SalientRegion(
            start_time_sec=3.2,
            end_time_sec=3.8,
            duration_sec=0.6,
            salience_score=0.85,
            linguistic_phenomenon="Intonation Pitch Modulation",
            phonetic_explanation="Rising-falling melodic pitch contour at clause ending (Central Indian intonation).",
        ),
        SalientRegion(
            start_time_sec=5.1,
            end_time_sec=5.6,
            duration_sec=0.5,
            salience_score=0.79,
            linguistic_phenomenon="Softened Retroflex Flap",
            phonetic_explanation="Intervocalic retroflex articulation with moderated burst aspiration [ɽ].",
        ),
    ]

    return PredictionResponse(
        is_mock_prototype=True,
        predicted_influence=pred_lang,
        language_family=FAMILY_MAP.get(pred_lang, "Indo-Aryan (Central MP)"),
        confidence=0.78,
        all_scores={
            "Central_MP": 0.78,
            "Northern_Hindi": 0.12,
            "Western_Gujarati": 0.06,
            "Southern_Tamil": 0.04,
        },
        salient_regions=salient_regions,
        timestamps=timestamps,
        saliency_curve=saliency_list,
    )


from src.asr.adaptation import WhisperAccentAdaptor

asr_adaptor = WhisperAccentAdaptor()


@app.post("/api/downstream-asr", response_model=DownstreamASRResponse)
async def downstream_asr_demo(
    detected_accent: str = Form("Central_MP"),
    reference_text: Optional[str] = Form(None),
):
    """
    Demonstrates downstream speech recognition adaptation using the predicted accent profile.
    """
    bench = asr_adaptor.benchmark_sample(
        regional_accent=detected_accent,
        reference_text=reference_text,
    )
    corrections = [
        f"{c['original_sound']}: {c['unadapted_error']} -> {c['adapted_correction']}"
        for c in bench.get("phonetic_corrections_analyzed", [])
    ]
    family_desc = FAMILY_MAP.get(detected_accent, "Indian English")

    return DownstreamASRResponse(
        baseline_transcript=bench["baseline_transcript"],
        accent_adapted_transcript=bench["adapted_transcript"],
        adaptation_strategy=f"Whisper Prompt Conditioning [{bench['conditioning_prompt']}]",
        detected_accent_profile=f"{detected_accent} ({family_desc})",
        phonetic_corrections_noted=corrections,
    )
