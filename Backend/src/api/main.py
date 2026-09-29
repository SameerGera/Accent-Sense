"""
AccentSense FastAPI service (inference only).

Endpoints:
  GET   /health              — real readiness of the local model artifacts
  POST  /api/predict         — accent prediction + temporal evidence
  POST  /api/downstream-asr  — static prompt-conditioning illustration

Inference runs the pretrained pipeline from ``src.models.service``:
mono 16 kHz audio -> YAMNet frame embeddings (local TF-Hub copy) ->
``fbadine/uk_ireland_accent_classification`` SavedModel -> per-frame class
probabilities -> mean aggregation over frames.

There is no mock fallback anywhere in this module: if the model is missing
or fails, the API answers 503 with a generic message (details go to the
server log, never to the client).
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from src.asr.adaptation import WhisperAccentAdaptor
from src.audio.io import AudioValidationError, decode_and_normalize
from src.config import (
    CLASS_DESCRIPTIONS,
    CLASSES,
    CORS_HEADERS,
    CORS_METHODS,
    CORS_ORIGINS,
    MAX_UPLOAD_BYTES,
    NON_SPEECH_CLASS,
    SCORE_NOTE,
)
from src.models.service import (
    ModelError,
    ModelNotBootstrappedError,
    get_service,
)

logger = logging.getLogger("accentsense")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# App lifecycle: warm the model in the background so the first request does
# not pay the full TensorFlow load cost. A failed warm-up (e.g. artifacts
# not bootstrapped yet) is reported by /health and surfaces as 503 on
# /api/predict — the server still starts so operators can see the error.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    def _warm_up() -> None:
        try:
            get_service().ensure_loaded()
        except Exception:
            logger.warning("Model warm-up failed; /health reports the reason.")

    threading.Thread(target=_warm_up, name="model-warmup", daemon=True).start()
    yield


app = FastAPI(
    title="AccentSense API",
    description="UK/Ireland/Scotland/Wales accent classification with temporal evidence",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# CORS — narrowed: only the methods and headers this API actually uses.
# Origins are configurable via the CORS_ORIGINS env var (comma-separated).
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=CORS_METHODS,
    allow_headers=CORS_HEADERS,
)

# Static, honest label attached to every prediction.
ASR_ADAPTOR = WhisperAccentAdaptor()


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class EvidenceRegion(BaseModel):
    """One contiguous high-evidence region of the recording.

    ``model_score`` is the mean per-frame probability of the predicted class
    inside the region — not a gradient/saliency attribution and not a
    phonological measurement.
    """

    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    model_score: float
    label: str
    detail: str


class PredictionResponse(BaseModel):
    predicted_influence: str
    language_family: str
    model_score: float
    all_scores: dict[str, float]
    evidence_regions: list[EvidenceRegion]
    timestamps: list[float]
    evidence_curve: list[float]
    speech_frame_ratio: float
    is_sufficient_speech: bool
    model_score_note: str


class DownstreamASRResponse(BaseModel):
    baseline_transcript: str
    accent_adapted_transcript: str
    adaptation_strategy: str
    detected_accent_profile: str
    phonetic_corrections_noted: list[str]
    is_static_example: bool
    disclaimer: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check(response: Response) -> dict:
    """Real readiness: actually loads/verifies the local model artifacts."""
    model_info = get_service().status()
    ready = bool(model_info.get("inference_ready"))
    response.status_code = 200 if ready else 503
    return {
        "status": "healthy" if ready else "degraded",
        "service": "AccentSense ML Engine",
        "device": "cpu",
        "inference_ready": ready,
        "supported_classes": list(CLASSES),
        "model": model_info,
    }


async def _read_bounded(file: UploadFile) -> bytes:
    """Read the upload with a hard byte cap (never buffer unbounded input)."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large (>{MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB). "
                    f"Maximum is {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/api/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
async def predict_speech(request: Request, file: UploadFile = File(...)):
    """Classify the uploaded recording and return temporal evidence.

    Content is sniffed by the decoders (extensions are never trusted);
    audio is never silently truncated — oversized/too-long input is
    rejected with a clear 4xx. Model problems are 503, never a fallback
    prediction.
    """
    payload = await _read_bounded(file)

    try:
        waveform = await run_in_threadpool(decode_and_normalize, payload)
    except AudioValidationError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    service = get_service()
    try:
        prediction = await run_in_threadpool(service.infer, waveform)
    except ModelNotBootstrappedError as exc:
        logger.error("Predict rejected, model artifacts missing: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="The accent model is not installed on this server.",
        ) from exc
    except ModelError as exc:
        logger.error("Predict failed in model inference: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="The accent model failed to process this recording.",
        ) from exc
    except Exception:
        logger.exception("Predict failed with an unexpected model error")
        raise HTTPException(
            status_code=503,
            detail="The accent model failed to process this recording.",
        ) from None

    timestamps, curve = service.temporal_evidence(prediction)
    regions = await run_in_threadpool(service.evidence_regions, prediction)
    all_scores = dict(
        zip(CLASSES, prediction.aggregated_probabilities, strict=True)
    )

    return PredictionResponse(
        predicted_influence=prediction.predicted_class,
        language_family=CLASS_DESCRIPTIONS.get(prediction.predicted_class, ""),
        model_score=all_scores[prediction.predicted_class],
        all_scores=all_scores,
        evidence_regions=[EvidenceRegion(**region) for region in regions],
        timestamps=timestamps,
        evidence_curve=curve,
        speech_frame_ratio=prediction.speech_frame_ratio,
        is_sufficient_speech=prediction.predicted_class != NON_SPEECH_CLASS,
        model_score_note=SCORE_NOTE,
    )


@app.post("/api/downstream-asr", response_model=DownstreamASRResponse)
@limiter.limit("10/minute")
async def downstream_asr_demo(
    request: Request,
    detected_accent: str = Form("Northern"),
    reference_text: str | None = Form(None),
):
    """Static illustration of accent-aware Whisper prompt conditioning.

    No ASR model is executed; the response says so explicitly. Rejects
    unknown classes and the "Not a speech" state (no accent to adapt to).
    """
    if detected_accent not in CLASSES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown accent '{detected_accent}'. "
                f"Supported: {', '.join(CLASSES)}."
            ),
        )
    if detected_accent == NON_SPEECH_CLASS:
        raise HTTPException(
            status_code=400,
            detail=(
                "The recording was classified as insufficient speech; "
                "there is no accent profile to adapt to."
            ),
        )

    bench = ASR_ADAPTOR.benchmark_sample(
        regional_accent=detected_accent,
        reference_text=reference_text,
    )
    corrections = [
        f"{c['original_sound']}: {c['unadapted_error']} -> {c['adapted_correction']}"
        for c in bench["phonetic_corrections_analyzed"]
    ]
    family_desc = CLASS_DESCRIPTIONS.get(detected_accent, "")

    return DownstreamASRResponse(
        baseline_transcript=bench["baseline_transcript"],
        accent_adapted_transcript=bench["adapted_transcript"],
        adaptation_strategy=f"Whisper prompt conditioning [{bench['conditioning_prompt']}]",
        detected_accent_profile=f"{detected_accent} ({family_desc})",
        phonetic_corrections_noted=corrections,
        is_static_example=True,
        disclaimer=(
            "Static illustration: no speech-recognition model was executed for "
            "this request. Transcripts are fixed benchmark examples shipped "
            "with the repository, and the phonetic notes are documented "
            "patterns from the literature — not measurements from this system."
        ),
    )
