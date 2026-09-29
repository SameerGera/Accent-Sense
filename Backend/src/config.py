"""
Central configuration for the AccentSense backend.

All paths are resolved relative to this file so the application behaves the
same whether it is launched from the repository root or from ``Backend``.

Importing this module also pins every model cache to a project-local
directory. This must happen before ``tensorflow``, ``tensorflow_hub`` or
``huggingface_hub`` are imported anywhere in the process, so no code may
import those libraries before ``src.config``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout (CWD-independent)
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

# Project-local model area. Everything the models need at inference time
# lives here; global HF / TF-Hub caches are never used.
MODELS_DIR = BACKEND_DIR / "models"
YAMNET_DIR = MODELS_DIR / "yamnet"
CLASSIFIER_DIR = MODELS_DIR / "uk_ireland_accent_classification"
MANIFEST_PATH = MODELS_DIR / "model_manifest.json"

# Secondary caches used only during `bootstrap_models.py` downloads. Kept
# inside the project so even the download step never touches ~/.cache.
HF_HOME_DIR = MODELS_DIR / ".hf-cache"
TFHUB_CACHE_DIR = MODELS_DIR / ".tfhub-cache"

# ---------------------------------------------------------------------------
# Cache isolation — force project-local caches before TF/HF initialize.
# ---------------------------------------------------------------------------
os.environ["HF_HOME"] = str(HF_HOME_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_HOME_DIR / "hub")
os.environ["TFHUB_CACHE_DIR"] = str(TFHUB_CACHE_DIR)
# Never allow accidental network access through huggingface_hub during
# normal inference. bootstrap_models.py explicitly flips this back to "0"
# for the duration of the one-time download, before it imports
# huggingface_hub (which reads the flag at import time).
os.environ["HF_HUB_OFFLINE"] = "1"

# ---------------------------------------------------------------------------
# Class taxonomy — EXACT output order of the pretrained classifier.
# Source: keras.io "English speaker accent recognition using transfer
# learning" (class_names) — verified against the model artifact at load
# time. Never reorder or remap.
# ---------------------------------------------------------------------------
CLASSES: tuple[str, ...] = (
    "Irish",
    "Midlands",
    "Northern",
    "Scottish",
    "Southern",
    "Welsh",
    "Not a speech",
)

NON_SPEECH_CLASS = "Not a speech"

CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

# Short human descriptions surfaced as `language_family` in API responses.
# The "Not a speech" entry is intentionally not an accent description.
CLASS_DESCRIPTIONS: dict[str, str] = {
    "Irish": "Irish English (Republic of Ireland & Northern Ireland)",
    "Midlands": "Midlands English (English Midlands)",
    "Northern": "Northern English (Northern England)",
    "Scottish": "Scottish English (Scotland)",
    "Southern": "Southern English (Southern England)",
    "Welsh": "Welsh English (Wales)",
    "Not a speech": "Insufficient speech evidence — the recording does not contain enough speech to assign an accent",
}

# Honest interpretation of the number the API reports as `model_score`.
# Sent verbatim to clients so nobody mistakes it for a calibrated confidence.
SCORE_NOTE = (
    "Mean softmax probability of the top class across all frames — a model "
    "score, not a calibrated confidence. The model's published validation "
    "accuracy is ~51% on a 7-way, non speaker-disjoint utterance split "
    "(keras.io 'English speaker accent recognition using transfer learning')."
)

# ---------------------------------------------------------------------------
# Audio constraints
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000  # YAMNet requirement
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_AUDIO_DURATION_SEC = 30.0
MIN_AUDIO_DURATION_SEC = 1.0  # YAMNet needs ~0.96 s for its first frame
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000
FFMPEG_TIMEOUT_SEC = 30

# ---------------------------------------------------------------------------
# CORS — only the methods/headers this API actually uses.
# Override origins with a comma-separated CORS_ORIGINS env var for deploys.
# ---------------------------------------------------------------------------
_DEFAULT_ORIGINS = (
    "http://localhost:5173,http://localhost:5174,"
    "http://localhost:4173,http://localhost:3000"
)
CORS_ORIGINS: list[str] = [
    o.strip() for o in os.getenv("CORS_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()
]
CORS_METHODS: list[str] = ["GET", "POST"]
CORS_HEADERS: list[str] = ["Content-Type", "Accept"]
