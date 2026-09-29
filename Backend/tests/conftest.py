"""Shared fixtures for the AccentSense backend test suite.

Tests run against the REAL model artifacts (Backend/models). There are no
mocked predictions anywhere in this suite: model-dependent tests fail
loudly if bootstrap has not been run, because "the model is missing" is a
broken state, not a passing one.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.config import SAMPLE_RATE  # noqa: E402


def make_wav_bytes(
    duration_sec: float = 3.0,
    sr: int = SAMPLE_RATE,
    channels: int = 1,
    freq: float = 220.0,
    amplitude: float = 0.3,
) -> bytes:
    """Synthesize a PCM_16 WAV in memory (pure tone or stereo variants)."""
    num = int(duration_sec * sr)
    t = np.linspace(0.0, duration_sec, num, endpoint=False)
    wave = amplitude * np.sin(2.0 * np.pi * freq * t)
    if channels > 1:
        wave = np.stack([wave] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient with rate limiting disabled for the suite."""
    from fastapi.testclient import TestClient

    from src.api.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def service():
    """The real singleton model service (loads once for the whole suite)."""
    from src.models.service import get_service

    svc = get_service()
    svc.ensure_loaded()
    return svc


@pytest.fixture(scope="session")
def tone_wav() -> bytes:
    return make_wav_bytes(duration_sec=3.0)


@pytest.fixture(scope="session")
def speech_wav() -> bytes:
    """A real speech recording from the local corpus (1.5 to 25 s).

    Skipped only when no local corpus file is usable — synthetic-tone tests
    still exercise the full model path either way.
    """
    candidates = sorted((BACKEND_DIR / "data" / "audio").glob("*.wav"))
    for path in candidates:
        try:
            info = sf.info(path)
        except (sf.LibsndfileError, OSError):
            continue
        duration = info.frames / float(info.samplerate)
        if 1.5 <= duration <= 25.0:
            return path.read_bytes()
    pytest.skip("No local speech WAV between 1.5s and 25s in data/audio/")
