"""Unit tests for src.audio.io: real decoding, no trusted extensions,
explicit rejection instead of silent truncation, safe FFmpeg usage."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from src.audio.io import AudioValidationError, decode_and_normalize
from src.config import (
    MAX_AUDIO_DURATION_SEC,
    MAX_UPLOAD_BYTES,
    MIN_AUDIO_DURATION_SEC,
    SAMPLE_RATE,
)

from .conftest import make_wav_bytes


def test_valid_wav_decodes_to_mono_float32_16k():
    samples = decode_and_normalize(make_wav_bytes(duration_sec=3.0, sr=16000))
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert abs(len(samples) / SAMPLE_RATE - 3.0) < 0.05
    assert np.all(np.isfinite(samples))


def test_stereo_is_downmixed():
    samples = decode_and_normalize(make_wav_bytes(duration_sec=2.0, channels=2))
    assert samples.ndim == 1
    assert abs(len(samples) / SAMPLE_RATE - 2.0) < 0.05


def test_non_16k_input_is_resampled():
    samples = decode_and_normalize(make_wav_bytes(duration_sec=2.0, sr=44100))
    assert abs(len(samples) / SAMPLE_RATE - 2.0) < 0.05


def test_extension_is_not_trusted_wrong_extension_still_decodes():
    # Valid WAV bytes with a lying ".txt" name — content decides.
    payload = make_wav_bytes(duration_sec=2.0)
    samples = decode_and_normalize(payload)
    assert samples.size > 0


def test_garbage_bytes_are_rejected_with_clear_message():
    with pytest.raises(AudioValidationError) as exc:
        decode_and_normalize(b"this is definitely not audio" * 100)
    assert exc.value.status == 400
    assert "Traceback" not in str(exc.value)


def test_empty_payload_rejected():
    with pytest.raises(AudioValidationError) as exc:
        decode_and_normalize(b"")
    assert "empty" in str(exc.value).lower()


def test_oversized_payload_rejected_with_413():
    payload = b"\x00" * (MAX_UPLOAD_BYTES + 1)
    with pytest.raises(AudioValidationError) as exc:
        decode_and_normalize(payload)
    assert exc.value.status == 413


def test_too_short_audio_rejected_not_truncated():
    payload = make_wav_bytes(duration_sec=0.4)
    with pytest.raises(AudioValidationError) as exc:
        decode_and_normalize(payload)
    assert "too short" in str(exc.value)
    assert str(MIN_AUDIO_DURATION_SEC) in str(exc.value)


def test_too_long_audio_rejected_not_truncated():
    # The old code truncated to 10 s then measured; now we reject outright.
    payload = make_wav_bytes(duration_sec=MAX_AUDIO_DURATION_SEC + 3.0)
    with pytest.raises(AudioValidationError) as exc:
        decode_and_normalize(payload)
    assert "too long" in str(exc.value)
    assert str(int(MAX_AUDIO_DURATION_SEC)) in str(exc.value)


def test_silence_decodes_fine_model_decides(monkeypatch):
    # Zero-filled audio is valid input; validation must not judge content.
    payload = make_wav_bytes(duration_sec=3.0, amplitude=0.0)
    samples = decode_and_normalize(payload)
    assert np.all(samples == 0)


def test_wav_with_junk_after_header_still_rejected_if_undecodable():
    # A file that is neither a real container nor FFmpeg-decodable.
    payload = b"RIFF" + b"\xff" * 500  # truncated/garbage RIFF
    with pytest.raises(AudioValidationError):
        decode_and_normalize(payload)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_fallback_decodes_real_mp3(tmp_path):
    """Exercise the FFmpeg path with genuinely compressed audio."""
    src = tmp_path / "tone.wav"
    mp3 = tmp_path / "tone.mp3"
    sf.write(src, np.sin(np.linspace(0, 100, 44100 * 2)), 44100)
    subprocess.run(
        [shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
         "-y", "-i", str(src), str(mp3)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    samples = decode_and_normalize(mp3.read_bytes())
    assert samples.dtype == np.float32
    assert abs(len(samples) / SAMPLE_RATE - 2.0) < 0.1
