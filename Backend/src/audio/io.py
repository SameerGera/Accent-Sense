"""
Secure audio decoding and validation for uploaded recordings.

Design rules:
- The filename/extension is never trusted; content is sniffed by the
  decoders themselves (libsndfile, then FFmpeg as fallback).
- FFmpeg is invoked with an argument array (never a shell string), with
  ``-nostdin`` (no input interaction), a subprocess timeout, and temporary
  files that are always cleaned up.
- Uploaded bytes, decoded sample rate, decoded duration and finiteness are
  all validated *before* anything reaches the model. Audio is never
  silently truncated: oversized/long/empty input is rejected with a clear
  error instead of being altered.
"""

from __future__ import annotations

import io
import logging
import math
import shutil

# nosec B404 — subprocess is required to decode compressed uploads. Every
# invocation below uses an argument list (never shell=True), an absolute
# binary path from shutil.which, a timeout, and a neutral temp file; the
# caller's filename never reaches the command line. See ARCHITECTURE.md.
import subprocess  # nosec B404 — justification above
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from src.config import (
    FFMPEG_TIMEOUT_SEC,
    MAX_AUDIO_DURATION_SEC,
    MAX_SAMPLE_RATE,
    MAX_UPLOAD_BYTES,
    MIN_AUDIO_DURATION_SEC,
    MIN_SAMPLE_RATE,
    SAMPLE_RATE,
)

logger = logging.getLogger("accentsense.audio")


class AudioValidationError(Exception):
    """Raised when an upload cannot be accepted as valid model input.

    ``status`` is the HTTP status the API layer should respond with.
    The message is safe to show to the client.
    """

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _check_duration(num_samples: int, sample_rate: int) -> float:
    duration = num_samples / float(sample_rate)
    if duration > MAX_AUDIO_DURATION_SEC:
        raise AudioValidationError(
            f"Audio too long ({duration:.1f}s). "
            f"Maximum duration is {MAX_AUDIO_DURATION_SEC:.0f}s."
        )
    if duration < MIN_AUDIO_DURATION_SEC:
        raise AudioValidationError(
            f"Audio too short ({duration:.1f}s). "
            f"Minimum duration is {MIN_AUDIO_DURATION_SEC:.0f}s."
        )
    return duration


def _check_finite(samples: np.ndarray) -> None:
    if not np.all(np.isfinite(samples)):
        raise AudioValidationError("Decoded audio contains non-finite samples.")


def _to_mono_float32(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


def _resample_to_16k(samples: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE:
        return samples
    from scipy.signal import resample_poly

    divisor = math.gcd(int(source_rate), SAMPLE_RATE)
    up = SAMPLE_RATE // divisor
    down = int(source_rate) // divisor
    resampled = resample_poly(samples, up, down).astype(np.float32)
    return np.ascontiguousarray(resampled)


def _decode_with_soundfile(payload: bytes) -> tuple[np.ndarray, int] | None:
    """Try libsndfile (WAV/FLAC/OGG/MP3/CAF/AIFF...). Returns None on failure."""
    try:
        data, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, ValueError):
        return None
    if data.shape[0] == 0:
        raise AudioValidationError("The file contains no audio samples.")
    if not (MIN_SAMPLE_RATE <= int(sample_rate) <= MAX_SAMPLE_RATE):
        raise AudioValidationError(
            f"Unsupported sample rate ({int(sample_rate)} Hz). "
            f"Supported range: {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE} Hz."
        )
    samples = _to_mono_float32(data)
    _check_finite(samples)
    _check_duration(len(samples), int(sample_rate))
    return samples, int(sample_rate)


def _probe_duration(path: Path) -> float | None:
    """Best-effort header-level duration check so we can reject very long
    inputs before decoding them fully. Returns None when unavailable."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(  # nosec B603 — arg list, abs path, timeout, no shell
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            timeout=FFMPEG_TIMEOUT_SEC,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.decode("utf-8", "replace").strip())
    except ValueError:
        return None


def _decode_with_ffmpeg(payload: bytes) -> np.ndarray:
    """Decode any FFmpeg-supported container to mono float32 16 kHz PCM.

    The original bytes are written to a temp file with a neutral name
    (the caller's filename never reaches the command line), decoded with an
    argument array + timeout, and cleaned up unconditionally.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioValidationError(
            "This audio format could not be decoded. "
            "Please upload WAV, FLAC or OGG, or install FFmpeg for MP3/M4A/WebM support."
        )

    with tempfile.TemporaryDirectory(prefix="accentsense-upload-") as tmp_dir:
        in_path = Path(tmp_dir) / "input"
        in_path.write_bytes(payload)

        probed = _probe_duration(in_path)
        if probed is not None and probed > MAX_AUDIO_DURATION_SEC:
            raise AudioValidationError(
                f"Audio too long ({probed:.1f}s). "
                f"Maximum duration is {MAX_AUDIO_DURATION_SEC:.0f}s."
            )

        try:
            result = subprocess.run(  # nosec B603 — arg list, abs path, timeout, no shell
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-y",
                    "-i", str(in_path),
                    "-f", "f32le",
                    "-acodec", "pcm_f32le",
                    "-ac", "1",
                    "-ar", str(SAMPLE_RATE),
                    "pipe:1",
                ],
                capture_output=True,
                timeout=FFMPEG_TIMEOUT_SEC,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning("FFmpeg decode timed out after %ss", FFMPEG_TIMEOUT_SEC)
            raise AudioValidationError(
                "Audio decoding took too long. Please upload a shorter file."
            ) from exc
        except OSError as exc:
            logger.error("Failed to execute FFmpeg: %s", exc)
            raise AudioValidationError("Audio decoding is unavailable on this server.") from exc

        if result.returncode != 0 or not result.stdout:
            stderr_tail = result.stderr.decode("utf-8", "replace")[-500:]
            logger.warning("FFmpeg rejected upload: %s", stderr_tail)
            raise AudioValidationError(
                "Could not decode the uploaded file as audio."
            )

        samples = np.frombuffer(result.stdout, dtype=np.float32).copy()

    if samples.size == 0:
        raise AudioValidationError("The file contains no audio samples.")
    _check_finite(samples)
    _check_duration(len(samples), SAMPLE_RATE)
    return samples


def decode_and_normalize(payload: bytes) -> np.ndarray:
    """Validate raw upload bytes and return mono float32 audio at 16 kHz.

    Raises AudioValidationError (with an HTTP-safe message) for anything
    that must be rejected. Never truncates or otherwise alters user audio
    to make it fit — rejection is explicit.
    """
    if not payload:
        raise AudioValidationError("The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise AudioValidationError(
            f"File too large ({len(payload) / 1024 / 1024:.1f} MB). "
            f"Maximum is {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
            status=413,
        )

    decoded = _decode_with_soundfile(payload)
    if decoded is not None:
        samples, source_rate = decoded
        samples = _resample_to_16k(samples, source_rate)
    else:
        samples = _decode_with_ffmpeg(payload)

    # Resampling can change length by a sample; re-validate the final signal.
    _check_finite(samples)
    _check_duration(len(samples), SAMPLE_RATE)
    return samples
