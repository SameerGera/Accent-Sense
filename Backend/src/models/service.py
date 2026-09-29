"""
Pretrained accent inference service.

Data flow (see docs/ARCHITECTURE.md for the decision record):

    mono float32 @ 16 kHz
        -> YAMNet (TensorFlow Hub, local copy)
        -> per-frame embeddings (N x 1024)
        -> fbadine/uk_ireland_accent_classification (local SavedModel)
        -> per-frame class probabilities (N x 7)
        -> mean aggregation over frames  (the reference implementation's
           strategy: keras.io example, predictions.mean(axis=0).argmax())
        -> prediction + temporal evidence

Both models are loaded exactly once per process, from project-local paths
only. There is no network access in this module: TensorFlow, TensorFlow
Hub and the artifacts are all local. ``src.config`` must be imported
before this module loads TensorFlow (it is, transitively, via package
imports) so that cache isolation env vars are in place.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    CLASSES,
    CLASSIFIER_DIR,
    MANIFEST_PATH,
    MODELS_DIR,
    NON_SPEECH_CLASS,
    YAMNET_DIR,
)

logger = logging.getLogger("accentsense.model")

# YAMNet AudioSet class-map index 0 == "Speech". The training pipeline of
# the pretrained classifier labeled every frame whose YAMNet top-1 event
# was not Speech as "Not a speech"; we reuse the identical rule.
YAMNET_SPEECH_CLASS_INDEX = 0


class ModelError(Exception):
    """Server-side model failure. Message is safe for logs, not for clients."""


class ModelNotBootstrappedError(ModelError):
    """Required model artifacts are missing from Backend/models/."""


@dataclass(frozen=True)
class FrameEvidence:
    """One YAMNet temporal frame and what the models said about it."""

    timestamp_sec: float
    probabilities: tuple[float, ...]  # len == len(CLASSES)
    is_speech: bool  # YAMNet top-1 AudioSet event == Speech


@dataclass(frozen=True)
class Prediction:
    """Deterministic typed result of one inference call."""

    predicted_class: str
    predicted_index: int
    aggregated_probabilities: tuple[float, ...]  # mean over frames, len == 7
    speech_frame_ratio: float
    frames: tuple[FrameEvidence, ...]
    duration_sec: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise ModelNotBootstrappedError(
            f"Model manifest not found at {MANIFEST_PATH}. "
            "Run `python bootstrap_models.py` once to download the model artifacts."
        )
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"Model manifest is unreadable: {exc}") from exc
    for section in ("yamnet", "accent_classifier"):
        if section not in manifest.get("models", {}):
            raise ModelError(f"Model manifest is missing the '{section}' section.")
    return manifest


def _verify_artifacts(section: dict[str, Any]) -> None:
    """Checksum every declared artifact; never load tampered/corrupt files."""
    root = MODELS_DIR / section["local_path"]
    if not root.is_dir():
        raise ModelNotBootstrappedError(
            f"Model artifacts missing at {root}. "
            "Run `python bootstrap_models.py` to download them."
        )
    for entry in section["files"]:
        path = root / entry["path"]
        if not path.is_file():
            raise ModelNotBootstrappedError(
                f"Expected artifact {path} is missing. Re-run `python bootstrap_models.py`."
            )
        actual = _sha256(path)
        if actual != entry["sha256"]:
            raise ModelError(
                f"Checksum mismatch for {path.name}: expected {entry['sha256'][:12]}…, "
                f"got {actual[:12]}…. Re-run `python bootstrap_models.py`."
            )


class AccentModelService:
    """Loads YAMNet + the pretrained classifier once and runs inference.

    Not constructed directly by callers: use :func:`get_service`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: str | None = None
        self._load_seconds: float | None = None
        self._tf: Any = None
        self._yamnet: Any = None
        self._classifier_signature: Any = None
        self._classifier_input_name: str | None = None
        self._manifest: dict[str, Any] | None = None
        self._frame_hop_seconds: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._loaded

    def status(self) -> dict[str, Any]:
        """Best-effort readiness snapshot for /health (never raises)."""
        info: dict[str, Any] = {
            "yamnet_loaded": self._yamnet is not None,
            "accent_classifier_loaded": self._classifier_signature is not None,
            "manifest_present": MANIFEST_PATH.is_file(),
            "artifacts_present": YAMNET_DIR.is_dir() and CLASSIFIER_DIR.is_dir(),
            "inference_ready": self._loaded,
            "load_seconds": self._load_seconds,
            "error": self._load_error,
        }
        if not self._loaded and self._load_error is None:
            try:
                self.ensure_loaded()
            except Exception as exc:  # status() must never throw
                self._load_error = f"{type(exc).__name__}: {exc}"
                info["error"] = self._load_error
            info.update(
                yamnet_loaded=self._yamnet is not None,
                accent_classifier_loaded=self._classifier_signature is not None,
                inference_ready=self._loaded,
                load_seconds=self._load_seconds,
            )
        return info

    def ensure_loaded(self) -> None:
        """Idempotent, thread-safe model load from local artifacts only."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            started = time.monotonic()
            try:
                self._load()
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Model load failed")
                raise
            self._load_seconds = round(time.monotonic() - started, 2)
            self._load_error = None
            self._loaded = True
            logger.info("Models loaded in %.2fs (local artifacts only)", self._load_seconds)

    def _load(self) -> None:
        manifest = _load_manifest()
        yamnet_meta = manifest["models"]["yamnet"]
        classifier_meta = manifest["models"]["accent_classifier"]

        # Validate class order before touching the artifacts.
        declared = tuple(classifier_meta["output"]["classes"])
        if declared != CLASSES:
            raise ModelError(
                "Manifest class order does not match the built-in class order: "
                f"{declared!r} != {CLASSES!r}"
            )
        hop = float(yamnet_meta["output"]["frame_hop_seconds"])
        if not 0 < hop <= 5:
            raise ModelError(f"Implausible YAMNet frame hop in manifest: {hop}")
        self._frame_hop_seconds = hop

        _verify_artifacts(yamnet_meta)
        _verify_artifacts(classifier_meta)

        # Heavy imports happen here — never at module import time, so the
        # API process can start and answer /health even without artifacts.
        import tensorflow as tf
        import tensorflow_hub as hub

        self._tf = tf
        yamnet_path = str(MODELS_DIR / yamnet_meta["local_path"])
        classifier_path = str(MODELS_DIR / classifier_meta["local_path"])

        logger.info("Loading YAMNet from %s", yamnet_path)
        self._yamnet = hub.load(yamnet_path)

        logger.info("Loading accent classifier from %s", classifier_path)
        loaded = tf.saved_model.load(classifier_path)
        signature = _extract_classifier_signature(tf, loaded)
        self._classifier_signature = signature
        self._classifier_input_name = _classifier_input_name(signature)

        self._validate_models()
        self._manifest = manifest

    def _validate_models(self) -> None:
        """Probe the real artifacts: shapes, class count, finiteness."""
        tf = self._tf
        probe = np.zeros((3, 1024), dtype=np.float32)
        out = self._classify_embeddings(probe)
        if out.shape != (3, len(CLASSES)):
            raise ModelError(
                f"Classifier output shape mismatch: expected {(3, len(CLASSES))}, got {out.shape}"
            )
        if not np.all(np.isfinite(out)):
            raise ModelError("Classifier produced non-finite values on probe input.")
        row_sums = out.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            raise ModelError(f"Classifier outputs do not sum to 1 (got {row_sums}).")

        # YAMNet smoke probe: ~2 s of audio must yield frames of 1024-dim
        # embeddings and 521 AudioSet scores.
        silence = np.zeros(2 * 16000, dtype=np.float32)
        scores, embeddings, _ = self._yamnet(tf.convert_to_tensor(silence))
        emb = embeddings.numpy()
        scr = scores.numpy()
        if emb.ndim != 2 or emb.shape[1] != 1024:
            raise ModelError(f"YAMNet embedding shape unexpected: {emb.shape}")
        if scr.ndim != 2 or scr.shape[1] != 521:
            raise ModelError(f"YAMNet score shape unexpected: {scr.shape}")
        if emb.shape[0] == 0:
            raise ModelError("YAMNet produced no frames for 2 s of audio.")
        if emb.shape[0] != scr.shape[0]:
            raise ModelError("YAMNet scores/embeddings frame counts differ.")
        if not (np.all(np.isfinite(emb)) and np.all(np.isfinite(scr))):
            raise ModelError("YAMNet produced non-finite values.")
        logger.info(
            "Model validation OK: YAMNet %s frames per 2 s, classifier %s",
            emb.shape[0],
            (len(CLASSES),),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _classify_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        tf = self._tf
        tensor = tf.convert_to_tensor(embeddings, dtype=tf.float32)
        outputs = self._classifier_signature(**{self._classifier_input_name: tensor})
        result = next(iter(outputs.values()))
        return result.numpy()

    def infer(self, waveform: np.ndarray) -> Prediction:
        """Run the full pipeline on mono float32 audio at 16 kHz."""
        self.ensure_loaded()
        if waveform.ndim != 1 or waveform.size == 0:
            raise ModelError("infer() requires a non-empty 1-D waveform.")
        if not np.all(np.isfinite(waveform)):
            raise ModelError("infer() received non-finite samples.")

        with self._lock:  # one TF forward pass at a time
            tf = self._tf
            scores, embeddings, _ = self._yamnet(tf.convert_to_tensor(waveform))
            embeddings_np = embeddings.numpy()
            scores_np = scores.numpy()

            if embeddings_np.shape[0] == 0:
                raise ModelError(
                    "YAMNet produced no frames for this audio; it is too short to analyse."
                )
            probabilities = self._classify_embeddings(embeddings_np)  # (N, 7)

        if probabilities.shape != (embeddings_np.shape[0], len(CLASSES)):
            raise ModelError(
                f"Classifier output {probabilities.shape} does not match "
                f"{embeddings_np.shape[0]} frames x {len(CLASSES)} classes."
            )
        if not np.all(np.isfinite(probabilities)):
            raise ModelError("Classifier produced non-finite probabilities.")

        # Speech evidence: exactly the rule used in the model's training.
        is_speech = scores_np.argmax(axis=1) == YAMNET_SPEECH_CLASS_INDEX
        speech_frame_ratio = float(is_speech.mean())

        # Reference aggregation: mean over all frames, then argmax.
        # (keras.io example: class_names[predictions.mean(axis=0).argmax()])
        aggregated = probabilities.mean(axis=0)
        predicted_index = int(aggregated.argmax())

        frames = tuple(
            FrameEvidence(
                timestamp_sec=round(i * self._frame_hop_seconds, 3),
                probabilities=tuple(round(float(p), 4) for p in probabilities[i]),
                is_speech=bool(is_speech[i]),
            )
            for i in range(probabilities.shape[0])
        )

        return Prediction(
            predicted_class=CLASSES[predicted_index],
            predicted_index=predicted_index,
            aggregated_probabilities=tuple(round(float(p), 4) for p in aggregated),
            speech_frame_ratio=round(speech_frame_ratio, 4),
            frames=frames,
            duration_sec=round(len(waveform) / 16000.0, 3),
        )

    # ------------------------------------------------------------------
    # Temporal evidence (derived from real per-frame outputs)
    # ------------------------------------------------------------------
    @staticmethod
    def temporal_evidence(prediction: Prediction) -> tuple[list[float], list[float]]:
        """Return (timestamps, curve) where the curve is the aggregated
        class's per-frame probability — model evidence over time, not a
        gradient attribution."""
        timestamps = [frame.timestamp_sec for frame in prediction.frames]
        curve = [frame.probabilities[prediction.predicted_index] for frame in prediction.frames]
        return timestamps, curve

    @staticmethod
    def evidence_regions(prediction: Prediction, max_regions: int = 4) -> list[dict[str, Any]]:
        """Contiguous high-evidence regions derived from per-frame outputs.

        A region requires per-frame probability >= max(75th percentile,
        0.50) for the predicted class and at least two consecutive frames,
        so a uniformly uncertain model produces no 'high evidence' claims.
        """
        if not prediction.frames:
            return []
        curve = np.array(
            [frame.probabilities[prediction.predicted_index] for frame in prediction.frames],
            dtype=np.float64,
        )
        threshold = max(float(np.percentile(curve, 75)), 0.50)
        mask = curve >= threshold

        regions: list[dict[str, Any]] = []
        start: int | None = None
        for idx, high in enumerate(mask):
            if high and start is None:
                start = idx
            elif not high and start is not None:
                regions.append(_build_region(prediction, start, idx, curve))
                start = None
        if start is not None:
            regions.append(_build_region(prediction, start, len(mask), curve))

        regions = [r for r in regions if r is not None]
        regions.sort(key=lambda r: r["model_score"], reverse=True)
        return regions[:max_regions]


def _build_region(
    prediction: Prediction, start: int, end: int, curve: np.ndarray
) -> dict[str, Any] | None:
    if end - start < 2:  # at least ~2 frames (>= ~1 s with default hop)
        return None
    segment = prediction.frames[start:end]
    mean_score = float(curve[start:end].mean())
    speech_ratio = float(np.mean([f.is_speech for f in segment]))
    pred = prediction.predicted_class

    if pred == NON_SPEECH_CLASS:
        label = "Non-speech evidence"
        detail = (
            f"The model scored these frames “Not a speech” at {mean_score:.0%} on average; "
            f"YAMNet detected speech in {speech_ratio:.0%} of them."
        )
    elif speech_ratio < 0.5:
        label = "Mostly non-speech frames"
        detail = (
            f"Model score for {pred}: {mean_score:.0%}, but YAMNet detected speech in only "
            f"{speech_ratio:.0%} of these frames — weak evidence, treat with caution."
        )
    else:
        label = f"High evidence for {pred}"
        detail = (
            f"Mean model score for {pred} across {len(segment)} frames: {mean_score:.0%} "
            f"(speech detected in {speech_ratio:.0%} of them)."
        )

    return {
        "start_time_sec": round(float(segment[0].timestamp_sec), 2),
        "end_time_sec": round(float(segment[-1].timestamp_sec), 2),
        "duration_sec": round(float(segment[-1].timestamp_sec - segment[0].timestamp_sec), 2),
        "model_score": round(mean_score, 3),
        "label": label,
        "detail": detail,
    }


def _extract_classifier_signature(tf: Any, loaded: Any) -> Any:
    signatures = getattr(loaded, "signatures", {}) or {}
    if "serving_default" in signatures:
        return signatures["serving_default"]
    if len(signatures) == 1:
        return next(iter(signatures.values()))
    if signatures:
        raise ModelError(
            f"Classifier SavedModel has no serving_default signature (found: {list(signatures)})."
        )
    raise ModelError("Classifier SavedModel exposes no callable signatures.")


def _classifier_input_name(signature: Any) -> str:
    _, kwargs = signature.structured_input_signature
    if not kwargs:
        raise ModelError("Classifier signature has no named inputs.")
    if len(kwargs) > 1:
        raise ModelError(f"Classifier signature has unexpected inputs: {list(kwargs)}")
    name, spec = next(iter(kwargs.items()))
    if spec.dtype != _float32():
        raise ModelError(f"Classifier input dtype is {spec.dtype}, expected float32.")
    shape = spec.shape.as_list()
    if len(shape) != 2 or shape[1] != 1024:
        raise ModelError(f"Classifier input shape {shape} is not (None, 1024).")
    return name


def _float32():
    import tensorflow as tf

    return tf.float32


# Module-level singleton: constructed cheaply, loaded lazily.
_service = AccentModelService()
_service_lock = threading.Lock()


def get_service() -> AccentModelService:
    """Return the process-wide service instance (loads on first use)."""
    return _service
