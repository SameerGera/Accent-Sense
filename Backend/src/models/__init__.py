"""Model package: the pretrained accent inference service."""

from .service import (
    AccentModelService,
    FrameEvidence,
    ModelError,
    ModelNotBootstrappedError,
    Prediction,
    get_service,
)

__all__ = [
    "AccentModelService",
    "FrameEvidence",
    "ModelError",
    "ModelNotBootstrappedError",
    "Prediction",
    "get_service",
]
