"""Artifact integrity, provenance and cache-isolation tests.

These verify the supply-chain guarantees: pinned revision, SHA-256
checksums, exact class order, and that every cache points inside
Backend/models/ with huggingface_hub pinned offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from src.config import (
    CLASSES,
    HF_HOME_DIR,
    MANIFEST_PATH,
    MODELS_DIR,
    NON_SPEECH_CLASS,
    TFHUB_CACHE_DIR,
    YAMNET_DIR,
)

EXPECTED_CLASSES = (
    "Irish",
    "Midlands",
    "Northern",
    "Scottish",
    "Southern",
    "Welsh",
    "Not a speech",
)
PINNED_REVISION = "ebe681a49e9506acdc4cf5312cb79b291275ab67"


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST_PATH.is_file(), (
        "model_manifest.json missing — run `python bootstrap_models.py` first."
    )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_class_order_is_exact_and_never_remapped():
    assert CLASSES == EXPECTED_CLASSES
    assert NON_SPEECH_CLASS == "Not a speech"
    assert CLASSES[-1] == NON_SPEECH_CLASS  # real class, stays in position


def test_manifest_pins_immutable_revision(manifest: dict):
    clf = manifest["models"]["accent_classifier"]
    assert clf["revision"] == PINNED_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", clf["revision"])
    assert clf["source"] == "huggingface:fbadine/uk_ireland_accent_classification"


def test_manifest_records_exact_class_order(manifest: dict):
    assert tuple(manifest["models"]["accent_classifier"]["output"]["classes"]) == EXPECTED_CLASSES


def test_manifest_declares_provenance_and_loader(manifest: dict):
    clf = manifest["models"]["accent_classifier"]
    assert "keras.io" in clf["provenance"]
    assert "OpenSLR-83" in clf["provenance"]
    loader = manifest["compatibility"]["loader"]
    assert "trust_remote_code" in loader and "no trust_remote_code" in loader
    assert manifest["models"]["yamnet"]["source"].startswith("TensorFlow Hub")


def test_manifest_frame_hop_is_plausible(manifest: dict):
    hop = manifest["models"]["yamnet"]["output"]["frame_hop_seconds"]
    assert 0.4 < hop < 0.6, f"unexpected YAMNet frame hop {hop}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_every_declared_artifact_matches_its_checksum(manifest: dict):
    for section in manifest["models"].values():
        root = MODELS_DIR / section["local_path"]
        assert root.is_dir(), f"missing artifact dir {root}"
        assert section["files"], f"manifest lists no files for {section['local_path']}"
        for entry in section["files"]:
            path = root / entry["path"]
            assert path.is_file(), f"missing artifact {path}"
            assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            assert _sha256(path) == entry["sha256"], f"checksum mismatch: {path}"


def test_saved_model_and_yamnet_present():
    assert (MODELS_DIR / "uk_ireland_accent_classification" / "saved_model.pb").is_file()
    assert YAMNET_DIR.is_dir()
    assert any(
        (YAMNET_DIR / name).is_file() for name in ("saved_model.pb", "tfhub_module.pb")
    )


def test_caches_are_project_local_and_offline():
    # config sets these at import time, before any TF/HF import.
    assert os.environ["HF_HOME"] == str(HF_HOME_DIR)
    assert os.environ["TFHUB_CACHE_DIR"] == str(TFHUB_CACHE_DIR)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    # All cache paths live inside Backend/models (no ~/.cache usage).
    for var in ("HF_HOME", "HF_HUB_CACHE", "TFHUB_CACHE_DIR"):
        resolved = Path(os.environ[var]).resolve()
        assert str(resolved).startswith(str(MODELS_DIR.resolve())), (
            f"{var}={resolved} escapes {MODELS_DIR}"
        )


def _fresh_service_with_manifest(monkeypatch, manifest: dict, tmp_path):
    """Build a not-yet-loaded service that reads a (possibly tampered) manifest."""
    import src.models.service as service_module

    fake_path = tmp_path / "model_manifest.json"
    fake_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(service_module, "MANIFEST_PATH", fake_path)
    return service_module.AccentModelService()


def test_missing_manifest_is_model_not_bootstrapped(monkeypatch, tmp_path):
    import src.models.service as service_module
    from src.models.service import ModelNotBootstrappedError

    monkeypatch.setattr(service_module, "MANIFEST_PATH", tmp_path / "absent.json")
    service = service_module.AccentModelService()
    with pytest.raises(ModelNotBootstrappedError):
        service.ensure_loaded()


def test_tampered_checksum_is_rejected_before_any_model_load(manifest, monkeypatch, tmp_path):
    """Flip one artifact checksum in the manifest → refuse to load."""
    import copy

    from src.models.service import ModelError

    tampered = copy.deepcopy(manifest)
    tampered["models"]["accent_classifier"]["files"][0]["sha256"] = "0" * 64
    service = _fresh_service_with_manifest(monkeypatch, tampered, tmp_path)
    with pytest.raises(ModelError, match="Checksum mismatch"):
        service.ensure_loaded()


def test_reordered_classes_in_manifest_is_rejected(manifest, monkeypatch, tmp_path):
    import copy

    from src.models.service import ModelError

    tampered = copy.deepcopy(manifest)
    classes = tampered["models"]["accent_classifier"]["output"]["classes"]
    tampered["models"]["accent_classifier"]["output"]["classes"] = list(reversed(classes))
    service = _fresh_service_with_manifest(monkeypatch, tampered, tmp_path)
    with pytest.raises(ModelError, match="class order"):
        service.ensure_loaded()


def test_implausible_frame_hop_is_rejected(manifest, monkeypatch, tmp_path):
    import copy

    from src.models.service import ModelError

    tampered = copy.deepcopy(manifest)
    tampered["models"]["yamnet"]["output"]["frame_hop_seconds"] = 42.0
    service = _fresh_service_with_manifest(monkeypatch, tampered, tmp_path)
    with pytest.raises(ModelError, match="frame hop"):
        service.ensure_loaded()
