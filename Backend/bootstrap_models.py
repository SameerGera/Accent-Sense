"""
One-time bootstrap: download and verify the pretrained model artifacts
into the project-local model area (Backend/models/).

This is the ONLY component that may access the network. Normal API
inference loads exclusively from local paths and never downloads.

Usage:
    python bootstrap_models.py            # download (idempotent) + verify + write manifest
    python bootstrap_models.py --force    # re-download everything
    python bootstrap_models.py --check    # offline verification of existing artifacts

Artifacts:
    models/uk_ireland_accent_classification/   HF SavedModel, pinned revision
    models/yamnet/                             TF Hub google/yamnet/1 copy
    models/model_manifest.json                 checksums, provenance, I/O contract
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src import config  # noqa: E402  (pins cache env vars first)

# Pinned, immutable HF revision of the accent classifier.
HF_MODEL_ID = "fbadine/uk_ireland_accent_classification"
HF_REVISION = "ebe681a49e9506acdc4cf5312cb79b291275ab67"

# YAMNet handles, in preference order. tensorflow_hub resolves both to the
# same google/yamnet/1 artifact (tfhub.dev handles redirect to Kaggle).
YAMNET_HANDLES = (
    "https://tfhub.dev/google/yamnet/1",
    "https://www.kaggle.com/models/google/yamnet/frameworks/TfHub",
)

YAMNET_SPEECH_CLASS_INDEX = 0  # AudioSet class map row 0 == "Speech"
EXPECTED_CLASSES = (
    "Irish", "Midlands", "Northern", "Scottish", "Southern", "Welsh", "Not a speech",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path) -> list[dict]:
    """All files under root (relative paths + checksums), skipping the
    huggingface_hub local_dir bookkeeping cache."""
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts[:2] == (".cache", "huggingface"):
            continue
        entries.append(
            {"path": rel.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    return entries


def download_hf_classifier() -> None:
    # huggingface_hub reads HF_HUB_OFFLINE at import time; config pinned it
    # to "1" for inference, so clear it before importing the hub library.
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download

    target = config.CLASSIFIER_DIR
    target.mkdir(parents=True, exist_ok=True)
    print(f"[hf] Downloading {HF_MODEL_ID} @ {HF_REVISION[:12]} -> {target}")
    snapshot_download(
        repo_id=HF_MODEL_ID,
        revision=HF_REVISION,
        local_dir=str(target),
        allow_patterns=[
            "saved_model.pb",
            "keras_metadata.pb",
            "variables/*",
            "README.md",
            "*.png",
        ],
    )
    if not (target / "saved_model.pb").is_file():
        raise SystemExit("[hf] Download finished but saved_model.pb is missing.")


def download_yamnet() -> None:
    import tensorflow_hub as hub

    target = config.YAMNET_DIR
    last_error: Exception | None = None
    for handle in YAMNET_HANDLES:
        try:
            print(f"[yamnet] Resolving {handle} ...")
            resolved = Path(hub.resolve(handle))
            print(f"[yamnet] Cached at {resolved}, copying -> {target}")
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(resolved, target, dirs_exist_ok=True)
            if not any(
                (target / name).is_file() for name in ("saved_model.pb", "tfhub_module.pb")
            ):
                raise RuntimeError("Resolved YAMNet has no SavedModel entry point.")
            print(f"[yamnet] OK (handle used: {handle})")
            return
        except Exception as exc:  # try the next handle
            last_error = exc
            print(f"[yamnet] Handle failed: {exc}")
    raise SystemExit(f"[yamnet] Could not download YAMNet: {last_error}")


def probe_and_measure(tf, yamnet, classifier_dir: Path) -> dict:
    """Run the real models to discover/verify the I/O contract, and measure
    the YAMNet frame hop from known-duration inputs."""
    import numpy as np

    # --- classifier signature -------------------------------------------------
    loaded = tf.saved_model.load(str(classifier_dir))
    if "serving_default" not in loaded.signatures:
        raise SystemExit("[probe] Classifier SavedModel has no serving_default signature.")
    signature = loaded.signatures["serving_default"]
    _, kwargs = signature.structured_input_signature
    input_name, input_spec = next(iter(kwargs.items()))
    in_shape = input_spec.shape.as_list()
    if len(in_shape) != 2 or in_shape[1] != 1024:
        raise SystemExit(f"[probe] Classifier input {input_name}{in_shape} != (None, 1024).")

    probe = np.zeros((2, 1024), dtype=np.float32)
    outputs = signature(**{input_name: tf.convert_to_tensor(probe)})
    probs = next(iter(outputs.values())).numpy()
    if probs.shape != (2, len(EXPECTED_CLASSES)):
        raise SystemExit(f"[probe] Classifier output {probs.shape} != (2, {len(EXPECTED_CLASSES)}).")
    if not np.all(np.isfinite(probs)):
        raise SystemExit("[probe] Classifier produced non-finite outputs.")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
        raise SystemExit(f"[probe] Classifier rows do not sum to 1: {probs.sum(axis=1)}")
    print(f"[probe] classifier: {input_name}{in_shape} -> {probs.shape}, rows sum to 1")

    # --- YAMNet frame hop -----------------------------------------------------
    counts: dict[float, int] = {}
    for seconds in (10.0, 20.0):
        wav = np.zeros(int(seconds * 16000), dtype=np.float32)
        scores, embeddings, _ = yamnet(tf.convert_to_tensor(wav))
        n_frames = int(embeddings.shape[0])
        if int(scores.shape[1]) != 521 or int(embeddings.shape[1]) != 1024:
            raise SystemExit(
                f"[probe] YAMNet shapes wrong: scores {scores.shape}, embeddings {embeddings.shape}"
            )
        if n_frames == 0:
            raise SystemExit(f"[probe] YAMNet produced 0 frames for {seconds}s.")
        counts[seconds] = n_frames

    delta_frames = counts[20.0] - counts[10.0]
    if delta_frames <= 0:
        raise SystemExit(f"[probe] YAMNet frame counts did not scale with duration: {counts}")
    hop = 10.0 / delta_frames
    hop = round(hop, 4)
    print(f"[probe] YAMNet frames for 10s/20s: {counts} -> measured hop {hop}s/frame")

    return {
        "input": {
            "name": input_name,
            "shape": in_shape,
            "dtype": str(input_spec.dtype.name),
        },
        "output": {
            "shape": [None, len(EXPECTED_CLASSES)],
            "classes": list(EXPECTED_CLASSES),
        },
        "measured": {
            "frames_10s": counts[10.0],
            "frames_20s": counts[20.0],
        },
        "hop": hop,
    }


def write_manifest(probe: dict, yamnet_handle_used: str) -> None:
    from importlib.metadata import version

    import numpy as np
    import tensorflow as tf
    import tensorflow_hub

    manifest = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "models": {
            "yamnet": {
                "source": "TensorFlow Hub google/yamnet/1 (AudioSet YAMNet)",
                "handles": list(YAMNET_HANDLES),
                "handle_used": yamnet_handle_used,
                "license": "Apache-2.0 (TensorFlow models/research/audioset)",
                "local_path": "yamnet",
                "files": collect_files(config.YAMNET_DIR),
                "output": {
                    "scores_shape": [None, 521],
                    "embeddings_shape": [None, 1024],
                    "frame_hop_seconds": probe["hop"],
                    "frame_hop_source": (
                        "measured at bootstrap from 10s/20s inputs "
                        f"({probe['measured']['frames_10s']} / "
                        f"{probe['measured']['frames_20s']} frames)"
                    ),
                    "speech_class_index": YAMNET_SPEECH_CLASS_INDEX,
                    "speech_class_name": "Speech",
                },
            },
            "accent_classifier": {
                "source": f"huggingface:{HF_MODEL_ID}",
                "revision": HF_REVISION,
                "license": "Apache-2.0",
                "provenance": (
                    "Keras example 'English speaker accent recognition using transfer "
                    "learning' by Fadi Badine (keras.io); trained on OpenSLR-83 "
                    "(British Isles English Accents, 120 speakers)"
                ),
                "local_path": "uk_ireland_accent_classification",
                "files": collect_files(config.CLASSIFIER_DIR),
                "input": probe["input"],
                "output": probe["output"],
                "reference_aggregation": "mean over YAMNet frames, then argmax",
            },
        },
        "compatibility": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "tensorflow_hub": tensorflow_hub.__version__,
            "numpy": np.__version__,
            "huggingface_hub": version("huggingface_hub"),
            "loader": "tf.saved_model.load (Keras-3-safe; no from_pretrained_keras, no trust_remote_code)",
        },
    }
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] wrote {config.MANIFEST_PATH}")


def yamnet_handle_used() -> str:
    """Recover which handle succeeded from the populated cache dir."""
    cache = config.TFHUB_CACHE_DIR
    if not cache.is_dir():
        return "unknown"
    for entry in sorted(cache.iterdir()):
        if entry.is_dir() and (
            (entry / "saved_model.pb").is_file() or (entry / "tfhub_module.pb").is_file()
        ):
            return f"cached:{entry.name}"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument(
        "--check", action="store_true", help="offline verification of existing artifacts"
    )
    args = parser.parse_args()

    if args.check:
        from src.models.service import get_service

        get_service().ensure_loaded()
        print("[check] All artifacts verified; models load and validate offline.")
        return

    # Network is allowed only below this line (bootstrap-time only).
    os.environ["HF_HUB_OFFLINE"] = "0"

    hf_done = (config.CLASSIFIER_DIR / "saved_model.pb").is_file()
    yamnet_done = any(
        (config.YAMNET_DIR / name).is_file()
        for name in ("saved_model.pb", "tfhub_module.pb")
    )

    if args.force:
        hf_done = yamnet_done = False
        shutil.rmtree(config.CLASSIFIER_DIR, ignore_errors=True)
        shutil.rmtree(config.YAMNET_DIR, ignore_errors=True)
        config.MANIFEST_PATH.unlink(missing_ok=True)

    if not hf_done:
        download_hf_classifier()
    else:
        print(f"[hf] reusing {config.CLASSIFIER_DIR}")

    if not yamnet_done:
        download_yamnet()
    else:
        print(f"[yamnet] reusing {config.YAMNET_DIR}")

    # Import TF only after downloads (keeps download step fast) — cache env
    # was already pinned by src.config.
    import tensorflow as tf
    import tensorflow_hub as hub

    print("[probe] loading models to verify the artifact contract ...")
    yamnet = hub.load(str(config.YAMNET_DIR))
    probe = probe_and_measure(tf, yamnet, config.CLASSIFIER_DIR)
    write_manifest(probe, yamnet_handle_used())

    # Final self-check through the production service (checksums + probes).
    from src.models.service import get_service

    get_service().ensure_loaded()
    print("[done] Bootstrap complete. Normal inference will now run offline.")


if __name__ == "__main__":
    main()
