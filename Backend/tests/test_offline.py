"""Offline-guarantee tests: bootstrap --check and socket-blocked inference.

These prove the two core supply-chain rules:
  1. Artifacts verify and load with zero network access.
  2. Normal inference runs even when the network is hard-disabled.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

# Runs in a subprocess with sockets crippled BEFORE any project import, so
# an accidental network touch (HF download, TF-Hub fetch, telemetry) fails
# the test instead of silently succeeding.
_SOCKET_BLOCKED_SCRIPT = """
import json
import socket

def _blocked(*args, **kwargs):
    raise RuntimeError("NETWORK ACCESS ATTEMPTED DURING INFERENCE")

socket.socket = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
socket.gethostbyname = _blocked

import numpy as np

from src.models.service import get_service

rng = np.random.default_rng(0)
wave = (0.1 * rng.standard_normal(16000 * 4)).astype(np.float32)
prediction = get_service().infer(wave)
result = {
    "predicted_class": prediction.predicted_class,
    "frames": len(prediction.frames),
    "scores_sum": round(sum(prediction.aggregated_probabilities), 4),
    "finite": bool(np.all(np.isfinite(prediction.aggregated_probabilities))),
}
print("RESULT:" + json.dumps(result))
"""


def _run(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "no_proxy": "*",
        }
    )
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, *args],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_bootstrap_check_verifies_artifacts_offline():
    proc = _run(["bootstrap_models.py", "--check"])
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "All artifacts verified" in proc.stdout
    # Must not have hit the network in offline mode.
    assert "Downloading" not in proc.stdout


def test_inference_works_with_sockets_disabled():
    proc = _run(["-c", _SOCKET_BLOCKED_SCRIPT])
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    marker = [line for line in proc.stdout.splitlines() if line.startswith("RESULT:")]
    assert marker, f"no result in stdout: {proc.stdout}"
    result = json.loads(marker[0][len("RESULT:"):])
    assert result["frames"] > 0
    assert abs(result["scores_sum"] - 1.0) < 0.01
    assert result["finite"] is True


def test_config_pins_offline_before_hub_import():
    """src.config must flip HF offline mode at import, before any hub code."""
    proc = _run(
        [
            "-c",
            (
                "import os, src.config; "
                "assert os.environ['HF_HUB_OFFLINE'] == '1', os.environ.get('HF_HUB_OFFLINE'); "
                "assert os.environ['HF_HOME'].endswith('models/.hf-cache'); "
                "print('OK')"
            ),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
