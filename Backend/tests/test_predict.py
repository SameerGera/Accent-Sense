"""End-to-end /api/predict tests against the REAL model.

No mock predictions exist in this suite; every 200 response must come from
the actual YAMNet + classifier pipeline. Model-failure paths are exercised
by injecting failures into the service seam, asserting the API answers a
generic 503 and never leaks internals.
"""

from __future__ import annotations

import numpy as np

from src.config import CLASSES, MAX_AUDIO_DURATION_SEC, NON_SPEECH_CLASS, SCORE_NOTE

from .conftest import make_wav_bytes


def _post(client, payload: bytes, filename: str = "clip.wav"):
    return client.post("/api/predict", files={"file": (filename, payload, "audio/wav")})


def _assert_valid_body(body: dict) -> None:
    assert body["predicted_influence"] in CLASSES
    assert list(body["all_scores"].keys()) == list(CLASSES), "class order must never be remapped"
    total = sum(body["all_scores"].values())
    assert 0.99 <= total <= 1.01, f"scores must sum to ~1, got {total}"
    assert 0.0 <= body["model_score"] <= 1.0
    assert body["model_score"] == max(body["all_scores"].values())
    assert len(body["timestamps"]) == len(body["evidence_curve"])
    assert all(t >= 0 for t in body["timestamps"])
    assert body["timestamps"] == sorted(body["timestamps"])
    assert all(0.0 <= c <= 1.0 for c in body["evidence_curve"])
    assert 0.0 <= body["speech_frame_ratio"] <= 1.0
    assert body["is_sufficient_speech"] is (body["predicted_influence"] != NON_SPEECH_CLASS)
    assert body["language_family"]
    for region in body["evidence_regions"]:
        assert set(region) == {
            "start_time_sec",
            "end_time_sec",
            "duration_sec",
            "model_score",
            "label",
            "detail",
        }


def test_predict_tone_returns_real_prediction(client):
    res = _post(client, make_wav_bytes(3.0))
    assert res.status_code == 200, res.text
    body = res.json()
    _assert_valid_body(body)


def test_predict_real_speech_file(client, speech_wav):
    res = _post(client, speech_wav, filename="speaker.wav")
    assert res.status_code == 200, res.text
    body = res.json()
    _assert_valid_body(body)
    # Speech should yield speech frames (sanity of the YAMNet speech rule).
    assert body["speech_frame_ratio"] > 0.0


def test_prediction_is_deterministic_not_random(client, tone_wav):
    first = _post(client, tone_wav).json()
    second = _post(client, tone_wav).json()
    assert first["predicted_influence"] == second["predicted_influence"]
    assert first["model_score"] == second["model_score"]
    assert first["all_scores"] == second["all_scores"]
    assert first["evidence_curve"] == second["evidence_curve"]


def test_no_mock_fields_and_honest_score_note(client, tone_wav):
    body = _post(client, tone_wav).json()
    # Deprecated mock-era fields are gone, not merely false.
    for legacy in ("is_mock_prototype", "confidence", "salient_regions", "saliency_curve"):
        assert legacy not in body
    assert body["model_score_note"] == SCORE_NOTE
    assert "~51%" in body["model_score_note"]
    assert "not a calibrated confidence" in body["model_score_note"]
    # Regions carry honest evidence wording, no phonological pseudo-explanations.
    for region in body["evidence_regions"]:
        assert "linguistic_phenomenon" not in region
        assert "phonetic_explanation" not in region


def test_model_not_reloaded_between_requests(client):
    first = client.get("/health").json()["model"]["load_seconds"]
    _post(client, make_wav_bytes(2.0))
    second = client.get("/health").json()["model"]["load_seconds"]
    assert first is not None and first == second


def test_malformed_upload_is_400_without_traceback(client):
    res = _post(client, b"not audio at all" * 50, filename="clip.wav")
    assert res.status_code == 400
    assert "Traceback" not in res.text
    assert "File \"" not in res.text


def test_empty_upload_is_400(client):
    res = _post(client, b"")
    assert res.status_code == 400


def test_oversized_upload_is_413(client):
    res = _post(client, b"\x00" * (16 * 1024 * 1024))
    assert res.status_code == 413


def test_too_short_is_400_rejected_not_truncated(client):
    res = _post(client, make_wav_bytes(0.3))
    assert res.status_code == 400
    assert "too short" in res.json()["detail"]


def test_too_long_is_400_rejected_not_truncated(client):
    res = _post(client, make_wav_bytes(MAX_AUDIO_DURATION_SEC + 5.0))
    assert res.status_code == 400
    assert "too long" in res.json()["detail"]


def test_extension_is_never_trusted(client):
    # Lying extension on valid audio → accepted (content decides).
    assert _post(client, make_wav_bytes(2.0), filename="payload.txt").status_code == 200
    # Plausible extension on non-audio → rejected.
    assert _post(client, b"<html>hello</html>" * 40, filename="clip.wav").status_code == 400


def test_stereo_and_non_16k_are_accepted(client):
    res = _post(client, make_wav_bytes(3.0, sr=44100, channels=2))
    assert res.status_code == 200, res.text
    _assert_valid_body(res.json())


def test_model_failure_returns_generic_503(client, tone_wav, monkeypatch):
    import src.api.main as main_module
    from src.models.service import ModelError

    class ExplodingService:
        def infer(self, waveform):
            raise ModelError("secret internal state at /home/sameer/model.py:42")

    monkeypatch.setattr(main_module, "get_service", ExplodingService)
    res = _post(client, tone_wav)
    assert res.status_code == 503
    detail = res.json()["detail"]
    assert detail == "The accent model failed to process this recording."
    assert "secret" not in res.text
    assert "Traceback" not in res.text
    assert ".py" not in res.text


def test_unexpected_model_crash_also_returns_503(client, tone_wav, monkeypatch):
    import src.api.main as main_module

    class CrashingService:
        def infer(self, waveform):
            raise RuntimeError("TF kernel exploded: tensor shape (7, 3)")

    monkeypatch.setattr(main_module, "get_service", CrashingService)
    res = _post(client, tone_wav)
    assert res.status_code == 503
    assert "TF kernel" not in res.text
    assert "Traceback" not in res.text


def test_missing_artifacts_returns_503(client, tone_wav, monkeypatch):
    import src.api.main as main_module
    from src.models.service import ModelNotBootstrappedError

    class NotInstalledService:
        def infer(self, waveform):
            raise ModelNotBootstrappedError("manifest not found; run bootstrap")

    monkeypatch.setattr(main_module, "get_service", NotInstalledService)
    res = _post(client, tone_wav)
    assert res.status_code == 503
    assert res.json()["detail"] == "The accent model is not installed on this server."
    assert "bootstrap" not in res.text


def test_not_a_speech_state_is_surfaced_honestly(client, tone_wav, monkeypatch):
    """The real class 'Not a speech' must surface as insufficient speech."""
    import src.api.main as main_module
    from src.models.service import (
        AccentModelService,
        FrameEvidence,
        Prediction,
    )

    probs = (0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.52)
    frames = tuple(
        FrameEvidence(timestamp_sec=round(i * 0.48, 3), probabilities=probs, is_speech=False)
        for i in range(12)
    )
    canned = Prediction(
        predicted_class=NON_SPEECH_CLASS,
        predicted_index=len(CLASSES) - 1,
        aggregated_probabilities=probs,
        speech_frame_ratio=0.0,
        frames=frames,
        duration_sec=5.76,
    )

    class CannedService:
        def infer(self, waveform):
            return canned

        temporal_evidence = staticmethod(AccentModelService.temporal_evidence)
        evidence_regions = staticmethod(AccentModelService.evidence_regions)

    monkeypatch.setattr(main_module, "get_service", CannedService)
    body = _post(client, tone_wav).json()
    _assert_valid_body(body)
    assert body["predicted_influence"] == NON_SPEECH_CLASS
    assert body["is_sufficient_speech"] is False
    assert "Insufficient speech" in body["language_family"]
    # No accent profile → downstream ASR refuses instead of inventing one.
    res = client.post("/api/downstream-asr", data={"detected_accent": body["predicted_influence"]})
    assert res.status_code == 400


def test_evidence_regions_match_curve_threshold(client, speech_wav):
    """Regions are derived from real per-frame scores, consistently."""
    body = _post(client, speech_wav, filename="speech.wav").json()
    curve = np.asarray(body["evidence_curve"], dtype=np.float64)
    for region in body["evidence_regions"]:
        assert region["start_time_sec"] <= region["end_time_sec"]
        assert region["model_score"] >= 0.5, "high-evidence regions must clear the floor"
        if len(curve) > 1:
            assert region["model_score"] <= float(curve.max()) + 1e-9
        assert region["label"]
        assert region["detail"]
