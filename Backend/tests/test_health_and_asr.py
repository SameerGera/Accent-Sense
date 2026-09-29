"""Endpoint tests for /health and /api/downstream-asr (honest semantics)."""

from __future__ import annotations

from src.config import CLASSES


class TestHealth:
    def test_health_reports_real_readiness(self, client):
        res = client.get("/health")
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["status"] == "healthy"
        assert body["inference_ready"] is True
        assert body["device"] == "cpu"
        assert body["supported_classes"] == list(CLASSES)
        model = body["model"]
        assert model["yamnet_loaded"] is True
        assert model["accent_classifier_loaded"] is True
        assert model["manifest_present"] is True
        assert model["artifacts_present"] is True
        assert model["error"] is None
        assert isinstance(model["load_seconds"], (int, float))

    def test_health_does_not_reload_model(self, client):
        first = client.get("/health").json()["model"]["load_seconds"]
        second = client.get("/health").json()["model"]["load_seconds"]
        assert first is not None and first == second

    def test_health_503_when_service_reports_not_ready(self, client, monkeypatch):
        import src.api.main as main_module
        from src.models.service import AccentModelService

        broken = AccentModelService.__new__(AccentModelService)
        broken._loaded = False
        broken._load_error = "ModelNotBootstrappedError: missing"
        broken._yamnet = None
        broken._classifier_signature = None
        broken._load_seconds = None
        broken._lock = None  # status() won't touch it: _load_error already set

        monkeypatch.setattr(main_module, "get_service", lambda: broken)
        res = client.get("/health")
        assert res.status_code == 503
        body = res.json()
        assert body["status"] == "degraded"
        assert body["inference_ready"] is False
        assert body["model"]["error"]


class TestDownstreamASR:
    def test_known_accent_returns_static_example(self, client):
        res = client.post("/api/downstream-asr", data={"detected_accent": "Northern"})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["is_static_example"] is True
        assert "no speech-recognition model was executed" in body["disclaimer"]
        assert body["baseline_transcript"]
        assert body["accent_adapted_transcript"]
        assert body["phonetic_corrections_noted"]
        assert "Northern" in body["detected_accent_profile"]

    def test_every_accent_class_has_benchmark_data(self, client):
        for accent in CLASSES:
            res = client.post("/api/downstream-asr", data={"detected_accent": accent})
            if accent == "Not a speech":
                # Insufficient speech has no accent profile to adapt to.
                assert res.status_code == 400, accent
                assert "insufficient speech" in res.json()["detail"].lower()
            else:
                assert res.status_code == 200, f"{accent}: {res.text}"
                assert res.json()["is_static_example"] is True

    def test_unknown_accent_rejected(self, client):
        res = client.post("/api/downstream-asr", data={"detected_accent": "RP"})
        assert res.status_code == 400
        assert "Unknown accent" in res.json()["detail"]

    def test_response_never_leaks_stack_traces(self, client):
        res = client.post("/api/downstream-asr", data={"detected_accent": "Atlantis"})
        assert res.status_code == 400
        assert "Traceback" not in res.text
        assert "File \"" not in res.text
