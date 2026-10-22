import cv2
import pytest
from fastapi.testclient import TestClient

from boji_detector.server import create_app
from synth import SynthSpec, make_perception


class FakePerceiver:
    closed = False

    def perceive(self, bgr):
        return make_perception(SynthSpec())

    def close(self):
        FakePerceiver.closed = True


def _png() -> bytes:
    ok, buf = cv2.imencode(".png", make_perception(SynthSpec()).image)
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("BOJI_SERVICE_TOKEN", raising=False)
    with TestClient(create_app(perceiver_factory=FakePerceiver)) as c:
        yield c


def test_health_reports_loaded_models(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_analyze_returns_report_and_messages(client):
    r = client.post("/v1/analyze", content=_png(), headers={"Content-Type": "image/png"})
    assert r.status_code == 200
    body = r.json()
    assert body["report"]["metrics"]["abs"]["values"]["pack_count"] == 6
    assert body["report"]["total"]["score"] is not None
    codes = {c for m in body["report"]["metrics"].values() for c in m["reasons"]} | set(body["report"]["quality"])
    assert set(body["messages"]) == codes
    for text in body["messages"].values():
        assert text["zh"] and text["en"]


def test_token_is_required_when_configured(monkeypatch):
    monkeypatch.setenv("BOJI_SERVICE_TOKEN", "s3cret")
    with TestClient(create_app(perceiver_factory=FakePerceiver)) as c:
        assert c.post("/v1/analyze", content=_png()).status_code == 401
        assert c.post("/v1/analyze", content=_png(), headers={"X-Boji-Token": "wrong"}).status_code == 401
        assert c.post("/v1/analyze", content=_png(), headers={"X-Boji-Token": "s3cret"}).status_code == 200


@pytest.mark.parametrize("body,status,detail", [
    (b"", 400, "empty_body"),
    (b"definitely not an image", 400, "unreadable_image"),
])
def test_bad_uploads_are_rejected(client, body, status, detail):
    r = client.post("/v1/analyze", content=body)
    assert r.status_code == status and r.json()["detail"] == detail


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setenv("BOJI_MAX_UPLOAD_MB", "0.5")
    r = client.post("/v1/analyze", content=b"x" * (600 * 1024))
    assert r.status_code == 413


def test_perceiver_is_closed_on_shutdown():
    FakePerceiver.closed = False
    with TestClient(create_app(perceiver_factory=FakePerceiver)):
        pass
    assert FakePerceiver.closed
