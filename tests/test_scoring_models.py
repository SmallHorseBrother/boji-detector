import json

import pytest

from boji_detector import models
from boji_detector.cli import main
from boji_detector.scoring import Curve, DEFAULT_CONFIG, load_config, logistic_score, total_score
from boji_detector.types import Metric, STATUS_OK, STATUS_UNAVAILABLE


def test_logistic_score_is_monotone_and_bounded():
    curve = Curve(mid=1.0, scale=0.1)
    values = [logistic_score(v, curve) for v in (0.0, 0.9, 1.0, 1.1, 5.0)]
    assert values == sorted(values)
    assert values[2] == pytest.approx(50.0)
    assert 0.0 <= values[0] < 1.0 and 99.0 < values[-1] <= 100.0


def test_total_renormalises_available_weights():
    metrics = {
        "abs": Metric("abs", STATUS_OK, score=80.0),
        "v_taper": Metric("v_taper", STATUS_OK, score=40.0),
        "arms": Metric("arms", STATUS_UNAVAILABLE),
        "symmetry": Metric("symmetry", STATUS_OK, score=60.0),
    }
    total = total_score(metrics)
    w = DEFAULT_CONFIG.weights
    expected = (80 * w["abs"] + 40 * w["v_taper"] + 60 * w["symmetry"]) / (w["abs"] + w["v_taper"] + w["symmetry"])
    assert total["score"] == pytest.approx(expected)
    assert total["missing"] == ["arms"]


def test_config_file_overrides_curves_and_weights(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"abs": {"mid": 0.3, "scale": 0.05}, "weights": {"abs": 1.0}}), encoding="utf-8")
    config = load_config(path)
    assert config.abs == Curve(0.3, 0.05)
    assert config.weights == {"abs": 1.0}
    assert config.v_taper == DEFAULT_CONFIG.v_taper


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"bogus": {"mid": 1, "scale": 1}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_model_reports_how_to_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv(models.ENV_MODELS_DIR, str(tmp_path))
    with pytest.raises(models.ModelMissingError, match="fetch-models"):
        models.require("pose")
    assert models.status() == {"pose": "missing", "segmenter": "missing"}


def test_checksum_mismatch_is_detected(tmp_path, monkeypatch):
    monkeypatch.setenv(models.ENV_MODELS_DIR, str(tmp_path))
    (tmp_path / models.MODELS["pose"].filename).write_bytes(b"not a model")
    assert models.status()["pose"] == "checksum mismatch"


def test_mirror_base_url_is_used(monkeypatch):
    monkeypatch.setenv(models.ENV_BASE_URL, "https://mirror.example/models/")
    url = models.download_url(models.MODELS["segmenter"])
    assert url == "https://mirror.example/models/selfie_multiclass_256x256.tflite"


def test_doctor_fails_without_models(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(models.ENV_MODELS_DIR, str(tmp_path))
    assert main(["doctor"]) == 1
    assert "missing" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [
    {"abs": {"mid": 0.5, "scale": 0}},
    {"abs": {"mid": "nan", "scale": 0.1}},
    {"weights": {"abs": -1.0}},
    {"weights": {"abs": 0.0}},
])
def test_invalid_config_values_are_rejected(tmp_path, bad):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_total_needs_enough_of_the_weights():
    only_symmetry = {
        "abs": Metric("abs", STATUS_UNAVAILABLE),
        "v_taper": Metric("v_taper", STATUS_UNAVAILABLE),
        "arms": Metric("arms", STATUS_UNAVAILABLE),
        "symmetry": Metric("symmetry", STATUS_OK, score=99.0),
    }
    total = total_score(only_symmetry)
    assert total["score"] is None
    assert total["missing"] == ["abs", "arms", "v_taper"]
