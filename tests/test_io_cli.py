import json
import os

import cv2
import numpy as np
import pytest
from PIL import Image

from boji_detector import models, perception_mp
from boji_detector.analyze import analyze_image, load_image
from boji_detector.cli import main
from boji_detector.silhouette import clean_mask, run_at, walk_extent
from synth import SynthSpec, make_perception


class FakePerceiver:
    """Stands in for MediaPipe: returns the synthetic torso for any image."""

    def __init__(self, *args, **kwargs):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed = True

    def perceive(self, bgr):
        return make_perception(SynthSpec())


class NoPersonPerceiver(FakePerceiver):
    def perceive(self, bgr):
        return None


def _write_png(path, image):
    ok, buf = cv2.imencode(".png", image)
    assert ok
    buf.tofile(str(path))


def test_cli_analyze_writes_json_and_overlay(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(perception_mp, "MediaPipePerceiver", FakePerceiver)
    photo = tmp_path / "front.png"
    _write_png(photo, make_perception(SynthSpec()).image)
    out = tmp_path / "out"
    assert main(["analyze", str(photo), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "腹肌清晰度" in text and "pack_count=6" in text
    data = json.loads((out / "front.json").read_text(encoding="utf-8"))
    assert data["metrics"]["abs"]["values"]["pack_count"] == 6
    assert (out / "front.overlay.png").exists()


def test_cli_json_format_and_english(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(perception_mp, "MediaPipePerceiver", NoPersonPerceiver)
    photo = tmp_path / "empty.png"
    _write_png(photo, np.full((120, 100, 3), 90, np.uint8))
    assert main(["analyze", str(photo), "--format", "json", "--lang", "en"]) == 0
    data = json.loads(capsys.readouterr().out)
    report = data[str(photo)]
    assert report["total"]["score"] is None
    assert report["quality"] == ["pose_not_found"]


def test_analyze_image_closes_the_perceiver_it_creates(monkeypatch):
    created = []

    class Tracking(FakePerceiver):
        def __init__(self, *a, **k):
            super().__init__()
            created.append(self)

    monkeypatch.setattr(perception_mp, "MediaPipePerceiver", Tracking)
    report, perception = analyze_image(make_perception(SynthSpec()).image)
    assert report.metrics["abs"].values["pack_count"] == 6
    assert perception is not None and created and created[0].closed


def test_load_image_applies_exif_rotation_and_caps_size(tmp_path):
    path = tmp_path / "rotated.jpg"
    img = Image.new("RGB", (400, 200), (200, 100, 50))
    exif = img.getexif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees clockwise to display
    img.save(path, exif=exif)
    loaded = load_image(path, max_side=100)
    assert loaded.shape[:2] == (100, 50)  # portrait after EXIF, longest side capped


def test_clean_mask_fills_small_holes_but_keeps_real_gaps():
    mask = np.zeros((100, 100), bool)
    mask[10:90, 10:60] = True
    mask[20:23, 20:23] = False      # small hole (noise): filled
    mask[40:70, 30:40] = False      # large enclosed gap (arm-torso loop): kept
    mask[5:8, 80:83] = True         # speck: dropped
    cleaned = clean_mask(mask)
    assert cleaned[21, 21]
    assert not cleaned[55, 35]
    assert not cleaned[6, 81]
    assert not clean_mask(np.zeros((10, 10), bool)).any()


def test_run_at_and_walk_extent():
    mask = np.zeros((5, 20), bool)
    mask[2, 4:12] = True
    assert run_at(mask, 2, 6) == (4, 11)
    assert run_at(mask, 2, 13) == (4, 11)            # within tolerance of the run
    assert run_at(mask, 2, 18) is None
    assert run_at(mask, 9, 6) is None
    assert walk_extent(mask, np.array([6.0, 2.0]), np.array([1.0, 0.0]), 30) == 6
    full = np.ones((5, 5), bool)
    assert walk_extent(full, np.array([2.0, 2.0]), np.array([0.0, 0.0]), 4) == 4


@pytest.mark.skipif(
    not os.environ.get("BOJI_SAMPLE_IMAGE") or models.status() != {"pose": "ok", "segmenter": "ok"},
    reason="set BOJI_SAMPLE_IMAGE and fetch models to run the real-model check",
)
def test_real_photo_with_mediapipe():
    report, perception = analyze_image(os.environ["BOJI_SAMPLE_IMAGE"])
    assert perception is not None
    assert perception.skin_mask is not None and perception.skin_mask.any()
    assert report.metrics["v_taper"].available


def test_cli_keeps_going_after_a_bad_file_and_avoids_name_clashes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(perception_mp, "MediaPipePerceiver", FakePerceiver)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    image = make_perception(SynthSpec()).image
    _write_png(a / "front.png", image)
    _write_png(b / "front.png", image)
    out = tmp_path / "out"
    missing = tmp_path / "nope.jpg"
    code = main(["analyze", str(missing), str(a / "front.png"), str(b / "front.png"),
                 "--out", str(out), "--format", "json", "--no-overlay"])
    assert code == 1
    data = json.loads(capsys.readouterr().out)
    assert "error" in data[str(missing)]
    assert sorted(f.name for f in out.iterdir()) == ["front-2.json", "front.json"]


class _StubMP:
    class ImageFormat:
        SRGB = 1

    @staticmethod
    def Image(image_format, data):
        return data


class _StubSegmenter:
    def __init__(self):
        self.shapes = []

    def segment(self, image):
        from types import SimpleNamespace

        self.shapes.append(image.shape)
        categories = np.full(image.shape[:2], perception_mp.CATEGORY_BODY_SKIN, np.uint8)
        return SimpleNamespace(category_mask=SimpleNamespace(numpy_view=lambda: categories))


def _stub_perceiver():
    perceiver = object.__new__(perception_mp.MediaPipePerceiver)
    perceiver._mp = _StubMP
    perceiver._segmenter = _StubSegmenter()
    return perceiver


def test_segmenter_gets_a_true_square_even_at_the_photo_edge():
    perceiver = _stub_perceiver()
    person = np.zeros((300, 200), bool)
    person[20:290, 0:120] = True                     # body runs off the left edge
    skin, clothes = perceiver._segment(np.zeros((300, 200, 3), np.uint8), person)
    (shape,) = perceiver._segmenter.shapes
    assert shape[0] == shape[1]
    assert skin[150, 60] and not skin[150, 190]
    assert not clothes.any()


def test_segmenter_skips_a_tiny_person_mask():
    perceiver = _stub_perceiver()
    person = np.zeros((50, 50), bool)
    person[10, 10] = True
    assert perceiver._segment(np.zeros((50, 50, 3), np.uint8), person) == (None, None)
    assert perceiver._segmenter.shapes == []


MODELS_READY = models.status() == {"pose": "ok", "segmenter": "ok"}


def _grace_hopper() -> str:
    from matplotlib import cbook

    return str(cbook.get_sample_data("grace_hopper.jpg", asfileobj=False))


@pytest.mark.skipif(not MODELS_READY, reason="run `boji fetch-models` to enable real-model tests")
def test_real_models_load_from_a_non_ascii_folder_and_masks_are_owned(tmp_path):
    import shutil

    folder = tmp_path / "模型"
    folder.mkdir()
    for name in ("pose", "segmenter"):
        shutil.copy(models.model_path(name), folder / models.MODELS[name].filename)
    portrait = load_image(_grace_hopper())
    other = np.ascontiguousarray(portrait[:, ::-1])
    with perception_mp.MediaPipePerceiver(
        pose_model=folder / models.MODELS["pose"].filename,
        segmenter_model=folder / models.MODELS["segmenter"].filename,
    ) as perceiver:
        first = perceiver.perceive(portrait)
        perceiver.perceive(other)                    # must not disturb the first result
        again = perceiver.perceive(portrait)
    assert first is not None and again is not None
    assert set(np.unique(first.skin_mask.astype(np.uint8))) <= {0, 1}
    assert np.array_equal(first.person_mask, again.person_mask)
    assert np.array_equal(first.skin_mask, again.skin_mask)


@pytest.mark.skipif(not MODELS_READY, reason="run `boji fetch-models` to enable real-model tests")
def test_head_and_shoulders_portrait_is_not_scored_as_a_torso():
    report, perception = analyze_image(_grace_hopper())
    assert perception is not None
    assert report.metrics["v_taper"].reasons == ("shoulders_cut_off",)
    assert report.metrics["abs"].reasons == ("abdomen_out_of_frame",)
    assert report.total["score"] is None
