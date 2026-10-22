import json
from dataclasses import replace

import numpy as np
import pytest

from boji_detector import abdomen, arms, body, proportion, quality, symmetry
from boji_detector.analyze import analyze_perception
from boji_detector.messages import MESSAGES, describe
from boji_detector.render import render_overlay
from boji_detector.types import Landmark, Perception, STATUS_UNAVAILABLE
from boji_detector.upright import rotate_perception
from synth import SynthSpec, make_perception


def test_report_is_json_serialisable_and_complete():
    report, _ = analyze_perception(make_perception(SynthSpec()))
    data = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert set(data["metrics"]) == {"v_taper", "symmetry", "arms", "abs"}
    assert data["total"]["score"] is not None
    assert data["total"]["calibration"] == "uncalibrated-v0"
    assert abs(sum(data["total"]["weights_used"].values()) - 1.0) < 1e-6


def test_tilted_photo_is_deskewed_to_the_same_result():
    straight, _ = analyze_perception(make_perception(SynthSpec()))
    tilted_input = rotate_perception(make_perception(SynthSpec()), 8.0)
    tilted, upright = analyze_perception(tilted_input)
    assert tilted.deskew_deg == pytest.approx(-8.0, abs=0.5)
    s, t = straight.metrics, tilted.metrics
    assert t["v_taper"].values["ratio"] == pytest.approx(s["v_taper"].values["ratio"], rel=0.04)
    assert t["abs"].values["pack_count"] == s["abs"].values["pack_count"]
    lean = body.build_body_frame(upright.landmarks)[0].lean_deg
    assert abs(lean) < 0.5


def test_missing_pose_gives_unavailable_metrics():
    p = make_perception(SynthSpec())
    empty = Perception(image=p.image, landmarks={}, person_mask=p.person_mask)
    report, _ = analyze_perception(empty)
    assert all(m.status == STATUS_UNAVAILABLE for m in report.metrics.values())
    assert report.total["score"] is None
    assert body.R_POSE_NOT_FOUND in report.quality


def test_hips_out_of_view_are_estimated():
    p = make_perception(SynthSpec())
    landmarks = dict(p.landmarks)
    for name in ("left_hip", "right_hip"):
        lm = landmarks[name]
        landmarks[name] = Landmark(lm.x, lm.y, lm.z, 0.05)
    report, _ = analyze_perception(replace(p, landmarks=landmarks))
    assert body.R_HIPS_ESTIMATED in report.quality
    assert report.metrics["v_taper"].status != STATUS_UNAVAILABLE


def test_total_skips_unavailable_metrics():
    report, _ = analyze_perception(make_perception(replace(SynthSpec(), shirt=True)))
    assert report.metrics["abs"].status == STATUS_UNAVAILABLE
    assert "abs" in report.total["missing"]
    assert "abs" not in report.total["weights_used"]


def test_every_reason_code_has_a_message():
    modules = (abdomen, arms, body, proportion, quality, symmetry)
    codes = {
        value for module in modules for name, value in vars(module).items()
        if name.startswith("R_") and isinstance(value, str)
    }
    assert codes <= set(MESSAGES), codes - set(MESSAGES)
    assert describe("arms_raised", "en").startswith("Arms")


def test_overlay_never_modifies_the_input_image():
    # No face landmarks are visible in the synthetic photo, the path that once
    # drew straight onto the caller's array.
    report, upright = analyze_perception(make_perception(SynthSpec()))
    before = upright.image.copy()
    for blur in (True, False):
        render_overlay(upright, report, blur_face=blur)
        assert np.array_equal(upright.image, before)


def test_overlay_keeps_size_and_pixelates_a_visible_face():
    p = make_perception(SynthSpec())
    face = {name: Landmark(320.0, 60.0 + i, 0.0, 0.99)
            for i, name in enumerate(("nose", "left_eye", "right_eye", "left_ear", "right_ear"))}
    face["left_ear"] = Landmark(290.0, 60.0, 0.0, 0.99)
    face["right_ear"] = Landmark(350.0, 60.0, 0.0, 0.99)
    p = replace(p, landmarks={**p.landmarks, **face})
    textured = p.image.copy()
    textured[30:90, 290:350] = np.random.default_rng(1).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    p = replace(p, image=textured)
    report, upright = analyze_perception(p)
    out = render_overlay(upright, report, blur_face=True)
    assert out.shape == upright.image.shape
    patch_in = upright.image[40:80, 300:340].astype(int)
    patch_out = out[40:80, 300:340].astype(int)
    assert np.abs(np.diff(patch_out, axis=1)).mean() < np.abs(np.diff(patch_in, axis=1)).mean()


def test_turned_body_is_flagged_from_the_silhouette():
    p = make_perception(SynthSpec())
    straight, _ = analyze_perception(p)
    assert proportion.R_BODY_ROTATED not in straight.quality
    assert straight.metrics["symmetry"].values["rotation_index"] < 0.05
    # Joints shifted against the silhouette, as when the torso turns.
    shifted = {k: Landmark(v.x + 30.0, v.y, v.z, v.visibility) for k, v in p.landmarks.items()}
    turned, _ = analyze_perception(replace(p, landmarks=shifted))
    assert proportion.R_BODY_ROTATED in turned.quality


def _scaled(p: Perception, factor: float) -> Perception:
    import cv2

    size = (int(p.width * factor), int(p.height * factor))

    def mask(m):
        return None if m is None else cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0

    landmarks = {k: Landmark(v.x * factor, v.y * factor, v.z, v.visibility) for k, v in p.landmarks.items()}
    return Perception(
        image=cv2.resize(p.image, size, interpolation=cv2.INTER_CUBIC),
        landmarks=landmarks,
        person_mask=mask(p.person_mask),
        skin_mask=mask(p.skin_mask),
        clothes_mask=mask(p.clothes_mask),
    )


@pytest.mark.parametrize("factor", [0.7, 1.6])
def test_metrics_do_not_depend_on_resolution(factor):
    base, _ = analyze_perception(make_perception(SynthSpec()))
    scaled, _ = analyze_perception(_scaled(make_perception(SynthSpec()), factor))
    b, s = base.metrics, scaled.metrics
    assert s["v_taper"].values["ratio"] == pytest.approx(b["v_taper"].values["ratio"], rel=0.03)
    assert s["arms"].values["upper_ratio"] == pytest.approx(b["arms"].values["upper_ratio"], rel=0.06)
    assert s["abs"].values["pack_count"] == b["abs"].values["pack_count"]
    assert s["abs"].values["definition_index"] == pytest.approx(b["abs"].values["definition_index"], abs=0.05)


def test_hips_out_of_view_do_not_turn_camera_roll_into_shoulder_asymmetry():
    def report_for(drop):
        p = make_perception(replace(SynthSpec(), shoulder_drop=(0.0, drop)))
        hidden = {k: (Landmark(v.x, v.y, v.z, 0.05) if k.endswith("_hip") else v) for k, v in p.landmarks.items()}
        return analyze_perception(replace(p, landmarks=hidden))[0].metrics["symmetry"]

    level, rolled = report_for(0.0), report_for(14.0)
    assert body.R_HIPS_ESTIMATED in rolled.reasons and rolled.status == "low_confidence"
    assert abs(rolled.score - level.score) < 3.0


def test_small_deskew_does_not_make_the_photo_blurry():
    straight, _ = analyze_perception(make_perception(SynthSpec()))
    assert quality.R_BLURRY not in straight.quality
    rotated, _ = analyze_perception(rotate_perception(make_perception(SynthSpec()), 3.0))
    assert quality.R_BLURRY not in rotated.quality


def test_camera_tilted_beyond_deskew_lowers_every_metric():
    report, _ = analyze_perception(rotate_perception(make_perception(SynthSpec()), 30.0))
    assert quality.R_CAMERA_TILTED in report.quality
    for metric in report.metrics.values():
        if metric.available:
            assert metric.status == "low_confidence"
            assert quality.R_CAMERA_TILTED in metric.reasons


def test_to_original_maps_upright_geometry_back_onto_the_input():
    from boji_detector.types import map_points

    tilted = rotate_perception(make_perception(SynthSpec()), 8.0)
    report, upright = analyze_perception(tilted)
    for name in ("left_shoulder", "right_hip", "left_elbow"):
        moved = upright.landmarks[name]
        (x, y), = map_points([[moved.x, moved.y]], report.to_original)
        assert x == pytest.approx(tilted.landmarks[name].x, abs=1.0)
        assert y == pytest.approx(tilted.landmarks[name].y, abs=1.0)
    assert report.original_size == (tilted.width, tilted.height)


def test_analyze_image_reports_coordinates_of_the_full_size_photo(monkeypatch):
    from boji_detector import perception_mp
    from boji_detector.analyze import analyze_image
    from boji_detector.types import map_points

    class ScaledPerceiver:
        def perceive(self, bgr):
            factor = bgr.shape[1] / 640.0
            return _scaled(make_perception(SynthSpec()), factor)

        def close(self):
            pass

    base, _ = analyze_perception(make_perception(SynthSpec()))
    (bx0, by), (bx1, _) = base.metrics["v_taper"].geometry["shoulder_line"]
    big = _scaled(make_perception(SynthSpec()), 2.5)  # 1600 x 2250, capped to 1600 tall for analysis
    report, _ = analyze_image(big.image, perceiver=ScaledPerceiver())
    assert report.original_size == (big.width, big.height)
    assert max(report.image_size) <= 1600
    line = report.metrics["v_taper"].geometry["shoulder_line"]
    (x0, y0), (x1, _) = map_points(line, report.to_original)
    assert x0 == pytest.approx(2.5 * bx0, rel=0.03)
    assert x1 == pytest.approx(2.5 * bx1, rel=0.03)
    assert y0 == pytest.approx(2.5 * by, rel=0.03)


def test_array_input_is_validated_and_capped():
    from boji_detector.analyze import prepare_array

    grey = np.full((3000, 1000), 128, np.uint8)
    bgr, factor = prepare_array(grey)
    assert bgr.shape == (1600, 533, 3) and factor == pytest.approx(1600 / 3000)
    with pytest.raises(ValueError):
        prepare_array(np.zeros((10, 10, 3), np.float32))
    with pytest.raises(ValueError):
        prepare_array(np.zeros((10, 10, 2), np.uint8))


def test_head_is_pixelated_even_without_face_landmarks():
    # The synthetic neck runs to the top of the photo; no face points are visible.
    p = make_perception(SynthSpec())
    textured = p.image.copy()
    textured[20:80, 290:350] = np.random.default_rng(2).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    p = replace(p, image=textured)
    report, upright = analyze_perception(p)
    out = render_overlay(upright, report, blur_face=True)
    before = np.abs(np.diff(upright.image[25:75, 295:345].astype(int), axis=1)).mean()
    after = np.abs(np.diff(out[25:75, 295:345].astype(int), axis=1)).mean()
    assert after < 0.5 * before
