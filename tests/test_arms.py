from dataclasses import replace

import pytest

from boji_detector.arms import R_SLEEVES, arms_metric
from boji_detector.body import build_body_frame
from boji_detector.types import STATUS_OK, STATUS_UNAVAILABLE
from synth import SynthSpec, make_perception


def _arms(spec: SynthSpec):
    p = make_perception(spec)
    frame, _ = build_body_frame(p.landmarks)
    return arms_metric(p, frame)


def test_upper_arm_ratio_matches_drawn_thickness():
    spec = SynthSpec()
    m = _arms(spec)
    assert m.status == STATUS_OK
    expected = 2 * spec.upper_arm_half[0] / spec.upper_arm_len
    assert m.values["upper_ratio"] == pytest.approx(expected, rel=0.08)
    assert m.values["balance"] == pytest.approx(1.0, abs=0.04)


def test_each_side_is_measured_separately():
    m = _arms(replace(SynthSpec(), upper_arm_half=(30.0, 40.0)))
    left = m.values["upper_ratio_image_left"]
    right = m.values["upper_ratio_image_right"]
    assert left < right
    assert m.values["balance"] == pytest.approx(30.0 / 40.0, abs=0.06)


def test_thicker_arms_score_higher():
    thin = _arms(replace(SynthSpec(), upper_arm_half=(28.0, 28.0)))
    thick = _arms(replace(SynthSpec(), upper_arm_half=(42.0, 42.0)))
    assert thick.score > thin.score


def test_sleeves_make_arms_unavailable():
    m = _arms(replace(SynthSpec(), sleeves=True))
    assert m.status == STATUS_UNAVAILABLE
    assert R_SLEEVES in m.reasons


def test_visible_edges_win_over_mirrored_guesses():
    # Landmarks off the arm's centre line: where the arm touches the torso the
    # mirrored outer half is wrong, so samples with both edges visible must win.
    spec = SynthSpec()
    p = make_perception(spec)
    shifted = {}
    for name, lm in p.landmarks.items():
        if name.endswith(("_shoulder", "_elbow", "_wrist")):
            outward = 1 if lm.x > spec.cx else -1
            lm = replace(lm, x=lm.x + 8 * outward)
        shifted[name] = lm
    p = replace(p, landmarks=shifted)
    frame, _ = build_body_frame(p.landmarks)
    m = arms_metric(p, frame)
    expected = 2 * spec.upper_arm_half[0] / spec.upper_arm_len
    assert m.values["upper_ratio"] == pytest.approx(expected, rel=0.08)
    assert m.values["upper_width_estimated_image_left"] is False
