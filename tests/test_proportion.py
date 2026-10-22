import math
from dataclasses import replace

import pytest

from boji_detector.body import build_body_frame
from boji_detector.proportion import R_ARMS_RAISED, R_ARMS_TOUCHING, v_taper_metric
from boji_detector.symmetry import symmetry_metric
from boji_detector.types import STATUS_OK, STATUS_UNAVAILABLE
from synth import SynthSpec, make_perception


def _frame(spec: SynthSpec):
    p = make_perception(spec)
    frame, _ = build_body_frame(p.landmarks)
    assert frame is not None
    return p, frame


def test_v_taper_matches_drawn_widths():
    spec = SynthSpec()
    p, frame = _frame(spec)
    m = v_taper_metric(p, frame)
    assert m.status == STATUS_OK
    assert m.values["shoulder_width_px"] == pytest.approx(2 * spec.deltoid_half, rel=0.04)
    assert m.values["waist_width_px"] == pytest.approx(2 * spec.waist_half, rel=0.03)
    assert m.values["ratio"] == pytest.approx(spec.deltoid_half / spec.waist_half, rel=0.05)
    assert 0.0 <= m.score <= 100.0


def test_wider_shoulders_score_higher():
    narrow = v_taper_metric(*_frame(replace(SynthSpec(), deltoid_half=150.0)))
    wide = v_taper_metric(*_frame(replace(SynthSpec(), deltoid_half=172.0)))
    assert wide.values["ratio"] > narrow.values["ratio"]
    assert wide.score > narrow.score


def test_waist_unavailable_when_arms_touch_torso():
    m = v_taper_metric(*_frame(replace(SynthSpec(), arm_gap=-40.0)))
    assert m.status == STATUS_UNAVAILABLE
    assert R_ARMS_TOUCHING in m.reasons


def test_waist_uses_rows_where_arms_leave_a_gap():
    # Arms touch near the elbow and wrist but clear the narrow waist.
    spec = replace(SynthSpec(), arm_gap=-12.0)
    m = v_taper_metric(*_frame(spec))
    assert m.values["waist_width_px"] == pytest.approx(2 * spec.waist_half, rel=0.03)


def test_shoulders_unavailable_when_arms_raised():
    m = v_taper_metric(*_frame(replace(SynthSpec(), arms_raised=True)))
    assert m.status == STATUS_UNAVAILABLE
    assert R_ARMS_RAISED in m.reasons


def test_symmetric_body_has_high_silhouette_iou():
    m = symmetry_metric(*_frame(SynthSpec()), arms=None)
    assert m.status == STATUS_OK
    assert m.values["silhouette_iou"] > 0.95
    assert abs(m.values["shoulder_tilt_deg"]) < 0.5


def test_uneven_arms_lower_symmetry():
    even = symmetry_metric(*_frame(SynthSpec()), arms=None)
    uneven = symmetry_metric(*_frame(replace(SynthSpec(), upper_arm_half=(30.0, 44.0))), arms=None)
    assert uneven.values["silhouette_iou"] < even.values["silhouette_iou"]
    assert uneven.score < even.score


def test_shoulder_tilt_is_reported():
    spec = replace(SynthSpec(), shoulder_drop=(0.0, 14.0))
    m = symmetry_metric(*_frame(spec), arms=None)
    expected = math.degrees(math.atan2(14.0, 2 * (spec.deltoid_half - 50.0)))
    assert abs(m.values["shoulder_tilt_deg"]) == pytest.approx(expected, abs=0.3)


def test_gap_rows_at_the_waist_never_become_the_measurement():
    # A sharp notch whose centre rows have no torso run: smoothing fills those
    # rows with the notch's narrow neighbours, so they hold the smallest
    # smoothed width. Picking one of them used to crash on the missing edges.
    spec = SynthSpec()
    p = make_perception(spec)
    waist_row = int(spec.shoulder_y + spec.waist_t * spec.torso_len)
    cx = int(spec.cx)
    mask = p.person_mask.copy()
    for d in range(-3, 4):
        y = waist_row + d
        half = int(spec.waist_half)
        cut = 4 * (3 - abs(d))
        mask[y, cx + half - cut + 1: cx + half + 2] = False
        mask[y, cx - half - 1: cx - half + cut] = False
    mask[waist_row - 1: waist_row + 2, :] = False
    p = replace(p, person_mask=mask)
    frame, _ = build_body_frame(p.landmarks)
    m = v_taper_metric(p, frame)
    assert m.values["waist_width_px"] == pytest.approx(2 * spec.waist_half, rel=0.05)
    (_, y), _ = m.geometry["waist_line"]
    assert not (waist_row - 1 <= y <= waist_row + 1)
