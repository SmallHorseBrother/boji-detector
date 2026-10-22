"""Photos that end before the body does: cut-offs, tilted crops, hands on thighs."""

from dataclasses import replace

import numpy as np
import pytest

from boji_detector.analyze import analyze_perception
from boji_detector.arms import R_ARM_CUT_OFF
from boji_detector.body import build_body_frame
from boji_detector.proportion import (
    R_ARMS_TOUCHING,
    R_SHOULDERS_CUT_OFF,
    R_WAIST_AT_EDGE,
    R_WAIST_OUT_OF_FRAME,
    v_taper_metric,
)
from boji_detector.silhouette import clean_mask
from boji_detector.types import STATUS_OK, STATUS_UNAVAILABLE, Landmark, Perception
from boji_detector.upright import rotate_perception
from synth import SynthSpec, make_perception


def _crop(p: Perception, rows: slice = slice(None), cols: slice = slice(None)) -> Perception:
    """A photo framed tighter: every array cut, landmarks shifted, all pixels real."""
    x0 = cols.start or 0
    y0 = rows.start or 0
    landmarks = {k: Landmark(v.x - x0, v.y - y0, v.z, v.visibility) for k, v in p.landmarks.items()}

    def cut(a):
        return None if a is None else np.ascontiguousarray(a[rows, cols])

    return Perception(image=cut(p.image), landmarks=landmarks, person_mask=cut(p.person_mask),
                      skin_mask=cut(p.skin_mask), clothes_mask=cut(p.clothes_mask))


def _axis_row(p: Perception, t: float) -> int:
    frame, _ = build_body_frame(p.landmarks)
    return int(frame.shoulder_mid[1] + t * (frame.hip_mid[1] - frame.shoulder_mid[1]))


@pytest.mark.parametrize("tilt", [5.0, 10.0])
@pytest.mark.parametrize("cut_t", [0.75, 0.85, 0.95])
def test_tilted_photo_cut_below_the_navel_keeps_the_true_waist(tilt, cut_t):
    straight, _ = analyze_perception(make_perception(SynthSpec()))
    truth = straight.metrics["v_taper"].values
    tilted = rotate_perception(make_perception(SynthSpec()), tilt)
    photo = _crop(tilted, rows=slice(0, _axis_row(tilted, cut_t)))
    report, _ = analyze_perception(photo)
    v = report.metrics["v_taper"]
    if v.available:
        assert v.values["waist_width_px"] == pytest.approx(truth["waist_width_px"], rel=0.05)
        assert v.values["ratio"] == pytest.approx(truth["ratio"], rel=0.05)
        assert report.metrics["abs"].values["waist_px_used"] == pytest.approx(truth["waist_width_px"], rel=0.05)
    else:
        assert R_WAIST_OUT_OF_FRAME in v.reasons


@pytest.mark.parametrize("cols", [slice(170, 470), slice(170, None)])
def test_shoulders_cut_by_the_photo_edge_are_not_measured(cols):
    report, _ = analyze_perception(_crop(make_perception(SynthSpec()), cols=cols))
    v = report.metrics["v_taper"]
    assert v.status == STATUS_UNAVAILABLE
    assert R_SHOULDERS_CUT_OFF in v.reasons


def test_arm_cut_by_the_photo_edge_is_not_measured():
    # Elbows stay inside the photo (offset ~160 px) but the arms' outer edges do not.
    report, _ = analyze_perception(_crop(make_perception(SynthSpec()), cols=slice(155, 485)))
    arms = report.metrics["arms"]
    assert arms.status == STATUS_UNAVAILABLE
    assert R_ARM_CUT_OFF in arms.reasons


def test_photo_cut_above_the_waist_says_waist_out_of_frame():
    p = make_perception(SynthSpec())
    m = v_taper_metric(*_frame(_crop(p, rows=slice(0, _axis_row(p, 0.6)))))
    assert m.status == STATUS_UNAVAILABLE
    assert m.reasons == (R_WAIST_OUT_OF_FRAME,)


def test_waist_still_narrowing_at_the_photo_edge_is_flagged():
    # Flat from the waist down: the photo cannot show whether it narrows further.
    spec = replace(SynthSpec(), waist_t=0.6, hip_half=95.0)
    p = make_perception(spec)
    m = v_taper_metric(*_frame(_crop(p, rows=slice(0, _axis_row(p, 0.8)))))
    assert m.available
    assert R_WAIST_AT_EDGE in m.reasons


def test_hands_touching_thighs_keep_the_arm_gap_open():
    spec = SynthSpec()
    p = make_perception(spec)
    mask = p.person_mask.copy()
    wrist_rows = slice(int(p.landmarks["left_wrist"].y) - 6, int(p.landmarks["left_wrist"].y) + 6)
    mask[wrist_rows, :] |= np.any(mask[wrist_rows], axis=0)[None, :]  # bridge every column the hands span
    mask[wrist_rows, int(spec.cx - 160): int(spec.cx + 160)] = True
    cleaned = replace(p, person_mask=clean_mask(mask))
    m = v_taper_metric(*_frame(cleaned))
    assert m.status == STATUS_OK, m.reasons
    assert R_ARMS_TOUCHING not in m.reasons
    assert m.values["waist_width_px"] == pytest.approx(2 * spec.waist_half, rel=0.03)


def _frame(p: Perception):
    frame, _ = build_body_frame(p.landmarks)
    return p, frame
