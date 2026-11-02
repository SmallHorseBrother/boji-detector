"""Web-photo fixes: back views, covered torsos and impossible waist/hip ratios."""
from dataclasses import replace

import numpy as np

from boji_detector.analyze import analyze_perception
from boji_detector.body import R_BACK_VIEW, build_body_frame
from boji_detector.hips import R_WAIST_HIP_IMPLAUSIBLE, waist_hip_metric
from boji_detector.scoring import DEFAULT_CONFIG, FEMALE_CONFIG, logistic_score, total_score
from boji_detector.types import STATUS_OK, STATUS_UNAVAILABLE, Landmark, Metric
from synth import SynthSpec, make_perception


def _seen_from_behind(p):
    """Same silhouette, landmarks as MediaPipe reports them for a person facing away."""
    swapped = {}
    for name, lm in p.landmarks.items():
        if name.startswith("left_"):
            twin = p.landmarks["right_" + name[5:]]
        elif name.startswith("right_"):
            twin = p.landmarks["left_" + name[6:]]
        else:
            twin = lm
        swapped[name] = Landmark(twin.x, twin.y, twin.z, twin.visibility)
    nose = swapped["nose"]
    swapped["nose"] = Landmark(nose.x, nose.y, 200.0, nose.visibility)
    return replace(p, landmarks=swapped)


def test_front_view_is_not_a_back_view():
    p = make_perception(SynthSpec())
    frame, _ = build_body_frame(p.landmarks)
    assert frame.back_view is False


def test_mirrored_left_right_alone_is_not_a_back_view():
    # A mirror selfie flips the sides but the nose stays in front of the shoulders.
    p = _seen_from_behind(make_perception(SynthSpec()))
    nose = p.landmarks["nose"]
    p = replace(p, landmarks={**p.landmarks, "nose": Landmark(nose.x, nose.y, -200.0, nose.visibility)})
    frame, _ = build_body_frame(p.landmarks)
    assert frame.back_view is False


def test_back_view_has_no_abs_for_either_profile():
    p = _seen_from_behind(make_perception(SynthSpec()))
    for config in (DEFAULT_CONFIG, FEMALE_CONFIG):
        report, _ = analyze_perception(p, config)
        assert report.metrics["abs"].status == STATUS_UNAVAILABLE
        assert R_BACK_VIEW in report.metrics["abs"].reasons
        assert R_BACK_VIEW in report.quality


def test_female_total_needs_the_abs():
    p = _seen_from_behind(make_perception(SynthSpec()))
    report, _ = analyze_perception(p, FEMALE_CONFIG)
    assert report.total["score"] is None
    assert "abs" in report.total["missing"]


def test_male_total_without_abs_is_unchanged():
    metrics = {
        "abs": Metric("abs", STATUS_UNAVAILABLE),
        "v_taper": Metric("v_taper", STATUS_OK, score=80.0),
        "arms": Metric("arms", STATUS_OK, score=80.0),
        "symmetry": Metric("symmetry", STATUS_OK, score=80.0),
    }
    assert total_score(metrics)["score"] is not None


def test_covered_abdomen_also_drops_the_waist_hip_ratio():
    # Loose clothes make a silhouette, not a waist.
    p = make_perception(SynthSpec())
    bare = replace(p, skin_mask=np.zeros_like(p.person_mask), clothes_mask=p.person_mask.copy())
    report, _ = analyze_perception(bare, FEMALE_CONFIG)
    assert report.metrics["abs"].status == STATUS_UNAVAILABLE
    assert report.metrics["waist_hip"].status == STATUS_UNAVAILABLE
    assert report.total["score"] is None


def test_impossible_ratio_is_unavailable():
    # Hips far wider than a waist can be: another person or the background in the silhouette.
    p = make_perception(replace(SynthSpec(), hip_half=240.0))
    frame, _ = build_body_frame(p.landmarks)
    m = waist_hip_metric(p, frame, FEMALE_CONFIG)
    assert m.status == STATUS_UNAVAILABLE
    assert R_WAIST_HIP_IMPLAUSIBLE in m.reasons


def test_waist_hip_curve_does_not_zero_a_lean_athlete():
    # Lean athletes measure 0.80-0.90 from the front; web photos of them scored 0-7.
    assert logistic_score(-0.87, FEMALE_CONFIG.waist_hip) > 15
    assert logistic_score(-0.82, FEMALE_CONFIG.waist_hip) > 30
    assert logistic_score(-0.68, FEMALE_CONFIG.waist_hip) > 75
