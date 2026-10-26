import json
from dataclasses import replace

import cv2
import pytest
from fastapi.testclient import TestClient

from boji_detector.abdomen import line_definition_index
from boji_detector.analyze import analyze_perception
from boji_detector.body import build_body_frame
from boji_detector.hips import R_HIPS_OUT_OF_FRAME, hand_boxes, measure_hips, waist_hip_metric
from boji_detector.scoring import DEFAULT_CONFIG, FEMALE_CONFIG, load_config, profile_config
from boji_detector.server import create_app
from boji_detector.types import STATUS_UNAVAILABLE, Landmark
from synth import SynthSpec, make_perception


def _frame(spec: SynthSpec):
    p = make_perception(spec)
    frame, _ = build_body_frame(p.landmarks)
    assert frame is not None
    return p, frame


def test_waist_hip_ratio_matches_drawn_widths():
    spec = SynthSpec()
    m = waist_hip_metric(*_frame(spec), FEMALE_CONFIG)
    assert m.values["waist_width_px"] == pytest.approx(2 * spec.waist_half, rel=0.04)
    assert m.values["hip_width_px"] == pytest.approx(2 * spec.hip_half, rel=0.04)
    assert m.values["ratio"] == pytest.approx(spec.waist_half / spec.hip_half, rel=0.06)
    assert "hip_line" in m.geometry and "waist_line" in m.geometry


def test_curvier_hips_score_higher():
    straight = waist_hip_metric(*_frame(replace(SynthSpec(), hip_half=105.0)), FEMALE_CONFIG)
    curvy = waist_hip_metric(*_frame(replace(SynthSpec(), hip_half=135.0)), FEMALE_CONFIG)
    assert curvy.values["ratio"] < straight.values["ratio"]
    assert curvy.score > straight.score


def test_hips_out_of_photo_are_unavailable():
    spec = replace(SynthSpec(), height=600)
    m = waist_hip_metric(*_frame(spec), FEMALE_CONFIG)
    assert m.status == STATUS_UNAVAILABLE


def test_hand_boxes_cover_fingertips_and_block_hip_rows():
    p, frame = _frame(SynthSpec())
    hip_y = frame.hip_mid[1]
    x_edge = frame.mid_x(hip_y) + SynthSpec().hip_half - 5
    marks = dict(p.landmarks)
    marks.update({
        "right_wrist": Landmark(x_edge, hip_y - 40, 0.0, 0.9),
        "right_index": Landmark(x_edge, hip_y - 10, 0.0, 0.9),
        "right_pinky": Landmark(x_edge + 10, hip_y - 12, 0.0, 0.9),
    })
    boxes = hand_boxes(replace(p, landmarks=marks))
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert y1 > hip_y + 10           # fingertips reach past the knuckles
    free, _ = measure_hips(p, frame)
    assert free is not None
    blocked, reasons = measure_hips(replace(p, landmarks=marks), frame)
    # Rows the hand covers are skipped: the hip width comes from other rows, or nothing is left.
    if blocked is None:
        assert reasons
    else:
        assert not (y0 <= blocked["row"] <= y1)


def test_line_index_weights_alba_and_both_edges():
    assert line_definition_index(0.0, {"image_left": 0.0, "image_right": 0.0}) == 0.0
    full = line_definition_index(0.02, {"image_left": 0.02, "image_right": 0.02})
    assert full == pytest.approx(0.02)
    one_side = line_definition_index(0.02, {"image_left": 0.02, "image_right": 0.0})
    assert one_side < full
    assert line_definition_index(-1.0, {"image_left": -1.0}) == 0.0


def test_profiles_and_config_overrides():
    assert profile_config(None) is DEFAULT_CONFIG
    assert profile_config("male") is DEFAULT_CONFIG
    assert profile_config("female") is FEMALE_CONFIG
    with pytest.raises(ValueError):
        profile_config("other")
    assert "waist_hip" in FEMALE_CONFIG.weights and "arms" not in FEMALE_CONFIG.weights


def test_load_config_keeps_the_base_profile(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"waist_hip": {"mid": -0.7, "scale": 0.02}}), encoding="utf-8")
    cfg = load_config(path, FEMALE_CONFIG)
    assert cfg.profile == "female" and cfg.waist_hip.mid == -0.7
    path.write_text(json.dumps({"profile": "female"}), encoding="utf-8")
    assert load_config(path).profile == "female"


def test_male_report_is_unchanged_and_female_adds_waist_hip():
    p = make_perception(SynthSpec())
    male, _ = analyze_perception(p)
    female, _ = analyze_perception(p, FEMALE_CONFIG)
    assert male.profile == "male" and "waist_hip" not in male.metrics
    assert female.profile == "female" and "waist_hip" in female.metrics
    assert male.metrics["v_taper"].score != female.metrics["v_taper"].score
    assert "line_index" in female.metrics["abs"].values
    assert set(female.total["weights_used"]) <= set(FEMALE_CONFIG.weights)
    json.dumps(female.to_dict())


class FakePerceiver:
    def perceive(self, bgr):
        return make_perception(SynthSpec())

    def close(self):
        pass


def _png() -> bytes:
    ok, buf = cv2.imencode(".png", make_perception(SynthSpec()).image)
    assert ok
    return buf.tobytes()


def test_server_scores_the_requested_profile(monkeypatch):
    monkeypatch.delenv("BOJI_SERVICE_TOKEN", raising=False)
    with TestClient(create_app(perceiver_factory=FakePerceiver)) as client:
        male = client.post("/v1/analyze", content=_png()).json()["report"]
        female = client.post("/v1/analyze?sex=female", content=_png()).json()["report"]
        bad = client.post("/v1/analyze?sex=x", content=_png())
    assert male["profile"] == "male" and "waist_hip" not in male["metrics"]
    assert female["profile"] == "female" and "waist_hip" in female["metrics"]
    assert bad.status_code == 400


def test_hips_out_of_frame_reason_has_text():
    from boji_detector.messages import MESSAGES

    for code in (R_HIPS_OUT_OF_FRAME, "hips_blocked_by_arms", "hips_at_image_edge"):
        assert MESSAGES[code][0] and MESSAGES[code][1]


def test_raised_arm_symmetry_compares_the_torso_only():
    from boji_detector.symmetry import symmetry_metric

    raised = SynthSpec(arms_raised=True)
    p, frame = _frame(raised)
    m = symmetry_metric(p, frame, None)
    assert m.values["torso_only"] is True
    assert m.values["silhouette_iou"] > 0.9          # the synthetic torso is symmetric
    assert m.score > 80
    down = symmetry_metric(*_frame(SynthSpec()), None)
    assert down.values["torso_only"] is False


def test_forearm_merging_into_the_hips_is_skipped():
    spec = SynthSpec()
    p, frame = _frame(spec)
    clean, _ = measure_hips(p, frame)
    # Paint a forearm-sized blob merging into the image-left hip, below the wrist line.
    mask = p.person_mask.copy()
    hip_y = int(frame.hip_mid[1])
    x_left = int(frame.mid_x(hip_y) - spec.hip_half)
    mask[hip_y + 5: hip_y + 40, x_left - 45: x_left + 2] = True
    merged, _ = measure_hips(replace(p, person_mask=mask), frame)
    assert clean is not None and merged is not None
    assert merged["width"] == pytest.approx(clean["width"], rel=0.05)
