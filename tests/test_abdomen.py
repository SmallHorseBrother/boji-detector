from dataclasses import replace

import numpy as np
import pytest

from boji_detector.abdomen import R_ABDOMEN_COVERED, abs_metric
from boji_detector.body import build_body_frame
from boji_detector.proportion import v_taper_metric
from boji_detector.types import STATUS_OK, STATUS_UNAVAILABLE
from synth import SynthSpec, abs_boundaries, make_perception


def _abs(spec: SynthSpec):
    p = make_perception(spec)
    frame, _ = build_body_frame(p.landmarks)
    waist = v_taper_metric(p, frame).values.get("waist_width_px")
    return abs_metric(p, frame, waist_px=waist)


@pytest.mark.parametrize("rows,packs", [(2, 4), (3, 6), (4, 8)])
def test_counts_blocks(rows, packs):
    m = _abs(replace(SynthSpec(), abs_rows=rows))
    assert m.status == STATUS_OK
    assert m.values["pack_count"] == packs
    assert m.values["blocks_image_left"] == m.values["blocks_image_right"] == rows


def test_linea_alba_found_on_midline():
    spec = SynthSpec()
    m = _abs(spec)
    assert m.values["alba_found"] is True
    (x_top, _), (x_bottom, _) = m.geometry["linea_alba"]
    assert x_top == pytest.approx(spec.cx, abs=3)
    assert x_bottom == pytest.approx(spec.cx, abs=3)


def test_grooves_sit_on_drawn_rows():
    spec = SynthSpec()
    m = _abs(spec)
    expected = abs_boundaries(spec)
    for side in ("image_left", "image_right"):
        rows = sorted(np.mean([pt[1] for pt in line]) for line in m.geometry["grooves"][side])
        for want in expected:
            assert min(abs(r - want) for r in rows) < 5, (side, rows, expected)


def test_clear_six_pack_is_well_enclosed():
    m = _abs(SynthSpec())
    assert m.values["mean_enclosure"] > 0.8
    assert m.values["open_angle_deg"] < 72


def test_smooth_belly_has_no_blocks_and_low_score():
    m = _abs(replace(SynthSpec(), abs_rows=0))
    assert m.status == STATUS_OK
    assert m.values["pack_count"] == 0
    assert m.score < 20


def test_deeper_grooves_score_higher():
    faint = _abs(replace(SynthSpec(), groove_depth=0.08))
    clear = _abs(replace(SynthSpec(), groove_depth=0.22))
    assert clear.values["definition_index"] > faint.values["definition_index"]
    assert clear.score > faint.score


def test_exposure_change_barely_moves_the_index():
    base = _abs(SynthSpec())
    dark = _abs(replace(SynthSpec(), exposure=0.6))
    assert dark.values["pack_count"] == base.values["pack_count"]
    assert dark.values["definition_index"] == pytest.approx(base.values["definition_index"], rel=0.08)


def test_side_light_keeps_the_blocks():
    m = _abs(replace(SynthSpec(), light_gradient=0.6))
    assert m.values["pack_count"] == 6


def test_covered_abdomen_is_unavailable():
    m = _abs(replace(SynthSpec(), shirt=True))
    assert m.status == STATUS_UNAVAILABLE
    assert R_ABDOMEN_COVERED in m.reasons
