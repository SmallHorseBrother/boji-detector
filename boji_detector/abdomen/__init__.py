"""Abdominal definition: linea alba, tendinous intersections and per-block separation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..body import R_HIPS_ESTIMATED, BodyFrame
from ..proportion import R_BODY_ROTATED, WAIST_BAND, rotation_reasons, torso_row
from ..scoring import DEFAULT_CONFIG, ScoringConfig, logistic_score
from ..types import Metric, Perception, status_for, unavailable
from .blocks import Block, definition_index, side_blocks
from .grooves import SIDES, Alba, Groove, find_alba, find_grooves, find_lateral
from .workspace import WORK_WAIST, Workspace, build_workspace

R_ABDOMEN_COVERED = "abdomen_covered"
R_ABDOMEN_OUT_OF_FRAME = "abdomen_out_of_frame"
R_WAIST_ESTIMATED = "waist_estimated"
R_LOW_RESOLUTION = "low_resolution"
R_OVEREXPOSED = "overexposed"
R_UNDEREXPOSED = "underexposed"
R_SIDE_LIGHT = "strong_side_light"
R_ALBA_NOT_FOUND = "linea_alba_not_found"
R_NOT_MEASURABLE = "abdomen_not_measurable"

LOW_CONFIDENCE = frozenset({
    R_WAIST_ESTIMATED, R_LOW_RESOLUTION, R_OVEREXPOSED, R_UNDEREXPOSED,
    R_SIDE_LIGHT, R_HIPS_ESTIMATED, R_BODY_ROTATED,
})

MIN_WAIST_PX = 120.0
CORE_T = (0.35, 0.80)
CORE_HALF = 0.30
MIN_CORE_SKIN = 0.6
MIN_CORE_IN_PHOTO = 0.8
FALLBACK_WAIST_PER_SHOULDER = 0.85  # waist width / shoulder-joint distance, typical adult
OVEREXPOSED_SHARE = 0.08
UNDEREXPOSED_MEDIAN = 40
SIDE_LIGHT_LOG = 0.35

__all__ = ["abs_metric", "R_ABDOMEN_COVERED", "R_ABDOMEN_OUT_OF_FRAME"]


def _estimate_waist(p: Perception, frame: BodyFrame) -> float:
    """Narrowest complete torso row, else a typical share of the shoulder span."""
    rows = range(int(frame.y_at(WAIST_BAND[0])), int(frame.y_at(WAIST_BAND[1])) + 1, 2)
    widths = [r.width for y in rows if (r := torso_row(p, frame, y)) is not None and r.usable]
    if len(widths) >= 4:
        return float(np.percentile(widths, 10))
    return FALLBACK_WAIST_PER_SHOULDER * frame.shoulder_joint_dist


def _core_shares(ws: Workspace) -> tuple[float, float]:
    """(share of the central abdomen inside the photo, share of that which is skin)."""
    h, w = ws.shape
    want0, want1 = ws.y_work(CORE_T[0]), ws.y_work(CORE_T[1])
    r0, r1 = max(0, int(want0)), min(h - 1, int(want1))
    if r1 <= r0:
        return 0.0, 0.0
    rows = np.arange(r0, r1 + 1)
    xs = np.arange(w)[None, :]
    core = np.abs(xs - ws.mid_x(rows)[:, None]) <= CORE_HALF * ws.waist
    expected = core.sum() * (want1 - want0 + 1) / max(1, rows.size)
    in_photo = core & ws.photo[r0:r1 + 1]
    photo_share = float(in_photo.sum() / expected) if expected else 0.0
    skin_share = float(ws.surface[r0:r1 + 1][in_photo].mean()) if in_photo.any() else 0.0
    return min(1.0, photo_share), skin_share


def _lighting(ws: Workspace, alba: Alba) -> tuple[dict[str, float], list[str]]:
    valid = ws.valid
    if not valid.any():
        return {}, []
    grey = ws.gray[valid]
    rows, cols = np.nonzero(valid)
    left = cols < alba.x(rows)
    side_light = 0.0
    if left.any() and (~left).any():
        logs = ws.log_l[valid]
        side_light = float(logs[~left].mean() - logs[left].mean())
    relief = ws.relief[valid]
    contrast = float(1.4826 * np.median(np.abs(relief - np.median(relief))))
    reasons = []
    if float(np.mean(grey >= 250)) > OVEREXPOSED_SHARE:
        reasons.append(R_OVEREXPOSED)
    if float(np.median(grey)) < UNDEREXPOSED_MEDIAN:
        reasons.append(R_UNDEREXPOSED)
    if abs(side_light) > SIDE_LIGHT_LOG:
        reasons.append(R_SIDE_LIGHT)
    return {"side_light": side_light, "texture_contrast": contrast}, reasons


def _geometry(ws: Workspace, alba: Alba, grooves: dict[str, list[Groove]],
              laterals: dict[str, float], blocks: list[Block]) -> dict[str, Any]:
    h, w = ws.shape
    roi = ws.to_image(0, ws.row_range[0]) + ws.to_image(w - 1, ws.row_range[1])
    rows = [g.row for gs in grooves.values() for g in gs]
    # Draw the vertical lines over the detected grooves when there are any.
    r0, r1 = (min(rows), max(rows)) if len(rows) >= 2 else ws.row_range
    return {
        "roi": roi,
        "linea_alba": [ws.to_image(float(alba.x(r0)), r0), ws.to_image(float(alba.x(r1)), r1)],
        "grooves": {
            side: [[ws.to_image(x, y) for x, y in g.points] for g in gs]
            for side, gs in grooves.items()
        },
        "lateral_lines": {
            side: [ws.to_image(float(alba.x(r0) + SIDES[side] * laterals[side]), r0),
                   ws.to_image(float(alba.x(r1) + SIDES[side] * laterals[side]), r1)]
            for side in laterals
        },
        "blocks": [
            {
                "side": b.side,
                "index": b.index,
                "outline": [
                    ws.to_image(float(alba.x(b.top)), b.top),
                    ws.to_image(float(alba.x(b.top) + SIDES[b.side] * b.lateral), b.top),
                    ws.to_image(float(alba.x(b.bottom) + SIDES[b.side] * b.lateral), b.bottom),
                    ws.to_image(float(alba.x(b.bottom)), b.bottom),
                ],
                "enclosure": b.enclosure,
                "depth": b.depth,
            }
            for b in blocks
        ],
    }


def abs_metric(
    p: Perception,
    frame: BodyFrame,
    waist_px: float | None,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> Metric:
    key = "abs"
    reasons: list[str] = []
    if waist_px is None:
        waist_px = _estimate_waist(p, frame)
        reasons.append(R_WAIST_ESTIMATED)
    if waist_px < 20:
        return unavailable(key, R_NOT_MEASURABLE)
    if waist_px < MIN_WAIST_PX:
        reasons.append(R_LOW_RESOLUTION)
    if frame.hips_estimated:
        reasons.append(R_HIPS_ESTIMATED)
    reasons += rotation_reasons(frame)

    ws = build_workspace(p, frame, waist_px)
    if ws is None:
        return unavailable(key, R_NOT_MEASURABLE)
    photo_share, skin_share = _core_shares(ws)
    if photo_share < MIN_CORE_IN_PHOTO:
        return unavailable(key, R_ABDOMEN_OUT_OF_FRAME)
    if skin_share < MIN_CORE_SKIN:
        return unavailable(key, R_ABDOMEN_COVERED)

    alba = find_alba(ws)
    if not alba.found:
        reasons.append(R_ALBA_NOT_FOUND)
    grooves, laterals, blocks = {}, {}, []
    counts = {}
    for side, sign in SIDES.items():
        side_grooves = find_grooves(ws, alba, sign)
        grooves[side] = side_grooves
        if len(side_grooves) >= 2:
            lateral, _ = find_lateral(ws, alba, sign, side_grooves[0].row, side_grooves[-1].row)
        else:
            lateral, _ = find_lateral(ws, alba, sign, *ws.row_range)
        laterals[side] = lateral
        found = side_blocks(ws, alba, side, sign, side_grooves, lateral)
        counts[side] = len(found)
        blocks += found

    light_values, light_reasons = _lighting(ws, alba)
    reasons += light_reasons
    strengths = [g.strength for gs in grooves.values() for g in gs] + [alba.strength]
    per_side = float(np.mean(list(counts.values())))
    index = definition_index(blocks, max(strengths), per_side)
    enclosure = float(np.mean([b.enclosure for b in blocks])) if blocks else 0.0
    values: dict[str, Any] = {
        "definition_index": index,
        "pack_count": int(sum(counts.values())),
        "blocks_image_left": counts["image_left"],
        "blocks_image_right": counts["image_right"],
        "mean_enclosure": enclosure,
        "open_angle_deg": 360.0 * (1.0 - enclosure),
        "mean_groove_depth": float(np.mean([b.depth for b in blocks])) if blocks else 0.0,
        "mean_relief": float(np.mean([b.relief for b in blocks])) if blocks else 0.0,
        "alba_found": bool(alba.found),
        "alba_strength": alba.strength,
        "waist_px_used": float(waist_px),
        "work_scale": WORK_WAIST / float(waist_px),
        **light_values,
    }
    all_reasons = tuple(dict.fromkeys(reasons))
    return Metric(
        key=key,
        status=status_for(all_reasons, LOW_CONFIDENCE),
        score=logistic_score(index, config.abs),
        values=values,
        reasons=all_reasons,
        geometry=_geometry(ws, alba, grooves, laterals, blocks),
    )
