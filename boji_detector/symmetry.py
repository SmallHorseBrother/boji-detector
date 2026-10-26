"""Left-right symmetry: mirrored torso overlap, shoulder tilt, arm balance.

The overlap compares the torso only (armpits to waist): how the arms hang, a
hand in a pocket or a phone held up for a mirror selfie is pose, not body
asymmetry. Arm size differences count separately through the arm balance.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .body import R_HIPS_ESTIMATED, BodyFrame
from .proportion import (
    R_ARMS_TOUCHING,
    R_BODY_ROTATED,
    R_SHOULDERS_NOT_FOUND,
    arms_down,
    rotation_reasons,
    torso_row,
)
from .scoring import DEFAULT_CONFIG, ScoringConfig, logistic_score
from .silhouette import run_at
from .types import Metric, Perception, status_for, unavailable

R_BODY_CUT_OFF = "body_cut_off"

LOW_CONFIDENCE = frozenset({R_BODY_ROTATED, R_HIPS_ESTIMATED, R_BODY_CUT_OFF, R_ARMS_TOUCHING})

SYMMETRY_BAND = (0.15, 0.85)    # armpits to waist, fraction of the shoulder->hip axis
AXIS_SEARCH = 0.04              # axis shift search, fraction of shoulder-joint distance
SYMMETRY_MAX_WIDTH = 640
MIN_KEPT_ROWS = 0.5             # below this share of whole rows the overlap is low confidence
MIN_ROWS = 10


def _mirror_iou(mask: np.ndarray, axes: np.ndarray) -> float:
    w = mask.shape[1]
    xs = np.arange(w)[None, :]
    idx = np.rint(2.0 * axes[:, None] - xs).astype(int)
    inside = (idx >= 0) & (idx < w)
    mirrored = np.zeros_like(mask)
    rows = np.broadcast_to(np.arange(mask.shape[0])[:, None], idx.shape)
    mirrored[inside] = mask[rows[inside], idx[inside]]
    union = np.logical_or(mask, mirrored).sum()
    return float(np.logical_and(mask, mirrored).sum() / union) if union else 0.0


def _rows_touching_photo_edge(p: Perception, y0: int, y1: int) -> np.ndarray:
    """Rows where the silhouette reaches the image border or a non-photo pixel."""
    frame = p.frame_mask if p.frame_mask is not None else np.ones((p.height, p.width), bool)
    outside = np.pad(~frame, 1, constant_values=True).astype(np.uint8)
    near_outside = cv2.dilate(outside, np.ones((3, 3), np.uint8))[1:-1, 1:-1].astype(bool)
    return (p.person_mask & near_outside)[y0: y1 + 1].any(axis=1)


def _torso_runs(p: Perception, frame: BodyFrame, rows: np.ndarray) -> np.ndarray:
    """Mask rows keeping only the silhouette run that contains the body midline."""
    out = np.zeros((rows.size, p.width), dtype=bool)
    for i, y in enumerate(rows):
        span = run_at(p.person_mask, int(y), int(frame.mid_x(y)))
        if span is not None:
            out[i, span[0]: span[1] + 1] = True
    return out


def _arm_merged(p: Perception, frame: BodyFrame, y: int) -> bool:
    """An arm resting against the side merges into the torso run on that row."""
    row = torso_row(p, frame, y)
    return row is not None and not row.arms_clear


def silhouette_symmetry(p: Perception, frame: BodyFrame) -> dict[str, float] | None:
    y0 = max(0, int(frame.y_at(SYMMETRY_BAND[0])))
    y1 = min(p.height - 1, int(frame.y_at(SYMMETRY_BAND[1])))
    if y1 - y0 < MIN_ROWS:
        return None
    all_rows = np.arange(y0, y1 + 1)
    in_photo = ~_rows_touching_photo_edge(p, y0, y1)
    keep = in_photo & ~np.array([_arm_merged(p, frame, int(y)) for y in all_rows])
    arms_merged = keep.sum() < MIN_ROWS
    if arms_merged:
        # Arms rest against the sides all the way down: compare the run with
        # them rather than give no symmetry at all, and say so.
        keep = in_photo
    if keep.sum() < MIN_ROWS:
        return None
    rows = all_rows[keep]
    band = _torso_runs(p, frame, rows)
    step = int(math.ceil(max(1.0, band.shape[1] / SYMMETRY_MAX_WIDTH)))
    small = band[::step, ::step]
    axes = np.array([frame.mid_x(y) for y in rows[::step]]) / step
    reach = max(1, int(AXIS_SEARCH * frame.shoulder_joint_dist / step))
    best = max(
        ((_mirror_iou(small, axes + dx), dx * step) for dx in range(-reach, reach + 1)),
        key=lambda item: item[0],
    )
    return {
        "iou": best[0],
        "axis_offset_px": float(best[1]),
        "rows_used": float(in_photo.mean()),
        "arms_merged": arms_merged,
    }


def symmetry_metric(
    p: Perception,
    frame: BodyFrame,
    arms: Metric | None,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> Metric:
    key = "symmetry"
    # Raised or bent arms move the shoulders and change arm widths on purpose:
    # shoulder tilt and arm balance only count when both arms hang down.
    relaxed = arms_down(frame) is not False
    sil = silhouette_symmetry(p, frame)
    if sil is None:
        return unavailable(key, R_SHOULDERS_NOT_FOUND)
    left, right = frame.image_left.shoulder, frame.image_right.shoulder
    tilt = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    values: dict[str, Any] = {
        "silhouette_iou": sil["iou"],
        "axis_offset_px": sil["axis_offset_px"],
        "rows_used": sil["rows_used"],
        "shoulder_tilt_deg": tilt,
        "rotation_index": frame.rotation_index,
        "arms_down": relaxed,
    }
    reasons = rotation_reasons(frame)
    parts = [logistic_score(sil["iou"], config.symmetry_iou)]
    if frame.hips_estimated:
        # Without hips there is no torso axis to deskew against, so a rolled
        # camera would read as uneven shoulders; leave tilt out of the score.
        reasons.append(R_HIPS_ESTIMATED)
    elif relaxed:
        parts.append(logistic_score(-abs(tilt), config.symmetry_tilt))
    if sil["rows_used"] < MIN_KEPT_ROWS:
        reasons.append(R_BODY_CUT_OFF)
    if sil["arms_merged"]:
        reasons.append(R_ARMS_TOUCHING)
    if relaxed and arms is not None and arms.available and arms.values.get("balance") is not None:
        values["arm_balance"] = arms.values["balance"]
        parts.append(logistic_score(arms.values["balance"], config.symmetry_balance))
    all_reasons = tuple(reasons)
    return Metric(
        key=key,
        status=status_for(all_reasons, LOW_CONFIDENCE),
        score=float(np.mean(parts)),
        values=values,
        reasons=all_reasons,
        geometry={"shoulder_points": [left.tolist(), right.tolist()]},
    )
