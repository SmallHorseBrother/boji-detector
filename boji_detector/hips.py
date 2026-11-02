"""Waist/hip ratio: narrowest waist width / widest hip width, from the silhouette.

Measured on the female profile. The hip band sits around the hip joints, where
the pelvis and upper thighs are widest in a front photo. Rows where a hanging
arm or hand merges into the hip silhouette are skipped, like the waist rows.
"""

from __future__ import annotations

import numpy as np

from .body import R_HIPS_ESTIMATED, BodyFrame
from .proportion import LOW_CONFIDENCE as WAIST_LOW_CONFIDENCE
from .proportion import R_WAIST_OUT_OF_FRAME, measure_waist, rotation_reasons, torso_row
from .scoring import DEFAULT_CONFIG, ScoringConfig, logistic_score
from .silhouette import median_smooth
from .types import Metric, Perception, status_for, unavailable

R_HIPS_OUT_OF_FRAME = "hips_out_of_frame"
R_HIPS_BLOCKED = "hips_blocked_by_arms"
R_HIPS_AT_EDGE = "hips_at_image_edge"
R_WAIST_HIP_IMPLAUSIBLE = "waist_hip_implausible"

LOW_CONFIDENCE = WAIST_LOW_CONFIDENCE | {R_HIPS_AT_EDGE}

HIP_BAND = (0.90, 1.10)          # fraction of the shoulder->hip axis (hip joints at 1.0); lower rows reach the thigh gap
MIN_HIP_IN_PHOTO = 0.6           # share of the hip band that must be inside the photo
MIN_VALID_HIP_ROWS = 6
HAND_POINTS = ("wrist", "pinky", "index", "thumb")
MIN_HAND_VISIBILITY = 0.3
FINGER_REACH = 1.0               # fingertips lie about one wrist->knuckle length past the knuckles
HAND_PAD = 0.15                  # extra margin, as a share of the hand box size
# Half the width of a forearm, as a share of the shoulder-joint distance: a run
# edge this close to the arm centre line is the arm, not the hip.
ARM_HALF_WIDTH = 0.20
# Hips widen gradually row by row; an edge that jumps outward by more than this
# share of the shoulder-joint distance from the last kept row is something else
# (a forearm or hand) merging into the silhouette.
EDGE_JUMP = 0.06
# Below this no real waist/hip width ratio exists: the web photos under it had a
# side view, a pose or another person in the outline.
MIN_PLAUSIBLE_RATIO = 0.6


def hand_boxes(p: Perception) -> list[tuple[float, float, float, float]]:
    """(x0, y0, x1, y1) around each visible hand, fingertips included."""
    boxes = []
    for side in ("left", "right"):
        pts = {}
        for name in HAND_POINTS:
            lm = p.landmarks.get(f"{side}_{name}")
            if lm is not None and lm.visibility >= MIN_HAND_VISIBILITY:
                pts[name] = np.array([lm.x, lm.y], dtype=np.float64)
        if "wrist" not in pts or len(pts) < 2:
            continue
        wrist = pts["wrist"]
        knuckles = [pts[n] for n in ("pinky", "index") if n in pts]
        tips = [k + FINGER_REACH * (k - wrist) for k in knuckles]
        cloud = np.array(list(pts.values()) + tips)
        x0, y0 = cloud.min(axis=0)
        x1, y1 = cloud.max(axis=0)
        pad = HAND_PAD * max(x1 - x0, y1 - y0, 1.0)
        boxes.append((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
    return boxes


def _touches_hand(boxes, y: float, x0: float, x1: float) -> bool:
    return any(by0 <= y <= by1 and bx0 <= x1 and bx1 >= x0 for bx0, by0, bx1, by1 in boxes)


def _touches_arm(frame: BodyFrame, y: float, x0: float, x1: float) -> bool:
    margin = ARM_HALF_WIDTH * frame.shoulder_joint_dist
    left = frame.arm_x_at(frame.image_left, y)
    right = frame.arm_x_at(frame.image_right, y)
    return (left is not None and x0 <= left + margin) or (right is not None and x1 >= right - margin)


def _without_edge_jumps(torso, candidates: list[bool], frame: BodyFrame) -> list[bool]:
    """Drop rows whose left or right edge jumps outward from the last kept row."""
    jump = EDGE_JUMP * frame.shoulder_joint_dist
    kept, last = [], None
    for t, ok in zip(torso, candidates):
        if ok and last is not None and (t.x0 < last[0] - jump or t.x1 > last[1] + jump):
            ok = False
        if ok:
            last = (t.x0, t.x1)
        kept.append(ok)
    return kept


def measure_hips(p: Perception, frame: BodyFrame) -> tuple[dict[str, float] | None, list[str]]:
    band = np.arange(int(frame.y_at(HIP_BAND[0])), int(frame.y_at(HIP_BAND[1])) + 1)
    if band.size == 0:
        return None, [R_HIPS_OUT_OF_FRAME]
    in_photo = p.in_frame(np.array([int(frame.mid_x(y)) for y in band]), band)
    if in_photo.mean() < MIN_HIP_IN_PHOTO:
        return None, [R_HIPS_OUT_OF_FRAME]
    rows = band[in_photo]
    torso = [torso_row(p, frame, int(y)) for y in rows]
    hands = hand_boxes(p)
    # A hanging hand next to the thigh can merge into the hip run below the
    # wrist, where the arm centre line no longer flags it.
    candidates = [
        t is not None and t.usable and not _touches_hand(hands, float(y), t.x0, t.x1)
        and not _touches_arm(frame, float(y), t.x0, t.x1)
        for y, t in zip(rows, torso)
    ]
    widths = np.array(
        [t.width if keep else np.nan for t, keep in zip(torso, _without_edge_jumps(torso, candidates, frame))],
        dtype=float,
    )
    if int(np.sum(~np.isnan(widths))) < MIN_VALID_HIP_ROWS:
        cut = sum(1 for t in torso if t is not None and t.clipped)
        return None, [R_HIPS_OUT_OF_FRAME if cut > rows.size / 2 else R_HIPS_BLOCKED]
    smooth = np.where(np.isnan(widths), np.nan, median_smooth(widths))
    i = int(np.nanargmax(smooth))
    reasons = []
    if not in_photo[-1] and i >= rows.size - 2:
        # Widest row is the last one inside the photo: the hips may continue below.
        reasons.append(R_HIPS_AT_EDGE)
    best = torso[i]
    return {"width": float(smooth[i]), "row": float(rows[i]), "x0": float(best.x0), "x1": float(best.x1)}, reasons


def waist_hip_metric(p: Perception, frame: BodyFrame, config: ScoringConfig = DEFAULT_CONFIG) -> Metric:
    key = "waist_hip"
    if frame.hips_estimated:
        return unavailable(key, R_HIPS_ESTIMATED)
    waist, waist_reasons = measure_waist(p, frame)
    if waist is None:
        return unavailable(key, *(waist_reasons or [R_WAIST_OUT_OF_FRAME]))
    hips, hip_reasons = measure_hips(p, frame)
    if hips is None:
        return unavailable(key, *hip_reasons)
    reasons = tuple(dict.fromkeys(waist_reasons + hip_reasons + rotation_reasons(frame)))
    ratio = waist["width"] / hips["width"]
    if ratio < MIN_PLAUSIBLE_RATIO:
        return unavailable(key, R_WAIST_HIP_IMPLAUSIBLE)
    return Metric(
        key=key,
        status=status_for(reasons, LOW_CONFIDENCE),
        score=logistic_score(-ratio, config.waist_hip),
        values={"ratio": ratio, "waist_width_px": waist["width"], "hip_width_px": hips["width"]},
        reasons=reasons,
        geometry={
            "waist_line": [[waist["x0"], waist["row"]], [waist["x1"], waist["row"]]],
            "hip_line": [[hips["x0"], hips["row"]], [hips["x1"], hips["row"]]],
        },
    )
