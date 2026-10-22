"""Arm fullness: limb width across the bone line divided by limb length.

Widths are read from the silhouette on lines perpendicular to the
shoulder->elbow (and elbow->wrist) segment. When the inner side of the arm is
merged with the torso, the outer half-width is mirrored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .body import BodyFrame, Side
from .proportion import R_BODY_ROTATED, arms_down, rotation_reasons
from .scoring import DEFAULT_CONFIG, ScoringConfig, logistic_score
from .silhouette import walk
from .types import Metric, Perception, status_for, unavailable

R_SLEEVES = "sleeves_cover_arms"
R_ARMS_NOT_VISIBLE = "arms_not_visible"
R_ONE_ARM = "one_arm_only"
R_ARM_CUT_OFF = "arm_cut_off"
LOW_CONFIDENCE = frozenset({R_ONE_ARM, R_BODY_ROTATED})

UPPER_SAMPLES = (0.45, 0.55, 0.65, 0.75)
FOREARM_SAMPLES = (0.25, 0.35, 0.45)
MERGE_FACTOR = 1.6        # inner extent this much longer than outer => merged with torso
SLEEVE_FRACTION = 0.3     # share of a sample line covered by clothes


def _inside(p: Perception, point: np.ndarray) -> bool:
    x, y = int(round(point[0])), int(round(point[1]))
    return 0 <= x < p.width and 0 <= y < p.height and bool(p.person_mask[y, x])


def _clothes_share(p: Perception, a: np.ndarray, b: np.ndarray) -> float:
    if p.clothes_mask is None:
        return 0.0
    n = max(2, int(np.linalg.norm(b - a)))
    xs = np.clip(np.rint(np.linspace(a[0], b[0], n)).astype(int), 0, p.width - 1)
    ys = np.clip(np.rint(np.linspace(a[1], b[1], n)).astype(int), 0, p.height - 1)
    return float(p.clothes_mask[ys, xs].mean())


def measure_segment(
    p: Perception, frame: BodyFrame, a: np.ndarray, b: np.ndarray, samples: tuple[float, ...]
) -> dict[str, Any] | None:
    """Median width across the segment a->b; None when no sample lands on the body."""
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 5:
        return None
    u = d / length
    normal = np.array([-u[1], u[0]])
    max_len = int(0.6 * length)
    found: list[dict[str, Any]] = []
    cut = 0
    for t in samples:
        c = a + t * d
        if not _inside(p, c):
            continue
        outward = normal if (c[0] - frame.mid_x(c[1])) * normal[0] > 0 else -normal
        outer, outer_cut = walk(p.person_mask, c, outward, max_len, p.in_frame)
        inner, inner_cut = walk(p.person_mask, c, -outward, max_len, p.in_frame)
        if outer_cut or inner_cut:
            cut += 1  # the photo ends before the arm does
            continue
        merged = inner >= max_len or inner > MERGE_FACTOR * outer
        if merged:
            inner = outer
        p0, p1 = c - inner * outward, c + outer * outward
        found.append({
            "width": outer + inner - 1,
            "merged": merged,
            "line": [p0.tolist(), p1.tolist()],
            "clothes": _clothes_share(p, p0, p1),
        })
    if not found:
        return {"cut_off": True} if cut else None
    # Where the arm touches the torso its inner edge is hidden and the width is
    # a mirrored guess around the landmark line, which is rarely centred. Use
    # samples with both edges visible whenever there are any.
    complete = [f for f in found if not f["merged"]]
    used = complete or found
    width = float(np.median([f["width"] for f in used]))
    return {
        "width": width,
        "length": length,
        "ratio": width / length,
        "estimated": not complete,
        "lines": [f["line"] for f in used],
        "clothes": float(np.mean([f["clothes"] for f in found])),
    }


def _in_frame(p: Perception, point: np.ndarray | None) -> bool:
    return point is not None and 0 <= point[0] < p.width and 0 <= point[1] < p.height


def measure_side(p: Perception, frame: BodyFrame, side: Side) -> dict[str, Any] | None:
    if not _in_frame(p, side.elbow):
        return None
    upper = measure_segment(p, frame, side.shoulder, side.elbow, UPPER_SAMPLES)
    if upper is None or upper.get("cut_off"):
        return upper
    fore = None
    if _in_frame(p, side.wrist):
        fore = measure_segment(p, frame, side.elbow, side.wrist, FOREARM_SAMPLES)
        if fore is not None and fore.get("cut_off"):
            fore = None
    return {"upper": upper, "fore": fore}


def arms_metric(p: Perception, frame: BodyFrame, config: ScoringConfig = DEFAULT_CONFIG) -> Metric:
    key = "arms"
    measured: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for name, side in frame.sides().items():
        result = measure_side(p, frame, side)
        if result is None:
            skipped.append(R_ARMS_NOT_VISIBLE)
        elif result.get("cut_off"):
            skipped.append(R_ARM_CUT_OFF)
        elif result["upper"]["clothes"] > SLEEVE_FRACTION:
            skipped.append(R_SLEEVES)
        else:
            measured[name] = result
    if not measured:
        return unavailable(key, *dict.fromkeys(skipped))
    reasons = rotation_reasons(frame)
    if len(measured) == 1:
        reasons += [R_ONE_ARM, *skipped]
    values: dict[str, Any] = {"arms_down": bool(arms_down(frame))}
    upper_ratios, fore_ratios, lines = [], [], []
    for name, result in measured.items():
        values[f"upper_ratio_{name}"] = result["upper"]["ratio"]
        values[f"upper_width_px_{name}"] = result["upper"]["width"]
        values[f"upper_width_estimated_{name}"] = result["upper"]["estimated"]
        upper_ratios.append(result["upper"]["ratio"])
        lines += result["upper"]["lines"]
        if result["fore"] is not None:
            values[f"forearm_ratio_{name}"] = result["fore"]["ratio"]
            fore_ratios.append(result["fore"]["ratio"])
            lines += result["fore"]["lines"]
    upper_ratio = float(np.mean(upper_ratios))
    values["upper_ratio"] = upper_ratio
    values["forearm_ratio"] = float(np.mean(fore_ratios)) if fore_ratios else None
    values["balance"] = min(upper_ratios) / max(upper_ratios) if len(upper_ratios) == 2 else None
    all_reasons = tuple(reasons)
    return Metric(
        key=key,
        status=status_for(all_reasons, LOW_CONFIDENCE),
        score=logistic_score(upper_ratio, config.arms),
        values=values,
        reasons=all_reasons,
        geometry={"width_lines": lines},
    )
