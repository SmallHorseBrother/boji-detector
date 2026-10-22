"""V-taper: bideltoid width / narrowest waist width, read from the silhouette.

Rows whose torso run ends at the edge of the photo (image border, or the
empty corners a deskew rotation adds) are never used as measurements: the
body continues beyond what the photo shows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .body import R_HIPS_ESTIMATED, BodyFrame
from .scoring import DEFAULT_CONFIG, ScoringConfig, logistic_score
from .silhouette import median_smooth, run_at
from .types import Metric, Perception, status_for, unavailable

R_ARMS_RAISED = "arms_raised"
R_ARMS_TOUCHING = "arms_touching_torso"
R_ELBOWS_MISSING = "elbows_not_visible"
R_SHOULDERS_NOT_FOUND = "shoulders_not_found"
R_SHOULDERS_CUT_OFF = "shoulders_cut_off"
R_WAIST_AT_EDGE = "waist_at_image_edge"
R_WAIST_OUT_OF_FRAME = "waist_out_of_frame"
R_BODY_ROTATED = "body_rotated"

LOW_CONFIDENCE = frozenset({R_ELBOWS_MISSING, R_WAIST_AT_EDGE, R_HIPS_ESTIMATED, R_BODY_ROTATED})

SHOULDER_BAND = (-0.12, 0.30)   # in upper-arm lengths around the shoulder joints
WAIST_BAND = (0.45, 0.95)       # fraction of the shoulder->hip axis
MIN_ARM_DROP = 0.6              # elbow drop / upper-arm length when arms hang down
MIN_VALID_WAIST_FRACTION = 0.25
MIN_VALID_WAIST_ROWS = 8
MIN_WAIST_IN_PHOTO = 0.5        # share of the waist band that must be inside the photo
CUT_TOLERANCE = 0.97            # clipped rows this close to the widest row may hide the real maximum
EDGE_ROWS = 0.05                # rows this close to a photo edge count as "at the edge"
ROTATION_LIMIT = 0.18           # torso centre offset / torso half-width


@dataclass(frozen=True)
class TorsoRow:
    x0: int
    x1: int
    arms_clear: bool            # neither edge is an arm merged into the torso
    clipped: bool               # the run ends at the edge of the photo

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def usable(self) -> bool:
        return self.arms_clear and not self.clipped


def arms_down(frame: BodyFrame) -> bool | None:
    """True when both elbows hang below the shoulders, None when unknown."""
    states = []
    for side in frame.sides().values():
        if side.elbow is None:
            return None
        length = float(np.linalg.norm(side.elbow - side.shoulder))
        states.append(side.elbow[1] - side.shoulder[1] >= MIN_ARM_DROP * length)
    return all(states)


def rotation_reasons(frame: BodyFrame) -> list[str]:
    if frame.rotation_index is not None and frame.rotation_index > ROTATION_LIMIT:
        return [R_BODY_ROTATED]
    return []


def torso_row(p: Perception, frame: BodyFrame, y: int) -> TorsoRow | None:
    span = run_at(p.person_mask, y, int(frame.mid_x(y)))
    if span is None:
        return None
    x0, x1 = span
    arm_left = frame.arm_x_at(frame.image_left, y)
    arm_right = frame.arm_x_at(frame.image_right, y)
    arms_clear = (arm_left is None or x0 > arm_left) and (arm_right is None or x1 < arm_right)
    return TorsoRow(x0=x0, x1=x1, arms_clear=arms_clear, clipped=p.run_clipped(y, x0, x1))


def torso_rotation(p: Perception, frame: BodyFrame) -> float | None:
    """How far the waist silhouette centre sits from the joint midline.

    When the body turns, the silhouette shifts against the shoulder/hip joints.
    MediaPipe depth (z) proved too noisy for this on still photos.
    """
    y0 = max(0, int(frame.y_at(WAIST_BAND[0])))
    y1 = min(p.height - 1, int(frame.y_at(WAIST_BAND[1])))
    offsets = []
    for y in range(y0, y1 + 1, 2):
        row = torso_row(p, frame, y)
        if row is None or not row.usable or row.x1 <= row.x0:
            continue
        offsets.append(abs((row.x0 + row.x1) / 2.0 - frame.mid_x(y)) / ((row.x1 - row.x0) / 2.0))
    return float(np.median(offsets)) if len(offsets) >= MIN_VALID_WAIST_ROWS // 2 else None


def _only_measured(smooth: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Smoothing fills gaps from neighbours; never pick a row that had no measurement."""
    return np.where(np.isnan(raw), np.nan, smooth)


def measure_shoulders(p: Perception, frame: BodyFrame) -> tuple[dict[str, float] | None, list[str]]:
    upper = frame.upper_arm_len() or 0.6 * frame.shoulder_joint_dist
    ys = [frame.image_left.shoulder[1], frame.image_right.shoulder[1]]
    y0 = max(0, int(min(ys) + SHOULDER_BAND[0] * upper))
    y1 = min(p.height - 1, int(max(ys) + SHOULDER_BAND[1] * upper))
    rows = np.arange(y0, y1 + 1)
    spans = [run_at(p.person_mask, int(y), int(frame.mid_x(y))) for y in rows]
    clipped = np.array([s is not None and p.run_clipped(int(y), *s) for y, s in zip(rows, spans)], bool)
    widths = np.array([np.nan if s is None else s[1] - s[0] + 1 for s in spans], dtype=float)
    measured = np.where(clipped, np.nan, widths)
    if rows.size == 0 or np.all(np.isnan(measured)):
        return None, [R_SHOULDERS_CUT_OFF if clipped.any() else R_SHOULDERS_NOT_FOUND]
    smooth = _only_measured(median_smooth(measured), measured)
    i = int(np.nanargmax(smooth))
    if clipped.any() and np.nanmax(widths[clipped]) >= CUT_TOLERANCE * smooth[i]:
        # A row cut by the photo edge is already about as wide as the widest
        # complete row, so the real shoulder width is unknown.
        return None, [R_SHOULDERS_CUT_OFF]
    x0, x1 = spans[i]
    return {"width": float(smooth[i]), "row": float(rows[i]), "x0": float(x0), "x1": float(x1)}, []


def measure_waist(p: Perception, frame: BodyFrame) -> tuple[dict[str, float] | None, list[str]]:
    band = np.arange(int(frame.y_at(WAIST_BAND[0])), int(frame.y_at(WAIST_BAND[1])) + 1)
    in_photo = p.in_frame(np.array([int(frame.mid_x(y)) for y in band]), band)
    if band.size == 0 or in_photo.mean() < MIN_WAIST_IN_PHOTO:
        return None, [R_WAIST_OUT_OF_FRAME]
    rows = band[in_photo]
    torso = [torso_row(p, frame, int(y)) for y in rows]
    widths = np.array([t.width if t is not None and t.usable else np.nan for t in torso], dtype=float)
    cut = np.array([t is not None and t.clipped for t in torso], bool)
    valid = int(np.sum(~np.isnan(widths)))
    if valid < max(MIN_VALID_WAIST_ROWS, MIN_VALID_WAIST_FRACTION * rows.size):
        too_few_in_photo = cut.sum() > rows.size / 2
        return None, [R_WAIST_OUT_OF_FRAME if too_few_in_photo else R_ARMS_TOUCHING]
    smooth = _only_measured(median_smooth(widths), widths)
    i = int(np.nanargmin(smooth))

    # Rows next to a photo edge: either outside the photo or clipped by it.
    edge = ~in_photo.copy()
    edge[np.flatnonzero(in_photo)[cut]] = True
    reach = max(3, int(EDGE_ROWS * band.size))
    near = np.convolve(edge.astype(int), np.ones(2 * reach + 1, int), mode="same")[in_photo] > 0
    candidates = smooth[near & ~np.isnan(smooth)]
    reasons = []
    if candidates.size and np.min(candidates) <= smooth[i] + max(1.0, 0.01 * smooth[i]):
        reasons.append(R_WAIST_AT_EDGE)
    best = torso[i]
    return {"width": float(smooth[i]), "row": float(rows[i]), "x0": float(best.x0), "x1": float(best.x1)}, reasons


def v_taper_metric(p: Perception, frame: BodyFrame, config: ScoringConfig = DEFAULT_CONFIG) -> Metric:
    key = "v_taper"
    down = arms_down(frame)
    if down is False:
        return unavailable(key, R_ARMS_RAISED)
    reasons = [] if down else [R_ELBOWS_MISSING]
    if frame.hips_estimated:
        reasons.append(R_HIPS_ESTIMATED)
    reasons += rotation_reasons(frame)
    shoulders, shoulder_reasons = measure_shoulders(p, frame)
    if shoulders is None:
        return unavailable(key, *shoulder_reasons)
    waist, waist_reasons = measure_waist(p, frame)
    if waist is None:
        return unavailable(key, *waist_reasons)
    reasons += waist_reasons
    ratio = shoulders["width"] / waist["width"]
    all_reasons = tuple(reasons)
    return Metric(
        key=key,
        status=status_for(all_reasons, LOW_CONFIDENCE),
        score=logistic_score(ratio, config.v_taper),
        values={
            "ratio": ratio,
            "shoulder_width_px": shoulders["width"],
            "waist_width_px": waist["width"],
        },
        reasons=all_reasons,
        geometry={
            "shoulder_line": [[shoulders["x0"], shoulders["row"]], [shoulders["x1"], shoulders["row"]]],
            "waist_line": [[waist["x0"], waist["row"]], [waist["x1"], waist["row"]]],
        },
    )
