"""Body frame: named joints grouped by image side, plus torso axis helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .types import Landmark

MIN_VISIBILITY = 0.5
MIN_LIMB_VISIBILITY = 0.3     # elbows/wrists; later code also checks they are in frame
MIN_HIP_VISIBILITY = 0.3
# Shoulder-joint to hip-joint length relative to the shoulder-joint distance,
# used only when the hips are not detected.
TORSO_PER_SHOULDER = 1.4

R_POSE_NOT_FOUND = "pose_not_found"
R_HIPS_ESTIMATED = "hips_estimated"


@dataclass(frozen=True)
class Side:
    """Joints of one arm/hip, all on the same image side."""

    shoulder: np.ndarray
    elbow: np.ndarray | None
    wrist: np.ndarray | None
    hip: np.ndarray | None


@dataclass(frozen=True)
class BodyFrame:
    image_left: Side
    image_right: Side
    shoulder_mid: np.ndarray
    hip_mid: np.ndarray
    hips_estimated: bool
    # Offset of the torso silhouette centre from the joint midline, as a share
    # of the torso half-width; filled in once the silhouette is measured.
    rotation_index: float | None = None

    @property
    def torso_len(self) -> float:
        return float(np.linalg.norm(self.hip_mid - self.shoulder_mid))

    @property
    def shoulder_joint_dist(self) -> float:
        return float(np.linalg.norm(self.image_right.shoulder - self.image_left.shoulder))

    @property
    def lean_deg(self) -> float:
        """Angle of the shoulder->hip axis from vertical, positive when hips sit right."""
        d = self.hip_mid - self.shoulder_mid
        return math.degrees(math.atan2(d[0], d[1]))

    def y_at(self, t: float) -> float:
        return float(self.shoulder_mid[1] + t * (self.hip_mid[1] - self.shoulder_mid[1]))

    def mid_x(self, y: float) -> float:
        dy = self.hip_mid[1] - self.shoulder_mid[1]
        if abs(dy) < 1e-6:
            return float(self.shoulder_mid[0])
        t = (y - self.shoulder_mid[1]) / dy
        return float(self.shoulder_mid[0] + t * (self.hip_mid[0] - self.shoulder_mid[0]))

    def sides(self) -> dict[str, Side]:
        return {"image_left": self.image_left, "image_right": self.image_right}

    def upper_arm_len(self) -> float | None:
        lengths = [
            float(np.linalg.norm(s.elbow - s.shoulder))
            for s in (self.image_left, self.image_right)
            if s.elbow is not None
        ]
        return float(np.mean(lengths)) if lengths else None

    def arm_x_at(self, side: Side, y: float) -> float | None:
        """x of the arm centre line (shoulder->elbow->wrist) at row y."""
        chain = [p for p in (side.shoulder, side.elbow, side.wrist) if p is not None]
        for a, b in zip(chain, chain[1:]):
            lo, hi = sorted((a[1], b[1]))
            if lo <= y <= hi and hi - lo > 1e-6:
                t = (y - a[1]) / (b[1] - a[1])
                return float(a[0] + t * (b[0] - a[0]))
        return None


def _point(lm: Landmark | None, min_vis: float) -> np.ndarray | None:
    if lm is None or lm.visibility < min_vis:
        return None
    return np.array([lm.x, lm.y], dtype=np.float64)


def _side(landmarks: Mapping[str, Landmark], prefix: str) -> Side | None:
    shoulder = _point(landmarks.get(f"{prefix}_shoulder"), MIN_VISIBILITY)
    if shoulder is None:
        return None
    return Side(
        shoulder=shoulder,
        elbow=_point(landmarks.get(f"{prefix}_elbow"), MIN_LIMB_VISIBILITY),
        wrist=_point(landmarks.get(f"{prefix}_wrist"), MIN_LIMB_VISIBILITY),
        hip=_point(landmarks.get(f"{prefix}_hip"), MIN_HIP_VISIBILITY),
    )


def build_body_frame(
    landmarks: Mapping[str, Landmark] | None,
) -> tuple[BodyFrame | None, tuple[str, ...]]:
    if not landmarks:
        return None, (R_POSE_NOT_FOUND,)
    subject_left = _side(landmarks, "left")
    subject_right = _side(landmarks, "right")
    if subject_left is None or subject_right is None:
        return None, (R_POSE_NOT_FOUND,)
    image_left, image_right = sorted(
        (subject_left, subject_right), key=lambda s: s.shoulder[0]
    )
    shoulder_mid = (image_left.shoulder + image_right.shoulder) / 2.0
    reasons: list[str] = []
    if image_left.hip is not None and image_right.hip is not None:
        hip_mid = (image_left.hip + image_right.hip) / 2.0
        hips_estimated = False
    else:
        span = float(np.linalg.norm(image_right.shoulder - image_left.shoulder))
        hip_mid = shoulder_mid + np.array([0.0, TORSO_PER_SHOULDER * span])
        hips_estimated = True
        reasons.append(R_HIPS_ESTIMATED)
    if hip_mid[1] - shoulder_mid[1] < 10:
        return None, (R_POSE_NOT_FOUND,)
    frame = BodyFrame(
        image_left=image_left,
        image_right=image_right,
        shoulder_mid=shoulder_mid,
        hip_mid=hip_mid,
        hips_estimated=hips_estimated,
    )
    return frame, tuple(reasons)
