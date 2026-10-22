"""Photo-level quality checks that apply to every metric."""

from __future__ import annotations

import cv2
import numpy as np

from .body import R_HIPS_ESTIMATED, BodyFrame
from .proportion import rotation_reasons
from .types import Perception
from .upright import MAX_DESKEW_DEG

R_TORSO_TOO_SMALL = "torso_too_small"
R_BLURRY = "blurry"
R_CAMERA_TILTED = "camera_tilted"
R_NO_SKIN_MODEL = "skin_mask_unavailable"

MIN_SHOULDER_JOINT_PX = 80.0
BLUR_REFERENCE_PX = 200.0     # shoulder-joint distance the blur check is normalised to
MIN_SHARPNESS = 6.0


def sharpness(p: Perception, frame: BodyFrame) -> float | None:
    """Variance of the Laplacian over the torso, at a fixed body scale."""
    y0 = max(0, int(frame.y_at(-0.05)))
    y1 = min(p.height, int(frame.y_at(0.95)))
    half = 0.6 * frame.shoulder_joint_dist
    x0 = max(0, int(frame.shoulder_mid[0] - half))
    x1 = min(p.width, int(frame.shoulder_mid[0] + half))
    if y1 - y0 < 10 or x1 - x0 < 10:
        return None
    crop = cv2.cvtColor(p.image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    mask = p.person_mask[y0:y1, x0:x1]
    scale = BLUR_REFERENCE_PX / frame.shoulder_joint_dist
    size = (max(8, int((x1 - x0) * scale)), max(8, int((y1 - y0) * scale)))
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    lap = cv2.Laplacian(crop.astype(np.float32), cv2.CV_32F)
    return float(lap[mask].var()) if mask.any() else None


def global_quality(
    p: Perception, frame: BodyFrame, deskew_deg: float, sharp: float | None
) -> tuple[str, ...]:
    """Photo-level reasons. ``sharp`` must be measured before any deskew,
    since rotating resamples the image and softens it."""
    reasons: list[str] = []
    if frame.shoulder_joint_dist < MIN_SHOULDER_JOINT_PX:
        reasons.append(R_TORSO_TOO_SMALL)
    if frame.hips_estimated:
        reasons.append(R_HIPS_ESTIMATED)
    reasons += rotation_reasons(frame)
    if camera_too_tilted(frame, deskew_deg):
        reasons.append(R_CAMERA_TILTED)
    if sharp is not None and sharp < MIN_SHARPNESS:
        reasons.append(R_BLURRY)
    if p.skin_mask is None:
        reasons.append(R_NO_SKIN_MODEL)
    return tuple(reasons)


def camera_too_tilted(frame: BodyFrame, deskew_deg: float) -> bool:
    return deskew_deg == 0.0 and abs(frame.lean_deg) > MAX_DESKEW_DEG
