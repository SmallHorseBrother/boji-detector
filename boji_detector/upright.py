"""Rotate the photo so the shoulder->hip axis is vertical before measuring rows."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .body import BodyFrame
from .types import Landmark, Perception

MIN_DESKEW_DEG = 1.0
MAX_DESKEW_DEG = 25.0


def _warp_mask(mask: np.ndarray | None, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    warped = cv2.warpAffine(mask.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped.astype(bool)


def rotation_matrix(width: int, height: int, degrees_ccw: float) -> tuple[np.ndarray, tuple[int, int]]:
    """2x3 matrix rotating about the centre into a canvas that fits the whole photo."""
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees_ccw, 1.0)
    corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]],
                       dtype=np.float64) @ matrix.T
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    matrix[:, 2] -= lo
    return matrix, (int(math.ceil(hi[0] - lo[0])), int(math.ceil(hi[1] - lo[1])))


def rotate_perception(p: Perception, degrees_ccw: float) -> Perception:
    """Rotate everything counter-clockwise (as displayed), growing the canvas to fit.

    The canvas corners added by the rotation are marked as outside the photo
    in ``frame_mask``, so measurements can tell a body edge from a photo edge.
    """
    matrix, size = rotation_matrix(p.width, p.height, degrees_ccw)
    image = cv2.warpAffine(p.image, matrix, size, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    landmarks = {}
    for name, lm in p.landmarks.items():
        x, y = matrix @ np.array([lm.x, lm.y, 1.0])
        landmarks[name] = Landmark(float(x), float(y), lm.z, lm.visibility)
    frame = p.frame_mask if p.frame_mask is not None else np.ones((p.height, p.width), bool)
    return Perception(
        image=image,
        landmarks=landmarks,
        person_mask=_warp_mask(p.person_mask, matrix, size),
        skin_mask=_warp_mask(p.skin_mask, matrix, size),
        clothes_mask=_warp_mask(p.clothes_mask, matrix, size),
        frame_mask=_warp_mask(frame, matrix, size),
    )


def deskew(p: Perception, frame: BodyFrame) -> tuple[Perception, float, np.ndarray | None]:
    """Upright perception, applied rotation in degrees and its 2x3 matrix (None if unrotated)."""
    lean = frame.lean_deg
    if abs(lean) < MIN_DESKEW_DEG or abs(lean) > MAX_DESKEW_DEG:
        return p, 0.0, None
    matrix, _ = rotation_matrix(p.width, p.height, -lean)
    return rotate_perception(p, -lean), -lean, matrix
