"""Abdomen crop for a second opinion by a vision model.

Callers that want a vision model to double-check the abdominal blocks can ask the
service for this crop instead of sending the whole photo: it covers the abdomen
(ribs to below the navel, flank to flank) and never the face.
"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

MAX_SIDE = 512
MARGIN = 0.15          # of the ROI size, added on every side
JPEG_QUALITY = 85


def _to_original(point: list[float], m: list[list[float]]) -> tuple[float, float]:
    x, y = point
    return m[0][0] * x + m[0][1] * y + m[0][2], m[1][0] * x + m[1][1] * y + m[1][2]


def abdomen_crop(bgr: np.ndarray, report: dict[str, Any], max_side: int = MAX_SIDE) -> bytes | None:
    """JPEG bytes of the abdomen in the original photo, or None when it was not measured."""
    metric = report.get("metrics", {}).get("abs") or {}
    roi = (metric.get("geometry") or {}).get("roi")
    if metric.get("status") == "unavailable" or not roi or len(roi) != 4:
        return None
    m = report.get("to_original") or [[1, 0, 0], [0, 1, 0]]
    x0, y0, x1, y1 = (float(v) for v in roi)
    corners = [_to_original(p, m) for p in ([x0, y0], [x1, y0], [x1, y1], [x0, y1])]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if w < 8 or h < 8:
        return None
    img_h, img_w = bgr.shape[:2]
    left = int(max(0, min(xs) - MARGIN * w))
    right = int(min(img_w, max(xs) + MARGIN * w))
    top = int(max(0, min(ys) - MARGIN * h))
    bottom = int(min(img_h, max(ys) + MARGIN * h))
    crop = bgr[top:bottom, left:right]
    if crop.size == 0:
        return None
    scale = min(1.0, max_side / max(crop.shape[:2]))
    if scale < 1.0:
        crop = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


def abdomen_crop_b64(bgr: np.ndarray, report: dict[str, Any]) -> str | None:
    data = abdomen_crop(bgr, report)
    return base64.b64encode(data).decode("ascii") if data else None
