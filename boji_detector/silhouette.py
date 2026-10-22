"""Silhouette helpers: mask clean-up and horizontal runs."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

# Holes larger than this share of the body are real gaps (for example between
# an arm and the torso when the hand touches the thigh) and stay open.
MAX_FILLED_HOLE = 0.005


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Close small gaps, keep the largest blob and fill only its small holes."""
    m = mask.astype(np.uint8)
    if not m.any():
        return mask.astype(bool)
    k = max(3, int(round(min(m.shape) * 0.006)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if count <= 1:
        return m.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = labels == largest
    holes = (~blob).astype(np.uint8)
    n, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    h, w = blob.shape
    limit = MAX_FILLED_HOLE * blob.sum()
    x, y, bw, bh, area = (hole_stats[:, i] for i in range(5))
    touches_border = (x == 0) | (y == 0) | (x + bw == w) | (y + bh == h)
    fill = ~touches_border & (area <= limit)
    fill[0] = False  # label 0 is the blob itself
    return blob | fill[hole_labels]


def run_at(mask: np.ndarray, y: int, x: int, tolerance: int = 3) -> tuple[int, int] | None:
    """Inclusive [x0, x1] of the mask run in row y that contains column x."""
    h, w = mask.shape
    if not (0 <= y < h):
        return None
    row = mask[y]
    x = int(round(x))
    if not (0 <= x < w) or not row[x]:
        candidates = [
            c for d in range(1, tolerance + 1) for c in (x - d, x + d)
            if 0 <= c < w and row[c]
        ]
        if not candidates:
            return None
        x = candidates[0]
    left = row[:x][::-1]
    right = row[x:]
    x0 = x - (int(np.argmin(left)) if not left.all() else len(left))
    x1 = x + (int(np.argmin(right)) if not right.all() else len(right)) - 1
    return x0, x1


def median_smooth(values: np.ndarray, size: int = 5) -> np.ndarray:
    """Median filter that ignores NaN entries."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    half = size // 2
    for i in range(len(values)):
        window = values[max(0, i - half): i + half + 1]
        window = window[~np.isnan(window)]
        if window.size:
            out[i] = float(np.median(window))
    return out


InFrame = Callable[[np.ndarray, np.ndarray], np.ndarray]


def walk(
    mask: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    max_len: int,
    in_frame: InFrame | None = None,
) -> tuple[int, bool]:
    """(pixels travelled before leaving the mask, whether the walk hit the photo edge).

    A walk that stops because the photo ends did not find the body's edge, so
    callers must not use its length as a measurement.
    """
    h, w = mask.shape
    steps = np.arange(max_len + 1)
    xs = np.rint(start[0] + steps * direction[0]).astype(int)
    ys = np.rint(start[1] + steps * direction[1]).astype(int)
    if in_frame is None:
        frame = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    else:
        frame = in_frame(xs, ys)
    hits = np.zeros(len(steps), dtype=bool)
    xi, yi = np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1)
    hits[frame] = mask[yi[frame], xi[frame]]
    misses = np.nonzero(~hits)[0]
    if not misses.size:
        return max_len, False
    first = int(misses[0])
    return first, not bool(frame[first])


def walk_extent(mask: np.ndarray, start: np.ndarray, direction: np.ndarray, max_len: int) -> int:
    return walk(mask, start, direction, max_len)[0]
