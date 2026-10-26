"""Locate the linea alba, the horizontal grooves and the lateral rectus edges.

All coordinates are workspace pixels; distances are fractions of the waist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .workspace import Workspace

GROOVE_FLOOR = 0.010          # minimum valley response for a groove
ALBA_T = (0.30, 0.85)         # rows searched for the linea alba
ALBA_SEARCH = 0.12            # half-width around the body midline
ALBA_TRACK = 0.05             # per-band deviation allowed from the alba peak
ALBA_BAND = 0.08              # band height used to trace the alba
GROOVE_COLUMNS = (0.06, 0.27) # distance from the alba for horizontal profiles
GROOVE_MIN_GAP = 0.09
RELATIVE_PROMINENCE = 0.12    # of the side's strongest groove (usually the pec fold)
GROOVE_TRACE = (0.03, 0.44)
GROOVE_TRACE_ROWS = 0.04
LATERAL_SEARCH = (0.18, 0.42)
LATERAL_DEFAULT = 0.30
SIDES = {"image_left": -1, "image_right": 1}


@dataclass(frozen=True)
class Alba:
    found: bool
    strength: float
    a: float                  # x = a + b * y
    b: float

    def x(self, y: np.ndarray | float) -> np.ndarray:
        return self.a + self.b * np.asarray(y, dtype=np.float64)


@dataclass(frozen=True)
class Groove:
    row: float                # mean row of the traced line
    strength: float
    points: np.ndarray        # (N, 2) x, y along the groove


def _sample(field: np.ndarray, valid: np.ndarray, rows: np.ndarray, xs: np.ndarray):
    """Values of field at (rows, xs) broadcast together, plus their validity."""
    h, w = field.shape
    rr = np.broadcast_to(rows, np.broadcast_shapes(rows.shape, xs.shape))
    xi = np.rint(np.broadcast_to(xs, rr.shape)).astype(int)
    inside = (xi >= 0) & (xi < w) & (rr >= 0) & (rr < h)
    xi = np.clip(xi, 0, w - 1)
    ri = np.clip(rr, 0, h - 1)
    ok = inside & valid[ri, xi]
    return np.where(ok, field[ri, xi], 0.0), ok


def _masked_mean(values: np.ndarray, ok: np.ndarray, axis: int, min_share: float) -> np.ndarray:
    count = ok.sum(axis=axis)
    total = np.where(ok, values, 0.0).sum(axis=axis)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return np.where(count >= min_share * ok.shape[axis], mean, np.nan)


def _rows(ws: Workspace, t0: float, t1: float) -> np.ndarray:
    lo = max(ws.row_range[0], int(round(ws.y_work(t0))))
    hi = min(ws.row_range[1], int(round(ws.y_work(t1))))
    return np.arange(lo, hi + 1)


def find_alba(ws: Workspace) -> Alba:
    rows = _rows(ws, *ALBA_T)
    w = ws.waist
    mid = ws.mid_x(rows)
    if rows.size < 5:
        return Alba(False, 0.0, float(np.mean(mid)) if mid.size else 0.0, 0.0)
    offsets = np.arange(-ALBA_SEARCH * w, ALBA_SEARCH * w + 1)
    vals, ok = _sample(ws.vertical, ws.valid, rows[:, None], mid[:, None] + offsets[None, :])
    profile = np.nan_to_num(_masked_mean(vals, ok, axis=0, min_share=0.3))
    profile = gaussian_filter1d(profile, 1.5)
    peak = int(np.argmax(profile))
    strength = float(profile[peak])
    found = strength >= GROOVE_FLOOR and strength - float(np.median(profile)) >= 0.5 * GROOVE_FLOOR
    slope = float(np.polyfit(rows, mid, 1)[0])
    if not found:
        return Alba(False, strength, float(mid[0] - slope * rows[0]), slope)

    band = max(4, int(ALBA_BAND * w))
    track = np.arange(-ALBA_TRACK * w, ALBA_TRACK * w + 1)
    pts_y, pts_x = [], []
    for start in range(0, rows.size, band):
        r = rows[start:start + band]
        centre = mid[start:start + band] + offsets[peak]
        v, k = _sample(ws.vertical, ws.valid, r[:, None], centre[:, None] + track[None, :])
        local = _masked_mean(v, k, axis=0, min_share=0.5)
        if np.all(np.isnan(local)) or np.nanmax(local) < GROOVE_FLOOR:
            continue
        j = int(np.nanargmax(local))
        pts_y.append(float(np.mean(r)))
        pts_x.append(float(np.mean(centre) + track[j]))
    if len(pts_y) >= 3:
        b, a = np.polyfit(pts_y, pts_x, 1)
        return Alba(True, strength, float(a), float(b))
    x_peak = mid + offsets[peak]
    return Alba(True, strength, float(x_peak[0] - slope * rows[0]), slope)


def side_profile(ws: Workspace, alba: Alba, sign: int, rows: np.ndarray) -> np.ndarray:
    w = ws.waist
    offsets = np.arange(GROOVE_COLUMNS[0] * w, GROOVE_COLUMNS[1] * w + 1)
    xs = alba.x(rows)[:, None] + sign * offsets[None, :]
    vals, ok = _sample(ws.horizontal, ws.valid, rows[:, None], xs)
    return _masked_mean(vals, ok, axis=1, min_share=0.5)


def trace_groove(ws: Workspace, alba: Alba, sign: int, row: int) -> np.ndarray:
    w = ws.waist
    reach = max(2, int(GROOVE_TRACE_ROWS * w))
    window = np.arange(row - reach, row + reach + 1)
    points = []
    for d in np.arange(GROOVE_TRACE[0] * w, GROOVE_TRACE[1] * w, 2.0):
        x = float(alba.x(row) + sign * d)
        vals, ok = _sample(ws.horizontal, ws.valid, window, np.full(window.shape, x))
        if ok.any() and vals.max() >= 0.5 * GROOVE_FLOOR:
            points.append((x, float(window[int(np.argmax(vals))])))
        else:
            points.append((x, float(row)))
    return np.array(points)


def find_grooves(ws: Workspace, alba: Alba, sign: int) -> list[Groove]:
    rows = _rows(ws, 0.24, 0.92)
    if rows.size < 5:
        return []
    profile = gaussian_filter1d(np.nan_to_num(side_profile(ws, alba, sign, rows)), 1.0)
    top = float(profile.max()) if profile.size else 0.0
    if top < GROOVE_FLOOR:
        return []
    peaks, props = find_peaks(
        profile,
        height=GROOVE_FLOOR,
        distance=max(1, int(GROOVE_MIN_GAP * ws.waist)),
        prominence=max(0.6 * GROOVE_FLOOR, RELATIVE_PROMINENCE * top),
    )
    grooves = []
    for i, peak in enumerate(peaks):
        row = int(rows[peak])
        pts = trace_groove(ws, alba, sign, row)
        grooves.append(Groove(row=float(np.mean(pts[:, 1])), strength=float(props["peak_heights"][i]), points=pts))
    return grooves


def find_lateral(ws: Workspace, alba: Alba, sign: int, r0: float, r1: float) -> tuple[float, bool, float]:
    """(distance from the alba, found, valley strength) of one lateral rectus edge."""
    w = ws.waist
    rows = np.arange(int(r0), int(r1) + 1)
    if rows.size < 3:
        return LATERAL_DEFAULT * w, False, 0.0
    offsets = np.arange(LATERAL_SEARCH[0] * w, LATERAL_SEARCH[1] * w + 1)
    xs = alba.x(rows)[:, None] + sign * offsets[None, :]
    vals, ok = _sample(ws.vertical, ws.valid, rows[:, None], xs)
    profile = gaussian_filter1d(np.nan_to_num(_masked_mean(vals, ok, axis=0, min_share=0.5)), 1.5)
    j = int(np.argmax(profile))
    strength = float(profile[j])
    if strength < GROOVE_FLOOR:
        return LATERAL_DEFAULT * w, False, strength
    return float(offsets[j]), True, strength
