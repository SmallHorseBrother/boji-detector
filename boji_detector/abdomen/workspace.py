"""Scale-normalised crop around the abdomen.

The crop is resized so the waist is WORK_WAIST pixels wide; every filter size
and search window below is therefore a fixed fraction of the waist, whatever
the photo resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..body import BodyFrame
from ..types import Perception
from .filters import SIGMAS, flatten, log_gray, valley_responses

WORK_WAIST = 240.0
ROI_T = (0.24, 0.92)          # analysed rows along the shoulder->hip axis
WINDOW_HALF = 0.45            # analysed half-width, fraction of the waist
FLATTEN_SIGMA = 0.20          # fraction of the waist
EDGE_EROSION = 0.03           # fraction of the waist trimmed off the skin edge


@dataclass(frozen=True)
class Workspace:
    x0: int
    y0: int
    scale: float
    frame: BodyFrame
    gray: np.ndarray          # uint8 grey crop, for exposure checks
    log_l: np.ndarray
    surface: np.ndarray       # skin (or silhouette) mask
    photo: np.ndarray         # pixels that come from the photo itself
    valid: np.ndarray         # eroded surface inside the analysed window
    relief: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray
    row_range: tuple[int, int]

    @property
    def waist(self) -> float:
        return WORK_WAIST

    @property
    def shape(self) -> tuple[int, int]:
        return self.gray.shape

    def y_work(self, t: float) -> float:
        return (self.frame.y_at(t) - self.y0) * self.scale

    def mid_x(self, y_work: np.ndarray | float) -> np.ndarray:
        y_img = np.asarray(y_work, dtype=np.float64) / self.scale + self.y0
        mid = np.vectorize(self.frame.mid_x)(y_img)
        return (mid - self.x0) * self.scale

    def to_image(self, x: float, y: float) -> list[float]:
        return [float(x / self.scale + self.x0), float(y / self.scale + self.y0)]


def _resize(a: np.ndarray, size: tuple[int, int], scale: float, nearest: bool) -> np.ndarray:
    if nearest:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(a, size, interpolation=interp)


def build_workspace(p: Perception, frame: BodyFrame, waist_px: float) -> Workspace | None:
    scale = WORK_WAIST / waist_px
    pad = (3 * max(SIGMAS) + 4) / scale
    top, bottom = frame.y_at(ROI_T[0]), frame.y_at(ROI_T[1])
    mids = [frame.mid_x(top), frame.mid_x(bottom)]
    half = WINDOW_HALF * waist_px
    x0 = max(0, int(min(mids) - half - pad))
    x1 = min(p.width, int(max(mids) + half + pad) + 1)
    y0 = max(0, int(top - pad))
    y1 = min(p.height, int(bottom + pad) + 1)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    size = (max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale))))
    crop = _resize(p.image[y0:y1, x0:x1], size, scale, nearest=False)
    surface = _resize(p.surface_mask[y0:y1, x0:x1].astype(np.uint8), size, scale, nearest=True) > 0
    full = p.frame_mask if p.frame_mask is not None else np.ones((p.height, p.width), bool)
    photo = _resize(full[y0:y1, x0:x1].astype(np.uint8), size, scale, nearest=True) > 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    log_l = log_gray(crop)

    k = 2 * int(round(EDGE_EROSION * WORK_WAIST)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(surface.astype(np.uint8), kernel).astype(bool)
    relief = flatten(log_l, eroded, FLATTEN_SIGMA * WORK_WAIST)
    horizontal, vertical = valley_responses(relief, eroded)

    h, w = gray.shape
    r0 = max(0, int(round((top - y0) * scale)))
    r1 = min(h - 1, int(round((bottom - y0) * scale)))
    ys = np.arange(h)
    mid = (np.array([frame.mid_x(y / scale + y0) for y in ys]) - x0) * scale
    xs = np.arange(w)[None, :]
    window = np.abs(xs - mid[:, None]) <= WINDOW_HALF * WORK_WAIST
    window &= ((ys >= r0) & (ys <= r1))[:, None]
    return Workspace(
        x0=x0, y0=y0, scale=scale, frame=frame, gray=gray, log_l=log_l,
        surface=surface, photo=photo, valid=eroded & window, relief=relief,
        horizontal=horizontal, vertical=vertical, row_range=(r0, r1),
    )
