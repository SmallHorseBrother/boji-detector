"""Shared immutable data types.

Coordinates are image pixels: x grows to the right, y grows downward.
Landmark names follow MediaPipe Pose (``left_*`` is the subject's left side).
Metrics report sides as ``image_left`` / ``image_right`` so that mirror
selfies and front photos read the same way.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping

import numpy as np

STATUS_OK = "ok"
STATUS_LOW = "low_confidence"
STATUS_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass(frozen=True)
class Perception:
    """Everything the metric modules need from the image models."""

    image: np.ndarray                       # HxWx3 BGR uint8
    landmarks: Mapping[str, Landmark]       # pixel coordinates
    person_mask: np.ndarray                 # HxW bool silhouette
    skin_mask: np.ndarray | None = None     # HxW bool body skin (face excluded)
    clothes_mask: np.ndarray | None = None  # HxW bool
    # Pixels that come from the photo itself; None means all of them. After a
    # deskew rotation the corners of the enlarged canvas are not photo pixels.
    frame_mask: np.ndarray | None = None

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    def in_frame(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """True where integer pixel (xs, ys) lies on real photo pixels."""
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        inside = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
        if self.frame_mask is None:
            return inside
        out = np.zeros(np.broadcast_shapes(xs.shape, ys.shape), dtype=bool)
        xi = np.clip(xs, 0, self.width - 1)
        yi = np.clip(ys, 0, self.height - 1)
        out[...] = inside & self.frame_mask[yi, xi]
        return out

    def run_clipped(self, y: int, x0: int, x1: int) -> bool:
        """Whether the run [x0, x1] in row y ends at the edge of the photo."""
        return not bool(self.in_frame(np.array([x0 - 1, x1 + 1]), np.array([y, y])).all())

    @property
    def surface_mask(self) -> np.ndarray:
        """Pixels where muscle texture may be read: skin when known."""
        if self.skin_mask is not None:
            return self.skin_mask & self.person_mask
        return self.person_mask


@dataclass(frozen=True)
class Metric:
    key: str
    status: str
    score: float | None = None
    values: Mapping[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    geometry: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status != STATUS_UNAVAILABLE


def unavailable(key: str, *reasons: str) -> Metric:
    return Metric(key=key, status=STATUS_UNAVAILABLE, reasons=tuple(reasons))


def status_for(reasons: tuple[str, ...], low_confidence: frozenset[str]) -> str:
    return STATUS_LOW if any(r in low_confidence for r in reasons) else STATUS_OK


def with_low_confidence(metric: Metric, *codes: str) -> Metric:
    """Copy of an available metric with extra reasons that lower its confidence."""
    if not metric.available or not codes:
        return metric
    reasons = tuple(dict.fromkeys(metric.reasons + codes))
    return replace(metric, status=STATUS_LOW, reasons=reasons)


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def map_points(points: Iterable[Iterable[float]], matrix=IDENTITY) -> list[list[float]]:
    """Apply a 2x3 affine matrix to (x, y) points."""
    m = np.asarray(matrix, dtype=np.float64)
    pts = np.asarray(list(points), dtype=np.float64).reshape(-1, 2)
    mapped = pts @ m[:, :2].T + m[:, 2]
    return mapped.tolist()


@dataclass(frozen=True)
class Report:
    """Analysis result. Geometry is in analysed-image pixels (after the size
    cap and deskew); ``to_original`` maps those points onto the input photo."""

    version: str
    image_size: tuple[int, int]             # (width, height) of the analysed image
    deskew_deg: float
    quality: tuple[str, ...]
    metrics: Mapping[str, Metric]
    total: Mapping[str, Any]
    to_original: tuple[tuple[float, float, float], tuple[float, float, float]] = IDENTITY
    original_size: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return round(number, 4) if np.isfinite(number) else None  # strict JSON has no NaN
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
