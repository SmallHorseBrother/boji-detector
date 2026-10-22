"""Top-level analysis: perception -> upright frame -> metrics -> report."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np

from . import __version__
from .abdomen import abs_metric
from .arms import arms_metric
from .body import R_POSE_NOT_FOUND, build_body_frame
from .proportion import torso_rotation, v_taper_metric
from .quality import R_CAMERA_TILTED, global_quality, sharpness
from .scoring import DEFAULT_CONFIG, ScoringConfig, total_score
from .symmetry import symmetry_metric
from .types import IDENTITY, Perception, Report, unavailable, with_low_confidence
from .upright import deskew

METRIC_KEYS = ("v_taper", "symmetry", "arms", "abs")
MAX_SIDE = 1600


def _empty_report(p: Perception, reasons: tuple[str, ...], config: ScoringConfig) -> Report:
    metrics = {key: unavailable(key, *reasons) for key in METRIC_KEYS}
    return Report(
        version=__version__,
        image_size=(p.width, p.height),
        deskew_deg=0.0,
        quality=reasons,
        metrics=metrics,
        total=total_score(metrics, config),
        original_size=(p.width, p.height),
    )


def _as_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return tuple(tuple(float(v) for v in row) for row in np.asarray(matrix, dtype=np.float64))  # type: ignore[return-value]


def analyze_perception(p: Perception, config: ScoringConfig = DEFAULT_CONFIG) -> tuple[Report, Perception]:
    """Measure every metric; returns the report and the upright perception it used.

    Report geometry is in the upright perception's pixels; ``to_original``
    maps it back onto ``p``.
    """
    frame, frame_reasons = build_body_frame(p.landmarks)
    if frame is None:
        return _empty_report(p, frame_reasons, config), p
    sharp = sharpness(p, frame)
    upright, angle, matrix = deskew(p, frame)
    if angle:
        frame, frame_reasons = build_body_frame(upright.landmarks)
        if frame is None:
            return _empty_report(p, frame_reasons, config), p
    frame = replace(frame, rotation_index=torso_rotation(upright, frame))
    v_taper = v_taper_metric(upright, frame, config)
    arms = arms_metric(upright, frame, config)
    symmetry = symmetry_metric(upright, frame, arms, config)
    waist = v_taper.values.get("waist_width_px") if v_taper.available else None
    abdomen = abs_metric(upright, frame, waist, config)
    metrics = {"v_taper": v_taper, "symmetry": symmetry, "arms": arms, "abs": abdomen}
    quality = global_quality(upright, frame, angle, sharp)
    if R_CAMERA_TILTED in quality:
        metrics = {k: with_low_confidence(m, R_CAMERA_TILTED) for k, m in metrics.items()}
    to_original = _as_tuple(cv2.invertAffineTransform(matrix)) if matrix is not None else IDENTITY
    report = Report(
        version=__version__,
        image_size=(upright.width, upright.height),
        deskew_deg=angle,
        quality=quality,
        metrics=metrics,
        total=total_score(metrics, config),
        to_original=to_original,
        original_size=(p.width, p.height),
    )
    return report, upright


def read_image(path: str | Path | BinaryIO) -> np.ndarray:
    """Full-size BGR image from a path or file object, EXIF orientation applied
    (HEIC if pillow-heif is installed)."""
    from PIL import Image, ImageOps

    try:
        import pillow_heif  # optional

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def prepare_array(image: np.ndarray, max_side: int = MAX_SIDE) -> tuple[np.ndarray, float]:
    """Validate a BGR(A)/grey uint8 array and cap its size; returns (image, scale)."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("expected a uint8 image array (BGR, BGRA or greyscale)")
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image
    else:
        raise ValueError(f"unsupported image shape {image.shape}")
    h, w = bgr.shape[:2]
    factor = min(1.0, max_side / max(h, w))
    if factor < 1.0:
        bgr = cv2.resize(bgr, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(bgr), factor


def load_image(path: str | Path, max_side: int = MAX_SIDE) -> np.ndarray:
    """Read a photo as BGR, honouring EXIF orientation and capping its size."""
    return prepare_array(read_image(path), max_side)[0]


def analyze_image(
    image: str | Path | np.ndarray,
    config: ScoringConfig = DEFAULT_CONFIG,
    perceiver=None,
) -> tuple[Report, Perception | None]:
    """Run perception and analysis on a file path or a BGR array.

    ``report.to_original`` maps geometry onto the photo as given, before the
    size cap and deskew.
    """
    from .perception_mp import MediaPipePerceiver

    original = image if isinstance(image, np.ndarray) else read_image(image)
    bgr, factor = prepare_array(original)
    original_size = (int(original.shape[1]), int(original.shape[0]))
    owned = perceiver is None
    perceiver = perceiver or MediaPipePerceiver()
    try:
        p = perceiver.perceive(bgr)
    finally:
        if owned:
            perceiver.close()
    if p is None:
        blank = Perception(image=bgr, landmarks={}, person_mask=np.zeros(bgr.shape[:2], bool))
        report = _empty_report(blank, (R_POSE_NOT_FOUND,), config)
        scale_back = _as_tuple(np.asarray(IDENTITY, dtype=np.float64) / factor)
        return replace(report, to_original=scale_back, original_size=original_size), None
    report, upright = analyze_perception(p, config)
    to_original = _as_tuple(np.asarray(report.to_original, dtype=np.float64) / factor)
    return replace(report, to_original=to_original, original_size=original_size), upright
