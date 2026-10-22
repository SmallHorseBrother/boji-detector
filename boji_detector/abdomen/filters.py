"""Illumination flattening and multi-scale valley (groove) responses.

Muscle separations show up as narrow dark valleys in log-luminance. Working in
log space makes a global exposure change an additive constant, which the
large-scale flattening removes. A valley's response is the scale-normalised
positive second derivative across it, so a Gaussian groove of depth A (log
units) responds with roughly 0.385 * A at its best scale.
"""

from __future__ import annotations

import cv2
import numpy as np

SIGMAS = (2.0, 3.0, 4.5, 6.0)
LOG_OFFSET = 4.0


def log_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.log(gray + LOG_OFFSET)


def masked_blur(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that only averages pixels inside mask (normalised convolution)."""
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(values.astype(np.float32) * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    safe = np.maximum(den, 1e-3)
    return np.where(den > 1e-3, num / safe, 0.0).astype(np.float32)


def flatten(log_l: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Remove shading slower than sigma, leaving local relief."""
    return np.where(mask, log_l - masked_blur(log_l, mask, sigma), 0.0).astype(np.float32)


def _second_derivative(g: np.ndarray, axis: int) -> np.ndarray:
    out = np.zeros_like(g)
    if axis == 0:
        out[1:-1] = g[2:] - 2.0 * g[1:-1] + g[:-2]
    else:
        out[:, 1:-1] = g[:, 2:] - 2.0 * g[:, 1:-1] + g[:, :-2]
    return out


def valley_responses(
    relief: np.ndarray, mask: np.ndarray, sigmas: tuple[float, ...] = SIGMAS
) -> tuple[np.ndarray, np.ndarray]:
    """(horizontal-groove, vertical-groove) responses, zero outside the mask interior."""
    horizontal = np.zeros_like(relief, dtype=np.float32)
    vertical = np.zeros_like(relief, dtype=np.float32)
    for s in sigmas:
        g = masked_blur(relief, mask, s)
        horizontal = np.maximum(horizontal, s * s * np.maximum(_second_derivative(g, 0), 0.0))
        vertical = np.maximum(vertical, s * s * np.maximum(_second_derivative(g, 1), 0.0))
    k = 2 * int(round(max(sigmas))) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    interior = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    return np.where(interior, horizontal, 0.0), np.where(interior, vertical, 0.0)
