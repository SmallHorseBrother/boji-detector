"""Synthetic front-relaxed torso photos with known ground truth.

The generator draws a stylised shirtless torso straight into a Perception
(image, landmarks and masks), so metric modules can be tested without the
MediaPipe models and with exact expected values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from boji_detector.types import Landmark, Perception

DELTOID_RADIUS = 46.0
JOINT_INSET = 50.0
SKIN_TINT = np.array([0.80, 0.90, 1.05])  # BGR multipliers on luminance


@dataclass(frozen=True)
class SynthSpec:
    width: int = 640
    height: int = 900
    cx: float = 320.0
    shoulder_y: float = 200.0
    torso_len: float = 440.0
    deltoid_half: float = 165.0
    chest_half: float = 128.0
    waist_half: float = 95.0
    hip_half: float = 112.0
    waist_t: float = 0.70
    arm_gap: float = 18.0
    upper_arm_half: tuple[float, float] = (34.0, 34.0)  # image-left, image-right
    forearm_half: float = 26.0
    upper_arm_len: float = 220.0
    forearm_len: float = 200.0
    shoulder_drop: tuple[float, float] = (0.0, 0.0)      # image-left, image-right
    abs_rows: int = 3
    groove_depth: float = 0.22
    pec_depth: float = 0.25
    light_gradient: float = 0.0
    exposure: float = 1.0
    pants_t: float = 0.95
    shirt: bool = False
    sleeves: bool = False
    arms_raised: bool = False
    noise: float = 2.0
    seed: int = 7


def torso_half(spec: SynthSpec, y: np.ndarray | float) -> np.ndarray:
    t = (np.asarray(y, dtype=np.float64) - spec.shoulder_y) / spec.torso_len
    knots_t = [-2.0, -0.12, -0.06, -0.02, 0.15, spec.waist_t, 1.0, 3.0]
    knots_h = [48.0, 55.0, 108.0, 120.0, spec.chest_half, spec.waist_half,
               spec.hip_half, spec.hip_half]
    return np.interp(t, knots_t, knots_h)


def abs_boundaries(spec: SynthSpec) -> list[float]:
    """Rows of the pec fold and every tendinous intersection, top to bottom."""
    pec = spec.shoulder_y + 0.30 * spec.torso_len
    if spec.abs_rows <= 0:
        return [pec]
    block = {2: 0.12, 3: 0.11, 4: 0.095}.get(spec.abs_rows, 0.1) * spec.torso_len
    return [pec + i * block for i in range(spec.abs_rows + 1)]


def rectus_half(spec: SynthSpec) -> float:
    return 0.58 * spec.waist_half


def _arm_chain(spec: SynthSpec, sign: int, drop: float) -> list[np.ndarray]:
    joint = np.array([spec.cx + sign * (spec.deltoid_half - JOINT_INSET), spec.shoulder_y + drop])
    if spec.arms_raised:
        ang = math.radians(10)
        elbow = joint + np.array([sign * spec.upper_arm_len * math.cos(ang),
                                  -spec.upper_arm_len * math.sin(ang)])
        wrist = elbow + np.array([0.0, -spec.forearm_len])
        return [joint, elbow, wrist]
    side = 0 if sign < 0 else 1
    half = spec.upper_arm_half[side]

    def hang(start: np.ndarray, length: float, own_half: float) -> np.ndarray:
        y = start[1] + length
        for _ in range(4):
            off = float(torso_half(spec, y)) + spec.arm_gap + own_half
            dx = abs(off - abs(start[0] - spec.cx))
            y = start[1] + math.sqrt(max(length ** 2 - dx ** 2, 1.0))
        return np.array([spec.cx + sign * off, y])

    elbow = hang(joint, spec.upper_arm_len, half)
    wrist = hang(elbow, spec.forearm_len, spec.forearm_half)
    return [joint, elbow, wrist]


def _capsule(mask: np.ndarray, a: np.ndarray, b: np.ndarray, half: float) -> None:
    pa = tuple(int(round(v)) for v in a)
    pb = tuple(int(round(v)) for v in b)
    cv2.line(mask, pa, pb, 1, thickness=int(round(2 * half)), lineType=cv2.LINE_8)
    for p in (pa, pb):
        cv2.circle(mask, p, int(round(half)), 1, thickness=-1)


def _gauss(d: np.ndarray, width: float) -> np.ndarray:
    return np.exp(-(d ** 2) / (2.0 * width ** 2))


def _window(d: np.ndarray, half: float, soft: float = 4.0) -> np.ndarray:
    return np.clip((half - d) / soft + 0.5, 0.0, 1.0)


def _abs_field(spec: SynthSpec, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    field = np.zeros_like(xx)
    dx = np.abs(xx - spec.cx)
    pec_y = abs_boundaries(spec)[0] - 0.0008 * (xx - spec.cx) ** 2
    pec_depth = spec.pec_depth if spec.abs_rows > 0 else spec.pec_depth * 0.5
    field -= pec_depth * _gauss(yy - pec_y, 5.0) * _window(dx, 0.62 * spec.deltoid_half, 12.0)
    if spec.abs_rows <= 0:
        return field
    bounds = abs_boundaries(spec)
    rh = rectus_half(spec)
    depth = spec.groove_depth
    for yb in bounds[1:]:
        field -= depth * _gauss(yy - yb, 4.0) * _window(dx, rh + 2.0)
    top, bottom = bounds[0] - 0.02 * spec.torso_len, bounds[-1] + 0.06 * spec.torso_len
    span = _window(np.abs(yy - (top + bottom) / 2.0), (bottom - top) / 2.0, 6.0)
    field -= depth * _gauss(dx, 4.0) * span
    lat_bottom = bounds[-1] + 0.04 * spec.torso_len
    lat_span = _window(np.abs(yy - (bounds[0] + lat_bottom) / 2.0), (lat_bottom - bounds[0]) / 2.0, 6.0)
    field -= 0.8 * depth * _gauss(dx - rh, 4.0) * lat_span
    dome = 0.07 * depth / 0.22
    for y0, y1 in zip(bounds, bounds[1:]):
        v = (yy - (y0 + y1) / 2.0) / ((y1 - y0) / 2.0)
        u = (dx - rh / 2.0) / (rh / 2.0)
        inside = (np.abs(u) < 1) & (np.abs(v) < 1)
        field += np.where(inside, dome * np.cos(np.pi / 2 * u) * np.cos(np.pi / 2 * v), 0.0)
    return field


def make_perception(spec: SynthSpec = SynthSpec()) -> Perception:
    h, w = spec.height, spec.width
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    half = torso_half(spec, yy[:, 0])
    torso = np.abs(xx - spec.cx) <= half[:, None]
    torso &= yy >= 0

    person = torso.astype(np.uint8)
    chains = {}
    for sign, side in ((-1, 0), (1, 1)):
        drop = spec.shoulder_drop[side]
        cap_c = (spec.cx + sign * (spec.deltoid_half - DELTOID_RADIUS),
                 spec.shoulder_y + 0.04 * spec.torso_len + drop)
        cv2.circle(person, tuple(int(round(v)) for v in cap_c), int(DELTOID_RADIUS), 1, -1)
        chain = _arm_chain(spec, sign, drop)
        _capsule(person, chain[0], chain[1], spec.upper_arm_half[side])
        _capsule(person, chain[1], chain[2], spec.forearm_half)
        chains[side] = chain
    person_mask = person.astype(bool)

    pants = torso & person_mask & (yy >= spec.shoulder_y + spec.pants_t * spec.torso_len)
    clothes = pants.copy()
    if spec.shirt:
        clothes |= torso & person_mask
    if spec.sleeves:
        sleeve = np.zeros((h, w), np.uint8)
        for side, chain in chains.items():
            end = chain[0] + 0.85 * (chain[1] - chain[0])
            _capsule(sleeve, chain[0], end, spec.upper_arm_half[side] + 4)
        clothes |= sleeve.astype(bool) & person_mask
    skin = person_mask & ~clothes

    rel = np.clip(np.abs(xx - spec.cx) / half[:, None], 0.0, 1.0)
    log_l = (math.log(150.0) - 0.45 * rel ** 4
             + spec.light_gradient * (xx - spec.cx) / (2.0 * spec.deltoid_half)
             + _abs_field(spec, xx, yy))
    lum = np.exp(log_l) * spec.exposure
    rng = np.random.default_rng(spec.seed)
    img = lum[..., None] * SKIN_TINT[None, None, :]
    background = np.array([92.0, 96.0, 100.0])
    img = np.where(person_mask[..., None], img, background)
    img = np.where(pants[..., None], np.array([70.0, 45.0, 35.0]), img)
    if spec.shirt or spec.sleeves:
        img = np.where((clothes & ~pants)[..., None], np.array([60.0, 60.0, 62.0]), img)
    img = img + rng.normal(0.0, spec.noise, img.shape)
    image = np.clip(img, 0, 255).astype(np.uint8)

    landmarks = {}
    for side, prefix in ((0, "right"), (1, "left")):  # subject's right is image-left
        joint, elbow, wrist = chains[side]
        for name, p in (("shoulder", joint), ("elbow", elbow), ("wrist", wrist)):
            landmarks[f"{prefix}_{name}"] = Landmark(float(p[0]), float(p[1]), 0.0, 0.99)
        sign = -1 if side == 0 else 1
        landmarks[f"{prefix}_hip"] = Landmark(
            spec.cx + sign * 0.72 * spec.hip_half, spec.shoulder_y + spec.torso_len, 0.0, 0.9)
    landmarks["nose"] = Landmark(spec.cx, spec.shoulder_y - 150, 0.0, 0.2)
    return Perception(image=image, landmarks=landmarks, person_mask=person_mask,
                      skin_mask=skin, clothes_mask=clothes)
