"""Mapping raw measurements to 0-100 scores.

Every curve here is a provisional, *uncalibrated* placeholder. Raw values are
always reported next to scores; calibration against expert ratings will
replace these numbers without changing the measurements.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .types import Metric

CALIBRATION = "uncalibrated-v0"
PROFILES = ("male", "female")
# A total needs metrics covering at least this share of the weights; one
# metric alone (say, symmetry on a head-and-shoulders portrait) is not a total.
MIN_TOTAL_COVERAGE = 0.5


@dataclass(frozen=True)
class Curve:
    """Logistic curve: score 50 at ``mid``; ``scale`` sets the slope."""

    mid: float
    scale: float


@dataclass(frozen=True)
class ScoringConfig:
    # "male": abs scored by separated blocks (packs); "female": by the vertical
    # lines (linea alba + lateral rectus edges), and the waist/hip ratio counts.
    profile: str = "male"
    v_taper: Curve = Curve(mid=1.62, scale=0.09)          # bideltoid / waist width
    arms: Curve = Curve(mid=0.36, scale=0.03)             # upper-arm width / length
    abs: Curve = Curve(mid=0.45, scale=0.08)              # definition index
    abs_lines: Curve = Curve(mid=0.009, scale=0.004)      # vertical line index (female profile)
    waist_hip: Curve = Curve(mid=-0.74, scale=0.03)       # minus waist / hip width
    symmetry_iou: Curve = Curve(mid=0.83, scale=0.04)     # mirrored torso IoU (armpits to waist)
    symmetry_balance: Curve = Curve(mid=0.88, scale=0.025)  # thinner / thicker arm
    symmetry_tilt: Curve = Curve(mid=-4.0, scale=1.2)     # minus |shoulder tilt| in degrees
    weights: Mapping[str, float] = field(
        default_factory=lambda: {"abs": 0.40, "v_taper": 0.30, "arms": 0.15, "symmetry": 0.15}
    )


DEFAULT_CONFIG = ScoringConfig()

# Women: the "vest line" instead of packs, a curvy waist/hip line and a modest
# shoulder/waist taper; arm size is measured but does not count towards the total.
FEMALE_CONFIG = ScoringConfig(
    profile="female",
    v_taper=Curve(mid=1.55, scale=0.08),
    weights={"abs": 0.35, "waist_hip": 0.35, "v_taper": 0.15, "symmetry": 0.15},
)


def profile_config(profile: str | None) -> ScoringConfig:
    if profile in (None, "", "male"):
        return DEFAULT_CONFIG
    if profile == "female":
        return FEMALE_CONFIG
    raise ValueError(f"unknown profile: {profile}")


def logistic_score(value: float, curve: Curve) -> float:
    z = (value - curve.mid) / curve.scale
    z = max(-60.0, min(60.0, z))
    return 100.0 / (1.0 + math.exp(-z))


def total_score(metrics: Mapping[str, Metric], config: ScoringConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    used = {
        key: weight for key, weight in config.weights.items()
        if key in metrics and metrics[key].available and metrics[key].score is not None
    }
    missing = sorted(set(config.weights) - set(used))
    weight_sum = sum(used.values())
    coverage = weight_sum / sum(config.weights.values())
    if weight_sum <= 0 or coverage < MIN_TOTAL_COVERAGE:
        return {"score": None, "weights_used": {}, "missing": missing, "calibration": CALIBRATION}
    normalized = {k: w / weight_sum for k, w in used.items()}
    score = sum(metrics[k].score * w for k, w in normalized.items())
    return {
        "score": score,
        "weights_used": normalized,
        "missing": missing,
        "calibration": CALIBRATION,
    }


def load_config(path: str | Path | None, base: ScoringConfig = DEFAULT_CONFIG) -> ScoringConfig:
    """Override curves and weights from a JSON file shaped like ScoringConfig."""
    if path is None:
        return base
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    updates: dict[str, Any] = {}
    for key, value in data.items():
        if key == "profile":
            updates["profile"] = profile_config(str(value)).profile
        elif key == "weights":
            weights = {str(k): float(v) for k, v in value.items()}
            if any(not math.isfinite(v) or v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
                raise ValueError("weights must be non-negative and not all zero")
            updates["weights"] = weights
        elif isinstance(getattr(DEFAULT_CONFIG, key, None), Curve):
            curve = Curve(mid=float(value["mid"]), scale=float(value["scale"]))
            if not (math.isfinite(curve.mid) and math.isfinite(curve.scale) and curve.scale > 0):
                raise ValueError(f"{key}: mid must be finite and scale positive")
            updates[key] = curve
        else:
            raise ValueError(f"unknown scoring key: {key}")
    return replace(base, **updates)
