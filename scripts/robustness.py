"""Check how stable the metrics are when the same photo is perturbed.

    python scripts/robustness.py path/to/photo.jpg [more photos...]

Each photo is re-analysed after mirroring, exposure and gamma changes, JPEG
compression, downscaling and small rotations. A metric that moves a lot under
a perturbation that does not change the body is a measurement problem.
Photos stay local; nothing is written unless --out is given.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boji_detector.analyze import analyze_image, analyze_perception, load_image  # noqa: E402
from boji_detector.perception_mp import MediaPipePerceiver  # noqa: E402
from boji_detector.render import render_overlay  # noqa: E402


def _rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _gamma(img: np.ndarray, g: float) -> np.ndarray:
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _soften(img: np.ndarray, keep: float) -> np.ndarray:
    """Keep only part of the local shading, like a softer, more frontal light."""
    f = img.astype(np.float32)
    base = cv2.GaussianBlur(f, (0, 0), max(img.shape[:2]) / 40.0)
    return np.clip(base + keep * (f - base), 0, 255).astype(np.uint8)


VARIANTS = {
    "original": lambda im: im,
    "mirror": lambda im: cv2.flip(im, 1),
    "exposure x0.7": lambda im: np.clip(im.astype(np.float32) * 0.7, 0, 255).astype(np.uint8),
    "exposure x1.25": lambda im: np.clip(im.astype(np.float32) * 1.25, 0, 255).astype(np.uint8),
    "gamma 0.8": lambda im: _gamma(im, 0.8),
    "gamma 1.25": lambda im: _gamma(im, 1.25),
    "jpeg q60": lambda im: _jpeg(im, 60),
    "scale 0.6": lambda im: cv2.resize(im, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA),
    "rotate +7": lambda im: _rotate(im, 7.0),
    "rotate -7": lambda im: _rotate(im, -7.0),
}
# Not invariances: same body and masks, weaker shadows. The abs index is
# expected to drop; this shows how much lighting matters.
LIGHTING = {"softer light x0.6": 0.6, "softer light x0.35": 0.35}
COLUMNS = (
    ("v_taper", "ratio"), ("arms", "upper_ratio"), ("symmetry", "silhouette_iou"),
    ("abs", "pack_count"), ("abs", "definition_index"), ("abs", "mean_enclosure"),
)


def _cell(report, key: str, field: str) -> str:
    metric = report.metrics[key]
    if not metric.available:
        return "n/a"
    value = metric.values.get(field)
    text = f"{value:.3f}" if isinstance(value, float) else str(value)
    return text + ("*" if metric.status != "ok" else "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--out", help="write overlays for every variant here")
    args = parser.parse_args()
    header = ["variant"] + [f"{k}.{f}" for k, f in COLUMNS] + ["total"]
    with MediaPipePerceiver() as perceiver:
        for path in args.images:
            base = load_image(path)
            print(f"\n== {path}")
            print(" | ".join(f"{h:>22s}" if i == 0 else f"{h:>14s}" for i, h in enumerate(header)))
            original = perceiver.perceive(base)
            runs = [(name, lambda fn=fn: analyze_image(fn(base), perceiver=perceiver)) for name, fn in VARIANTS.items()]
            if original is not None:
                runs += [
                    (name, lambda keep=keep: analyze_perception(replace(original, image=_soften(base, keep))))
                    for name, keep in LIGHTING.items()
                ]
            for name, run in runs:
                report, perception = run()
                total = report.total.get("score")
                cells = [_cell(report, k, f) for k, f in COLUMNS]
                cells.append(f"{total:.1f}" if total is not None else "n/a")
                print(f"{name:>22s} | " + " | ".join(f"{c:>14s}" for c in cells))
                flags = sorted({r for m in report.metrics.values() for r in m.reasons} | set(report.quality))
                if flags:
                    print(f"{'':>22s}   flags: {', '.join(flags)}")
                if args.out and perception is not None:
                    out = Path(args.out)
                    out.mkdir(parents=True, exist_ok=True)
                    target = out / f"{Path(path).stem}_{name.replace(' ', '_')}.png"
                    cv2.imencode(".png", render_overlay(perception, report))[1].tofile(str(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
