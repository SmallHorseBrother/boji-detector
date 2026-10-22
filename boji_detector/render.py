"""Annotated overlay: what was measured, where, and the resulting scores."""

from __future__ import annotations

import cv2
import numpy as np

from .scoring import CALIBRATION
from .types import Perception, Report

FACE_POINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "mouth_left", "mouth_right")
YELLOW, RED, CYAN, BLUE, ORANGE, GREEN, WHITE = (
    (0, 220, 255), (40, 40, 230), (230, 210, 40), (230, 120, 30), (0, 140, 255), (80, 200, 60), (255, 255, 255),
)


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


FACE_BLOCKS = 7               # pixel blocks across the head: too coarse to recognise a face
NECK_ABOVE_SHOULDERS = 0.15   # head region ends this far above the shoulders (x joint distance)


def _box_union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    arr = np.array(boxes)
    return int(arr[:, 0].min()), int(arr[:, 1].min()), int(arr[:, 2].max()), int(arr[:, 3].max())


def head_box(p: Perception, min_visibility: float = 0.3) -> tuple[int, int, int, int] | None:
    """Box around the head: from face landmarks and from the silhouette above the shoulders."""
    h, w = p.height, p.width
    boxes = []
    pts = [(lm.x, lm.y) for name in FACE_POINTS
           if (lm := p.landmarks.get(name)) is not None and lm.visibility >= min_visibility]
    if len(pts) >= 2:
        arr = np.array(pts)
        centre = arr.mean(axis=0)
        radius = max(20.0, 1.6 * float(np.max(np.linalg.norm(arr - centre, axis=1))))
        boxes.append((int(centre[0] - radius), int(centre[1] - 1.3 * radius),
                      int(centre[0] + radius), int(centre[1] + radius)))
    left, right = p.landmarks.get("left_shoulder"), p.landmarks.get("right_shoulder")
    if left is not None and right is not None:
        span = float(np.hypot(left.x - right.x, left.y - right.y))
        cut = int(min(left.y, right.y) - NECK_ABOVE_SHOULDERS * span)
        if cut > 0:
            ys, xs = np.nonzero(p.person_mask[:cut])
            if xs.size:
                pad = int(0.1 * (xs.max() - xs.min() + 1)) + 2
                boxes.append((int(xs.min()) - pad, int(ys.min()) - pad, int(xs.max()) + pad, cut))
    box = _box_union(boxes)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    return (x0, y0, x1, y1) if x1 - x0 >= 4 and y1 - y0 >= 4 else None


def pixelate_face(image: np.ndarray, p: Perception) -> np.ndarray:
    """Copy of image with the head pixelated; never modifies the input."""
    image = image.copy()
    box = head_box(p)
    if box is None:
        return image
    x0, y0, x1, y1 = box
    block = max(1, max(x1 - x0, y1 - y0) // FACE_BLOCKS)
    region = image[y0:y1, x0:x1]
    small = cv2.resize(region, (max(1, -(-(x1 - x0) // block)), max(1, -(-(y1 - y0) // block))),
                       interpolation=cv2.INTER_AREA)
    image[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return image


def _enclosure_colour(value: float) -> tuple[int, int, int]:
    v = float(np.clip(value, 0.0, 1.0))
    return int(40 + 40 * v), int(60 + 160 * v), int(230 - 170 * v)


def _draw_abs(canvas: np.ndarray, geometry: dict, t: int) -> None:
    overlay = canvas.copy()
    for block in geometry.get("blocks", []):
        quad = np.array([_pt(p) for p in block["outline"]], dtype=np.int32)
        cv2.fillPoly(overlay, [quad], _enclosure_colour(block["enclosure"]))
    cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, dst=canvas)
    for lines in geometry.get("grooves", {}).values():
        for line in lines:
            cv2.polylines(canvas, [np.array([_pt(p) for p in line], np.int32)], False, ORANGE, t)
    for line in geometry.get("lateral_lines", {}).values():
        cv2.line(canvas, _pt(line[0]), _pt(line[1]), CYAN, max(1, t // 2))
    if "linea_alba" in geometry:
        a, b = geometry["linea_alba"]
        cv2.line(canvas, _pt(a), _pt(b), BLUE, t)


def _panel_lines(report: Report) -> list[str]:
    m = report.metrics
    total = report.total.get("score")
    lines = [f"boji-detector {report.version} ({CALIBRATION})",
             f"TOTAL {total:.0f}" if total is not None else "TOTAL n/a"]

    def score(key: str) -> str:
        metric = m[key]
        if not metric.available:
            return "n/a (" + ", ".join(metric.reasons) + ")"
        flag = "" if metric.status == "ok" else " *"
        return f"{metric.score:.0f}{flag}"

    v = m["v_taper"].values
    lines.append(f"V-taper {score('v_taper')}" + (f"  ratio {v['ratio']:.2f}" if "ratio" in v else ""))
    a = m["abs"].values
    lines.append(f"Abs {score('abs')}" + (f"  packs {a['pack_count']}  idx {a['definition_index']:.2f}"
                                          if "pack_count" in a else ""))
    r = m["arms"].values
    lines.append(f"Arms {score('arms')}" + (f"  ratio {r['upper_ratio']:.2f}" if "upper_ratio" in r else ""))
    s = m["symmetry"].values
    lines.append(f"Symmetry {score('symmetry')}" + (f"  IoU {s['silhouette_iou']:.2f}" if s else ""))
    if report.quality:
        lines.append("! " + ", ".join(report.quality))
    return lines


def _draw_panel(canvas: np.ndarray, lines: list[str]) -> None:
    h, w = canvas.shape[:2]
    scale = max(0.4, min(w, h) / 1100)
    thick = max(1, int(round(scale * 1.6)))
    pad = int(10 * scale) + 4
    sizes = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0] for s in lines]
    line_h = max(s[1] for s in sizes) + pad
    box_w = max(s[0] for s in sizes) + 2 * pad
    box_h = line_h * len(lines) + pad
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (min(w - 1, box_w), min(h - 1, box_h)), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, dst=canvas)
    for i, text in enumerate(lines):
        y = pad + line_h * (i + 1) - pad // 2
        cv2.putText(canvas, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thick, cv2.LINE_AA)


def render_overlay(p: Perception, report: Report, blur_face: bool = True) -> np.ndarray:
    canvas = pixelate_face(p.image, p) if blur_face else p.image.copy()
    t = max(1, int(round(min(canvas.shape[:2]) / 450)))
    contours, _ = cv2.findContours(p.person_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, GREEN, t)
    metrics = report.metrics
    if metrics["abs"].available:
        _draw_abs(canvas, dict(metrics["abs"].geometry), t)
    for line in metrics["arms"].geometry.get("width_lines", []):
        cv2.line(canvas, _pt(line[0]), _pt(line[1]), CYAN, t)
    g = metrics["v_taper"].geometry
    if "shoulder_line" in g:
        cv2.line(canvas, _pt(g["shoulder_line"][0]), _pt(g["shoulder_line"][1]), YELLOW, t + 1)
        cv2.line(canvas, _pt(g["waist_line"][0]), _pt(g["waist_line"][1]), RED, t + 1)
    _draw_panel(canvas, _panel_lines(report))
    return canvas
