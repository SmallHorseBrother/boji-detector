"""Command line: `boji analyze`, `boji fetch-models`, `boji doctor`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, models
from .messages import describe
from .scoring import load_config
from .types import Report

LABELS = {
    "zh": {"v_taper": "倒三角（肩腰比）", "abs": "腹肌清晰度", "arms": "手臂围度", "symmetry": "左右对称", "total": "综合"},
    "en": {"v_taper": "V-taper", "abs": "Abs definition", "arms": "Arms", "symmetry": "Symmetry", "total": "Total"},
}
HEADLINE = {"v_taper": "ratio", "abs": "pack_count", "arms": "upper_ratio", "symmetry": "silhouette_iou"}


def _format(report: Report, name: str, lang: str) -> str:
    labels = LABELS[lang]
    lines = [f"== {name}"]
    total = report.total.get("score")
    lines.append(f"  {labels['total']}: {total:.1f}" if total is not None else f"  {labels['total']}: -")
    for key, metric in report.metrics.items():
        if metric.available:
            head = metric.values.get(HEADLINE[key])
            head_text = f"{HEADLINE[key]}={head:.3g}" if isinstance(head, (int, float)) else ""
            flag = "" if metric.status == "ok" else " (low confidence)"
            lines.append(f"  {labels[key]}: {metric.score:.1f}  {head_text}{flag}")
        else:
            lines.append(f"  {labels[key]}: -")
        lines += [f"      - {describe(code, lang)}" for code in metric.reasons]
    lines += [f"  ! {describe(code, lang)}" for code in report.quality]
    lines.append(f"  calibration: {report.total.get('calibration')}")
    return "\n".join(lines)


def _unique_stem(path: str, used: set[str]) -> str:
    stem = Path(path).stem or "photo"
    candidate, n = stem, 2
    while candidate in used:
        candidate, n = f"{stem}-{n}", n + 1
    used.add(candidate)
    return candidate


def cmd_analyze(args: argparse.Namespace) -> int:
    import cv2

    from .analyze import analyze_image
    from .perception_mp import MediaPipePerceiver
    from .render import render_overlay

    config = load_config(args.config)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    used: set[str] = set()
    failed = 0
    with MediaPipePerceiver() as perceiver:
        for path in args.images:
            try:
                report, perception = analyze_image(path, config, perceiver=perceiver)
            except Exception as exc:  # one unreadable photo must not stop a batch
                failed += 1
                results[path] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"{path}: {results[path]['error']}", file=sys.stderr)
                continue
            results[path] = report.to_dict()
            if args.format == "text":
                print(_format(report, path, args.lang))
            if out_dir:
                stem = _unique_stem(path, used)
                (out_dir / f"{stem}.json").write_text(
                    json.dumps(results[path], ensure_ascii=False, indent=2), encoding="utf-8")
                if perception is not None and not args.no_overlay:
                    overlay = render_overlay(perception, report, blur_face=not args.keep_face)
                    ok, buf = cv2.imencode(".png", overlay)
                    if ok:
                        buf.tofile(str(out_dir / f"{stem}.overlay.png"))
    if args.format == "json":
        json.dump(results, sys.stdout, ensure_ascii=False, indent=2)
        print()
    return 1 if failed else 0


def cmd_fetch(args: argparse.Namespace) -> int:
    for name in models.MODELS:
        path = models.fetch(name, force=args.force)
        print(f"{name}: {path}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"boji-detector {__version__}")
    print(f"models dir: {models.models_dir()}")
    states = models.status()
    for name, state in states.items():
        print(f"  {name}: {state}")
    try:
        import mediapipe

        print(f"mediapipe {mediapipe.__version__}")
    except ImportError:
        print("mediapipe: not installed")
        return 1
    return 0 if all(s == "ok" for s in states.values()) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boji", description="Explainable photo-based physique metrics")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analyse one or more photos")
    analyze.add_argument("images", nargs="+")
    analyze.add_argument("--out", help="folder for <name>.json and <name>.overlay.png")
    analyze.add_argument("--format", choices=("text", "json"), default="text")
    analyze.add_argument("--lang", choices=("zh", "en"), default="zh")
    analyze.add_argument("--config", help="JSON file overriding scoring curves and weights")
    analyze.add_argument("--no-overlay", action="store_true")
    analyze.add_argument("--keep-face", action="store_true", help="do not pixelate the face in overlays")
    analyze.set_defaults(func=cmd_analyze)

    fetch = sub.add_parser("fetch-models", help="download and verify model files")
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(func=cmd_fetch)

    doctor = sub.add_parser("doctor", help="check models and dependencies")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def _utf8_stdout() -> None:
    """Chinese text must survive non-UTF-8 consoles and pipes (cp1252, GBK)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except models.ModelMissingError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
