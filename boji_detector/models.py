"""Model assets: where they live, how to fetch them and how to verify them.

Model weights are not part of this repository. They are downloaded from
MediaPipe's public model storage and remain under their own model-card terms.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ENV_MODELS_DIR = "BOJI_MODELS_DIR"
ENV_BASE_URL = "BOJI_MODEL_BASE_URL"


@dataclass(frozen=True)
class ModelAsset:
    name: str
    filename: str
    url: str
    sha256: str


MODELS = {
    "pose": ModelAsset(
        name="pose",
        filename="pose_landmarker_heavy.task",
        url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        sha256="64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b",
    ),
    "segmenter": ModelAsset(
        name="segmenter",
        filename="selfie_multiclass_256x256.tflite",
        url="https://storage.googleapis.com/mediapipe-models/image_segmenter/"
            "selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
        sha256="c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
    ),
}


class ModelMissingError(RuntimeError):
    pass


def models_dir() -> Path:
    configured = os.environ.get(ENV_MODELS_DIR)
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "boji-detector" / "models"


def model_path(name: str) -> Path:
    return models_dir() / MODELS[name].filename


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(name: str) -> Path:
    path = model_path(name)
    if not path.exists():
        raise ModelMissingError(
            f"model '{name}' not found at {path}. Run `boji fetch-models` "
            f"or set {ENV_MODELS_DIR} to a folder that contains {MODELS[name].filename}."
        )
    return path


def download_url(asset: ModelAsset) -> str:
    base = os.environ.get(ENV_BASE_URL)
    return f"{base.rstrip('/')}/{asset.filename}" if base else asset.url


def fetch(name: str, force: bool = False) -> Path:
    """Download one model and verify its SHA-256; returns the local path."""
    asset = MODELS[name]
    target = model_path(name)
    if target.exists() and not force and sha256_of(target) == asset.sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(download_url(asset), timeout=60) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    actual = sha256_of(partial)
    if actual != asset.sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {asset.filename}: got {actual}")
    partial.replace(target)
    return target


def status() -> dict[str, str]:
    """'ok', 'missing' or 'checksum mismatch' for each model."""
    result = {}
    for name, asset in MODELS.items():
        path = model_path(name)
        if not path.exists():
            result[name] = "missing"
        elif sha256_of(path) != asset.sha256:
            result[name] = "checksum mismatch"
        else:
            result[name] = "ok"
    return result
