"""MediaPipe adapter: pose landmarks, person silhouette, skin and clothes masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import models
from .silhouette import clean_mask
from .types import Landmark, Perception

POSE_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
)
# selfie_multiclass_256x256 categories
CATEGORY_BODY_SKIN = 2
CATEGORY_CLOTHES = 4
SEGMENT_PAD = 0.08
MIN_SEGMENT_SIDE = 32


def _options(path: Path):
    """Model options from bytes: MediaPipe cannot open non-ASCII paths on Windows."""
    from mediapipe.tasks.python import BaseOptions

    return BaseOptions(model_asset_buffer=Path(path).read_bytes())


class MediaPipePerceiver:
    """Runs MediaPipe Pose Landmarker and the multiclass selfie segmenter."""

    def __init__(self, pose_model: Path | None = None, segmenter_model: Path | None = None):
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        self._mp = mp
        pose_path = pose_model or models.require("pose")
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=_options(pose_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                output_segmentation_masks=True,
            )
        )
        seg_path = segmenter_model or models.require("segmenter")
        self._segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=_options(seg_path),
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True,
                output_confidence_masks=False,
            )
        )

    def __enter__(self) -> "MediaPipePerceiver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._pose.close()
        self._segmenter.close()

    def _image(self, rgb: np.ndarray):
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def perceive(self, bgr: np.ndarray) -> Perception | None:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.detect(self._image(rgb))
        if not result.pose_landmarks:
            return None
        landmarks = {
            name: Landmark(
                x=float(lm.x * w),
                y=float(lm.y * h),
                z=float(lm.z * w),
                visibility=float(lm.visibility if lm.visibility is not None else 1.0),
            )
            for name, lm in zip(POSE_NAMES, result.pose_landmarks[0])
        }
        # numpy_view() does not own its memory; copy before the next MediaPipe call.
        raw = np.array(np.squeeze(result.segmentation_masks[0].numpy_view()), dtype=np.float32)
        if raw.shape != (h, w):
            raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        person = clean_mask(raw > 0.5)
        skin, clothes = self._segment(rgb, person)
        return Perception(image=bgr, landmarks=landmarks, person_mask=person,
                          skin_mask=skin, clothes_mask=clothes)

    def _segment(self, rgb: np.ndarray, person: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Run the segmenter on a square crop around the person for more resolution.

        The segmenter resizes its input to 256x256, so the crop is padded to a
        true square (also at the photo edges) to avoid stretching the body.
        """
        ys, xs = np.nonzero(person)
        if ys.size == 0:
            return None, None
        h, w = person.shape
        side = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * SEGMENT_PAD))
        if side < MIN_SEGMENT_SIDE:
            return None, None
        cx, cy = (xs.max() + xs.min()) // 2, (ys.max() + ys.min()) // 2
        sx0, sy0 = cx - side // 2, cy - side // 2
        x0, y0 = max(0, sx0), max(0, sy0)
        x1, y1 = min(w, sx0 + side), min(h, sy0 + side)
        square = np.zeros((side, side, 3), dtype=np.uint8)
        square[y0 - sy0:y1 - sy0, x0 - sx0:x1 - sx0] = rgb[y0:y1, x0:x1]
        result = self._segmenter.segment(self._image(square))
        categories = np.array(np.squeeze(result.category_mask.numpy_view()), dtype=np.uint8)
        if categories.shape != (side, side):
            categories = cv2.resize(categories, (side, side), interpolation=cv2.INTER_NEAREST)
        full = np.zeros((h, w), dtype=np.uint8)
        full[y0:y1, x0:x1] = categories[y0 - sy0:y1 - sy0, x0 - sx0:x1 - sx0]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        near_person = cv2.dilate(person.astype(np.uint8), kernel).astype(bool)
        skin = (full == CATEGORY_BODY_SKIN) & near_person
        clothes = (full == CATEGORY_CLOTHES) & near_person
        return skin, clothes
