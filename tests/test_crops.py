import numpy as np

from boji_detector.crops import MAX_SIDE, abdomen_crop


def _report(roi, status="ok", m=None):
    return {"metrics": {"abs": {"status": status, "geometry": {"roi": roi}}}, "to_original": m or [[1, 0, 0], [0, 1, 0]]}


def test_none_when_abs_unavailable_or_roi_missing():
    img = np.zeros((400, 300, 3), np.uint8)
    assert abdomen_crop(img, _report([10, 10, 100, 100], status="unavailable")) is None
    assert abdomen_crop(img, {"metrics": {}}) is None
    assert abdomen_crop(img, _report([10, 10, 12, 12])) is None


def test_crop_maps_back_to_the_original_and_is_bounded():
    import cv2

    img = np.zeros((2000, 1000, 3), np.uint8)
    img[900:1300, 300:700] = 255
    # Analysis image at half scale: the ROI (150,450)-(350,650) is (300,900)-(700,1300) originally.
    data = abdomen_crop(img, _report([150, 450, 350, 650], m=[[2, 0, 0], [0, 2, 0]]))
    crop = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert max(crop.shape[:2]) <= MAX_SIDE
    # Mostly the white abdomen box, with a margin of black around it.
    assert 0.5 < crop.mean() / 255 < 0.8
