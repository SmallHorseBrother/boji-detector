"""Human-readable text for reason codes (Chinese and English)."""

from __future__ import annotations

MESSAGES: dict[str, tuple[str, str]] = {
    "pose_not_found": ("未检测到人体姿态，请拍摄正面上半身照片", "No body pose found; use a front upper-body photo"),
    "hips_estimated": ("髋部不在画面内，躯干比例为估算", "Hips not visible; torso proportions are estimated"),
    "arms_raised": ("手臂未自然下垂，肩宽不可测", "Arms are raised; shoulder width cannot be measured"),
    "arms_touching_torso": ("手臂贴住躯干，腰宽不可测，请让手臂离身体一拳", "Arms touch the torso; keep a fist-width gap to measure the waist"),
    "elbows_not_visible": ("肘部不在画面内", "Elbows are not visible"),
    "shoulders_not_found": ("未找到肩部轮廓", "Shoulder outline not found"),
    "waist_at_image_edge": ("最窄腰线贴近画面底边，可能被裁掉", "Narrowest waist is at the image edge and may be cut off"),
    "body_rotated": ("身体没有正对镜头", "Body is turned away from the camera"),
    "sleeves_cover_arms": ("袖子遮住上臂", "Sleeves cover the upper arms"),
    "arms_not_visible": ("手臂不在画面内", "Arms are not visible"),
    "one_arm_only": ("只测到一侧手臂", "Only one arm could be measured"),
    "abdomen_covered": ("腹部被衣物遮挡", "Abdomen is covered by clothing"),
    "waist_estimated": ("腰宽为估算值", "Waist width is estimated"),
    "low_resolution": ("腹部像素太少，请靠近或用更高分辨率", "Abdomen is too small in the photo; move closer or use a higher resolution"),
    "overexposed": ("腹部过曝", "Abdomen is overexposed"),
    "underexposed": ("腹部过暗", "Abdomen is underexposed"),
    "strong_side_light": ("侧光很强，腹肌结果可比性下降", "Strong side light; abdominal results are less comparable"),
    "linea_alba_not_found": ("未找到腹白线", "Linea alba not found"),
    "abdomen_not_measurable": ("无法测量腹部", "Abdomen could not be measured"),
    "torso_too_small": ("人在画面中太小", "Person is too small in the frame"),
    "blurry": ("照片模糊", "Photo is blurry"),
    "camera_tilted": ("相机倾斜过大", "Camera is tilted too far"),
    "shoulders_cut_off": ("肩部被画面边缘截断，肩宽不可测", "Shoulders are cut off by the photo edge"),
    "waist_out_of_frame": ("腰部不在画面内，请拍到裤腰", "Waist is outside the photo; include the waistband"),
    "hips_out_of_frame": ("髋部不在画面内，测不了腰臀比，请拍到大腿根", "Hips are outside the photo; include the top of the thighs"),
    "hips_blocked_by_arms": ("手或手臂挡住了髋部轮廓，请让手臂离身体一拳", "Hands or arms block the hip outline; keep a fist-width gap"),
    "hips_at_image_edge": ("最宽的髋部贴近画面底边，可能被裁掉", "Widest hip row is at the image edge and may be cut off"),
    "arm_cut_off": ("手臂被画面边缘截断", "An arm is cut off by the photo edge"),
    "body_cut_off": ("身体大部分被画面边缘截断，对称性只看了部分区域", "Much of the body touches the photo edge; symmetry uses part of it"),
    "abdomen_out_of_frame": ("腹部不完全在画面内", "The abdomen is not fully inside the photo"),
    "skin_mask_unavailable": ("皮肤分割不可用，使用整体轮廓代替", "Skin segmentation unavailable; using the silhouette instead"),
}


def describe(code: str, lang: str = "zh") -> str:
    zh, en = MESSAGES.get(code, (code, code))
    return zh if lang == "zh" else en
