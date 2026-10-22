# boji-detector · 薄肌检测器

用一张正面照片，给出**可解释**的体型指标：倒三角（肩腰比）、左右对称、手臂围度、腹肌清晰度（定位到每一块）。
每个分数都带着原始测量值、标注图上的测量位置，以及"这次测量可不可信"的原因说明。

本项目是[北清薄肌俱乐部开源指南](https://github.com/SmallHorseBrother/PKU-THU-Lean-Club)的配套工具：指南里"同样的光线拍正面照"这一步，可以用它把照片变成可以对比的数字。

> 研究和自测用途，不是医疗诊断或体成分评估。
> 分数曲线**还没有**用人工评分校准（`calibration: uncalibrated-v0`），请主要看原始测量值，以及同一拍摄条件下的前后变化。

[English summary below](#english)

## 测什么

| 指标 | 原始测量 | 怎么测 | 受光照影响 |
|---|---|---|---|
| 倒三角 `v_taper` | `ratio` = 三角肌外缘宽 / 最窄腰宽 | 人体轮廓逐行宽度；自动剔除手臂贴身的行 | 否 |
| 左右对称 `symmetry` | 轮廓镜像 IoU、肩线倾斜角、左右上臂粗细比 | 以关节中线为轴镜像轮廓，轴位置在小范围内搜索 | 否 |
| 手臂 `arms` | `upper_ratio` = 上臂宽 / 上臂长（肩→肘） | 沿上臂骨线做 4 条垂线量宽度；内侧贴住躯干时用外侧半宽镜像 | 否 |
| 腹肌 `abs` | 块数、每块封闭度、沟深、`definition_index` | 找腹白线、腱划、腹直肌外缘，逐块测四条边有多少被"沟"围住 | **是**，见[已知局限](#已知局限) |

总分是各项分数的加权平均（默认 腹肌 0.40 / 倒三角 0.30 / 手臂 0.15 / 对称 0.15），不可测的项会被剔除后重新归一化。权重和曲线都可以用 `--config` 覆盖。

算法细节见下方[方法](#方法)。

## 安装

需要 Python 3.10–3.12。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"          # iPhone HEIC 照片再加 heif： pip install -e ".[dev,heif]"
boji fetch-models     # 下载并校验 MediaPipe 模型（约 47 MB）
boji doctor
```

模型默认放在 `~/.cache/boji-detector/models`，可用 `BOJI_MODELS_DIR` 指定目录。
如果访问 Google 存储较慢，把两个模型文件放到自己的镜像上，再设置 `BOJI_MODEL_BASE_URL`；下载后会按 SHA-256 校验。

## 使用

```bash
boji analyze photo.jpg --out out/          # 中文文字报告 + out/photo.json + out/photo.overlay.png
boji analyze a.jpg b.jpg --format json     # 批量，JSON 输出
boji analyze photo.jpg --lang en --keep-face
```

```python
from boji_detector import analyze_image
report, perception = analyze_image("photo.jpg")
print(report.metrics["abs"].values["pack_count"], report.total["score"])
```

标注图默认把头部打成粗马赛克（找不到五官时按肩部以上的轮廓打码）；黄色线是肩宽，红色线是最窄腰，青色短线是手臂宽度，蓝线是腹白线，橙线是腱划，色块是识别到的每一块腹肌（越绿封闭度越高）。

### 拍摄建议

- 正面站立，手臂自然下垂、离身体约一拳（贴身会让腰宽不可测）。
- 上身裸露到裤腰，裤腰不要盖住肚脐以下。
- 相机与胸口同高、正对身体，人占画面高度一半以上。
- 光源在头顶前上方；做前后对比时用**同一地点、同一光线、同一距离**。

## 输出

```jsonc
{
  "version": "0.1.0",
  "deskew_deg": 0.0,
  "quality": [],                               // 整张照片级别的问题，如 blurry / hips_estimated
  "metrics": {
    "abs": {
      "status": "ok",                           // ok | low_confidence | unavailable
      "score": 99.7,
      "values": { "pack_count": 5, "definition_index": 0.91, "mean_enclosure": 0.94,
                  "open_angle_deg": 22.8, "alba_found": true, "...": "..." },
      "reasons": [],                            // 代码见 boji_detector/messages.py
      "geometry": { "linea_alba": [[x, y], [x, y]], "grooves": {}, "blocks": [] }
    },
    "v_taper": {}, "arms": {}, "symmetry": {}
  },
  "total": { "score": 94.5, "weights_used": {}, "missing": [], "calibration": "uncalibrated-v0" }
}
```

`geometry` 是**分析图**的像素坐标（长边缩到 ≤1600 像素、并已摆正）。要画在用户原图上，用报告里的 `to_original`（2×3 仿射矩阵）换算：

```python
from boji_detector.types import map_points
points = map_points(report.metrics["v_taper"].geometry["shoulder_line"], report.to_original)
```

`original_size` 是原图宽高。

## 方法

### 感知

MediaPipe Pose Landmarker（heavy）给出 33 个关键点和人体轮廓；selfie_multiclass_256x256 在人物外接正方形裁剪上运行，分出身体皮肤和衣物。指标代码只依赖 `Perception` 数据结构，换模型不影响指标；关键点按**图像左右**分组，镜子自拍与正面照一样处理。躯干轴（肩中点→髋中点）倾斜 1°–25° 时，整图连同掩码和关键点先旋转摆正。

### 倒三角与对称（只用轮廓）

- 肩宽：肩关节上方 0.12 到下方 0.30 倍上臂长之间，逐行取包含躯干中线的轮廓段宽度，中值平滑后取最大。
- 腰宽：躯干轴 45%–95% 段逐行取宽度；某侧边缘越过该行手臂骨线（肩→肘→腕插值）说明手臂贴身，此行作废；有效行不足 25% 时不可测。取平滑后的最小值。
- 身体转向：腰段轮廓中心偏离关节中线的程度（相对半腰宽）> 0.18 记 `body_rotated`。MediaPipe 的深度 z 在静态照片上噪声太大，不采用。
- 对称：以关节中线为轴镜像轮廓，在 ±4% 肩距内平移轴找最大 IoU；加上肩线倾斜角和左右上臂宽度比。

### 手臂

沿肩→肘骨线 45%–75% 处做 4 条垂线，从骨线向两侧走到轮廓边界；内侧比外侧长 1.6 倍以上视为贴住躯干，改用外侧半宽镜像。`upper_ratio = 宽度中位数 / 上臂长`。垂线上 30% 以上是衣物时判为袖子遮挡。

### 腹肌

1. **工作区**：裁出躯干轴 24%–92%、中线 ±0.45 倍腰宽，缩放到腰宽 = 240 像素，之后所有尺度都是腰宽的比例。
2. **去光照**：`L = log(灰度 + 4)`，整体曝光变化变成常数；减去皮肤掩码内的大尺度模糊（σ = 0.2 倍腰宽）得到 `relief`，去掉侧光和柱面明暗。
3. **谷线响应**：σ ∈ {2, 3, 4.5, 6}，`σ² · max(∂²relief/∂y², 0)` 为横沟、`∂x²` 为竖沟。深度 A 的高斯沟响应约 0.385 A。
4. **结构**：腹白线 = 中线附近竖沟列均值的峰，分带追踪后直线拟合；腱划 = 左右分别在距腹白线 0.06–0.27 倍腰宽的列上找横沟行均值的峰（高度 ≥ 0.010、间距 ≥ 0.09 倍腰宽），允许左右错位；最上面一条通常是胸肌下沿，即第一排的上边界；腹直肌外缘 = 0.18–0.42 倍腰宽内的竖沟峰，找不到用 0.30。
5. **逐块**：相邻腱划之间、高 0.10–0.45 倍腰宽、皮肤覆盖 ≥ 70% 的区域为一块。封闭度 = 上下内外四条边上响应 ≥ 0.010 的比例（按边长加权），`open_angle_deg = 360 × (1 − 封闭度)`；另测沟深与隆起。
6. `definition_index = 平均(0.65 × 封闭度 + 0.35 × min(1, 沟深/0.05)) × min(1, 0.7 + 0.1 × 每侧块数)`；没有块时为 `0.35 × min(1, 最强沟/0.05)`。
7. **光照诊断**：过曝（≥250 的像素 > 8%）、过暗（中位灰度 < 40）、强侧光（腹白线两侧对数亮度差 > 0.35），并输出 `texture_contrast` 供日后校准把光照硬度当协变量。

### 计分

每项为 `100 / (1 + exp(−(x − mid) / scale))`。当前曲线（占位，待校准）：

| 指标 | mid | scale | 参考 |
|---|---|---|---|
| v_taper | 1.62 | 0.09 | 普通男性约 1.5，健身人群 1.7–1.9，顶尖体型 2.0+ |
| arms | 0.36 | 0.03 | |
| abs | 0.45 | 0.08 | |
| 对称：IoU / 臂比 / 肩倾斜 | 0.86 / 0.88 / −4° | 0.025 / 0.025 / 1.2 | |

校准计划：收集教练对同一批照片的分项评分，用保序回归或分段样条拟合"原始测量 → 分数"，公开交叉验证相关系数；届时只换 `calibration` 版本，测量不变。

### 扩展一个新区域

1. 新建模块，导出 `xxx_metric(p: Perception, frame: BodyFrame, config) -> Metric`。
2. 用关键点定义区域，尺寸一律写成腰宽或肢体长度的比例。
3. `values` 放原始测量，`geometry` 放原图坐标，`reasons` 用 `R_` 常量并在 `messages.py` 补中英文（有测试检查）。
4. 在 `tests/synth.py` 里画出该结构，按已知真值写测试。
5. 在 `analyze.py` 和 `ScoringConfig.weights` 登记。

## 与 T-Muscle Detector 的关系

本项目受 [T-Muscle Detector](https://github.com/gsr-04/T-muscle-detector)（MIT）启发："一块腹肌被沟围得越完整越清晰"这一思路来自它的开放角目标函数，`open_angle_deg = 360 × (1 − 封闭度)` 与之对应。代码是重写的，没有复用其实现。主要区别：

- 只用 MediaPipe 的两个模型（姿态 + 多类分割），不依赖 LIP/ATR 人体解析权重（这些权重基于学术数据集训练，商用条款需另行核实）。
- 腹肌：不再假设固定 6 块，可识别 4/6/8 块及左右不对称；使用对数亮度 + 去大尺度阴影 + 多尺度谷线响应，整体曝光变化不影响结果；搜索区以胸肌下沿为上界，不把胸肌算进腹部。
- 每个指标都有质量门禁和原因码，测不准时明确说"测不了"而不是给一个数。
- 单进程、无中间文件，单张约 2 秒（CPU）。

## 已知局限

1. **腹肌指标依赖阴影。** 同一个身体，光越柔和，沟越浅，`definition_index` 越低。在样例照片上，只保留 60% 的局部明暗时指数从 0.91 降到 0.88，保留 35% 时降到 0.75（块数不变）。做比较时请保持光线一致；比赛舞台的统一顶光反而让同场选手之间可比。
2. 只支持**正面放松**姿势。背面、侧面和健美规定动作在路线图里。
3. 分数曲线未校准。原始测量是稳定的（见下方鲁棒性），但"多少分算好"需要人工评分数据来定。
4. 单人照片；极端姿势、遮挡、宽松衣物会让 MediaPipe 出错，此时指标会标为 `unavailable` 或 `low_confidence`。
5. 被画面边缘截断的部位不做测量：肩部、手臂、腰部、腹部被截断时分别给出 `shoulders_cut_off` / `arm_cut_off` / `waist_out_of_frame` / `abdomen_out_of_frame`，可测项权重不足一半时不给总分。
6. 仓库不包含任何人像照片。自动化测试使用合成躯干图（`tests/synth.py`），真实照片用 `scripts/robustness.py` 在本地检查。

### 鲁棒性（同一张照片做扰动，`scripts/robustness.py`）

| 扰动 | 肩腰比 | 上臂比 | 腹肌块数 | definition_index |
|---|---|---|---|---|
| 原图 | 2.158 | 0.435 | 5 | 0.911 |
| 镜像 / 曝光 ×0.7、×1.25 / gamma 0.8、1.25 / JPEG q60 / 缩小 0.6 / 旋转 ±7° | 2.158–2.192 | 0.424–0.459 | 5 | 0.909–0.917 |

## 路线图

- 标注工具 + 教练评分数据集 → 非线性校准，并公开与人工评分的相关系数。
- 更多区域：胸肌、三角肌分离度、股四头（正面/侧面）、背阔（背面展背）、小腿。
- 健美规定动作识别（前展肱二头、前展背阔、侧展胸、背展肱二头……）与按组别加权。
- 输入身高 → 换算厘米。
- 进度对比模式：同一人多张照片对齐后逐项对比。

## 隐私

所有计算都在本地完成，不上传照片；只有指定 `--out` 时才写文件。标注图默认给头部打粗马赛克。请不要把任何人像照片提交到仓库（`.gitignore` 已忽略所有图片格式以及 `samples/`、`out/`）。

## 许可

代码采用 MIT 许可（见 `LICENSE`），只覆盖本仓库源代码。

- **模型权重不随仓库分发。** `boji fetch-models` 从 Google 公开的 MediaPipe 模型存储下载 `pose_landmarker_heavy.task`（Pose Landmarker heavy）和 `selfie_multiclass_256x256.tflite`（Selfie Multiclass Segmentation），它们适用各自模型卡的条款；再分发或商用前请自行核对。
- 依赖库：MediaPipe（Apache-2.0）、OpenCV（Apache-2.0）、NumPy（BSD-3-Clause）、SciPy（BSD-3-Clause）、Pillow（MIT-CMU）。
- 先前工作：用"被沟围住的程度"评价一块腹肌的思路来自 [T-Muscle Detector](https://github.com/gsr-04/T-muscle-detector)（MIT），本项目的封闭度与其开放角对应；未包含其任何代码。

---

## English

**boji-detector** ("boji" is 薄肌, lean muscle) turns one front-facing photo into explainable physique metrics: V-taper (bideltoid / waist width), left-right symmetry, arm fullness, and abdominal definition with per-block localisation. Every score ships with its raw measurements, the pixel geometry of what was measured, and reason codes when a measurement is unreliable or impossible.

- Install: `pip install -e ".[dev]" && boji fetch-models && boji doctor`
- Run: `boji analyze photo.jpg --out out/ --lang en`
- Method: see the 方法 section above (Chinese).
- Scores are **uncalibrated placeholders**; rely on raw values and before/after comparisons under identical lighting.
- The abdominal metric depends on lighting (see Known limitations above); the silhouette metrics do not.
- Inspired by T-Muscle Detector's open-angle idea; independent implementation using only MediaPipe models.
- Research and self-tracking only; not medical advice. MIT licensed code; model weights follow their own terms.
