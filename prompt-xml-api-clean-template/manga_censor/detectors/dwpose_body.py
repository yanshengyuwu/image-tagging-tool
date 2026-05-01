"""DWPose — 身体关键点 → 部位 mask 生成器。

模型: dw-ll_ucoco_384.onnx (pose) + yolox_l.onnx (person detection)
来源: https://huggingface.co/yzd-v/DWPose

通过 17 个 COCO 关键点生成身体部位的管状/多边形 mask:
  - arms: 左右上臂+前臂
  - legs: 左右大腿+小腿
  - feet: 左右脚踝区域
  - torso: 肩→髋围成的多边形
  - neck: 鼻→左肩/右肩 三角区域
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from .base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

MODEL_REPO = "yzd-v/DWPose"
CACHE_DIR = Path("model_cache/dwpose")

# COCO 17 关键点索引
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# 部位 → 需要连接的关键点对（limb segments）
PART_LIMBS = {
    "arms": [
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ],
    "legs": [
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ],
    "neck": [
        ("nose", "left_shoulder"),
        ("nose", "right_shoulder"),
    ],
}

# 需要面积的部位（torso, feet）单独处理
SUPPORTED_PARTS = {"arms", "legs", "feet", "torso", "neck"}

# 管状 mask 的粗细（像素，会根据图像尺寸自动缩放）
BASE_THICKNESS = 20


class DWPoseBodyDetector(BaseDetector):
    """DWPose 关键点检测 → 身体部位管状/多边形 mask。"""

    def __init__(self, part_name: str, conf: float = 0.3):
        if part_name not in SUPPORTED_PARTS:
            raise ValueError(f"[dwpose] 不支持的部位: {part_name}, 可选: {SUPPORTED_PARTS}")
        super().__init__(part_name, conf)
        self._yolox = None
        self._pose = None

    def load_model(self):
        """下载并加载 YOLOX + DWPose ONNX 模型。"""
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 下载 YOLOX-L (人体检测)
        yolox_path = hf_hub_download(
            MODEL_REPO,
            filename="yolox_l.onnx",
            cache_dir=str(CACHE_DIR),
        )
        # 下载 DW-LL pose (关键点)
        pose_path = hf_hub_download(
            MODEL_REPO,
            filename="dw-ll_ucoco_384.onnx",
            cache_dir=str(CACHE_DIR),
        )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._yolox = ort.InferenceSession(yolox_path, providers=providers)
        self._pose = ort.InferenceSession(pose_path, providers=providers)
        self._model = True  # 标记为已加载

        logger.info("[dwpose] YOLOX-L + DW-LL-384 已加载")

    def _detect_persons(self, image: np.ndarray):
        """用 YOLOX 检测人体 bbox。"""
        h, w = image.shape[:2]
        input_info = self._yolox.get_inputs()[0]
        input_name = input_info.name

        # YOLOX 输入尺寸
        in_h = input_info.shape[2] if isinstance(input_info.shape[2], int) else 640
        in_w = input_info.shape[3] if isinstance(input_info.shape[3], int) else 640

        # letterbox resize
        scale = min(in_h / h, in_w / w)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_w, pad_h = (in_w - new_w) // 2, (in_h - new_h) // 2

        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        blob = canvas.astype(np.float32).transpose(2, 0, 1)[np.newaxis]

        outputs = self._yolox.run(None, {input_name: blob})
        pred = outputs[0][0]  # (num, 7) or similar

        bboxes = []
        for row in pred:
            if len(row) < 6:
                continue
            # YOLOX 输出格式: cx, cy, w, h, obj_conf, cls_conf...
            obj_conf = float(row[4])
            if obj_conf < 0.3:
                continue

            cx, cy, bw, bh = row[:4]
            x1 = (cx - bw / 2 - pad_w) / scale
            y1 = (cy - bh / 2 - pad_h) / scale
            x2 = (cx + bw / 2 - pad_w) / scale
            y2 = (cy + bh / 2 - pad_h) / scale

            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            if x2 - x1 > 10 and y2 - y1 > 10:
                bboxes.append([int(x1), int(y1), int(x2), int(y2)])

        return bboxes

    def _estimate_keypoints(self, image: np.ndarray, bbox):
        """对单个人体 crop 估计 17 个关键点。"""
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        input_info = self._pose.get_inputs()[0]
        input_name = input_info.name
        in_h = input_info.shape[2] if isinstance(input_info.shape[2], int) else 384
        in_w = input_info.shape[3] if isinstance(input_info.shape[3], int) else 288

        # resize crop
        resized = cv2.resize(crop, (in_w, in_h))
        blob = resized.astype(np.float32) / 255.0
        # 标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        blob = (blob - mean) / std
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        outputs = self._pose.run(None, {input_name: blob})
        heatmaps = outputs[0][0]  # (17, hm_h, hm_w)

        crop_h, crop_w = y2 - y1, x2 - x1
        keypoints = []

        for i in range(min(17, heatmaps.shape[0])):
            hm = heatmaps[i]
            idx = np.argmax(hm)
            hy, hx = divmod(idx, hm.shape[1])
            conf = float(hm[hy, hx])

            # 映射回原图坐标
            px = x1 + hx / hm.shape[1] * crop_w
            py = y1 + hy / hm.shape[0] * crop_h

            keypoints.append((int(px), int(py), conf))

        return keypoints

    def _draw_limb_mask(self, mask, keypoints, limbs, thickness):
        """在 mask 上绘制管状肢体。"""
        count = 0
        for kp_a, kp_b in limbs:
            idx_a = KP.get(kp_a)
            idx_b = KP.get(kp_b)
            if idx_a is None or idx_b is None:
                continue
            if idx_a >= len(keypoints) or idx_b >= len(keypoints):
                continue

            xa, ya, ca = keypoints[idx_a]
            xb, yb, cb = keypoints[idx_b]

            if ca < self.conf or cb < self.conf:
                continue

            cv2.line(mask, (xa, ya), (xb, yb), 255, thickness)
            count += 1

        return count

    def _draw_torso(self, mask, keypoints):
        """绘制躯干多边形: 左肩→右肩→右髋→左髋。"""
        indices = [KP["left_shoulder"], KP["right_shoulder"],
                   KP["right_hip"], KP["left_hip"]]

        pts = []
        for idx in indices:
            if idx >= len(keypoints):
                return 0
            x, y, c = keypoints[idx]
            if c < self.conf:
                return 0
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
        return 1

    def _draw_feet(self, mask, keypoints, radius):
        """在脚踝位置画圆作为脚部 mask。"""
        count = 0
        for ankle_name in ["left_ankle", "right_ankle"]:
            idx = KP.get(ankle_name)
            if idx is None or idx >= len(keypoints):
                continue
            x, y, c = keypoints[idx]
            if c < self.conf:
                continue
            cv2.circle(mask, (x, y), radius, 255, -1)
            count += 1
        return count

    def _draw_neck(self, mask, keypoints, thickness):
        """绘制颈部: 鼻→左肩, 鼻→右肩 三角形。"""
        idx_nose = KP["nose"]
        idx_ls = KP["left_shoulder"]
        idx_rs = KP["right_shoulder"]

        for idx in [idx_nose, idx_ls, idx_rs]:
            if idx >= len(keypoints):
                return 0
            if keypoints[idx][2] < self.conf:
                return 0

        pts = np.array([
            [keypoints[idx_nose][0], keypoints[idx_nose][1]],
            [keypoints[idx_ls][0], keypoints[idx_ls][1]],
            [keypoints[idx_rs][0], keypoints[idx_rs][1]],
        ], dtype=np.int32).reshape(-1, 1, 2)

        cv2.fillPoly(mask, [pts], 255)
        return 1

    def detect(self, image: np.ndarray) -> DetectionResult:
        """检测人体关键点，生成目标部位的 mask。"""
        self.ensure_loaded()
        h, w = image.shape[:2]

        # 自适应粗细
        scale_factor = max(h, w) / 1000.0
        thickness = max(8, int(BASE_THICKNESS * scale_factor))
        foot_radius = max(12, int(30 * scale_factor))

        # 1. 检测人体
        bboxes = self._detect_persons(image)
        logger.info(f"[dwpose] {self.part_name}: YOLOX 检测到 {len(bboxes)} 个人体 bbox")
        for i, bbox in enumerate(bboxes):
            logger.info(f"[dwpose]   人体 #{i}: bbox={bbox}")
        if not bboxes:
            logger.info(f"[dwpose] {self.part_name}: 未检测到人体（YOLOX 是真人检测器，可能不适用于动漫图）")
            return self.empty_result(h, w)

        mask = np.zeros((h, w), dtype=np.uint8)
        total_count = 0

        for bbox in bboxes:
            # 2. 估计关键点
            keypoints = self._estimate_keypoints(image, bbox)
            if keypoints is None or len(keypoints) < 17:
                continue

            # 3. 根据部位绘制 mask
            if self.part_name == "arms":
                total_count += self._draw_limb_mask(mask, keypoints, PART_LIMBS["arms"], thickness)
            elif self.part_name == "legs":
                total_count += self._draw_limb_mask(mask, keypoints, PART_LIMBS["legs"], thickness)
            elif self.part_name == "torso":
                total_count += self._draw_torso(mask, keypoints)
            elif self.part_name == "feet":
                total_count += self._draw_feet(mask, keypoints, foot_radius)
            elif self.part_name == "neck":
                total_count += self._draw_neck(mask, keypoints, thickness)

        if total_count == 0:
            logger.info(f"[dwpose] {self.part_name}: 关键点置信度不足，无有效检测")
            return self.empty_result(h, w)

        # 高斯模糊平滑边缘
        kernel_size = max(3, thickness // 2) | 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        mask = (mask > 127).astype(np.uint8) * 255

        logger.info(f"[dwpose] {self.part_name}: 检测到 {total_count} 段, "
                     f"覆盖 {np.sum(mask > 0) / (h * w) * 100:.2f}%")

        return DetectionResult(
            part_name=self.part_name,
            mask=mask,
            confidence=0.8,  # 关键点模型无单一置信度，使用固定值
            count=total_count,
        )
