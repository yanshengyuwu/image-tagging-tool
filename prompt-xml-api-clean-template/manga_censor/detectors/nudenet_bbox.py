"""NudeNet v3 bbox 检测 + SAM2 精细化 — 臀部等部位。"""

import logging
from pathlib import Path

import cv2
import numpy as np

from .base import BaseDetector, DetectionResult
from .sam2_refiner import SAM2Refiner
from ..utils import bbox_to_mask

logger = logging.getLogger(__name__)

# NudeNet 支持的部位标签
NUDENET_BUTTOCKS_LABELS = {
    "BUTTOCKS_EXPOSED",
    "BUTTOCKS_COVERED",
    "FEMALE_BUTTOCKS_EXPOSED",
    "FEMALE_BUTTOCKS_COVERED",
    "MALE_BUTTOCKS_EXPOSED",
    "MALE_BUTTOCKS_COVERED",
}


class NudeNetBboxDetector(BaseDetector):
    """NudeNet v3 bbox 检测 + SAM2 精细化。

    用于臀部等无二次元专用分割模型的部位。
    NudeNet 对二次元误检率较高，置信度建议设低（0.15）。
    """

    def __init__(self, part_name: str = "buttocks", conf: float = 0.15,
                 target_labels: set[str] | None = None,
                 use_sam2: bool = True):
        super().__init__(part_name, conf)
        self.target_labels = target_labels or NUDENET_BUTTOCKS_LABELS
        self.use_sam2 = use_sam2
        self._sam2 = SAM2Refiner() if use_sam2 else None
        self._detector = None

    def load_model(self):
        """加载 NudeNet 检测器。"""
        try:
            from nudenet import NudeDetector
            self._detector = NudeDetector()
            logger.info(f"[{self.part_name}] NudeNet 检测器已加载")
        except Exception as e:
            logger.warning(f"[{self.part_name}] NudeNet 加载失败: {e}")
            logger.warning(f"[{self.part_name}] 臀部检测将不可用")
            self._detector = None

        if self.use_sam2:
            logger.info(f"[{self.part_name}] SAM2 精细化已启用（延迟加载）")

    def detect(self, image: np.ndarray) -> DetectionResult:
        """执行 NudeNet 检测 + SAM2 精细化。"""
        self.ensure_loaded()
        h, w = image.shape[:2]

        if self._detector is None:
            return self.empty_result(h, w)

        # 阶段1：NudeNet 检测
        try:
            # NudeNet 需要 BGR 或 RGB 输入
            results = self._detector.detect(image)
        except Exception as e:
            logger.warning(f"[{self.part_name}] NudeNet 检测失败: {e}")
            return self.empty_result(h, w)

        if not results:
            return self.empty_result(h, w)

        # 过滤目标标签
        boxes = []
        confs = []
        for det in results:
            label = det.get("class", "")
            score = det.get("score", 0)

            if label not in self.target_labels:
                continue
            if score < self.conf:
                continue

            x1 = max(0, int(det.get("box", [0, 0, 0, 0])[0]))
            y1 = max(0, int(det.get("box", [0, 0, 0, 0])[1]))
            x2 = min(w, int(det.get("box", [0, 0, 0, 0])[2]))
            y2 = min(h, int(det.get("box", [0, 0, 0, 0])[3]))

            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
                confs.append(score)

        if not boxes:
            return self.empty_result(h, w)

        # 阶段2：SAM2 精细化（或回退到 bbox 矩形）
        if self._sam2 and self.use_sam2:
            mask = self._sam2.refine(image, boxes)
        else:
            mask = bbox_to_mask(h, w, boxes)

        return DetectionResult(
            part_name=self.part_name,
            mask=mask,
            confidence=sum(confs) / len(confs),
            count=len(boxes),
        )
