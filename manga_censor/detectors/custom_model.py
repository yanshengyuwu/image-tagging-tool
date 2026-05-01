"""自定义本地 YOLO 模型检测器 — 支持用户加载任意 YOLO-seg/cls/det 模型。"""

import logging
from pathlib import Path

import cv2
import numpy as np

from .base import BaseDetector, DetectionResult
from ..utils import seg_to_mask

logger = logging.getLogger(__name__)


class CustomModelDetector(BaseDetector):
    """支持用户自定义的本地 YOLO 模型检测器。

    模型必须是从本地路径加载的 YOLO 模型（.pt 或 .onnx）。
    自动探测模型中的类别名称，并支持按类别 ID 过滤。
    """

    def __init__(self, model_path: str, conf: float = 0.25,
                 target_classes: list[int] | None = None,
                 part_name: str = "custom"):
        super().__init__(part_name, conf)
        self.model_path = Path(model_path)
        self.target_classes = target_classes or []
        self._class_names: list[str] | None = None

    @property
    def class_names(self) -> list[str] | None:
        """获取模型中的类别名称列表。"""
        return self._class_names

    def load_model(self):
        """加载 YOLO 模型并提取类别信息。"""
        from ultralytics import YOLO

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self._model = YOLO(str(self.model_path))
        # 提取类别名称（YOLOv8 模型通常有 .names 属性）
        names = getattr(self._model, "names", None)
        if names is None and hasattr(self._model, "model"):
            # 某些版本的模型嵌套在 .model 中
            names = getattr(self._model.model, "names", None)
        if isinstance(names, dict):
            self._class_names = [names.get(i, f"class_{i}") for i in range(len(names))]
        elif isinstance(names, (list, tuple)):
            self._class_names = list(names)
        else:
            self._class_names = []

        logger.info(f"[custom] 自定义模型已加载: {self.model_path} | 类别数: {len(self._class_names)}")

    def detect(self, image: np.ndarray) -> DetectionResult:
        """执行检测并返回 mask。

        如果模型是分割模型 (YOLO-seg)，返回像素级 mask。
        如果模型是检测模型 (YOLO-det)，返回 bbox 填充的 mask。
        """
        self.ensure_loaded()
        h, w = image.shape[:2]

        results = self._model.predict(image, conf=self.conf, verbose=False)

        mask = np.zeros((h, w), dtype=np.uint8)
        total_conf = 0.0
        count = 0

        for result in results:
            # 优先使用分割 mask（YOLO-seg）
            if result.masks is not None:
                for i, seg_mask in enumerate(result.masks.data):
                    cls_id = int(result.boxes.cls[i])
                    if self.target_classes and cls_id not in self.target_classes:
                        continue
                    seg_np = seg_mask.cpu().numpy()
                    part_mask = seg_to_mask(h, w, seg_np)
                    mask = np.maximum(mask, part_mask)
                    total_conf += float(result.boxes.conf[i])
                    count += 1
            # 否则使用检测框填充（YOLO-det）
            elif result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[i])
                    if self.target_classes and cls_id not in self.target_classes:
                        continue
                    xyxy = box.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    total_conf += float(box.conf[i])
                    count += 1

        avg_conf = total_conf / count if count > 0 else 0.0
        return DetectionResult(
            part_name=self.part_name,
            mask=mask,
            confidence=avg_conf,
            count=count,
        )
