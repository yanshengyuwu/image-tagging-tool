"""NSFW-API/NSFW_Segmentation — 高精度 NSFW YOLO11x 分割检测器。

三个独立模型，各自专注一个类别:
  - nsfw-seg-breast-x.pt  → breast, areola, nipple
  - nsfw-seg-vagina-x.pt  → vagina
  - nsfw-seg-penis-x.pt   → penis

来源: https://huggingface.co/NSFW-API/NSFW_Segmentation
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from .base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

MODEL_REPO = "NSFW-API/NSFW_Segmentation"
CACHE_DIR = Path("model_cache/nsfw_api_seg")

# 部位名 → 下载的模型文件名 + 目标类名列表
# 注意: vagina/penis 模型只有 1 个类，训练时类名为 "item"，
#       因此 target_classes 必须包含 "item" 以匹配模型实际输出。
PART_MODEL_MAP = {
    "nipple": {
        "file": "nsfw-seg-breast-x.pt",
        "target_classes": ["nipple", "areola"],  # breast 模型中与 nipple 相关的类
        "accept_all_single_class": False,
    },
    "pussy": {
        "file": "nsfw-seg-vagina-x.pt",
        "target_classes": ["vagina", "item"],  # 模型实际输出类名为 "item"
        "accept_all_single_class": True,  # 单类模型，接受所有检测
    },
    "penis": {
        "file": "nsfw-seg-penis-x.pt",
        "target_classes": ["penis", "item"],  # 模型实际输出类名为 "item"
        "accept_all_single_class": True,  # 单类模型，接受所有检测
    },
}


class NsfwApiSegDetector(BaseDetector):
    """NSFW-API YOLO11x seg 模型。直接输出像素级分割 mask。

    每个 part_name 对应一个专用模型文件。
    """

    def __init__(self, part_name: str, conf: float = 0.3):
        super().__init__(part_name, conf)
        self._target_classes = []

    def load_model(self):
        """从 HuggingFace 下载并加载 YOLO11x seg 模型。"""
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        model_info = PART_MODEL_MAP.get(self.part_name)
        if not model_info:
            raise ValueError(f"[nsfw_api_seg] 不支持的部位: {self.part_name}")

        model_filename = model_info["file"]
        self._target_classes = model_info["target_classes"]

        # 检查本地缓存
        local_path = CACHE_DIR / model_filename
        if not local_path.exists():
            logger.info(f"[nsfw_api_seg] 下载模型: {model_filename}")
            downloaded = hf_hub_download(
                MODEL_REPO,
                filename=model_filename,
                local_dir=str(CACHE_DIR),
            )
            local_path = Path(downloaded)

        self._model = YOLO(str(local_path))
        logger.info(f"[nsfw_api_seg] 模型已加载: {model_filename}, "
                     f"目标类: {self._target_classes}")

    def _multiscale_detect(self, image: np.ndarray):
        """多尺度推理: 640/1024/1280，取并集。"""
        all_masks = []
        all_confs = []
        h, w = image.shape[:2]

        for imgsz in [640, 1024, 1280]:
            # 跳过比图像大太多的尺度
            if imgsz > max(h, w) * 2:
                continue

            results = self._model.predict(
                image,
                imgsz=imgsz,
                conf=self.conf,
                verbose=False,
            )

            if not results:
                logger.info(f"[nsfw_api_seg] {self.part_name} @{imgsz}: 无结果")
                continue

            result = results[0]
            names = result.names or {}

            # 诊断: 打印模型所有类名
            if imgsz == 640:  # 只打印一次
                logger.info(f"[nsfw_api_seg] {self.part_name}: 模型类名映射 = {names}")
                logger.info(f"[nsfw_api_seg] {self.part_name}: 目标类名 = {self._target_classes}")

            if result.boxes is None or len(result.boxes) == 0:
                logger.info(f"[nsfw_api_seg] {self.part_name} @{imgsz}: 无检测 (boxes为空)")
                continue

            if not result.masks:
                logger.info(f"[nsfw_api_seg] {self.part_name} @{imgsz}: "
                           f"有 {len(result.boxes)} 个bbox但无masks")

            # 诊断: 打印所有检测结果（不管是否匹配目标）
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, f"class_{cls_id}")
                score = float(box.conf[0])
                is_target = cls_name in self._target_classes
                logger.info(f"[nsfw_api_seg] {self.part_name} @{imgsz}: "
                           f"class={cls_id}({cls_name}), conf={score:.3f}, "
                           f"目标匹配={'✓' if is_target else '✗'}")

            if not result.masks:
                continue

            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, f"class_{cls_id}")
                score = float(box.conf[0])

                if cls_name not in self._target_classes:
                    continue

                seg_mask = result.masks.data[i].cpu().numpy()
                # 将 mask resize 到原图尺寸
                if seg_mask.shape != (h, w):
                    seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_LINEAR)

                binary = (seg_mask > 0.5).astype(np.uint8) * 255
                all_masks.append(binary)
                all_confs.append(score)

        return all_masks, all_confs

    def detect(self, image: np.ndarray) -> DetectionResult:
        """多尺度检测并合并结果。"""
        self.ensure_loaded()
        h, w = image.shape[:2]

        masks, confs = self._multiscale_detect(image)

        if not masks:
            logger.info(f"[nsfw_api_seg] {self.part_name}: 未检测到")
            return self.empty_result(h, w)

        # 合并所有 mask
        merged = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            merged = np.maximum(merged, m)

        avg_conf = sum(confs) / len(confs) if confs else 0.0
        count = len(masks)

        logger.info(f"[nsfw_api_seg] {self.part_name}: 检测到 {count} 个 (多尺度), "
                     f"置信度={avg_conf:.3f}")

        return DetectionResult(
            part_name=self.part_name,
            mask=merged,
            confidence=avg_conf,
            count=count,
        )
