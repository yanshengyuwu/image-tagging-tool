"""增强型胸部检测器 — 针对巨乳/怀孕等极端体型的二次元图像。

策略: Anzhc n (高召回) + Anzhc m (高精度/大面积) + 多尺度推理
      使用 n 模型确保检出，m 模型补充面积和精度。
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from .base import BaseDetector, DetectionResult
from ..utils import seg_to_mask

logger = logging.getLogger(__name__)

CACHE_DIR = Path("model_cache/anzhc_seg")
REPO = "Anzhc/Anzhcs_YOLOs"

# 模型配置
MODEL_CONFIGS = {
    "n": {
        "filename": "Anzhc Breasts Seg v1 1024n.pt",
        "conf": 0.25,
        "weight": 0.6,  # 最终合并时的权重
    },
    "m": {
        "filename": "Anzhc Breasts Seg v1 1024m.pt",
        "conf": 0.25,
        "weight": 0.4,
    },
}


class EnhancedBreastsDetector(BaseDetector):
    """增强胸部检测器。

    同时使用 Anzhc n (高召回) 和 m (高精度) 模型，
    配合多尺度推理 (640/1024/1280)，最大化检出率。
    """

    def __init__(self, conf: float = 0.25):
        super().__init__("breasts", conf)
        self._models: dict[str, YOLO] = {}
        self._loaded = False

    def _download_model(self, filename: str) -> Path:
        """确保模型文件已下载。"""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        local_path = CACHE_DIR / filename
        if local_path.exists():
            return local_path

        logger.info(f"[enhanced_breasts] 下载模型: {filename}")
        downloaded = hf_hub_download(
            repo_id=REPO,
            filename=filename,
            local_dir=str(CACHE_DIR),
        )
        return Path(downloaded)

    def load_model(self):
        """加载所有子模型。"""
        for key, cfg in MODEL_CONFIGS.items():
            model_path = self._download_model(cfg["filename"])
            self._models[key] = YOLO(str(model_path))
            logger.info(f"[enhanced_breasts] {key} 模型已加载: {cfg['filename']}")

        self._model = True  # 标记已加载，使 is_loaded 返回 True
        self._loaded = True

    def _detect_with_model(
        self,
        image: np.ndarray,
        model: YOLO,
        conf: float,
        multiscale: bool = True,
    ) -> tuple[np.ndarray, float, int]:
        """使用单个模型执行多尺度检测。

        Returns:
            (mask, avg_conf, count)
        """
        h, w = image.shape[:2]
        imgsz_list = [640, 1024, 1280] if multiscale else [1024]

        all_masks = []
        all_confs = []

        for imgsz in imgsz_list:
            if imgsz > max(h, w) * 2:
                continue

            results = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)
            if not results:
                continue

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue

            if not result.masks:
                continue

            for i, box in enumerate(result.boxes):
                if i < len(result.masks.data):
                    seg_mask = result.masks.data[i].cpu().numpy()
                    mask = seg_to_mask(h, w, seg_mask)
                    all_masks.append(mask)
                    all_confs.append(float(box.conf[0]))

        if not all_masks:
            return np.zeros((h, w), dtype=np.uint8), 0.0, 0

        # 合并：取并集
        merged = np.zeros((h, w), dtype=np.uint8)
        for m in all_masks:
            merged = np.maximum(merged, m)

        avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
        return merged, avg_conf, len(all_masks)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """执行增强检测。

        流程:
        1. n 模型多尺度检测 (高召回)
        2. m 模型多尺度检测 (高精度补充)
        3. 加权合并两个 mask
        4. 二值化输出
        """
        if not self._loaded:
            self.load_model()
        h, w = image.shape[:2]

        # n 模型检测
        n_mask, n_conf, n_count = self._detect_with_model(
            image,
            self._models["n"],
            MODEL_CONFIGS["n"]["conf"],
            multiscale=True,
        )

        # m 模型检测
        m_mask, m_conf, m_count = self._detect_with_model(
            image,
            self._models["m"],
            MODEL_CONFIGS["m"]["conf"],
            multiscale=True,
        )

        # 合并策略: 取并集 (只要有任一模型检测到就算)
        # 原因: n 模型召回率高但面积偏小，m 模型面积大但召回率低
        merged_mask = np.maximum(n_mask, m_mask)

        # 统计
        has_detection = np.any(merged_mask > 0)
        mask_pixels = int(np.sum(merged_mask > 0))
        mask_ratio = mask_pixels / (h * w) * 100

        # 置信度: 取检测到的模型的平均
        confs = []
        if n_count > 0:
            confs.append(n_conf)
        if m_count > 0:
            confs.append(m_conf)
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        total_count = max(n_count, m_count)  # 取较大的实例数

        logger.info(
            f"[enhanced_breasts] n模型: {n_count}个 (conf={n_conf:.3f}), "
            f"m模型: {m_count}个 (conf={m_conf:.3f}), "
            f"合并: detected={has_detection}, pixels={mask_pixels} ({mask_ratio:.2f}%)"
        )

        return DetectionResult(
            part_name=self.part_name,
            mask=merged_mask,
            confidence=avg_conf,
            count=total_count,
        )
