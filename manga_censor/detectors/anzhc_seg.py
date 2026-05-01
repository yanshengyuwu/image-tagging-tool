"""Anzhc YOLO-seg 系列检测器 — 像素级分割（eyes/hair/breasts）。

注：面部(face)已改用 DeepGHS bbox + SAM2 精细化方案，不再使用此检测器。
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

from .base import BaseDetector, DetectionResult
from ..utils import seg_to_mask

logger = logging.getLogger(__name__)

# Anzhc 模型注册表：part_name → (HF repo, HF filename, 默认 conf)
ANZHC_MODELS = {
    # 头部部位
    "face": {
        "repo": "Anzhc/Anzhcs_YOLOs",
        "filename": "Anzhc Face seg 640 v3 y11n.pt",
        "conf": 0.5,
    },
    "eyes": {
        "repo": "Anzhc/Anzhcs_YOLOs",
        "filename": "Anzhc Eyes -seg-hd.pt",
        "conf": 0.5,
    },
    "mouth": {
        "repo": "Anzhc/Anzhcs_YOLOs",
        "filename": "Anzhc Mouth seg 640 v3 y11n.pt",
        "conf": 0.5,
    },
    "ears": {
        "repo": "Anzhc/Anzhcs_YOLOs",
        "filename": "Anzhc Ears seg 640 v3 y11n.pt",
        "conf": 0.5,
    },
    "hair": {
        "repo": "Anzhc/Anzhcs_YOLOs",
        "filename": "Anzhc HeadHair seg y8m.pt",
        "conf": 0.5,
    },
    # 上肢部位
    "hand": {
        "repo": "Anzhc/Anzhcs_YOLOs",
        "filename": "Anzhc Hand seg 640 v3 y11n.pt",
        "conf": 0.5,
    },
    "arms": {
        "repo": "Anzhc/Anzhcs_YOLOs",
        "filename": "Anzhc Arms seg y8m.pt",
        "conf": 0.5,
    },
    # 躯干部位
    "neck": {
        "repo": "Anzhc/Anzhcs_YOLOs",
        "filename": "Anzhc Neck seg y8m.pt",
        "conf": 0.5,
    },
    "torso": {
        "repo": "Anzhc/Anzhcs_YOLOs",
        "filename": "Anzhc Torso seg y8m.pt",
        "conf": 0.5,
    },
    "breasts": {
        "repo": "Anzhc/Anzhcs_YOLOs",
        "filename": "Anzhc Breasts Seg v1 1024m.pt",
        "conf": 0.5,
    },
    # 下肢部位
    "legs": {
        "repo": "Anzhc/Anzhcs_YOLOs",
        "filename": "Anzhc Legs seg y8m.pt",
        "conf": 0.5,
    },
    "feet": {
        "repo": "Anzhc/Anzhcs_YOLOs",
        "filename": "Anzhc Feet seg 640 v3 y11n.pt",
        "conf": 0.5,
    },
}

MODEL_CACHE_DIR = Path("model_cache/anzhc_seg")


class AnzhcSegDetector(BaseDetector):
    """基于 Anzhc YOLO-seg 模型的像素级分割检测器。"""

    def __init__(self, part_name: str, conf: float | None = None,
                 target_classes: list[int] | None = None):
        if part_name not in ANZHC_MODELS:
            raise ValueError(f"未知部位: {part_name}，可选: {list(ANZHC_MODELS.keys())}")

        model_info = ANZHC_MODELS[part_name]
        super().__init__(part_name, conf or model_info["conf"])
        self.repo = model_info["repo"]
        self.filename = model_info["filename"]
        self.target_classes = target_classes
        self._model_path: Path | None = None

    def _download_model(self) -> Path:
        """从 HuggingFace 下载模型文件。"""
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        local_path = MODEL_CACHE_DIR / self.filename
        if local_path.exists():
            logger.info(f"[{self.part_name}] 模型已缓存: {local_path}")
            return local_path

        logger.info(f"[{self.part_name}] 正在从 {self.repo} 下载 {self.filename}...")
        downloaded = hf_hub_download(
            repo_id=self.repo,
            filename=self.filename,
            local_dir=str(MODEL_CACHE_DIR),
        )
        logger.info(f"[{self.part_name}] 模型下载完成: {downloaded}")
        return Path(downloaded)

    def load_model(self):
        """加载 YOLO 模型。"""
        from ultralytics import YOLO

        self._model_path = self._download_model()
        self._model = YOLO(str(self._model_path))
        logger.info(f"[{self.part_name}] YOLO-seg 模型已加载")

    def detect(self, image: np.ndarray) -> DetectionResult:
        """执行像素级分割检测。"""
        self.ensure_loaded()
        h, w = image.shape[:2]

        results = self._model.predict(image, conf=self.conf, verbose=False)

        mask = np.zeros((h, w), dtype=np.uint8)
        total_conf = 0.0
        count = 0

        for result in results:
            if result.masks is None:
                continue
            for i, seg_mask in enumerate(result.masks.data):
                cls_id = int(result.boxes.cls[i])
                if self.target_classes and cls_id not in self.target_classes:
                    continue
                seg_np = seg_mask.cpu().numpy()
                part_mask = seg_to_mask(h, w, seg_np)
                mask = np.maximum(mask, part_mask)
                total_conf += float(result.boxes.conf[i])
                count += 1

        avg_conf = total_conf / count if count > 0 else 0.0
        return DetectionResult(
            part_name=self.part_name,
            mask=mask,
            confidence=avg_conf,
            count=count,
        )
