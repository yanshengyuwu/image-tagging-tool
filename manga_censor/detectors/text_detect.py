"""文字气泡检测器 — Anzhc text-seg + OCR 验证两阶段。"""

import logging
from pathlib import Path

import cv2
import numpy as np

from .base import BaseDetector, DetectionResult
from ..utils import seg_to_mask, extract_component_masks

logger = logging.getLogger(__name__)

# Anzhc text-seg 模型信息
ANZHC_TEXT_MODELS = {
    "text_bubble": {
        "huggingface_repo": "Anzhc/Anzhcs_YOLOs",
        "huggingface_filename": "Anzhcs-text-seg-v9-y11m.pt",
        "local_path": "model_cache/anzhc/Anzhcs-text-seg-v9-y11m.pt",
        "conf": 0.5,
        "imgsz": 1024,
    },
}


class TextBubbleDetector(BaseDetector):
    """文字气泡检测器：text-seg + OCR 验证。

    阶段1：Anzhc text-seg 输出像素级候选 mask
    阶段2：提取连通区域 → OCR 识别 → 过滤纹身/图案（OCR 返回空的区域被丢弃）

    OCR 模型优先使用 manga-ocr（日文漫画专用），
    可选 EasyOCR 作为中文/英文补充。
    """

    def __init__(self, part_name: str = "text_bubble", conf: float | None = None,
                 use_ocr: bool = True, ocr_min_length: int = 1):
        super().__init__(part_name, conf or 0.5)
        self.model_info = ANZHC_TEXT_MODELS.get(part_name, ANZHC_TEXT_MODELS["text_bubble"])
        self.use_ocr = use_ocr
        self.ocr_min_length = ocr_min_length  # OCR 结果最小有效字符数
        self._ocr_model = None
        self._ocr_type = None  # "manga_ocr" or "easyocr"

    def _download_model(self) -> Path:
        """下载 Anzhc text-seg 模型。"""
        local_path = Path(self.model_info["local_path"])
        if local_path.exists():
            logger.info(f"[{self.part_name}] 模型已缓存: {local_path}")
            return local_path

        repo = self.model_info["huggingface_repo"]
        filename = self.model_info["huggingface_filename"]

        logger.info(f"[{self.part_name}] 正在从 {repo} 下载 {filename}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(local_path.parent),
        )
        logger.info(f"[{self.part_name}] 模型下载完成: {downloaded}")
        return Path(downloaded)

    def load_model(self):
        """加载 text-seg 模型和 OCR 模型。"""
        # 加载 text-seg 模型
        model_path = self._download_model()
        from ultralytics import YOLO
        self._model = YOLO(str(model_path))
        logger.info(f"[{self.part_name}] text-seg 模型已加载: {model_path}")

        # 加载 OCR 模型
        if self.use_ocr:
            self._load_ocr_model()

    def _load_ocr_model(self):
        """加载 OCR 模型，优先 manga-ocr，失败则尝试 EasyOCR。"""
        if self._ocr_model is not None:
            return

        # 尝试 manga-ocr
        try:
            from manga_ocr import MangaOcr
            self._ocr_model = MangaOcr()
            self._ocr_type = "manga_ocr"
            logger.info(f"[{self.part_name}] OCR 模型已加载: manga-ocr")
            return
        except ImportError:
            logger.info(f"[{self.part_name}] manga-ocr 未安装，尝试 EasyOCR...")
        except Exception as e:
            logger.warning(f"[{self.part_name}] manga-ocr 加载失败: {e}，尝试 EasyOCR...")

        # 尝试 EasyOCR
        try:
            import easyocr
            self._ocr_model = easyocr.Reader(['ja', 'ch_sim', 'en'], gpu=True)
            self._ocr_type = "easyocr"
            logger.info(f"[{self.part_name}] OCR 模型已加载: EasyOCR (ja+ch_sim+en)")
            return
        except ImportError:
            logger.info(f"[{self.part_name}] EasyOCR 也未安装，OCR 验证将跳过")
        except Exception as e:
            logger.warning(f"[{self.part_name}] EasyOCR 加载失败: {e}，OCR 验证将跳过")

        self._ocr_model = None
        self._ocr_type = None

    def _ocr_verify_region(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[bool, str]:
        """对单个区域执行 OCR 验证。

        Args:
            image: 原图 (H, W, 3), BGR
            bbox: (x1, y1, x2, y2)

        Returns:
            (is_text, recognized_text) — 是否是有效文字，以及 OCR 识别结果
        """
        if self._ocr_model is None:
            # OCR 不可用，默认保留
            return True, ""

        x1, y1, x2, y2 = bbox
        # 添加少量 padding 让 OCR 更容易识别
        h, w = image.shape[:2]
        pad = max(2, int(min(x2 - x1, y2 - y1) * 0.05))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            return False, ""

        try:
            text = self._ocr_recognize(patch)
            # 判断是否有效文字
            if text and len(text.strip()) >= self.ocr_min_length:
                return True, text.strip()
            else:
                return False, text or ""
        except Exception as e:
            logger.debug(f"[{self.part_name}] OCR 识别异常: {e}")
            return True, ""  # OCR 异常时保守保留

    def _ocr_recognize(self, patch: np.ndarray) -> str:
        """对图像 patch 执行 OCR 识别。"""
        if self._ocr_type == "manga_ocr":
            # manga-ocr 期望 PIL RGB 输入
            from PIL import Image
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_patch)
            return self._ocr_model(pil_img)

        elif self._ocr_type == "easyocr":
            # EasyOCR 期望 BGR 或 RGB numpy
            results = self._ocr_model.readtext(patch)
            texts = [r[1] for r in results if r[2] > 0.3]  # 置信度 > 0.3
            return " ".join(texts)

        return ""

    def detect(self, image: np.ndarray) -> DetectionResult:
        """执行 text-seg + OCR 验证两阶段检测。"""
        self.ensure_loaded()
        h, w = image.shape[:2]

        # 阶段1：text-seg 输出候选 mask
        imgsz = self.model_info.get("imgsz", 1024)
        results = self._model.predict(image, conf=self.conf, imgsz=imgsz, verbose=False)

        # 合并所有 text 类别的 seg mask
        candidate_mask = np.zeros((h, w), dtype=np.uint8)
        total_conf = 0.0
        det_count = 0

        for result in results:
            if result.masks is not None:
                for i, seg_mask in enumerate(result.masks.data):
                    cls_id = int(result.boxes.cls[i]) if result.boxes is not None else 0
                    conf_val = float(result.boxes.conf[i]) if result.boxes is not None else 0.5
                    seg_np = seg_mask.cpu().numpy()
                    part_mask = seg_to_mask(h, w, seg_np)
                    candidate_mask = np.maximum(candidate_mask, part_mask)
                    total_conf += conf_val
                    det_count += 1

        if det_count == 0:
            return self.empty_result(h, w)

        avg_conf = total_conf / det_count

        # 如果没有候选区域，直接返回
        if not np.any(candidate_mask > 0):
            return self.empty_result(h, w)

        # 阶段2：OCR 验证（如果启用）
        if self.use_ocr and self._ocr_model is not None:
            # 提取连通区域
            components = extract_component_masks(candidate_mask)

            if not components:
                return self.empty_result(h, w)

            # 对每个连通区域执行 OCR 验证
            verified_mask = np.zeros((h, w), dtype=np.uint8)
            verified_count = 0

            for comp_mask, bbox in components:
                is_text, recognized = self._ocr_verify_region(image, bbox)
                if is_text:
                    verified_mask = np.maximum(verified_mask, comp_mask)
                    verified_count += 1
                    logger.debug(f"[{self.part_name}] OCR 验证通过: '{recognized[:30]}...' bbox={bbox}")
                else:
                    logger.debug(f"[{self.part_name}] OCR 过滤纹身/图案: bbox={bbox}")

            # 如果 OCR 过滤掉了所有区域，返回空
            if verified_count == 0:
                logger.info(f"[{self.part_name}] OCR 验证后所有区域被过滤（可能是纹身/图案）")
                return self.empty_result(h, w)

            logger.info(f"[{self.part_name}] OCR 验证: {verified_count}/{len(components)} 区域保留")
            return DetectionResult(
                part_name=self.part_name,
                mask=verified_mask,
                confidence=avg_conf,
                count=verified_count,
            )

        # OCR 未启用或不可用，直接返回候选 mask
        return DetectionResult(
            part_name=self.part_name,
            mask=candidate_mask,
            confidence=avg_conf,
            count=det_count,
        )
