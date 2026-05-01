"""NSFW 分割检测器 — ntd11 YOLO-seg 主力 + EraX bbox 二次确认。"""

import logging
from pathlib import Path

import cv2
import numpy as np

from .base import BaseDetector, DetectionResult
from ..utils import seg_to_mask

logger = logging.getLogger(__name__)

NTD11_MODEL_DIR = Path("model_cache/nsfw_detectors/ntd11")
NTD11_MODEL_NAME = "ntd11_anime_nsfw_segm_v5.pt"

ERAX_MODEL_DIR = Path("model_cache/nsfw_detectors/erax")
ERAX_REPO = "erax-ai/EraX-Anti-NSFW-V1.1"


class NsfwSegDetector(BaseDetector):
    """NSFW 像素级分割检测器。

    主力：ntd11_anime_nsfw_segm_v5（YOLO11s-seg）
      - 需手动下载：https://civitai.com/models/1313556
      - 放入 model_cache/nsfw_detectors/ntd11/

    备选：EraX-Anti-NSFW-V1.1（YOLO11n bbox）
      - 自动从 HuggingFace 下载

    两者的 mask 合并输出为单个 nsfw.png。
    """

    def __init__(self, conf: float = 0.3, use_erax: bool = True):
        super().__init__("nsfw", conf)
        self.use_erax = use_erax
        self._ntd11_model = None
        self._erax_model = None

    def _find_ntd11_model(self) -> Path | None:
        """查找 ntd11 模型文件。"""
        direct = NTD11_MODEL_DIR / NTD11_MODEL_NAME
        if direct.exists():
            logger.info(f"[nsfw] 找到 ntd11 模型（直接）: {direct}")
            return direct
        if NTD11_MODEL_DIR.exists():
            for f in NTD11_MODEL_DIR.glob("*.pt"):
                if "nsfw" in f.name.lower() and "seg" in f.name.lower():
                    logger.info(f"[nsfw] 找到 ntd11 模型（搜索）: {f}")
                    return f
        logger.warning(f"[nsfw] 未找到 ntd11 模型，目录: {NTD11_MODEL_DIR}")
        return None

    def _download_erax_model(self) -> Path | None:
        """从 HuggingFace 自动下载 EraX 模型。"""
        ERAX_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # 查找已下载的 .pt 文件
        for f in ERAX_MODEL_DIR.glob("*.pt"):
            logger.info(f"[nsfw] 找到已下载的 EraX 模型: {f}")
            return f

        logger.info("[nsfw] 正在从 HuggingFace 下载 EraX-Anti-NSFW-V1.1...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=ERAX_REPO,
                local_dir=str(ERAX_MODEL_DIR),
                allow_patterns=["*.pt"],
            )
            for f in ERAX_MODEL_DIR.glob("*.pt"):
                logger.info(f"[nsfw] EraX 模型下载完成: {f}")
                return f
        except Exception as e:
            logger.warning(f"[nsfw] EraX 模型下载失败: {e}")

        return None

    def load_model(self):
        """加载 NSFW 模型。"""
        from ultralytics import YOLO

        # 加载 ntd11（主力，需手动下载）
        ntd11_path = self._find_ntd11_model()
        if ntd11_path:
            try:
                self._ntd11_model = YOLO(str(ntd11_path))
                logger.info(f"[nsfw] ntd11 seg 模型已加载: {ntd11_path}")
            except Exception as e:
                logger.error(f"[nsfw] ntd11 模型加载失败: {e}")
                import traceback
                logger.error(f"[nsfw] 错误详情: {traceback.format_exc()}")
                self._ntd11_model = None
        else:
            logger.warning(
                "[nsfw] ntd11 模型未找到。"
                "请从 https://civitai.com/models/1313556 手动下载，"
                f"放入 {NTD11_MODEL_DIR}/"
            )

        # 加载 EraX（备选，自动下载）
        if self.use_erax:
            erax_path = self._download_erax_model()
            if erax_path:
                try:
                    self._erax_model = YOLO(str(erax_path))
                    logger.info(f"[nsfw] EraX 模型已加载: {erax_path}")
                except Exception as e:
                    logger.error(f"[nsfw] EraX 模型加载失败: {e}")
                    import traceback
                    logger.error(f"[nsfw] 错误详情: {traceback.format_exc()}")
                    self._erax_model = None
            else:
                logger.warning("[nsfw] EraX 模型下载/加载失败")

        # 至少有一个模型才算加载成功
        self._model = self._ntd11_model or self._erax_model
        if self._model is None:
            raise FileNotFoundError(
                "未找到任何 NSFW 模型。\n"
                f"ntd11: 请从 https://civitai.com/models/1313556 下载到 {NTD11_MODEL_DIR}/\n"
                f"EraX: 自动下载失败，请检查网络或手动下载 {ERAX_REPO}"
            )
        
        logger.info(f"[nsfw] 检测器准备就绪，ntd11: {self._ntd11_model is not None}, EraX: {self._erax_model is not None}")

    def _detect_ntd11(self, image: np.ndarray, h: int, w: int) -> np.ndarray:
        """ntd11 像素级分割。"""
        mask = np.zeros((h, w), dtype=np.uint8)
        if self._ntd11_model is None:
            logger.debug("[nsfw] ntd11 模型未加载，跳过")
            return mask

        try:
            logger.info(f"[nsfw] 使用 ntd11 进行检测，conf={self.conf}...")
            results = self._ntd11_model.predict(image, conf=self.conf, verbose=False)
            
            total_masks = 0
            for result in results:
                if result.masks is None:
                    logger.debug("[nsfw] ntd11 结果中无 masks")
                    continue
                logger.info(f"[nsfw] ntd11 检测到 {len(result.masks.data)} 个 mask")
                for i, seg_mask in enumerate(result.masks.data):
                    seg_np = seg_mask.cpu().numpy()
                    part_mask = seg_to_mask(h, w, seg_np)
                    mask = np.maximum(mask, part_mask)
                    total_masks += 1
            
            mask_pixels = np.sum(mask > 0)
            logger.info(f"[nsfw] ntd11 检测完成: {total_masks} 个 mask，像素数: {mask_pixels}")
            
        except Exception as e:
            logger.error(f"[nsfw] ntd11 检测失败: {e}")
            import traceback
            logger.error(f"[nsfw] 错误详情: {traceback.format_exc()}")

        return mask

    def _detect_erax(self, image: np.ndarray, h: int, w: int) -> np.ndarray:
        """EraX bbox 检测（补充 ntd11 可能遗漏的区域）。"""
        mask = np.zeros((h, w), dtype=np.uint8)
        if self._erax_model is None:
            logger.debug("[nsfw] EraX 模型未加载，跳过")
            return mask

        try:
            logger.info(f"[nsfw] 使用 EraX 进行检测，conf={self.conf}...")
            results = self._erax_model.predict(image, conf=self.conf, verbose=False)
            
            total_boxes = 0
            for result in results:
                if result.boxes is None:
                    logger.debug("[nsfw] EraX 结果中无 boxes")
                    continue
                logger.info(f"[nsfw] EraX 检测到 {len(result.boxes)} 个 bbox")
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    mask[y1:y2, x1:x2] = 255
                    total_boxes += 1
            
            mask_pixels = np.sum(mask > 0)
            logger.info(f"[nsfw] EraX 检测完成: {total_boxes} 个 bbox，像素数: {mask_pixels}")
            
        except Exception as e:
            logger.error(f"[nsfw] EraX 检测失败: {e}")
            import traceback
            logger.error(f"[nsfw] 错误详情: {traceback.format_exc()}")

        return mask

    def detect(self, image: np.ndarray) -> DetectionResult:
        """执行 NSFW 检测，合并 ntd11 seg + EraX bbox。"""
        self.ensure_loaded()
        h, w = image.shape[:2]

        logger.info(f"[nsfw] 开始 NSFW 检测，图像尺寸: {w}x{h}")

        ntd11_mask = self._detect_ntd11(image, h, w)
        erax_mask = self._detect_erax(image, h, w)

        # 合并两个 mask
        mask = np.maximum(ntd11_mask, erax_mask)
        count = 1 if np.any(mask > 0) else 0
        
        mask_pixels = np.sum(mask > 0)
        logger.info(f"[nsfw] 检测完成: 合并 mask 像素数: {mask_pixels} (占比: {mask_pixels/(h*w)*100:.2f}%)")

        return DetectionResult(
            part_name=self.part_name,
            mask=mask,
            confidence=float(self.conf) if count > 0 else 0.0,
            count=count,
        )
