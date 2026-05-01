"""NSFW 按部位细分检测器 — 共享模型实例，按类别过滤输出像素级 mask。

支持的部位：
  - penis      阴茎
  - pussy      女性生殖器
  - nipple     乳头
  - anus       肛门
  - testicles  睾丸

每个部位独立输出 mask，而非将所有 NSFW 内容合并到一起。

模型架构：
  - ntd11 (YOLO11-seg): 主力像素级分割模型，直接输出 seg mask
  - EraX (YOLO11-seg):  补充检测模型，也直接输出原生 seg mask
  - SAM2:               仅在 seg mask 完全不可用时回退使用（默认关闭）

EraX 与 ntd11 均为 YOLO-seg 架构，优先读取 result.masks.data 原生输出，
不再经过 SAM2 中间层，避免细小部位（anus, nipple）的误分割。
"""

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from .base import BaseDetector, DetectionResult
from .sam2_refiner import SAM2Refiner
from ..utils import (
    seg_to_mask,
    filter_mask_components,
    morph_close_open,
    dilate_mask,
    smooth_mask_edges,
)

logger = logging.getLogger(__name__)

NTD11_MODEL_DIR = Path("model_cache/nsfw_detectors/ntd11")
ERAX_MODEL_DIR = Path("model_cache/nsfw_detectors/erax")
ERAX_REPO = "erax-ai/EraX-Anti-NSFW-V1.1"

# ── ntd11 类别索引 ──
NTD11_CLASSES = {
    "nipple": [0],       # nipples
    "pussy": [1],        # pussy
    "anus": [2],         # anus
    "penis": [3],        # penis
    "testicles": [6],    # testicles
    # 附加类别（可选）
    "cross-section": [4],
    "x-ray": [5],
}

# ── EraX 类别索引 ──
ERAX_CLASSES = {
    "anus": [0],         # anus
    "nipple": [2],       # nipple
    "penis": [3],        # penis
    "pussy": [4],        # vagina
    # testicles 在 EraX 中无对应
}

    # ── 所有可用的 NSFW 细分部位 ──
NSFW_PART_NAMES = ["penis", "pussy", "nipple", "anus", "testicles"]

# ── 各部位推荐的 bbox 收缩比例（针对 SAM2 回退时减少遮盖范围）──
PART_SHRINK_RATIOS = {
    "anus": 0.40,      # 肛门 bbox 通常过大，收缩到 40%（更激进）
    "nipple": 0.85,    # 乳头 bbox 略大
    "penis": 0.80,     # 阴茎 bbox 偏大
    "pussy": 0.75,     # 女性生殖器 bbox 偏大
    "testicles": 0.80, # 睾丸 bbox 偏大
}

PART_POSTPROCESS = {
    "anus": {
        "min_area_ratio": 0.00001,
        "max_area_ratio": 0.0030,
        "min_width": 3,
        "min_height": 3,
        "max_aspect_ratio": 2.5,
        "close_kernel": 3,
        "open_kernel": 2,
        "dilate_kernel": 2,
        "smooth_blur": 3,
    },
    "nipple": {
        "min_area_ratio": 0.000008,
        "max_area_ratio": 0.0025,
        "min_width": 3,
        "min_height": 3,
        "max_aspect_ratio": 3.0,
        "close_kernel": 3,
        "open_kernel": 2,
        "dilate_kernel": 2,
        "smooth_blur": 3,
    },
    "pussy": {
        "min_area_ratio": 0.00003,
        "max_area_ratio": 0.0250,
        "min_width": 4,
        "min_height": 4,
        "max_aspect_ratio": 4.0,
        "close_kernel": 5,
        "open_kernel": 2,
        "dilate_kernel": 3,
        "smooth_blur": 3,
    },
    "penis": {
        "min_area_ratio": 0.00003,
        "max_area_ratio": 0.0300,
        "min_width": 4,
        "min_height": 4,
        "max_aspect_ratio": 8.0,
        "close_kernel": 3,
        "open_kernel": 2,
        "dilate_kernel": 3,
        "smooth_blur": 3,
    },
    "testicles": {
        "min_area_ratio": 0.00002,
        "max_area_ratio": 0.0150,
        "min_width": 3,
        "min_height": 3,
        "max_aspect_ratio": 3.5,
        "close_kernel": 3,
        "open_kernel": 2,
        "dilate_kernel": 2,
        "smooth_blur": 3,
    },
}


class _SharedNsfwModels:
    """NSFW 模型单例管理器，避免多个检测器重复加载同一模型。"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._ntd11_model = None
                    cls._instance._erax_model = None
                    cls._instance._sam2 = None
                    cls._instance._loaded = False
                    cls._instance._erax_model_size = "l"  # 默认使用 l 变体
        return cls._instance

    def _find_ntd11_model(self) -> Path | None:
        """查找 ntd11 模型文件。"""
        if NTD11_MODEL_DIR.exists():
            for f in NTD11_MODEL_DIR.glob("*.pt"):
                if "nsfw" in f.name.lower() and "seg" in f.name.lower():
                    return f
        return None

    def _find_erax_model(self, preferred_size: str = "l") -> Path | None:
        """查找或下载 EraX 模型。
        
        Args:
            preferred_size: 优先选择的模型大小 (n/s/m/l/x)
        """
        if ERAX_MODEL_DIR.exists():
            # 优先查找指定大小的模型
            for f in ERAX_MODEL_DIR.glob(f"*{preferred_size}*.pt"):
                logger.info(f"[nsfw_part] 找到 EraX {preferred_size} 模型: {f.name}")
                return f
            
            # 回退：按优先级查找 (x > l > m > s > n)
            for suffix in ["x", "l", "m", "s", "n"]:
                for f in ERAX_MODEL_DIR.glob(f"*{suffix}*.pt"):
                    logger.info(f"[nsfw_part] 回退到 EraX {suffix} 模型: {f.name}")
                    return f
            
            # 最后回退：任何 .pt
            for f in ERAX_MODEL_DIR.glob("*.pt"):
                logger.info(f"[nsfw_part] 使用 EraX 模型: {f.name}")
                return f

        # 自动下载指定大小的模型
        try:
            from huggingface_hub import snapshot_download
            ERAX_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"[nsfw_part] 正在下载 EraX {preferred_size} 模型...")
            snapshot_download(
                repo_id=ERAX_REPO,
                local_dir=str(ERAX_MODEL_DIR),
                allow_patterns=[f"*{preferred_size}*.pt"]
            )
            for f in ERAX_MODEL_DIR.glob(f"*{preferred_size}*.pt"):
                return f
        except Exception as e:
            logger.warning(f"[nsfw_part] EraX 下载失败: {e}")
        return None

    def load(self, use_erax: bool = True, use_sam2: bool = True, erax_size: str = "l"):
        """加载所有共享模型（幂等）。
        
        Args:
            use_erax: 是否使用 EraX 模型
            use_sam2: 是否使用 SAM2 精细化
            erax_size: EraX 模型大小 (n/s/m/l/x)
        """
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            from ultralytics import YOLO

            # 保存配置
            self._erax_model_size = erax_size

            # ntd11
            ntd11_path = self._find_ntd11_model()
            if ntd11_path:
                try:
                    self._ntd11_model = YOLO(str(ntd11_path))
                    logger.info(f"[nsfw_part] ntd11 seg 模型已加载: {ntd11_path}")
                except Exception as e:
                    logger.error(f"[nsfw_part] ntd11 加载失败: {e}")
            else:
                logger.warning("[nsfw_part] ntd11 模型未找到")

            # EraX
            if use_erax:
                erax_path = self._find_erax_model(preferred_size=erax_size)
                if erax_path:
                    try:
                        self._erax_model = YOLO(str(erax_path))
                        logger.info(f"[nsfw_part] EraX 模型已加载: {erax_path}")
                    except Exception as e:
                        logger.error(f"[nsfw_part] EraX 加载失败: {e}")
                else:
                    logger.warning("[nsfw_part] EraX 模型未找到")

            # SAM2
            if use_sam2:
                self._sam2 = SAM2Refiner()

            if self._ntd11_model is None and self._erax_model is None:
                raise FileNotFoundError(
                    "未找到任何 NSFW 模型。\n"
                    f"ntd11: 请从 https://civitai.com/models/1313556 下载到 {NTD11_MODEL_DIR}/\n"
                    f"EraX: 请检查网络或手动下载 {ERAX_REPO}"
                )

            self._loaded = True
    
    def set_erax_size(self, size: str):
        """设置 EraX 模型大小并重新加载。
        
        Args:
            size: 模型大小 (n/s/m/l/x)
        """
        if size not in ["n", "s", "m", "l", "x"]:
            raise ValueError(f"无效的模型大小: {size}，可选: n/s/m/l/x")
        
        with self._lock:
            if self._erax_model_size == size and self._erax_model is not None:
                logger.info(f"[nsfw_part] EraX 模型已是 {size} 变体，无需重新加载")
                return
            
            self._erax_model_size = size
            self._erax_model = None
            
            # 重新加载 EraX
            erax_path = self._find_erax_model(preferred_size=size)
            if erax_path:
                try:
                    from ultralytics import YOLO
                    self._erax_model = YOLO(str(erax_path))
                    logger.info(f"[nsfw_part] EraX 模型已切换到 {size} 变体: {erax_path}")
                except Exception as e:
                    logger.error(f"[nsfw_part] EraX {size} 加载失败: {e}")
            else:
                logger.warning(f"[nsfw_part] EraX {size} 模型未找到")

    @property
    def ntd11(self):
        return self._ntd11_model

    @property
    def erax(self):
        return self._erax_model

    @property
    def sam2(self):
        return self._sam2


class NsfwPartDetector(BaseDetector):
    """按 NSFW 部位细分的像素级检测器。

    每个实例只关注一个部位（如 penis / pussy / nipple），
    通过共享的 ntd11 + EraX 模型实例进行检测，按类别 ID 过滤。

    ntd11 与 EraX 均优先输出原生像素级 seg mask。
    仅在 seg mask 不可用时回退到 bbox（此时可选择 SAM2 精细化）。
    """

    def __init__(self, part_name: str, conf: float = 0.3,
                 use_erax: bool = False, use_sam2: bool = False,
                 allow_bbox_fallback: bool = False):
        if part_name not in NSFW_PART_NAMES:
            raise ValueError(f"未知 NSFW 部位: {part_name}，可选: {NSFW_PART_NAMES}")
        super().__init__(part_name, conf)
        self.use_erax = use_erax
        self.use_sam2 = use_sam2
        self.allow_bbox_fallback = allow_bbox_fallback
        self._ntd11_class_ids = NTD11_CLASSES.get(part_name, [])
        self._erax_class_ids = ERAX_CLASSES.get(part_name, [])
        self._shared: _SharedNsfwModels | None = None

    def load_model(self):
        """加载共享模型（首次调用时实际加载，后续幂等）。"""
        self._shared = _SharedNsfwModels()
        self._shared.load(use_erax=self.use_erax, use_sam2=self.use_sam2)
        self._model = self._shared.ntd11 or self._shared.erax
        logger.info(f"[{self.part_name}] NSFW 部位检测器就绪 "
                    f"(ntd11_classes={self._ntd11_class_ids}, "
                    f"erax_classes={self._erax_class_ids}, "
                    f"use_erax={self.use_erax}, use_sam2={self.use_sam2}, "
                    f"allow_bbox_fallback={self.allow_bbox_fallback})")

    def _postprocess_mask(self, mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """按部位特性精修 mask，减少碎片、过大区域和离谱形状。"""
        if not np.any(mask > 0):
            return mask

        cfg = PART_POSTPROCESS.get(self.part_name, {})
        image_area = h * w
        min_area = max(1, int(image_area * cfg.get("min_area_ratio", 0.0)))
        max_area_ratio = cfg.get("max_area_ratio")
        max_area = int(image_area * max_area_ratio) if max_area_ratio is not None else None

        refined = filter_mask_components(
            mask,
            min_area=min_area,
            max_area=max_area,
            min_width=cfg.get("min_width", 0),
            min_height=cfg.get("min_height", 0),
            max_aspect_ratio=cfg.get("max_aspect_ratio"),
        )
        refined = morph_close_open(
            refined,
            close_kernel=cfg.get("close_kernel", 0),
            open_kernel=cfg.get("open_kernel", 0),
        )
        refined = dilate_mask(refined, kernel_size=cfg.get("dilate_kernel", 0))
        refined = smooth_mask_edges(refined, blur_radius=cfg.get("smooth_blur", 0))
        return refined

    def _detect_ntd11_part(self, image: np.ndarray, h: int, w: int) -> np.ndarray:
        """使用 ntd11 seg 模型检测特定部位，返回像素级 mask。"""
        mask = np.zeros((h, w), dtype=np.uint8)
        if self._shared.ntd11 is None or not self._ntd11_class_ids:
            return mask

        try:
            results = self._shared.ntd11.predict(
                image, conf=self.conf, imgsz=1024, verbose=False
            )
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    # 诊断: 打印 ntd11 的所有检测结果
                    model_names = result.names if hasattr(result, 'names') else {}
                    for i in range(len(result.boxes)):
                        cls_id = int(result.boxes.cls[i])
                        conf_val = float(result.boxes.conf[i])
                        cls_name = model_names.get(cls_id, f"unknown_{cls_id}")
                        is_target = cls_id in self._ntd11_class_ids
                        logger.info(f"[{self.part_name}] ntd11 原始检测: class={cls_id}({cls_name}), "
                                    f"conf={conf_val:.3f}, 目标匹配={'✓' if is_target else '✗'}")
                else:
                    logger.info(f"[{self.part_name}] ntd11: 无任何检测结果 (boxes=None 或空)")

                if result.masks is None or result.boxes is None:
                    continue
                for i, seg_mask in enumerate(result.masks.data):
                    cls_id = int(result.boxes.cls[i])
                    if cls_id not in self._ntd11_class_ids:
                        continue
                    seg_np = seg_mask.cpu().numpy()
                    part_mask = seg_to_mask(h, w, seg_np)
                    mask = np.maximum(mask, part_mask)
                    logger.debug(f"[{self.part_name}] ntd11 seg: class={cls_id}, "
                                f"conf={float(result.boxes.conf[i]):.3f}")
        except Exception as e:
            logger.error(f"[{self.part_name}] ntd11 检测失败: {e}")

        return mask

    def _detect_erax_part(self, image: np.ndarray, h: int, w: int) -> tuple[np.ndarray, list]:
        """使用 EraX YOLO11-seg 模型检测特定部位，优先使用原生 seg mask。

        检测流程：
        1. 优先读取 result.masks.data 输出像素级 seg mask
        2. 若 seg mask 不可用，可尝试 bbox → SAM2
        3. 默认禁止 bbox 矩形 fallback，避免输出大块方形遮罩
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        boxes = []
        if self._shared.erax is None or not self._erax_class_ids:
            return mask, boxes

        try:
            results = self._shared.erax.predict(
                image, conf=self.conf, imgsz=1024, verbose=False
            )
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    logger.info(f"[{self.part_name}] EraX: 无任何检测结果")
                    continue

                # 诊断: 打印 EraX 的所有检测结果
                model_names = result.names if hasattr(result, 'names') else {}
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    cls_name = model_names.get(cls_id, f"unknown_{cls_id}")
                    is_target = cls_id in self._erax_class_ids
                    logger.info(f"[{self.part_name}] EraX 原始检测: class={cls_id}({cls_name}), "
                                f"conf={conf_val:.3f}, 目标匹配={'✓' if is_target else '✗'}")
                    if not is_target:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))

                # ── 优先：原生 seg mask（EraX 是 YOLO11-seg）──
                if result.masks is not None and len(result.masks) > 0:
                    seg_count = 0
                    for i, box in enumerate(result.boxes):
                        cls_id = int(box.cls[0])
                        if cls_id not in self._erax_class_ids:
                            continue
                        if i < len(result.masks.data):
                            seg_mask = result.masks.data[i].cpu().numpy()
                            part_mask = seg_to_mask(h, w, seg_mask)
                            mask = np.maximum(mask, part_mask)
                            seg_count += 1
                    if seg_count > 0:
                        logger.info(f"[{self.part_name}] EraX 原生 seg mask: {seg_count} 个，"
                                   f"跳过 SAM2")
                        return mask, boxes
                    else:
                        logger.info(f"[{self.part_name}] EraX: 有 masks 但无目标类别匹配，"
                                   f"不使用 bbox 作为最终 mask")
                else:
                    logger.info(f"[{self.part_name}] EraX: 无 seg mask 输出，"
                               f"仅在 SAM2 或显式允许 bbox fallback 时补充")
        except Exception as e:
            logger.error(f"[{self.part_name}] EraX 检测失败: {e}")

        # ── 回退：bbox → SAM2；默认不允许直接输出 bbox 矩形 ──
        if boxes and not np.any(mask > 0):
            shrink_ratio = PART_SHRINK_RATIOS.get(self.part_name)
            sam2 = self._shared.sam2
            if sam2 and self.use_sam2:
                try:
                    mask = sam2.refine(
                        image,
                        boxes,
                        shrink_ratio=shrink_ratio,
                        allow_bbox_fallback=self.allow_bbox_fallback,
                    )
                    if np.any(mask > 0):
                        logger.info(f"[{self.part_name}] EraX bbox → SAM2 精细化完成 ({len(boxes)} 个, "
                                    f"收缩比例={shrink_ratio})")
                    else:
                        logger.info(f"[{self.part_name}] EraX bbox → SAM2 未产生有效 mask，"
                                    f"已拒绝矩形 fallback")
                except Exception as e:
                    logger.warning(f"[{self.part_name}] SAM2 精细化失败: {e}")
            elif self.allow_bbox_fallback:
                from ..utils import bbox_to_mask, shrink_bboxes
                shrunk = shrink_bboxes(boxes, shrink_ratio) if shrink_ratio else boxes
                mask = bbox_to_mask(h, w, shrunk)
                logger.info(f"[{self.part_name}] EraX bbox → 收缩矩形 mask ({len(boxes)} 个, "
                           f"收缩比例={shrink_ratio})")
            else:
                logger.info(f"[{self.part_name}] EraX 仅得到 bbox，已拒绝矩形 fallback")

        return mask, boxes

    def detect(self, image: np.ndarray) -> DetectionResult:
        """执行按部位过滤的 NSFW 检测。

        优先使用 ntd11 像素级分割；如 ntd11 未检出则补充 EraX bbox + SAM2。
        """
        self.ensure_loaded()
        h, w = image.shape[:2]

        # 阶段1：ntd11 像素级分割
        ntd11_mask = self._detect_ntd11_part(image, h, w)
        ntd11_has = np.any(ntd11_mask > 0)

        # 阶段2：EraX bbox + SAM2（仅当 ntd11 未检出或作为补充时）
        erax_mask = np.zeros((h, w), dtype=np.uint8)
        erax_boxes = []
        if self.use_erax and self._erax_class_ids:
            erax_mask, erax_boxes = self._detect_erax_part(image, h, w)

        # 合并：ntd11 seg 优先，EraX 补充
        mask = np.maximum(ntd11_mask, erax_mask)
        mask = self._postprocess_mask(mask, h, w)
        count = 0
        if ntd11_has:
            # 粗略计数：ntd11 的连通区域
            _, labels = cv2.connectedComponents((ntd11_mask > 0).astype(np.uint8))
            count = labels.max()
        count = max(count, len(erax_boxes))


        mask_pixels = np.sum(mask > 0)
        logger.info(f"[{self.part_name}] 检测完成: count={count}, "
                    f"mask_pixels={mask_pixels} ({mask_pixels/(h*w)*100:.2f}%), "
                    f"ntd11={'有' if ntd11_has else '无'}, erax_boxes={len(erax_boxes)}")

        return DetectionResult(
            part_name=self.part_name,
            mask=mask,
            confidence=float(self.conf) if count > 0 else 0.0,
            count=count,
        )
