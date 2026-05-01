"""SAM2 精细化器 — 将 bbox/point prompt 转换为像素级 mask。"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SAM2_MODEL_DIR = Path("model_cache/sam2")

# 推荐模型：sam2.1_l（精度优先，约 200MB）
DEFAULT_SAM2_MODEL = "sam2.1_l.pt"

# 支持的模型列表
SAM2_MODELS = [
    "sam2.1_t.pt",   # ~78MB,  最快
    "sam2.1_s.pt",   # ~110MB, 快
    "sam2.1_b.pt",   # ~375MB, 中等
    "sam2.1_l.pt",   # ~594MB, 最好
]


class SAM2Refiner:
    """SAM2 精细化器：支持 bbox prompt 和 point prompt。

    使用 ultralytics SAM 接口加载 SAM2 模型。
    支持动态切换模型，模型按名称缓存。
    """

    _models = {}  # model_name -> model instance
    _current_model_name = None

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or DEFAULT_SAM2_MODEL

    def _get_model(self, model_name: str):
        """获取或加载指定模型。"""
        if model_name in self._models and self._models[model_name] is not None:
            return self._models[model_name]

        try:
            from ultralytics import SAM

            SAM2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_path = str(SAM2_MODEL_DIR / model_name)

            if Path(model_path).exists():
                model = SAM(model_path)
                logger.info(f"[SAM2] 从本地加载模型: {model_path}")
            else:
                logger.info(f"[SAM2] 自动下载模型: {model_name}")
                model = SAM(model_name)
                # 移动到缓存目录
                import shutil
                auto_path = Path(model_name)
                if auto_path.exists() and not Path(model_path).exists():
                    shutil.move(str(auto_path), model_path)

            self._models[model_name] = model
            self._current_model_name = model_name
            logger.info(f"[SAM2] 模型已加载: {model_name}")
            return model

        except Exception as e:
            logger.error(f"[SAM2] 模型加载失败 ({model_name}): {e}")
            import traceback
            logger.error(f"[SAM2] 错误详情: {traceback.format_exc()}")
            self._models[model_name] = None
            return None

    def load_model(self, model_name: str | None = None):
        """加载指定模型（兼容旧接口）。"""
        name = model_name or self.model_name
        return self._get_model(name)

    def refine(self, image: np.ndarray, boxes: list[tuple[int, int, int, int]],
               shrink_ratio: float | None = None,
               allow_bbox_fallback: bool = True) -> np.ndarray:
        """将 bbox 列表通过 SAM2 精细化为像素级 mask。

        Args:
            image: 原图 (H, W, 3), BGR
            boxes: [(x1, y1, x2, y2), ...] 整数坐标
            shrink_ratio: bbox 收缩比例（0.0-1.0），用于减少回退时的遮盖范围。
            allow_bbox_fallback: SAM2 不可用/失败/无 mask 时是否允许回退到 bbox 矩形。

        Returns:
            uint8 mask (H, W), 0 or 255
        """
        if not boxes:
            h, w = image.shape[:2]
            logger.warning("[SAM2] 输入的 boxes 为空，返回空 mask")
            return np.zeros((h, w), dtype=np.uint8)

        model = self._get_model(self.model_name)

        if model is None:
            h, w = image.shape[:2]
            if not allow_bbox_fallback:
                return np.zeros((h, w), dtype=np.uint8)

            logger.warning("[SAM2] 模型不可用，回退到 bbox 矩形 mask")
            from ..utils import bbox_to_mask, shrink_bboxes
            fallback_boxes = shrink_bboxes(boxes, shrink_ratio) if shrink_ratio else boxes
            return bbox_to_mask(h, w, fallback_boxes)

        h, w = image.shape[:2]
        merged_mask = np.zeros((h, w), dtype=np.uint8)

        try:
            bboxes = [list(box) for box in boxes]
            logger.info(f"[SAM2] 开始 bbox 推理，输入 {len(bboxes)} 个 bbox")
            results = model.predict(image, bboxes=bboxes, verbose=False)

            mask_count = 0
            for result in results:
                if result.masks is not None:
                    for seg_mask in result.masks.data:
                        seg_np = seg_mask.cpu().numpy()
                        from ..utils import seg_to_mask
                        part_mask = seg_to_mask(h, w, seg_np)
                        merged_mask = np.maximum(merged_mask, part_mask)
                        mask_count += 1

            logger.info(f"[SAM2] 成功生成 {mask_count} 个 mask")

        except Exception as e:
            logger.error(f"[SAM2] bbox 推理失败: {e}")
            import traceback
            logger.error(f"[SAM2] 错误详情: {traceback.format_exc()}")
            if not allow_bbox_fallback:
                return np.zeros((h, w), dtype=np.uint8)

            from ..utils import bbox_to_mask, shrink_bboxes
            fallback_boxes = shrink_bboxes(boxes, shrink_ratio) if shrink_ratio else boxes
            return bbox_to_mask(h, w, fallback_boxes)

        if not np.any(merged_mask > 0):
            if not allow_bbox_fallback:
                return np.zeros((h, w), dtype=np.uint8)

            logger.warning("[SAM2] 未生成任何 mask，回退到收缩后的 bbox 矩形")
            from ..utils import bbox_to_mask, shrink_bboxes
            fallback_boxes = shrink_bboxes(boxes, shrink_ratio) if shrink_ratio else boxes
            return bbox_to_mask(h, w, fallback_boxes)

        return merged_mask

    def predict_from_points(self, image: np.ndarray,
                           points: list[tuple[int, int]],
                           labels: list[int] | None = None,
                           model_name: str | None = None) -> np.ndarray:
        """通过 point prompt 生成 mask。

        Args:
            image: 原图 (H, W, 3), BGR
            points: [(x, y), ...] 点击坐标列表
            labels: [1, 0, 1, ...] 每个点的标签，1=前景(正点), 0=背景(负点)
                    为 None 时全部视为正点。
            model_name: 指定使用的模型名，None 使用默认模型

        Returns:
            uint8 mask (H, W), 0 or 255
        """
        if not points:
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        name = model_name or self.model_name
        model = self._get_model(name)

        if model is None:
            logger.error(f"[SAM2] 模型不可用 ({name})")
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        h, w = image.shape[:2]

        # 默认全部为正点
        if labels is None:
            labels = [1] * len(points)

        # 确保 points 和 labels 长度一致
        points = points[:len(labels)]
        labels = labels[:len(points)]

        try:
            # ultralytics SAM 的 points 参数格式
            # points: [[x, y], ...]
            # labels: [1, 0, ...]  1=前景, 0=背景
            points_arr = [[float(p[0]), float(p[1])] for p in points]
            labels_arr = [int(l) for l in labels]

            logger.info(f"[SAM2] 开始 point 推理，{len(points_arr)} 个点 "
                       f"(正点={sum(labels_arr)}, 负点={len(labels_arr)-sum(labels_arr)}), 模型={name}")

            results = model.predict(image, points=points_arr, labels=labels_arr, verbose=False)

            merged_mask = np.zeros((h, w), dtype=np.uint8)
            mask_count = 0

            for result in results:
                if result.masks is not None:
                    for seg_mask in result.masks.data:
                        seg_np = seg_mask.cpu().numpy()
                        from ..utils import seg_to_mask
                        part_mask = seg_to_mask(h, w, seg_np)
                        merged_mask = np.maximum(merged_mask, part_mask)
                        mask_count += 1

            logger.info(f"[SAM2] point 推理完成，生成 {mask_count} 个 mask")
            return merged_mask

        except Exception as e:
            logger.error(f"[SAM2] point 推理失败: {e}")
            import traceback
            logger.error(f"[SAM2] 错误详情: {traceback.format_exc()}")
            return np.zeros((h, w), dtype=np.uint8)

    def predict_from_bbox(self, image: np.ndarray,
                         bbox: tuple[int, int, int, int],
                         model_name: str | None = None) -> np.ndarray:
        """通过单个 bbox 矩形框生成 mask。

        Args:
            image: 原图 (H, W, 3), BGR
            bbox: (x1, y1, x2, y2) 矩形框坐标
            model_name: 指定使用的模型名，None 使用默认模型

        Returns:
            uint8 mask (H, W), 0 or 255
        """
        x1, y1, x2, y2 = bbox
        
        # 确保 bbox 有效
        if x2 <= x1 or y2 <= y1:
            h, w = image.shape[:2]
            logger.warning(f"[SAM2] 无效的 bbox: {bbox}")
            return np.zeros((h, w), dtype=np.uint8)

        name = model_name or self.model_name
        model = self._get_model(name)

        if model is None:
            logger.error(f"[SAM2] 模型不可用 ({name})")
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        h, w = image.shape[:2]

        try:
            logger.info(f"[SAM2] 开始 bbox 推理，bbox=[{x1}, {y1}, {x2}, {y2}], 模型={name}")
            
            # ultralytics SAM bbox 格式: [[x1, y1, x2, y2]]
            bboxes = [[float(x1), float(y1), float(x2), float(y2)]]
            results = model.predict(image, bboxes=bboxes, verbose=False)

            merged_mask = np.zeros((h, w), dtype=np.uint8)
            mask_count = 0

            for result in results:
                if result.masks is not None:
                    for seg_mask in result.masks.data:
                        seg_np = seg_mask.cpu().numpy()
                        from ..utils import seg_to_mask
                        part_mask = seg_to_mask(h, w, seg_np)
                        merged_mask = np.maximum(merged_mask, part_mask)
                        mask_count += 1

            logger.info(f"[SAM2] bbox 推理完成，生成 {mask_count} 个 mask")
            return merged_mask

        except Exception as e:
            logger.error(f"[SAM2] bbox 推理失败: {e}")
            import traceback
            logger.error(f"[SAM2] 错误详情: {traceback.format_exc()}")
            return np.zeros((h, w), dtype=np.uint8)

    def predict_from_bbox_and_points(self, image: np.ndarray,
                                     bbox: tuple[int, int, int, int],
                                     points: list[tuple[int, int]] | None = None,
                                     labels: list[int] | None = None,
                                     model_name: str | None = None) -> np.ndarray:
        """通过 bbox + points 混合提示生成 mask。

        Args:
            image: 原图 (H, W, 3), BGR
            bbox: (x1, y1, x2, y2) 矩形框坐标
            points: [(x, y), ...] 可选的额外点击点
            labels: [1, 0, ...] 点标签，1=前景, 0=背景
            model_name: 指定使用的模型名

        Returns:
            uint8 mask (H, W), 0 or 255
        """
        name = model_name or self.model_name
        model = self._get_model(name)

        if model is None:
            logger.error(f"[SAM2] 模型不可用 ({name})")
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        try:
            logger.info(f"[SAM2] 开始 bbox+points 混合推理，bbox=[{x1},{y1},{x2},{y2}], "
                       f"points={len(points) if points else 0}, 模型={name}")

            # ultralytics SAM 支持同时传入 bboxes 和 points
            bboxes = [[float(x1), float(y1), float(x2), float(y2)]]
            
            points_arr = None
            labels_arr = None
            if points and len(points) > 0:
                if labels is None:
                    labels = [1] * len(points)
                points_arr = [[float(p[0]), float(p[1])] for p in points]
                labels_arr = [int(l) for l in labels[:len(points)]]

            results = model.predict(
                image, 
                bboxes=bboxes, 
                points=points_arr, 
                labels=labels_arr, 
                verbose=False
            )

            merged_mask = np.zeros((h, w), dtype=np.uint8)
            mask_count = 0

            for result in results:
                if result.masks is not None:
                    for seg_mask in result.masks.data:
                        seg_np = seg_mask.cpu().numpy()
                        from ..utils import seg_to_mask
                        part_mask = seg_to_mask(h, w, seg_np)
                        merged_mask = np.maximum(merged_mask, part_mask)
                        mask_count += 1

            logger.info(f"[SAM2] bbox+points 混合推理完成，生成 {mask_count} 个 mask")
            return merged_mask

        except Exception as e:
            logger.error(f"[SAM2] bbox+points 推理失败: {e}")
            import traceback
            logger.error(f"[SAM2] 错误详情: {traceback.format_exc()}")
            return np.zeros((h, w), dtype=np.uint8)

    def get_available_models(self) -> list[str]:
        """返回可用的模型列表（本地已下载 + 支持的模型）。"""
        available = []
        for name in SAM2_MODELS:
            path = SAM2_MODEL_DIR / name
            available.append({
                'name': name,
                'cached': path.exists(),
                'default': name == DEFAULT_SAM2_MODEL
            })
        return available

    def is_available(self, model_name: str | None = None) -> bool:
        """检查 SAM2 是否可用。"""
        name = model_name or self.model_name
        model = self._get_model(name)
        return model is not None

    @classmethod
    def reset(cls):
        """重置所有缓存的模型（用于测试或强制重新加载）。"""
        cls._models.clear()
        cls._current_model_name = None
        logger.info("[SAM2] 所有模型缓存已重置")
