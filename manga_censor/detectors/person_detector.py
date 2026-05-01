"""人物检测器 — 使用 DeepGHS 动漫人物检测模型。

优先使用专门的动漫人物检测模型 (person_detect_v1.3_s)
如果不可用，降级为全图模式
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


@dataclass
class PersonInstance:
    """单个人物实例。"""
    person_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    area: int
    center: tuple[int, int]  # (cx, cy)


class PersonDetector:
    """人物检测器：使用 DeepGHS 动漫人物检测模型。"""

    def __init__(self, conf: float = 0.3):
        self.conf = conf
        self._session = None
        self._initialized = False
        self._fallback_mode = False
        self._model_path = Path("model_cache/anime_person_detection/person_detect_v1.3_s.onnx")

    def initialize(self):
        """加载动漫人物检测模型。"""
        if not self._initialized:
            try:
                if not self._model_path.exists():
                    logger.warning(f"[PersonDetector] 模型文件不存在: {self._model_path}")
                    logger.warning("[PersonDetector] 启用降级模式：返回全图作为单个人物区域")
                    self._fallback_mode = True
                    self._initialized = True
                    return
                
                logger.info(f"[PersonDetector] 加载动漫人物检测模型: {self._model_path.name}")
                self._session = ort.InferenceSession(
                    str(self._model_path),
                    providers=['CPUExecutionProvider']
                )
                self._initialized = True
                logger.info("[PersonDetector] 动漫人物检测模型已加载")
                
            except Exception as e:
                logger.warning(f"[PersonDetector] 模型加载失败: {e}")
                logger.warning("[PersonDetector] 启用降级模式：返回全图作为单个人物区域")
                self._fallback_mode = True
                self._initialized = True

    def _preprocess(self, image: np.ndarray, target_size: int = 640) -> tuple[np.ndarray, float]:
        """预处理图像用于 YOLO 推理。"""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        top = pad_h // 2
        left = pad_w // 2
        
        padded = cv2.copyMakeBorder(
            resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # CHW format
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, scale

    def _postprocess(self, output: np.ndarray, scale: float, orig_h: int, orig_w: int) -> list[PersonInstance]:
        """后处理 YOLO 输出。"""
        # output shape: (1, 5, 8400) -> (num_boxes, 5)
        # 5 = [x, y, w, h, conf]
        output = output[0].T  # (8400, 5)
        
        persons = []
        for detection in output:
            x_center, y_center, width, height, conf = detection
            
            if conf < self.conf:
                continue
            
            # Convert to original image coordinates
            x_center /= scale
            y_center /= scale
            width /= scale
            height /= scale
            
            # Convert to xyxy format
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Clip to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            area = (x2 - x1) * (y2 - y1)
            
            # Filter small boxes
            if area < (orig_h * orig_w * 0.005):
                continue
            
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            persons.append(PersonInstance(
                person_id=len(persons),
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                area=area,
                center=(cx, cy)
            ))
        
        return persons

    def detect_persons(self, image: np.ndarray) -> list[PersonInstance]:
        """
        检测图像中的所有人物，返回 PersonInstance 列表。

        Args:
            image: BGR 格式的 numpy 数组

        Returns:
            按面积降序排列的 PersonInstance 列表
        """
        if not self._initialized:
            self.initialize()

        h, w = image.shape[:2]

        # 降级模式：返回全图作为单个人物
        if self._fallback_mode:
            return [PersonInstance(
                person_id=0,
                bbox=(0, 0, w, h),
                confidence=1.0,
                area=h * w,
                center=(w // 2, h // 2)
            )]

        try:
            # 预处理
            input_tensor, scale = self._preprocess(image)
            
            # 推理
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: input_tensor})[0]
            
            # 后处理
            persons = self._postprocess(output, scale, h, w)

            # 如果没有检测到任何人物，返回全图作为默认
            if not persons:
                logger.info("[PersonDetector] 未检测到人物，返回全图作为默认区域")
                return [PersonInstance(
                    person_id=0,
                    bbox=(0, 0, w, h),
                    confidence=1.0,
                    area=h * w,
                    center=(w // 2, h // 2)
                )]

            # 按面积降序排列（最大的人物排在前面）
            persons.sort(key=lambda p: p.area, reverse=True)

            # 重新分配 ID
            for i, p in enumerate(persons):
                p.person_id = i

            logger.info(f"[PersonDetector] 检测到 {len(persons)} 个人物")
            return persons

        except Exception as e:
            logger.error(f"[PersonDetector] 推理失败: {e}，回退到全图模式")
            import traceback
            logger.error(traceback.format_exc())
            return [PersonInstance(
                person_id=0,
                bbox=(0, 0, w, h),
                confidence=1.0,
                area=h * w,
                center=(w // 2, h // 2)
            )]

    def get_person_mask(self, person: PersonInstance, h: int, w: int) -> np.ndarray:
        """根据人物 bbox 生成二值 mask。"""
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = person.bbox
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        mask[y1:y2, x1:x2] = 255
        return mask
