"""检测器抽象基类。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class DetectionResult:
    """单个部位的检测结果。"""
    part_name: str
    mask: np.ndarray  # H×W uint8, 0 or 255
    confidence: float  # 平均置信度
    count: int  # 检测到的实例数量


class BaseDetector(ABC):
    """所有检测器的抽象基类。"""

    def __init__(self, part_name: str, conf: float = 0.5):
        self.part_name = part_name
        self.conf = conf
        self._model = None

    @abstractmethod
    def load_model(self):
        """加载模型到内存。"""
        ...

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        对输入图像执行检测，返回二值 mask。

        Args:
            image: BGR 格式的 numpy 数组 (H, W, 3)

        Returns:
            DetectionResult，其中 mask 为与原图同尺寸的单通道二值图
        """
        ...

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self):
        if not self.is_loaded:
            self.load_model()

    def empty_result(self, h: int, w: int) -> DetectionResult:
        """返回空的检测结果（全黑 mask）。"""
        return DetectionResult(
            part_name=self.part_name,
            mask=np.zeros((h, w), dtype=np.uint8),
            confidence=0.0,
            count=0,
        )
