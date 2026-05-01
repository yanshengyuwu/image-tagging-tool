"""DeepGHS 动漫人物检测器 — 使用 imgutils 提供的最新模型。

相比旧的本地 ONNX 模型，优势：
- 自动下载/缓存最新模型
- 支持多版本多尺寸 (n/s/m/x, v0/v1/v1.1)
- 内置 NMS
- 更好的动漫人物检测精度
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PersonInstance:
    """单个人物实例。"""
    person_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    area: int
    center: tuple[int, int]  # (cx, cy)


class DeepGHSPersonDetector:
    """DeepGHS 人物检测器：使用 imgutils 的 detect_person。

    接口与旧的 PersonDetector 完全兼容，可直接替换。
    """

    def __init__(
        self,
        conf: float = 0.5,
        level: str = "m",
        version: str = "v1.1",
        iou_threshold: float = 0.5,
        max_detections: int | None = 15,
        min_area_ratio: float = 0.01,
    ):
        """
        Args:
            conf: 置信度阈值，默认 0.5（比旧模型的 0.3 更严格，减少假阳性）
            level: 模型尺寸 'n'/'s'/'m'/'x'，默认 'm'（平衡速度与精度）
            version: 模型版本 'v0'/'v1'/'v1.1'，默认 'v1.1'（最新）
            iou_threshold: NMS IoU 阈值
            max_detections: 最大检测数量，None 表示不限制
            min_area_ratio: 最小面积占全图比例，过滤碎片假阳性
        """
        self.conf = conf
        self.level = level
        self.version = version
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.min_area_ratio = min_area_ratio
        self._initialized = False

    def initialize(self):
        """惰性初始化：首次 detect 时才导入 imgutils。"""
        if self._initialized:
            return
        try:
            from imgutils.detect import detect_person
            self._detect_person = detect_person
            self._initialized = True
            logger.info(
                f"[DeepGHSPersonDetector] 已初始化 "
                f"(level={self.level}, version={self.version}, conf={self.conf})"
            )
        except ImportError as e:
            logger.error(f"[DeepGHSPersonDetector] imgutils 未安装: {e}")
            raise

    def detect_persons(self, image: np.ndarray) -> list[PersonInstance]:
        """检测图像中的所有人物。

        Args:
            image: BGR 格式的 numpy 数组

        Returns:
            按面积降序排列的 PersonInstance 列表
        """
        if not self._initialized:
            self.initialize()

        h, w = image.shape[:2]

        try:
            # imgutils 需要 PIL Image 或文件路径，不能直接传 numpy
            from PIL import Image
            pil_image = Image.fromarray(image[:, :, ::-1])  # BGR → RGB
            results = self._detect_person(
                pil_image,
                level=self.level,
                version=self.version,
                conf_threshold=self.conf,
                iou_threshold=self.iou_threshold,
            )
        except Exception as e:
            logger.error(f"[DeepGHSPersonDetector] 检测失败: {e}")
            # 回退到全图
            return [self._full_image_person(h, w)]

        if not results:
            logger.info("[DeepGHSPersonDetector] 未检测到人物，返回全图")
            return [self._full_image_person(h, w)]

        # 转换为 PersonInstance
        persons = []
        min_area = h * w * self.min_area_ratio
        for ((x1, y1, x2, y2), label, conf) in results:
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue
            persons.append(PersonInstance(
                person_id=0,  # 稍后重分配
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                area=area,
                center=((x1 + x2) // 2, (y1 + y2) // 2),
            ))

        if not persons:
            return [self._full_image_person(h, w)]

        # 按面积降序排列
        persons.sort(key=lambda p: p.area, reverse=True)

        # 限制最大检测数
        if self.max_detections is not None and len(persons) > self.max_detections:
            logger.warning(
                f"[DeepGHSPersonDetector] 检测到 {len(persons)} 个人物，"
                f"限制为 Top-{self.max_detections}"
            )
            persons = persons[:self.max_detections]

        # 重新分配 ID
        for i, p in enumerate(persons):
            p.person_id = i

        logger.info(f"[DeepGHSPersonDetector] 检测到 {len(persons)} 个人物")
        return persons

    @staticmethod
    def _full_image_person(h: int, w: int) -> PersonInstance:
        """返回全图作为单个人物（回退模式）。"""
        return PersonInstance(
            person_id=0,
            bbox=(0, 0, w, h),
            confidence=1.0,
            area=h * w,
            center=(w // 2, h // 2),
        )

    def get_person_mask(self, person: PersonInstance, h: int, w: int) -> np.ndarray:
        """根据人物 bbox 生成二值 mask。"""
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = person.bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        mask[y1:y2, x1:x2] = 255
        return mask
