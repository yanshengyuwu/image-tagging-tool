"""检测器模块。"""

from .base import BaseDetector, DetectionResult
from .deepghs_bbox import DeepghsBboxDetector
from .bbox_sam2 import BboxSam2Detector
from .anzhc_seg import AnzhcSegDetector
from .nsfw_seg import NsfwSegDetector
from .nsfw_part_seg import NsfwPartDetector, NSFW_PART_NAMES
from .text_detect import TextBubbleDetector
from .person_detector import PersonDetector, PersonInstance
from .gender_classifier import GenderClassifier
from .part_assignment import (
    assign_parts_to_persons,
    get_parts_by_person,
    merge_person_masks,
    calculate_iou
)
from .strategy_engine import StrategyEngine, StrategyConfig
from .gender_aware_pipeline import (
    GenderAwarePipeline,
    PipelineResult,
    PersonMaskResult
)

__all__ = [
    # 基础组件
    "BaseDetector",
    "DetectionResult",
    # 检测器
    "DeepghsBboxDetector",
    "BboxSam2Detector",
    "AnzhcSegDetector",
    "NsfwSegDetector",
    "NsfwPartDetector",
    "NSFW_PART_NAMES",
    "TextBubbleDetector",
    "PersonDetector",
    "PersonInstance",
    # 性别分类
    "GenderClassifier",
    # 部位分配
    "assign_parts_to_persons",
    "get_parts_by_person",
    "merge_person_masks",
    "calculate_iou",
    # 策略引擎
    "StrategyEngine",
    "StrategyConfig",
    # 性别感知 Pipeline
    "GenderAwarePipeline",
    "PipelineResult",
    "PersonMaskResult",
]
