"""性别感知的遮盖 Pipeline — 整合所有检测器，支持多人物场景。

工作流程：
1. 检测所有人物
2. 检测所有部位（完整流程）
3. 将部位分配给对应人物
4. 对每个人物进行性别分类
5. 应用策略引擎，筛选输出 mask
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from .base import BaseDetector, DetectionResult
from .person_detector import PersonDetector, PersonInstance
from .deepghs_person import DeepGHSPersonDetector
from .gender_classifier import GenderClassifier
from .strategy_engine import StrategyEngine, StrategyConfig
from .part_assignment import assign_parts_to_persons, get_parts_by_person, merge_person_masks
from .sam2_refiner import SAM2Refiner
from ..utils import cv2_imwrite

logger = logging.getLogger(__name__)


@dataclass
class PersonMaskResult:
    """单个人物的遮盖结果。"""
    person_id: int
    gender: str
    gender_confidence: float
    bboxes: Dict[str, Tuple[int, int, int, int]]
    masks: Dict[str, np.ndarray]
    final_mask: Optional[np.ndarray] = None  # 合并后的最终 mask


@dataclass
class PipelineResult:
    """完整 Pipeline 的处理结果。"""
    persons: List[PersonInstance]
    all_parts: Dict[str, DetectionResult]  # 所有检测到的部位
    person_results: List[PersonMaskResult]
    final_masks: List[np.ndarray]  # 所有人物合并后的 mask
    metadata: Dict = field(default_factory=dict)


class GenderAwarePipeline:
    """性别感知的遮盖 Pipeline。"""

    def __init__(
        self,
        detectors: Dict[str, BaseDetector],
        config_path: str = "mask_config.yaml",
        person_detector_type: str = "deepghs",
        person_conf: float = 0.5,
        person_level: str = "m",
        person_version: str = "v1.1",
        person_iou: float = 0.5,
        person_max_det: int | None = 15,
    ):
        """
        Args:
            detectors: 部位检测器字典 {part_name: detector}
            config_path: 策略配置文件路径
            person_detector_type: 人物检测器类型 'deepghs' 或 'legacy'
            person_conf: 人物检测置信度阈值
            person_level: DeepGHS 模型尺寸 n/s/m/x
            person_version: DeepGHS 模型版本 v0/v1/v1.1
            person_iou: NMS IoU 阈值
            person_max_det: 最大检测数量
        """
        self.detectors = detectors

        # 选择人物检测器
        if person_detector_type == "deepghs":
            logger.info(f"[gender_aware_pipeline] 使用 DeepGHS 人物检测器 "
                       f"(conf={person_conf}, level={person_level}, version={person_version})")
            self.person_detector = DeepGHSPersonDetector(
                conf=person_conf,
                level=person_level,
                version=person_version,
                iou_threshold=person_iou,
                max_detections=person_max_det,
            )
        else:
            logger.info("[gender_aware_pipeline] 使用旧版 ONNX 人物检测器")
            self.person_detector = PersonDetector(conf=person_conf)

        self.gender_classifier = GenderClassifier()
        self.strategy_engine = StrategyEngine(config_path)
        self.sam2_refiner = None  # 懒加载

        self._initialized = False

    def initialize(self):
        """初始化所有检测器。"""
        logger.info("[gender_aware_pipeline] 开始初始化...")

        # 初始化人物检测器
        self.person_detector.initialize()
        self.gender_classifier.initialize()

        # 初始化所有部位检测器
        for part_name, detector in self.detectors.items():
            if not detector.is_loaded:
                logger.info(f"[gender_aware_pipeline] 加载 {part_name} 检测器...")
                detector.load_model()

        # 懒加载 SAM2（仅在需要 full_body 时加载）
        try:
            self.sam2_refiner = SAM2Refiner()
            logger.info("[gender_aware_pipeline] SAM2 refiner 已就绪")
        except Exception as e:
            logger.warning(f"[gender_aware_pipeline] SAM2 初始化失败: {e}")
            self.sam2_refiner = None

        self._initialized = True
        logger.info(f"[gender_aware_pipeline] 初始化完成，共 {len(self.detectors)} 个部位检测器")

    def process(self, image: np.ndarray) -> PipelineResult:
        """处理单张图像。
        
        Args:
            image: BGR 图像
        
        Returns:
            PipelineResult
        """
        if not self._initialized:
            self.initialize()

        h, w = image.shape[:2]
        logger.info(f"[gender_aware_pipeline] 开始处理图像 {w}x{h}")

        # 步骤 1: 检测所有人物
        logger.info("[gender_aware_pipeline] 步骤1: 检测人物...")
        persons = self.person_detector.detect_persons(image)
        logger.info(f"[gender_aware_pipeline] 检测到 {len(persons)} 个人物")

        if not persons:
            logger.warning("[gender_aware_pipeline] 未检测到人物")
            return PipelineResult(
                persons=[],
                all_parts={},
                person_results=[],
                final_masks=[],
                metadata={"error": "no_persons_detected"}
            )

        # 步骤 2: 检测所有部位
        logger.info("[gender_aware_pipeline] 步骤2: 检测部位...")
        all_parts = {}
        # 排除全图型检测器（如 nsfw_seg），它们不适合 per-person 分配
        excluded_parts = {"nsfw"}
        for part_name, detector in self.detectors.items():
            if part_name in excluded_parts:
                logger.info(f"  跳过全图型检测器: {part_name} (不适合 per-person 分配)")
                continue
            logger.debug(f"  检测 {part_name}...")
            result = detector.detect(image)
            all_parts[part_name] = result
            logger.debug(f"  {part_name}: {result.count} 个检测结果")

        # 步骤 3: 部位分配（IoU 匹配）
        logger.info("[gender_aware_pipeline] 步骤3: 分配部位...")
        part_masks = {name: result.mask for name, result in all_parts.items()}
        person_parts = assign_parts_to_persons(part_masks, persons)

        # 步骤 4: 性别分类
        logger.info("[gender_aware_pipeline] 步骤4: 性别分类...")
        parts_by_person = get_parts_by_person(person_parts)
        gender_results = self.gender_classifier.batch_classify(image, persons, parts_by_person)

        # 打印性别分类结果（中文显示）
        _GENDER_ZH = {"male": "男", "female": "女"}
        for person_id, (gender, conf) in gender_results.items():
            zh = _GENDER_ZH.get(gender, gender)
            logger.info(f"  人物 {person_id}: {zh} ({gender}, 置信度: {conf:.2f})")

        # 步骤 5: 应用策略引擎（支持降级机制）
        logger.info("[gender_aware_pipeline] 步骤5: 应用遮盖策略...")
        person_results = []
        for person in persons:
            person_id = person.person_id
            gender, gender_conf = gender_results.get(person_id, ("unknown", 0.0))
            parts = person_parts.get(person_id, {})
            
            logger.info(f"[gender_aware_pipeline] 处理人物 {person_id}, "
                       f"性别: {gender} (置信度: {gender_conf:.2f}), "
                       f"可用部位: {list(parts.keys())}")

            # 检查是否为 full_body 模式
            strategy = self.strategy_engine.get_strategy(gender)
            if strategy.mode == "full_body":
                # full_body 模式：使用 person bbox + SAM2 生成完整人物 mask
                logger.info(f"[gender_aware_pipeline] 人物 {person_id} 使用 full_body 模式")
                x1, y1, x2, y2 = person.bbox
                
                if self.sam2_refiner is not None:
                    try:
                        person_mask = self.sam2_refiner.refine(
                            image,
                            [[x1, y1, x2, y2]],
                            allow_bbox_fallback=True
                        )
                        logger.info(f"[gender_aware_pipeline] 人物 {person_id} SAM2 生成 mask: "
                                   f"{person_mask.sum()} 像素")
                    except Exception as e:
                        logger.warning(f"[gender_aware_pipeline] 人物 {person_id} SAM2 失败: {e}，"
                                      f"回退到 bbox")
                        person_mask = self.person_detector.get_person_mask(person, h, w)
                else:
                    logger.info(f"[gender_aware_pipeline] 人物 {person_id} 无 SAM2，使用 bbox")
                    person_mask = self.person_detector.get_person_mask(person, h, w)
                
                # full_body 模式下，filtered_parts 只包含一个 "person" mask
                filtered_parts = {"person": person_mask}
            else:
                # custom 模式：应用策略筛选（支持降级机制）
                # 为了支持降级，需要先创建一个包含图像尺寸信息的 dummy mask
                if not parts:
                    # 如果没有任何部位，创建一个空 mask 用于获取图像尺寸
                    dummy_mask = np.zeros((h, w), dtype=np.uint8)
                    parts = {"_dummy": dummy_mask}
                
                filtered_parts = self.strategy_engine.apply_strategy(
                    gender, 
                    parts,
                    person_bbox=person.bbox,
                    enable_fallback=True
                )
                
                # 移除 dummy mask
                if "_dummy" in filtered_parts:
                    del filtered_parts["_dummy"]
            
            logger.info(f"[gender_aware_pipeline] 人物 {person_id} 筛选后部位: "
                       f"{list(filtered_parts.keys())}, "
                       f"总像素: {sum(m.sum() for m in filtered_parts.values())}")

            # 提取 bboxes
            bboxes = {}
            for part_name in filtered_parts:
                # 从 mask 中计算 bbox
                mask = filtered_parts[part_name]
                if mask.sum() > 0:
                    rows, cols = np.where(mask > 0)
                    if len(rows) > 0:
                        x1, y1 = cols.min(), rows.min()
                        x2, y2 = cols.max(), rows.max()
                        bboxes[part_name] = (x1, y1, x2, y2)

            # 合并 mask
            final_mask = None
            if self.strategy_engine.should_merge(gender):
                final_mask = merge_person_masks(filtered_parts, list(filtered_parts.keys()))
                if final_mask is not None:
                    logger.info(f"[gender_aware_pipeline] 人物 {person_id} 合并 mask: "
                               f"{final_mask.sum()} 像素")
            else:
                # 不合并，保留各部位的独立 mask
                pass

            person_result = PersonMaskResult(
                person_id=person_id,
                gender=gender,
                gender_confidence=gender_conf,
                bboxes=bboxes,
                masks=filtered_parts,
                final_mask=final_mask
            )
            person_results.append(person_result)

        # 步骤 6: 合并所有人物的结果
        final_masks = []
        for pr in person_results:
            if pr.final_mask is not None:
                final_masks.append(pr.final_mask)
            else:
                # 合并所有部位的 mask
                merged = np.zeros((h, w), dtype=np.uint8)
                for part_name, mask in pr.masks.items():
                    merged = np.maximum(merged, mask)
                if merged.sum() > 0:
                    final_masks.append(merged)

        logger.info(f"[gender_aware_pipeline] 处理完成，生成 {len(final_masks)} 个 mask")

        return PipelineResult(
            persons=persons,
            all_parts=all_parts,
            person_results=person_results,
            final_masks=final_masks,
            metadata={
                "gender_summary": {
                    pr.person_id: pr.gender for pr in person_results
                }
            }
        )

    def save_masks(
        self,
        result: PipelineResult,
        output_dir: str,
        prefix: str = "mask",
        invert: bool = False,
        merge_single: bool = False,
        save_individual: bool = False
    ) -> List[str]:
        """保存 mask 到文件。
        
        Args:
            result: PipelineResult
            output_dir: 输出目录
            prefix: 文件名前缀
            invert: 是否反相 mask
            merge_single: 是否合并所有 mask 为单个文件
            save_individual: 是否保存各部位独立的 mask
        
        Returns:
            保存的文件路径列表
        """
        import os
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        def _apply_invert(mask):
            if invert and mask is not None:
                return cv2.bitwise_not(mask)
            return mask

        saved_files = []

        if save_individual:
            # 保存每个部位的独立 mask
            for person_result in result.person_results:
                person_id = person_result.person_id
                gender = person_result.gender

                for part_name, mask in person_result.masks.items():
                    if mask.sum() > 0:
                        filename = f"{prefix}_{person_id}_{gender}_{part_name}.png"
                        filepath = output_path / filename
                        cv2_imwrite(str(filepath), _apply_invert(mask))
                        saved_files.append(str(filepath))

        if merge_single:
            # 合并所有人物的所有 mask 为单个文件
            all_masks = result.final_masks if result.final_masks else []
            if all_masks:
                final_merged = np.zeros_like(all_masks[0])
                for mask in all_masks:
                    final_merged = np.maximum(final_merged, mask)
                filepath = output_path / f"{prefix}_mask.png"
                cv2_imwrite(str(filepath), _apply_invert(final_merged))
                saved_files.append(str(filepath))
        elif not save_individual:
            # 默认行为：保存每个人物的合并 mask + 总合并
            for person_result in result.person_results:
                person_id = person_result.person_id
                gender = person_result.gender

                if person_result.final_mask is not None:
                    filename = f"{prefix}_{person_id}_{gender}_merged.png"
                    filepath = output_path / filename
                    cv2_imwrite(str(filepath), _apply_invert(person_result.final_mask))
                    saved_files.append(str(filepath))

            if result.final_masks:
                final_merged = np.zeros_like(result.final_masks[0])
                for mask in result.final_masks:
                    final_merged = np.maximum(final_merged, mask)
                filepath = output_path / f"{prefix}_all_merged.png"
                cv2_imwrite(str(filepath), _apply_invert(final_merged))
                saved_files.append(str(filepath))

        return saved_files

    def apply_to_image(
        self,
        image: np.ndarray,
        result: PipelineResult,
        color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """将 mask 应用到图像上（遮盖）。"""
        h, w = image.shape[:2]
        masked_image = image.copy()

        for mask in result.final_masks:
            if mask is not None and mask.sum() > 0:
                # 创建彩色遮盖
                color_mask = np.zeros_like(image)
                color_mask[mask > 0] = color
                masked_image = np.where(color_mask > 0, color_mask, masked_image)

        return masked_image
