"""策略引擎 — 根据性别（二元：male / female）和用户配置决定输出哪些 mask。

仅支持两种性别：male / female。任何未知输入统一归为 female。
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """策略配置数据类。"""
    mode: str  # "full_body" | "custom"
    custom_parts: List[str]
    merge_to_single_mask: bool


class StrategyEngine:
    """策略引擎 — 仅管理 male / female 两种性别的遮盖策略。"""

    VALID_GENDERS = ("male", "female")

    PRESET_STRATEGIES = {
        "full_body": {
            "parts": ["person"],
            "description": "完全遮盖整个人物",
        },
        "custom": {
            "parts": [],
            "description": "按部位列表自定义筛选",
        },
    }

    # 默认兜底策略
    _DEFAULT_MALE = StrategyConfig(
        mode="full_body", custom_parts=["person"], merge_to_single_mask=True
    )
    _DEFAULT_FEMALE = StrategyConfig(
        mode="custom",
        custom_parts=["face", "eyes"],
        merge_to_single_mask=True,
    )

    def __init__(self, config_path: str = "mask_config.yaml"):
        self.config_path = config_path
        self.strategies: Dict[str, StrategyConfig] = {}
        self._load_config()

    def _load_config(self):
        """从 YAML 加载 male_strategy / female_strategy。"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            gender_mask = config.get("gender_mask", {})

            for gender in self.VALID_GENDERS:
                key = f"{gender}_strategy"
                # 支持两种位置：顶层 或 gender_mask 下
                data = gender_mask.get(key) or config.get(key)
                if data:
                    self.strategies[gender] = self._parse_strategy(data)

            # 兜底
            if "male" not in self.strategies:
                self.strategies["male"] = self._DEFAULT_MALE
            if "female" not in self.strategies:
                self.strategies["female"] = self._DEFAULT_FEMALE

            logger.info(
                f"[strategy_engine] 已加载策略: "
                f"male={self.strategies['male'].mode}, "
                f"female={self.strategies['female'].mode}"
            )

        except FileNotFoundError:
            logger.warning(f"[strategy_engine] 配置文件不存在: {self.config_path}，使用默认策略")
            self._load_default_strategies()
        except Exception as e:
            logger.error(f"[strategy_engine] 加载配置失败: {e}，使用默认策略")
            self._load_default_strategies()

    def _parse_strategy(self, strategy_data: dict) -> StrategyConfig:
        """解析单个策略配置。"""
        mode = strategy_data.get("mode", "custom")
        merge = strategy_data.get("merge_to_single_mask", True)

        if mode == "full_body":
            parts = list(self.PRESET_STRATEGIES["full_body"]["parts"])
        else:
            if mode != "custom":
                logger.warning(f"[strategy_engine] 未知模式: {mode}，回退为 custom")
                mode = "custom"
            parts = list(strategy_data.get("custom_parts", []) or [])

        return StrategyConfig(
            mode=mode,
            custom_parts=parts,
            merge_to_single_mask=merge,
        )

    def _load_default_strategies(self):
        self.strategies["male"] = self._DEFAULT_MALE
        self.strategies["female"] = self._DEFAULT_FEMALE

    # ── 公共 API ─────────────────────────────────────────────

    def _normalize_gender(self, gender: str) -> str:
        """将任意输入规范化为 male / female。"""
        g = (gender or "").lower().strip()
        if g in self.VALID_GENDERS:
            return g
        # 任何未知/旧分类 → female
        logger.debug(f"[strategy_engine] 未知性别 '{gender}'，归为 female")
        return "female"

    def get_strategy(self, gender: str) -> StrategyConfig:
        """获取策略（强制二元）。"""
        g = self._normalize_gender(gender)
        return self.strategies.get(g, self._DEFAULT_FEMALE)

    def should_merge(self, gender: str) -> bool:
        return self.get_strategy(gender).merge_to_single_mask

    def get_target_parts(self, gender: str) -> List[str]:
        return self.get_strategy(gender).custom_parts

    # ── 策略应用 ─────────────────────────────────────────────

    def apply_strategy(
        self,
        gender: str,
        person_parts: Dict[str, np.ndarray],
        person_bbox: Optional[tuple] = None,
        enable_fallback: bool = True,
    ) -> Dict[str, np.ndarray]:
        """应用策略，筛选输出 mask（支持降级）。

        特殊标记 `__skip__`（出现在 custom_parts 中）表示该性别完全不遮盖，
        返回空字典、不触发任何降级逻辑。
        """
        g = self._normalize_gender(gender)
        strategy = self.get_strategy(g)

        # 显式跳过标记（前端 custom 且未勾选任何部位时由 app.py 设置）
        if strategy.mode == "custom" and strategy.custom_parts == ["__skip__"]:
            logger.info(f"[strategy_engine] 性别 {g} 策略为 __skip__，不遮盖")
            return {}

        # full_body 模式
        if strategy.mode == "full_body":
            if person_parts:
                logger.info(
                    f"[strategy_engine] 性别 {g} 使用 full_body，"
                    f"包含 {len(person_parts)} 个部位: {list(person_parts.keys())}"
                )
                return person_parts
            if enable_fallback and person_bbox:
                logger.warning(f"[strategy_engine] {g} full_body 无部位，降级使用 person_bbox")
                return self._create_bbox_mask(person_bbox, person_parts)
            logger.warning(f"[strategy_engine] {g} full_body 无部位且无法降级")
            return {}

        # custom 模式
        target_parts = set(strategy.custom_parts)
        if not target_parts:
            logger.warning(
                f"[strategy_engine] 性别 {g} custom 模式但 custom_parts 为空 → "
                f"不遮盖任何部位（应在上游修复）"
            )
            return {}

        logger.info(
            f"[strategy_engine] 性别 {g} custom 模式，目标部位: {target_parts}，"
            f"可用部位: {list(person_parts.keys())}"
        )

        filtered = {
            name: mask for name, mask in person_parts.items() if name in target_parts
        }

        if not filtered and enable_fallback:
            logger.warning(
                f"[strategy_engine] 目标部位未命中! 目标={target_parts}，可用={list(person_parts.keys())}"
            )
            if person_parts:
                logger.info("[strategy_engine] 降级 1: 使用所有可用部位")
                return person_parts
            if person_bbox:
                logger.info("[strategy_engine] 降级 2: 使用 person_bbox")
                return self._create_bbox_mask(person_bbox, person_parts)
            logger.error("[strategy_engine] 所有降级均失败")
            return {}

        return filtered

    def _create_bbox_mask(
        self,
        bbox: tuple,
        person_parts: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if person_parts:
            first_mask = next(iter(person_parts.values()))
            h, w = first_mask.shape[:2]
        else:
            logger.error("[strategy_engine] 无法确定图像尺寸，无法创建 bbox mask")
            return {}

        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        logger.info(
            f"[strategy_engine] 创建 bbox mask: ({x1},{y1},{x2},{y2}), size={w}x{h}"
        )
        return {"person_bbox": mask}
