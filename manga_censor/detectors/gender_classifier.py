"""性别分类器 - 基于 wd14 tagger 的动漫人物二元性别识别。

由于 deepghs 官方并未提供独立的 gender 分类模型，这里改用已被广泛验证的
**wd14 tagger**（通过 `imgutils.tagging.get_wd14_tags`）的 `1girl / 1boy` 等通用标签
来推断性别，规则如下：

  - 取 max(1girl, 2girls, 6+girls, multiple_girls) 作为 female 得分
  - 取 max(1boy, 2boys, 6+boys, multiple_boys) 作为 male 得分
  - boy 得分 > girl 且 ≥ confidence_threshold → 'male'
  - 否则（含两者均低）→ 'female'  （符合"未知归女性"的兜底规则）

模型在首次推理时由 imgutils 自动下载并缓存到 HuggingFace cache。
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# 性别指示标签（wd14 通用 tag 空间中的名字）
GIRL_TAGS = ("1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple_girls")
BOY_TAGS = ("1boy", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple_boys")


class GenderClassifier:
    """基于 wd14 tagger 的二元性别分类器。

    `classify()` 始终返回 'male' 或 'female'，绝不返回其他值。
    """

    DEFAULT_MODEL = "SwinV2_v3"  # wd14 模型变体，精度与速度权衡较好

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        general_threshold: float = 0.25,
    ):
        """初始化。

        Args:
            model_name: wd14 模型名，可选 'SwinV2_v3' / 'ConvNext_v3' / 'EVA02_Large' 等
            device: 计算设备（由 imgutils 内部自动处理）
            confidence_threshold: 判 male 所需的最低得分
            general_threshold: 传给 get_wd14_tags 的 general_threshold（决定哪些 tag 会进入返回字典）
        """
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.general_threshold = general_threshold
        self._available = False
        self._test_import()

    def _test_import(self):
        try:
            from imgutils.tagging import get_wd14_tags  # noqa: F401
            self._available = True
            logger.info(
                f"[GenderClassifier] wd14 tagger 可用 (model={self.model_name}, "
                f"general_threshold={self.general_threshold}, conf={self.confidence_threshold})"
            )
        except ImportError as e:
            logger.warning(f"[GenderClassifier] imgutils 不可用: {e}，将默认输出 female")
            self._available = False

    def initialize(self):
        """兼容 Pipeline 调用。"""
        if self._available:
            logger.info("[GenderClassifier] 已就绪（wd14 模型将在首次推理时自动下载）")
        else:
            logger.warning("[GenderClassifier] 不可用，将默认输出 female")

    # ── 核心 API ─────────────────────────────────────────────

    def classify(self, image, bbox: Tuple[int, int, int, int]) -> str:
        gender, _ = self._classify_with_conf(image, bbox)
        return gender

    def _classify_with_conf(self, image, bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        if not self._available:
            return "female", 0.5

        try:
            from imgutils.tagging import get_wd14_tags

            crop = self._extract_crop(image, bbox)
            if crop is None:
                return "female", 0.5

            # 只取 general tags（rating / character 这里不关心）
            rating, general, character = get_wd14_tags(
                crop,
                model_name=self.model_name,
                general_threshold=self.general_threshold,
                character_threshold=0.99,  # 极高阈值，基本排除 character 推理开销
            )

            girl_score = max((general.get(t, 0.0) for t in GIRL_TAGS), default=0.0)
            boy_score = max((general.get(t, 0.0) for t in BOY_TAGS), default=0.0)

            girl_score = float(girl_score)
            boy_score = float(boy_score)

            if boy_score > girl_score and boy_score >= self.confidence_threshold:
                logger.debug(
                    f"[GenderClassifier] male (boy={boy_score:.3f}, girl={girl_score:.3f})"
                )
                return "male", boy_score

            if girl_score >= self.confidence_threshold:
                logger.debug(
                    f"[GenderClassifier] female (girl={girl_score:.3f}, boy={boy_score:.3f})"
                )
                return "female", girl_score

            # 两者都低 → 兜底 female
            logger.debug(
                f"[GenderClassifier] female 兜底 (girl={girl_score:.3f}, boy={boy_score:.3f})"
            )
            return "female", max(0.5, girl_score)

        except Exception as e:
            logger.warning(f"[GenderClassifier] wd14 推理失败: {e}，默认 female")
            return "female", 0.5

    def _extract_crop(self, image, bbox: Tuple[int, int, int, int]) -> Optional[Image.Image]:
        try:
            if isinstance(image, np.ndarray):
                import cv2
                if image.ndim == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                pil_img = Image.fromarray(image_rgb)
            elif isinstance(image, Image.Image):
                pil_img = image.convert("RGB")
            else:
                logger.warning(f"[GenderClassifier] 不支持的图像类型: {type(image)}")
                return None

            x1, y1, x2, y2 = map(int, bbox)
            w, h = pil_img.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None

            crop = pil_img.crop((x1, y1, x2, y2))
            if crop.width < 20 or crop.height < 20:
                return None
            return crop
        except Exception as e:
            logger.warning(f"[GenderClassifier] 提取人物区域失败: {e}")
            return None

    # ── 批量接口 ─────────────────────────────────────────────

    def batch_classify(self, image, persons: list, parts_by_person=None) -> dict:
        results = {}
        for person in persons:
            gender, conf = self._classify_with_conf(image, person.bbox)
            results[person.person_id] = (gender, conf)
        return results

    # ── 调试接口 ─────────────────────────────────────────────

    def get_gender_details(self, image, bbox: Tuple[int, int, int, int]) -> dict:
        if not self._available:
            return {"error": "imgutils 不可用", "gender": "female", "confidence": 0.5}

        try:
            from imgutils.tagging import get_wd14_tags

            crop = self._extract_crop(image, bbox)
            if crop is None:
                return {"error": "无法提取区域", "gender": "female", "confidence": 0.5}

            rating, general, character = get_wd14_tags(
                crop,
                model_name=self.model_name,
                general_threshold=self.general_threshold,
                character_threshold=0.99,
            )
            girl_score = max((general.get(t, 0.0) for t in GIRL_TAGS), default=0.0)
            boy_score = max((general.get(t, 0.0) for t in BOY_TAGS), default=0.0)

            final_gender, final_conf = self._classify_with_conf(image, bbox)
            return {
                "gender": final_gender,
                "confidence": final_conf,
                "girl_score": float(girl_score),
                "boy_score": float(boy_score),
                "model": f"wd14/{self.model_name}",
                "all_gender_tags": {
                    **{t: general.get(t, 0.0) for t in GIRL_TAGS if t in general},
                    **{t: general.get(t, 0.0) for t in BOY_TAGS if t in general},
                },
            }
        except Exception as e:
            return {"error": str(e), "gender": "female", "confidence": 0.5}


# 便捷函数
def classify_gender(image, bbox: Tuple[int, int, int, int], **kwargs) -> str:
    return GenderClassifier(**kwargs).classify(image, bbox)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("GenderClassifier 测试 (wd14 tagger)")
    print("=" * 60)

    classifier = GenderClassifier()
    print(f"可用: {classifier._available}")
    print(f"模型: wd14/{classifier.model_name}")

    if len(sys.argv) > 1:
        img = Image.open(sys.argv[1])
        w, h = img.size
        details = classifier.get_gender_details(np.array(img), (0, 0, w, h))
        print("\n分析结果:")
        for k, v in details.items():
            print(f"  {k}: {v}")
    else:
        print("\n用法: python gender_classifier.py <image_path>")
