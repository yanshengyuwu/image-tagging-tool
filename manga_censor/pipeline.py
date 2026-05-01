"""Mask 生成 Pipeline — 每个部位独立输出 mask 文件，不修改原图。"""

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .detectors.base import BaseDetector, DetectionResult
from .detectors.anzhc_seg import AnzhcSegDetector, ANZHC_MODELS
from .detectors.bbox_sam2 import BboxSam2Detector, DEEPGHS_MODELS
from .detectors.nudenet_bbox import NudeNetBboxDetector
from .detectors.nsfw_seg import NsfwSegDetector
from .detectors.nsfw_part_seg import NsfwPartDetector, NSFW_PART_NAMES
from .detectors.text_detect import TextBubbleDetector
from .detectors.deepghs_nsfw import DeepghsNsfwDetector
from .detectors.nsfw_api_seg import NsfwApiSegDetector
from .detectors.dwpose_body import DWPoseBodyDetector
from .detectors.enhanced_breasts import EnhancedBreastsDetector
from .utils import save_mask, cv2_imread, filter_mask_components

logger = logging.getLogger(__name__)

PRECISE_NSFW_PARTS = {"penis", "pussy", "nipple", "anus", "testicles", "penis_hires", "pussy_hires", "nipple_hires"}

# ── 所有可用部位及其默认配置 ──────────────────────────────────────
#
#   type 字段决定使用哪个检测器：
#     "anzhc_seg"    → AnzhcSegDetector   (YOLO-seg 像素级分割)
#     "bbox_sam2"    → BboxSam2Detector    (DeepGHS ONNX bbox + SAM2 精细化)
#     "nudenet_bbox" → NudeNetBboxDetector  (NudeNet v3 bbox + SAM2 精细化)
#     "nsfw_seg"     → NsfwSegDetector      (ntd11 seg + EraX bbox)
#     "text_bubble"  → TextBubbleDetector   (Anzhc text-seg + OCR 验证)

ALL_PARTS = {
    # ── 面部：DeepGHS bbox + SAM2 精细化 ──
    "face": {
        "type": "bbox_sam2",
        "enabled": True,
        "conf": 0.4,
        "use_sam2": True,
    },

    # ── 眼睛：Anzhc YOLO-seg 像素级 ──
    "eyes": {
        "type": "anzhc_seg",
        "enabled": True,
        "conf": 0.5,
    },

    # ── 头发：Anzhc YOLO-seg 像素级 ──
    "hair": {
        "type": "anzhc_seg",
        "enabled": True,
        "conf": 0.5,
    },

    # ── 手部：DeepGHS bbox + SAM2 精细化 ──
    "hand": {
        "type": "bbox_sam2",
        "enabled": True,
        "conf": 0.4,
        "use_sam2": True,
    },

    # ── 胸部：Anzhc n+m 组合多尺度检测（增强巨乳/怀孕检出率）──
    "breasts": {
        "type": "enhanced_breasts",
        "enabled": True,
        "conf": 0.25,
    },

    # ── 臀部：NudeNet bbox + SAM2 精细化（二次元精度较低）──
    "buttocks": {
        "type": "nudenet_bbox",
        "enabled": False,
        "conf": 0.15,
        "use_sam2": True,
    },

    # ── NSFW（全部合并）：ntd11 seg + EraX bbox 二次确认 ──
    "nsfw": {
        "type": "nsfw_seg",
        "enabled": False,
        "conf": 0.3,
    },

    # ── NSFW 细分部位：使用 NSFW-API 高精度 seg 作为主链 ──
    "penis": {
        "type": "nsfw_api_seg",
        "enabled": True,
        "conf": 0.3,
    },
    "pussy": {
        "type": "nsfw_api_seg",
        "enabled": True,
        "conf": 0.3,
    },
    "nipple": {
        "type": "nsfw_api_seg",
        "enabled": True,
        "conf": 0.3,
    },
    "anus": {
        "type": "nsfw_part_seg",
        "enabled": True,
        "conf": 0.15,  # 降低阈值：肛门是小目标，0.3 经常漏检导致回退到 bbox
        "experimental": True,  # 标记为实验性
    },
    "testicles": {
        "type": "nsfw_part_seg",
        "enabled": False,  # 默认禁用，精度太低
        "conf": 0.3,
        "experimental": True,  # 标记为实验性
    },

    # ── 文字气泡：Anzhc text-seg + OCR 验证 ──
    "text_bubble": {
        "type": "text_bubble",
        "enabled": True,
        "conf": 0.5,
        "use_ocr": True,
    },

    # ── DWPose 身体部位（关键点 → 管状/多边形 mask）──
    "arms": {
        "type": "dwpose_body",
        "enabled": False,
        "conf": 0.3,
    },
    "legs": {
        "type": "dwpose_body",
        "enabled": False,
        "conf": 0.3,
    },
    "feet": {
        "type": "dwpose_body",
        "enabled": False,
        "conf": 0.3,
    },
    "torso": {
        "type": "dwpose_body",
        "enabled": False,
        "conf": 0.3,
    },
    "neck": {
        "type": "dwpose_body",
        "enabled": False,
        "conf": 0.3,
    },

    # ── deepghs NSFW（动漫专用 bbox + SAM2，与 nsfw_part_seg 互补）──
    "nipple_deepghs": {
        "type": "deepghs_nsfw",
        "enabled": False,
        "conf": 0.35,
        "use_sam2": True,
    },
    "penis_deepghs": {
        "type": "deepghs_nsfw",
        "enabled": False,
        "conf": 0.35,
        "use_sam2": True,
    },
    "pussy_deepghs": {
        "type": "deepghs_nsfw",
        "enabled": False,
        "conf": 0.35,
        "use_sam2": True,
    },

    # ── ntd11+EraX 备选链路（降级为可选）──
    "nipple_ntd11": {
        "type": "nsfw_part_seg",
        "enabled": False,
        "conf": 0.3,
    },
    "pussy_ntd11": {
        "type": "nsfw_part_seg",
        "enabled": False,
        "conf": 0.3,
    },
    "penis_ntd11": {
        "type": "nsfw_part_seg",
        "enabled": False,
        "conf": 0.3,
    },
}


# ── 前端别名 → 后端检测器名映射 ──────────────────────────────────
PART_ALIASES = {
    # bbox 变体 → 对应检测器
    "hand_bbox": "hand",
    "eye_bbox": "eyes",
    "face_bbox": "face",
    # 其他可能的别名
    "breast": "breasts",
    "eye": "eyes",
    # deepghs NSFW 目标部位映射
    "nipple_deepghs": "nipple_deepghs",
    "penis_deepghs": "penis_deepghs",
    "pussy_deepghs": "pussy_deepghs",
    # NSFW-API 高精度分割映射
    "nipple_hires": "nipple_hires",
    "pussy_hires": "pussy_hires",
    "penis_hires": "penis_hires",
}

# ── deepghs NSFW 部位名 → 实际检测目标映射 ──
DEEPGHS_PART_MAP = {
    "nipple_deepghs": "nipple",
    "penis_deepghs": "penis",
    "pussy_deepghs": "pussy",
}

# ── NSFW-API 部位名 → 实际检测目标映射 ──
NSFW_API_PART_MAP = {
    "nipple": "nipple",
    "pussy": "pussy",
    "penis": "penis",
    "nipple_hires": "nipple",
    "pussy_hires": "pussy",
    "penis_hires": "penis",
    "nipple_ntd11": "nipple",
    "pussy_ntd11": "pussy",
    "penis_ntd11": "penis",
}


class IndependentMaskPipeline:
    """多部位独立 mask 生成 pipeline。

    输入一张图像，为每个启用的部位生成独立的 mask 文件。
    原图始终不被修改。

    输出目录结构：
        output_dir/
          {image_stem}/
            face.png
            eyes.png
            hair.png
            hand.png
            breasts.png
            buttocks.png
            nsfw.png
            text_bubble.png
    """

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: 配置字典，格式参见 mask_config.yaml
        """
        self.config = config or {}
        self.detectors: dict[str, BaseDetector] = {}
        self._initialized = False

    def _create_detector(self, part_name: str, part_config: dict) -> BaseDetector:
        """根据配置创建检测器实例。"""
        dtype = part_config["type"]
        conf = part_config.get("conf", 0.5)

        if dtype == "anzhc_seg":
            return AnzhcSegDetector(part_name, conf=conf)

        elif dtype == "bbox_sam2":
            use_sam2 = part_config.get("use_sam2", True)
            return BboxSam2Detector(part_name, conf=conf, use_sam2=use_sam2)

        elif dtype == "nudenet_bbox":
            use_sam2 = part_config.get("use_sam2", True)
            return NudeNetBboxDetector(part_name=part_name, conf=conf, use_sam2=use_sam2)

        elif dtype == "nsfw_seg":
            use_erax = self.config.get("nsfw", {}).get("use_erax", True)
            return NsfwSegDetector(conf=conf, use_erax=use_erax)

        elif dtype == "nsfw_part_seg":
            nsfw_cfg = self.config.get("nsfw_detection", {})
            use_erax = nsfw_cfg.get("use_erax", True)
            use_sam2 = nsfw_cfg.get("use_sam2", True)
            allow_bbox_fallback = nsfw_cfg.get("allow_bbox_fallback", False)
            return NsfwPartDetector(
                part_name,
                conf=conf,
                use_erax=use_erax,
                use_sam2=use_sam2,
                allow_bbox_fallback=allow_bbox_fallback,
            )

        elif dtype == "text_bubble":
            use_ocr = part_config.get("use_ocr", True)
            return TextBubbleDetector(conf=conf, use_ocr=use_ocr)

        elif dtype == "enhanced_breasts":
            return EnhancedBreastsDetector(conf=conf)

        elif dtype == "dwpose_body":
            return DWPoseBodyDetector(part_name, conf=conf)

        elif dtype == "deepghs_nsfw":
            # 将 "nipple_deepghs" 等映射到实际检测目标 "nipple"
            target = DEEPGHS_PART_MAP.get(part_name, part_name)
            use_sam2 = part_config.get("use_sam2", True)
            nsfw_cfg = self.config.get("nsfw_detection", {})
            allow_bbox_fallback = nsfw_cfg.get("allow_bbox_fallback", False)
            return DeepghsNsfwDetector(
                target,
                conf=conf,
                use_sam2=use_sam2,
                allow_bbox_fallback=allow_bbox_fallback,
            )

        elif dtype == "nsfw_api_seg":
            # 将 "nipple_hires" 等映射到实际检测目标 "nipple"
            target = NSFW_API_PART_MAP.get(part_name, part_name)
            return NsfwApiSegDetector(target, conf=conf)

        else:
            raise ValueError(f"未知检测器类型: {dtype}")

    def initialize(self, enabled_parts: list[str] | None = None,
                   confidence_overrides: dict[str, float] | None = None):
        """初始化 pipeline，创建并加载所有启用的检测器。

        Args:
            enabled_parts: 要启用的部位列表，None 则使用默认配置
            confidence_overrides: 各部位的置信度覆盖
        """
        parts_config = {}

        # 前端别名转换：将前端使用的细分名称映射到实际检测器名称
        if enabled_parts is not None:
            resolved = set()
            for name in enabled_parts:
                resolved.add(PART_ALIASES.get(name, name))
            original_parts = enabled_parts
            enabled_parts = list(resolved)
            if set(original_parts) != set(enabled_parts):
                logger.info(f"[pipeline] 别名转换: {original_parts} → {enabled_parts}")

        # 前端传了 enabled_parts 时，前端选择优先
        frontend_override = enabled_parts is not None

        for name, default in ALL_PARTS.items():
            cfg = dict(default)  # 浅拷贝
            parts_config[name] = cfg

        # 仅当前端未指定时，才从 yaml 读取 enabled 覆盖
        if not frontend_override:
            yaml_parts = self.config.get("enabled_parts", {})
            for name, enabled in yaml_parts.items():
                if name in parts_config:
                    parts_config[name]["enabled"] = enabled
        else:
        # 前端指定了，严格按前端列表
            for name in parts_config:
                parts_config[name]["enabled"] = name in enabled_parts
            # 诊断: 打印匹配情况
            enabled_names = [n for n, c in parts_config.items() if c["enabled"]]
            print(f"  [pipeline] 前端请求部位: {enabled_parts}")
            print(f"  [pipeline] 匹配到的部位: {enabled_names}")
            if not enabled_names:
                print(f"  [pipeline] ⚠️ 没有任何部位被匹配！请检查部位名称是否正确")
                print(f"  [pipeline] 可用部位: {list(ALL_PARTS.keys())}")

        # yaml 的 confidence 作为默认值
        yaml_conf = self.config.get("confidence", {})
        for name, conf in yaml_conf.items():
            if name in parts_config:
                parts_config[name]["conf"] = conf

        # 前端传的 confidence 覆盖 yaml
        if confidence_overrides:
            for name, conf in confidence_overrides.items():
                if name in parts_config:
                    parts_config[name]["conf"] = conf

        # 创建检测器
        self.detectors.clear()
        for name, cfg in parts_config.items():
            if not cfg["enabled"]:
                continue
            try:
                detector = self._create_detector(name, cfg)
                detector.load_model()
                self.detectors[name] = detector
                logger.info(f"[pipeline] 检测器 '{name}' 已就绪")
                print(f"  [pipeline] ✅ 检测器 '{name}' ({cfg['type']}) 已就绪")
            except Exception as e:
                import traceback
                logger.error(f"[pipeline] 检测器 '{name}' 加载失败: {e}")
                print(f"  [pipeline] ❌ 检测器 '{name}' ({cfg['type']}) 加载失败: {e}")
                print(f"  [pipeline] 错误详情:\n{traceback.format_exc()}")

        self._initialized = True
        logger.info(f"[pipeline] 已初始化 {len(self.detectors)} 个检测器: "
                     f"{list(self.detectors.keys())}")

    def process_image(self, image_path: str, output_dir: str,
                      invert: bool = False,
                      merge_single: bool = False,
                      save_individual: bool = True,
                      save_json: bool = True) -> dict:
        """处理单张图像，为每个部位生成独立 mask 文件。

        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            invert: 是否反相（True 时需要遮盖的区域为黑色）
            merge_single: 是否合并为单个 mask 文件（True 时只输出合并 mask）
            save_individual: 是否保存每个部位的独立 mask（默认 True，当 merge_single=True 时被忽略）
            save_json: 是否保存检测结果 JSON（默认 True）

        Returns:
            处理报告字典
        """
        if not self._initialized:
            raise RuntimeError("Pipeline 未初始化，请先调用 initialize()")

        image = cv2_imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        img_name = Path(image_path).stem
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 每张图一个子目录存放各部位 mask（仅在需要时创建）
        mask_dir = out_path / img_name
        if save_individual and not merge_single:
            mask_dir.mkdir(parents=True, exist_ok=True)

        h, w = image.shape[:2]
        merged_mask = np.zeros((h, w), dtype=np.uint8)

        report = {
            "image": str(image_path),
            "size": {"width": w, "height": h},
            "parts": {},
            "timing": {},
            "invert": invert,
            "merge_single": merge_single,
        }

        if save_individual and not merge_single:
            report["mask_dir"] = str(mask_dir)

        total_start = time.time()

        for name, detector in self.detectors.items():
            part_start = time.time()
            try:
                logger.info(f"[pipeline] 开始检测部位: {name}")
                result = detector.detect(image)

                # 检查结果有效性
                h, w = image.shape[:2]
                mask_pixels = np.sum(result.mask > 0)
                mask_ratio = mask_pixels / (h * w) * 100

                logger.info(f"[pipeline] 部位 '{name}' 检测完成: "
                           f"count={result.count}, "
                           f"confidence={result.confidence:.4f}, "
                           f"mask_pixels={mask_pixels} ({mask_ratio:.2f}%)")

                # 合并到总 mask
                merge_mask = result.mask
                if name in PRECISE_NSFW_PARTS:
                    merge_mask = filter_mask_components(
                        merge_mask,
                        min_area=max(1, int(h * w * 0.000008)),
                        max_area=int(h * w * 0.04),
                        min_width=3,
                        min_height=3,
                    )
                merged_mask = np.maximum(merged_mask, merge_mask)

                report["parts"][name] = {
                    "count": result.count,
                    "confidence": round(result.confidence, 4),
                    "has_detection": result.count > 0,
                    "mask_pixels": int(mask_pixels),
                    "mask_ratio_percent": round(mask_ratio, 4),
                }

                # 保存独立 mask 文件（仅在非合并模式且启用独立保存时）
                if save_individual and not merge_single:
                    mask_file = mask_dir / f"{name}.png"
                    mask_to_save = (255 - result.mask) if invert else result.mask
                    save_mask(mask_to_save, mask_file)
                    report["parts"][name]["mask_file"] = str(mask_file)
                    logger.info(f"[pipeline] 保存独立 mask: {mask_file}")

            except Exception as e:
                logger.error(f"[pipeline] 检测器 '{name}' 处理失败: {e}")
                import traceback
                logger.error(f"[pipeline] 错误详情: {traceback.format_exc()}")
                report["parts"][name] = {
                    "error": str(e),
                    "has_detection": False,
                }
            finally:
                report["timing"][name] = round(time.time() - part_start, 3)

        # 反相处理合并 mask
        if invert:
            merged_mask = 255 - merged_mask

        # 保存合并后的 mask（与原图同名，放在输出根目录）
        merged_file = out_path / f"{img_name}.png"
        save_mask(merged_mask, merged_file)
        report["merged_mask"] = str(merged_file)

        report["timing"]["total"] = round(time.time() - total_start, 3)

        # 生成统计信息
        parts_ok = sum(1 for p in report["parts"].values() if p.get("has_detection"))
        total_mask_pixels = sum(p.get("mask_pixels", 0) for p in report["parts"].values() if p.get("has_detection"))
        total_mask_ratio = (total_mask_pixels / (h * w) * 100) if (h * w) > 0 else 0
        
        report["statistics"] = {
            "total_parts_detected": parts_ok,
            "total_parts_enabled": len(self.detectors),
            "detection_rate_percent": round(parts_ok / len(self.detectors) * 100, 2) if self.detectors else 0,
            "total_mask_pixels": int(total_mask_pixels),
            "total_mask_ratio_percent": round(total_mask_ratio, 4),
            "image_size": {"width": w, "height": h},
        }

        # 保存 JSON 报告
        if save_json:
            json_file = out_path / f"{img_name}.json"
            try:
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                report["json_file"] = str(json_file)
                logger.info(f"[pipeline] 保存检测报告: {json_file}")
            except Exception as e:
                logger.error(f"[pipeline] 保存 JSON 失败: {e}")

        output_desc = "合并mask" if merge_single else f"{mask_dir}/"
        logger.info(f"[pipeline] {img_name}: {parts_ok}/{len(self.detectors)} 部位检出 "
                     f"({total_mask_ratio:.2f}% 覆盖) → {output_desc} ({report['timing']['total']:.1f}s)")
        return report

    def process_batch(self, image_paths: list[str], output_dir: str,
                      progress_callback=None) -> list[dict]:
        """批量处理多张图像。"""
        reports = []
        total = len(image_paths)

        for i, img_path in enumerate(image_paths):
            try:
                report = self.process_image(img_path, output_dir)
                reports.append(report)
            except Exception as e:
                logger.error(f"[pipeline] 处理失败: {img_path} — {e}")
                reports.append({"image": img_path, "error": str(e)})

            if progress_callback:
                progress_callback(i + 1, total, img_path, reports[-1])

        return reports

    def merge_masks(self, mask_files: list[str], output_path: str,
                    invert: bool = False,
                    operation: str = "union") -> dict:
        """合并多个 mask 文件。

        Args:
            mask_files: mask 文件路径列表
            output_path: 输出合并后的 mask 路径
            invert: 是否反相输出
            operation: 合并操作，支持 "union"(并集) 或 "intersection"(交集)

        Returns:
            合并报告字典
        """
        if not mask_files:
            raise ValueError("mask_files 不能为空")

        if operation not in ("union", "intersection"):
            raise ValueError(f"不支持的操作: {operation}，仅支持 'union' 或 'intersection'")

        logger.info(f"[pipeline] 开始合并 {len(mask_files)} 个 mask 文件 (操作: {operation})")

        # 读取第一个 mask 作为基准
        first_mask = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
        if first_mask is None:
            raise ValueError(f"无法读取 mask 文件: {mask_files[0]}")

        h, w = first_mask.shape
        result_mask = first_mask.copy()

        # 合并其他 mask
        for mask_file in mask_files[1:]:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"[pipeline] 跳过无效文件: {mask_file}")
                continue

            if mask.shape != (h, w):
                logger.warning(f"[pipeline] 尺寸不匹配，调整大小: {mask_file}")
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            if operation == "union":
                result_mask = np.maximum(result_mask, mask)
            else:  # intersection
                result_mask = np.minimum(result_mask, mask)

        # 反相处理
        if invert:
            result_mask = 255 - result_mask

        # 保存结果
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_mask(result_mask, output_path)

        mask_pixels = np.sum(result_mask > 0)
        mask_ratio = (mask_pixels / (h * w) * 100) if (h * w) > 0 else 0

        report = {
            "operation": operation,
            "input_files": mask_files,
            "output_file": str(output_path),
            "size": {"width": w, "height": h},
            "mask_pixels": int(mask_pixels),
            "mask_ratio_percent": round(mask_ratio, 4),
            "invert": invert,
        }

        logger.info(f"[pipeline] mask 合并完成: {output_path} ({mask_ratio:.2f}% 覆盖)")
        return report

    def render_from_json(self, json_path: str, output_path: str,
                        parts_filter: list[str] | None = None,
                        invert: bool = False,
                        operation: str = "union") -> dict:
        """从 JSON 报告重新渲染 mask。

        Args:
            json_path: JSON 报告文件路径
            output_path: 输出 mask 路径
            parts_filter: 要包含的部位列表，None 表示全部
            invert: 是否反相输出
            operation: 合并操作，支持 "union"(并集) 或 "intersection"(交集)

        Returns:
            渲染报告字典
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

        # 读取 JSON 报告
        with open(json_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        if "parts" not in report:
            raise ValueError("JSON 文件格式错误: 缺少 'parts' 字段")

        # 收集要合并的 mask 文件
        mask_files = []
        included_parts = []

        for part_name, part_info in report["parts"].items():
            # 跳过没有检测结果的部位
            if not part_info.get("has_detection", False):
                continue

            # 应用部位过滤
            if parts_filter and part_name not in parts_filter:
                continue

            # 检查 mask 文件是否存在
            mask_file = part_info.get("mask_file")
            if not mask_file:
                logger.warning(f"[pipeline] 部位 '{part_name}' 没有 mask_file 字段")
                continue

            mask_file = Path(mask_file)
            if not mask_file.exists():
                logger.warning(f"[pipeline] mask 文件不存在: {mask_file}")
                continue

            mask_files.append(str(mask_file))
            included_parts.append(part_name)

        if not mask_files:
            raise ValueError("没有找到可用的 mask 文件")

        logger.info(f"[pipeline] 从 JSON 重新渲染: {len(mask_files)} 个部位 {included_parts}")

        # 使用 merge_masks 合并
        merge_report = self.merge_masks(mask_files, output_path, invert, operation)

        # 添加额外信息
        merge_report["source_json"] = str(json_path)
        merge_report["included_parts"] = included_parts
        merge_report["parts_filter"] = parts_filter

        logger.info(f"[pipeline] 从 JSON 渲染完成: {output_path}")
        return merge_report

    def get_status(self) -> dict:
        """获取 pipeline 状态。"""
        return {
            "initialized": self._initialized,
            "detectors": {
                name: {
                    "part_name": d.part_name,
                    "conf": d.conf,
                    "loaded": d.is_loaded,
                    "type": type(d).__name__,
                }
                for name, d in self.detectors.items()
            },
            "available_parts": {
                name: {
                    "type": cfg["type"],
                    "enabled": cfg["enabled"],
                    "conf": cfg["conf"],
                    "description": _PART_DESCRIPTIONS.get(name, ""),
                }
                for name, cfg in ALL_PARTS.items()
            },
        }


# ── 部位中文描述（前端展示用）──
_PART_DESCRIPTIONS = {
    "face": "面部（DeepGHS bbox + SAM2 精细化）",
    "eyes": "眼睛（Anzhc YOLO-seg 像素级）",
    "hair": "头发（Anzhc YOLO-seg 像素级）",
    "hand": "手部（DeepGHS bbox + SAM2 精细化）",
    "breasts": "胸部（Anzhc n+m 组合多尺度检测，增强巨乳/怀孕检出率）",
    "buttocks": "臀部（NudeNet bbox + SAM2，二次元精度较低）",
    "nsfw": "NSFW 全部敏感部位合并（ntd11 seg + EraX bbox）",
    "penis": "阴茎（NSFW-API YOLO11x 高精度 seg）",
    "pussy": "女性生殖器（NSFW-API YOLO11x 高精度 seg）",
    "nipple": "乳头（NSFW-API YOLO11x 高精度 seg）",
    "anus": "肛门（ntd11 YOLO-seg，实验性）",
    "testicles": "睾丸（ntd11 YOLO-seg，精度低，默认禁用）",
    "nipple_ntd11": "乳头（ntd11 备选链路）",
    "pussy_ntd11": "女性生殖器（ntd11 备选链路）",
    "penis_ntd11": "阴茎（ntd11 备选链路）",
    "text_bubble": "文字气泡（Anzhc text-seg + OCR 验证）",
    "arms": "手臂（DWPose 关键点 → 管状 mask）",
    "legs": "腿部（DWPose 关键点 → 管状 mask）",
    "feet": "脚部（DWPose 关键点 → 圆形 mask）",
    "torso": "躯干（DWPose 关键点 → 多边形 mask）",
    "neck": "颈部（DWPose 关键点 → 三角形 mask）",
    "nipple_deepghs": "乳头（deepghs 动漫专用 bbox + SAM2）",
    "penis_deepghs": "阴茎（deepghs 动漫专用 bbox + SAM2）",
    "pussy_deepghs": "外阴（deepghs 动漫专用 bbox + SAM2）",
    "nipple_hires": "乳头（NSFW-API YOLO11x 高精度 seg）",
    "pussy_hires": "外阴（NSFW-API YOLO11x 高精度 seg）",
    "penis_hires": "阴茎（NSFW-API YOLO11x 高精度 seg）",
}
