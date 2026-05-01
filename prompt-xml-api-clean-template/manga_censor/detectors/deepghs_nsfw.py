"""deepghs/anime_censor_detection — 动漫专用 NSFW bbox 检测器 + SAM2 精细化。

模型: censor_detect_v1.0_s (ONNX, ~10M)
标签: nipple_f, penis, pussy
来源: https://huggingface.co/deepghs/anime_censor_detection
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from .base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

MODEL_REPO = "deepghs/anime_censor_detection"
MODEL_VARIANT = "censor_detect_v1.0_s"
CACHE_DIR = Path("model_cache/deepghs_nsfw")

# deepghs 标签 → 后端部位名映射
LABEL_MAP = {
    "nipple_f": "nipple",
    "penis": "penis",
    "pussy": "pussy",
}


class DeepghsNsfwDetector(BaseDetector):
    """使用 deepghs anime_censor_detection ONNX 模型检测动漫 NSFW 部位。

    输出 bbox，可选配合 SAM2 精细化为像素级 mask。
    默认禁止 bbox 直接作为最终遮盖，避免生成大块方形区域。
    """

    def __init__(
        self,
        part_name: str,
        conf: float = 0.35,
        use_sam2: bool = True,
        allow_bbox_fallback: bool = False,
    ):
        super().__init__(part_name, conf)
        self.use_sam2 = use_sam2
        self.allow_bbox_fallback = allow_bbox_fallback
        self._sam2 = None
        self._meta = None
        self._input_size = 640
        self._sam2_load_attempted = False

    def load_model(self):
        """从 HuggingFace 下载并加载 ONNX 模型。"""
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        import json

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 下载模型
        model_path = hf_hub_download(
            MODEL_REPO,
            filename=f"{MODEL_VARIANT}/model.onnx",
            cache_dir=str(CACHE_DIR),
        )

        # 尝试下载元数据（可选，失败时使用默认标签）
        DEFAULT_LABELS = list(LABEL_MAP.keys())  # ["nipple_f", "penis", "pussy"]
        try:
            meta_path = hf_hub_download(
                MODEL_REPO,
                filename=f"{MODEL_VARIANT}/meta.json",
                cache_dir=str(CACHE_DIR),
            )
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
            self._labels = self._meta.get("labels", DEFAULT_LABELS)
            logger.info(f"[deepghs_nsfw] 从 meta.json 加载标签: {self._labels}")
        except Exception as e:
            logger.warning(f"[deepghs_nsfw] meta.json 下载/加载失败: {e}")
            logger.info(f"[deepghs_nsfw] 使用默认标签: {DEFAULT_LABELS}")
            self._meta = {"labels": DEFAULT_LABELS}
            self._labels = DEFAULT_LABELS

        # 加载 ONNX
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._model = ort.InferenceSession(model_path, providers=providers)

        input_info = self._model.get_inputs()[0]
        if input_info.shape and len(input_info.shape) == 4:
            self._input_size = input_info.shape[2] if isinstance(input_info.shape[2], int) else 640

        logger.info(f"[deepghs_nsfw] 模型已加载: {MODEL_VARIANT}, input_size={self._input_size}")

    def _preprocess(self, image: np.ndarray):
        """YOLOv8-style 预处理: letterbox resize + normalize。"""
        h, w = image.shape[:2]
        scale = min(self._input_size / h, self._input_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_w = (self._input_size - new_w) // 2
        pad_h = (self._input_size - new_h) // 2

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self._input_size, self._input_size, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        return blob, scale, pad_w, pad_h

    def _nms(self, detections: list, iou_threshold: float = 0.5) -> list:
        """非极大值抑制 (NMS) 去除重叠 bbox。"""
        if not detections:
            return []

        # 按置信度排序
        detections = sorted(detections, key=lambda x: x["score"], reverse=True)

        keep = []
        while detections:
            best = detections[0]
            keep.append(best)
            rest = []
            for det in detections[1:]:
                iou = self._iou(best["bbox"], det["bbox"])
                # 同类别且 IoU 高才抑制
                if best["label"] == det["label"] and iou > iou_threshold:
                    continue
                rest.append(det)
            detections = rest

        return keep

    @staticmethod
    def _iou(box1, box2):
        """计算两个 bbox 的 IoU。"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _postprocess(self, output, scale, pad_w, pad_h, orig_h, orig_w):
        """解析 ONNX 输出为 bbox 列表。

        修复: ONNX 输出格式为 (4+nc, num_anchors) 即 (7, 8400)，
        需要转置为 (num_anchors, 4+nc) 再遍历。
        """
        pred = output[0]  # shape: (7, 8400) 或 (8400, 7)

        # 确保格式为 (num_anchors, 4+num_classes)
        if pred.ndim == 3:
            # (1, 4+nc, num_anchors) 或 (1, num_anchors, 4+nc)
            pred = pred[0]
        
        # 如果第一维小于第二维，说明是 (4+nc, num_anchors)，需要转置
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T  # (7, 8400) -> (8400, 7)

        raw_detections = []
        num_classes = len(self._labels)

        for row in pred:
            if len(row) < 4 + num_classes:
                continue

            cx, cy, bw, bh = row[:4]
            class_scores = row[4:4 + num_classes]

            # YOLOv8 ONNX 输出的 class scores 是原始 logits，需要 sigmoid 转换为概率
            class_probs = 1.0 / (1.0 + np.exp(-class_scores.astype(np.float64)))

            cls_id = int(np.argmax(class_probs))
            score = float(class_probs[cls_id])

            if score < self.conf:
                continue

            label = self._labels[cls_id] if cls_id < len(self._labels) else f"class_{cls_id}"
            mapped = LABEL_MAP.get(label, label)

            # 转换回原图坐标
            x1 = (cx - bw / 2 - pad_w) / scale
            y1 = (cy - bh / 2 - pad_h) / scale
            x2 = (cx + bw / 2 - pad_w) / scale
            y2 = (cy + bh / 2 - pad_h) / scale

            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))

            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            raw_detections.append({
                "label": label,
                "mapped": mapped,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": score,
            })

        # 应用 NMS 去重
        detections = self._nms(raw_detections, iou_threshold=0.5)
        logger.debug(f"[deepghs_nsfw] 原始检测: {len(raw_detections)} 个, NMS 后: {len(detections)} 个")

        return detections

    def _ensure_sam2_ready(self):
        """懒加载 SAM2，避免在模型初始化阶段反复触发下载/加载。"""
        if not self.use_sam2 or self._sam2 is not None or self._sam2_load_attempted:
            return
        self._sam2_load_attempted = True
        try:
            from .sam2_refiner import SAM2Refiner

            self._sam2 = SAM2Refiner()
            logger.info("[deepghs_nsfw] SAM2 refiner 已挂载（按需推理）")
        except Exception as e:
            logger.warning(f"[deepghs_nsfw] SAM2 初始化失败，禁用 SAM2: {e}")
            self._sam2 = None

    def detect(self, image: np.ndarray) -> DetectionResult:
        """检测 NSFW 部位，返回与 self.part_name 匹配的部位 mask。"""
        self.ensure_loaded()
        self._ensure_sam2_ready()
        h, w = image.shape[:2]

        blob, scale, pad_w, pad_h = self._preprocess(image)

        input_name = self._model.get_inputs()[0].name
        outputs = self._model.run(None, {input_name: blob})

        detections = self._postprocess(outputs[0], scale, pad_w, pad_h, h, w)

        # 过滤出目标部位
        target = [d for d in detections if d["mapped"] == self.part_name]

        if not target:
            logger.info(f"[deepghs_nsfw] {self.part_name}: 未检测到")
            return self.empty_result(h, w)

        mask = np.zeros((h, w), dtype=np.uint8)
        total_conf = sum(det["score"] for det in target)
        avg_conf = total_conf / len(target) if target else 0.0

        # 收集所有 bbox，一次性调用 SAM2（避免重复推理）
        all_bboxes = [det["bbox"] for det in target]

        if self._sam2 is not None and all_bboxes:
            try:
                logger.info(f"[deepghs_nsfw] {self.part_name}: 对 {len(all_bboxes)} 个 bbox 进行批量 SAM2 精细化")
                sam_mask = self._sam2.refine(
                    image,
                    all_bboxes,
                    allow_bbox_fallback=self.allow_bbox_fallback,
                )
                if np.any(sam_mask > 0):
                    mask = sam_mask
                    logger.info(f"[deepghs_nsfw] {self.part_name}: SAM2 批量精细化完成")
                else:
                    logger.info(f"[deepghs_nsfw] {self.part_name}: SAM2 未产生有效 mask")
                    if self.allow_bbox_fallback:
                        # SAM2 失败时回退到 bbox
                        for bbox in all_bboxes:
                            x1, y1, x2, y2 = bbox
                            mask[y1:y2, x1:x2] = 255
                        logger.info(f"[deepghs_nsfw] {self.part_name}: 回退到 bbox 矩形遮盖")
            except Exception as e:
                logger.warning(f"[deepghs_nsfw] {self.part_name}: SAM2 批量精细化失败: {e}")
                if self.allow_bbox_fallback:
                    # SAM2 异常时回退到 bbox
                    for bbox in all_bboxes:
                        x1, y1, x2, y2 = bbox
                        mask[y1:y2, x1:x2] = 255
                    logger.info(f"[deepghs_nsfw] {self.part_name}: SAM2 异常，回退到 bbox 矩形遮盖")
        elif self.allow_bbox_fallback and all_bboxes:
            # 没有 SAM2 但允许 bbox fallback
            for bbox in all_bboxes:
                x1, y1, x2, y2 = bbox
                mask[y1:y2, x1:x2] = 255
            logger.info(f"[deepghs_nsfw] {self.part_name}: 无 SAM2，使用 bbox 矩形遮盖")
        else:
            logger.info(f"[deepghs_nsfw] {self.part_name}: 仅得到 bbox，已拒绝矩形 fallback")
        logger.info(f"[deepghs_nsfw] {self.part_name}: 检测到 {len(target)} 个, 置信度={avg_conf:.3f}")

        return DetectionResult(
            part_name=self.part_name,
            mask=mask,
            confidence=avg_conf,
            count=len(target),
        )
