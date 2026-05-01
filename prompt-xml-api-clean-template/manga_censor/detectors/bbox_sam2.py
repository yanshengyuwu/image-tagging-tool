"""DeepGHS ONNX bbox 检测 + SAM2 精细化 — 像素级 mask 输出。"""

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .base import BaseDetector, DetectionResult
from .sam2_refiner import SAM2Refiner
from ..utils import bbox_to_mask, get_onnx_providers

logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = Path("model_cache/yolov8_anime")

# DeepGHS 模型注册表
DEEPGHS_MODELS = {
    "face": {
        "path": "model_cache/yolov8_anime/face_detect_v1.4_s.onnx",
        "repo": "deepghs/anime_face_detection",
        "filename": "face_detect_v1.4_s.onnx",
        "conf": 0.4,
        "input_size": 640,
    },
    "hand": {
        "path": "model_cache/yolov8_anime/hand_detect_v1.0_s.onnx",
        "repo": "deepghs/anime_hand_detection",
        "filename": "hand_detect_v1.0_s.onnx",
        "conf": 0.4,
        "input_size": 640,
    },
    "eye": {
        "path": "model_cache/yolov8_anime/eye_detect_v1.0_s.onnx",
        "repo": "deepghs/anime_eye_detection",
        "filename": "eye_detect_v1.0_s.onnx",
        "conf": 0.4,
        "input_size": 640,
    },
}


class BboxSam2Detector(BaseDetector):
    """DeepGHS bbox 检测 + SAM2 精细化，输出像素级 mask。

    流程：
    1. DeepGHS ONNX 模型检测 bbox
    2. SAM2 以 bbox 为 prompt 精细分割
    3. 输出像素级二值 mask

    如果 SAM2 不可用，回退到 bbox 矩形 mask。
    """

    def __init__(self, part_name: str, conf: float | None = None,
                 use_sam2: bool = True):
        if part_name not in DEEPGHS_MODELS:
            raise ValueError(f"未知部位: {part_name}，可选: {list(DEEPGHS_MODELS.keys())}")

        model_info = DEEPGHS_MODELS[part_name]
        super().__init__(part_name, conf or model_info["conf"])
        self.model_path = Path(model_info["path"])
        self.input_size = model_info["input_size"]
        self.use_sam2 = use_sam2
        self._sam2 = SAM2Refiner() if use_sam2 else None

    def _download_model(self) -> Path:
        """从 HuggingFace 下载模型文件（如果本地不存在）。"""
        if self.model_path.exists():
            logger.info(f"[{self.part_name}] 模型已缓存: {self.model_path}")
            return self.model_path

        model_info = DEEPGHS_MODELS.get(self.part_name, {})
        repo = model_info.get("repo")
        filename = model_info.get("filename")

        if not repo or not filename:
            raise FileNotFoundError(
                f"模型文件不存在且无下载信息: {self.model_path}"
            )

        logger.info(f"[{self.part_name}] 正在从 {repo} 下载 {filename}...")
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(MODEL_CACHE_DIR),
        )
        logger.info(f"[{self.part_name}] 模型下载完成: {downloaded}")
        return Path(downloaded)

    def load_model(self):
        """加载 ONNX 检测模型（SAM2 延迟加载）。"""
        self.model_path = self._download_model()

        self._model = ort.InferenceSession(str(self.model_path), providers=get_onnx_providers())
        self._input_name = self._model.get_inputs()[0].name
        self._output_names = [o.name for o in self._model.get_outputs()]
        logger.info(f"[{self.part_name}] ONNX 检测模型已加载: {self.model_path}")

        if self.use_sam2:
            logger.info(f"[{self.part_name}] SAM2 精细化已启用（延迟加载）")

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple]:
        """预处理图像：letterbox resize + normalize。"""
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # letterbox padding
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_h, pad_w = (self.input_size - new_h) // 2, (self.input_size - new_w) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # HWC → CHW, BGR → RGB, normalize
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        return blob, scale, (pad_w, pad_h)

    def _postprocess(self, output: np.ndarray, orig_h: int, orig_w: int,
                     scale: float, pad: tuple) -> list[tuple[int, int, int, int, float]]:
        """后处理：解析 ONNX 输出，还原到原图坐标。"""
        pad_w, pad_h = pad
        detections = []

        if output.ndim == 3:
            data = output[0]
            if data.shape[0] < data.shape[1]:
                data = data.T
        else:
            data = output

        for row in data:
            if len(row) >= 5:
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]

                if len(row) == 5:
                    conf = float(row[4])
                else:
                    conf = float(np.max(row[4:]))

                # 如果 score > 1.0，说明是原始 logits，需要 sigmoid
                if conf > 1.0:
                    conf = 1.0 / (1.0 + np.exp(-conf))

                if conf < self.conf:
                    continue

                x1 = (cx - bw / 2 - pad_w) / scale
                y1 = (cy - bh / 2 - pad_h) / scale
                x2 = (cx + bw / 2 - pad_w) / scale
                y2 = (cy + bh / 2 - pad_h) / scale

                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(orig_w, int(x2))
                y2 = min(orig_h, int(y2))

                if x2 > x1 and y2 > y1:
                    detections.append((x1, y1, x2, y2, conf))

        return detections

    def detect(self, image: np.ndarray) -> DetectionResult:
        """执行 bbox 检测 + SAM2 精细化。"""
        self.ensure_loaded()
        h, w = image.shape[:2]

        # 阶段1：bbox 检测
        logger.info(f"[{self.part_name}] 开始 bbox 检测 (image {w}x{h})...")
        blob, scale, pad = self._preprocess(image)
        outputs = self._model.run(self._output_names, {self._input_name: blob})

        # 诊断日志：打印每个输出节点的 shape
        for idx, (name, out) in enumerate(zip(self._output_names, outputs)):
            logger.info(f"[{self.part_name}] ONNX output[{idx}] '{name}': "
                        f"shape={out.shape}, dtype={out.dtype}")

        detections = self._postprocess(outputs[0], h, w, scale, pad)

        logger.info(f"[{self.part_name}] bbox 检测完成: {len(detections)} 个检测结果")

        if not detections:
            logger.warning(f"[{self.part_name}] 未检测到任何目标")
            return self.empty_result(h, w)

        boxes = [(d[0], d[1], d[2], d[3]) for d in detections]
        confs = [d[4] for d in detections]

        # 阶段2：SAM2 精细化（或回退到 bbox 矩形）
        if self._sam2 and self.use_sam2:
            logger.info(f"[{self.part_name}] 使用 SAM2 精细化...")
            try:
                mask = self._sam2.refine(image, boxes)
                logger.info(f"[{self.part_name}] SAM2 精细化完成")
            except Exception as e:
                logger.error(f"[{self.part_name}] SAM2 精细化失败: {e}")
                logger.warning(f"[{self.part_name}] 回退到 bbox 矩形 mask")
                mask = bbox_to_mask(h, w, boxes)
        else:
            logger.info(f"[{self.part_name}] SAM2 已禁用，使用 bbox 矩形 mask")
            mask = bbox_to_mask(h, w, boxes)

        # 检查 mask 是否有效
        mask_pixels = np.sum(mask > 0)
        logger.info(f"[{self.part_name}] 最终 mask 像素数: {mask_pixels} (占比: {mask_pixels/(h*w)*100:.2f}%)")

        return DetectionResult(
            part_name=self.part_name,
            mask=mask,
            confidence=sum(confs) / len(confs),
            count=len(detections),
        )
