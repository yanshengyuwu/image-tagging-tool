"""deepghs ONNX bbox 检测器 — face/hand/eye/head/person。"""

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .base import BaseDetector, DetectionResult
from ..utils import bbox_to_mask, get_onnx_providers

logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = Path("model_cache/yolov8_anime")

# deepghs 模型注册表
DEEPGHS_MODELS = {
    "face_bbox": {
        "path": "model_cache/yolov8_anime/face_detect_v1.4_s.onnx",
        "repo": "deepghs/anime_face_detection",
        "filename": "face_detect_v1.4_s.onnx",
        "conf": 0.4,
        "input_size": 640,
    },
    "hand_bbox": {
        "path": "model_cache/yolov8_anime/hand_detect_v1.0_s.onnx",
        "repo": "deepghs/anime_hand_detection",
        "filename": "hand_detect_v1.0_s.onnx",
        "conf": 0.4,
        "input_size": 640,
    },
    "eye_bbox": {
        "path": "model_cache/yolov8_anime/eye_detect_v1.0_s.onnx",
        "repo": "deepghs/anime_eye_detection",
        "filename": "eye_detect_v1.0_s.onnx",
        "conf": 0.4,
        "input_size": 640,
    },
    "person_bbox": {
        "path": "model_cache/yolov8_anime/person_detect_v1.0_s.onnx",
        "repo": "deepghs/anime_person_detection",
        "filename": "person_detect_v1.0_s.onnx",
        "conf": 0.3,
        "input_size": 640,
    },
}


class DeepghsBboxDetector(BaseDetector):
    """基于 deepghs ONNX 模型的 bbox 检测器，输出 bbox → 二值 mask。"""

    def __init__(self, part_name: str, conf: float | None = None):
        if part_name not in DEEPGHS_MODELS:
            raise ValueError(f"未知部位: {part_name}，可选: {list(DEEPGHS_MODELS.keys())}")

        model_info = DEEPGHS_MODELS[part_name]
        super().__init__(part_name, conf or model_info["conf"])
        self.model_path = Path(model_info["path"])
        self.input_size = model_info["input_size"]

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
        
        # 尝试多个镜像源
        mirrors = [
            None,  # 默认源
            "https://hf-mirror.com",
            "https://huggingface.co",
        ]
        
        last_error = None
        for endpoint in mirrors:
            try:
                logger.info(f"[{self.part_name}] 尝试从 {endpoint or 'huggingface.co'} 下载...")
                downloaded = hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    local_dir=str(MODEL_CACHE_DIR),
                    endpoint=endpoint,
                )
                logger.info(f"[{self.part_name}] 模型下载完成: {downloaded}")
                return Path(downloaded)
            except Exception as e:
                last_error = e
                logger.warning(f"[{self.part_name}] 从 {endpoint or 'huggingface.co'} 下载失败: {e}")
                continue
        
        # 所有源都失败
        raise RuntimeError(
            f"[{self.part_name}] 模型下载失败，已尝试所有镜像源。最后错误: {last_error}"
        )

    def load_model(self):
        """加载 ONNX 模型（自动下载如果不存在）。"""
        self.model_path = self._download_model()

        self._model = ort.InferenceSession(str(self.model_path), providers=get_onnx_providers())
        self._input_name = self._model.get_inputs()[0].name
        self._output_names = [o.name for o in self._model.get_outputs()]
        logger.info(f"[{self.part_name}] ONNX 模型已加载: {self.model_path}")

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
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
                     scale: float, pad: tuple[float, float]) -> list[tuple[int, int, int, int, float]]:
        """后处理：解析 ONNX 输出，还原到原图坐标。"""
        pad_w, pad_h = pad
        detections = []

        # output shape: (1, N, 5+) or (1, 5+, N) — 需要适配
        if output.ndim == 3:
            data = output[0]
            # 如果列数 > 行数，说明是 (5+, N) 格式，需要转置
            if data.shape[0] < data.shape[1]:
                data = data.T
        else:
            data = output

        for row in data:
            if len(row) >= 5:
                # YOLOv8 格式: cx, cy, w, h, conf, [cls_scores...]
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]

                if len(row) == 5:
                    conf = row[4]
                else:
                    # 多类别：取最大类别分数
                    conf = float(np.max(row[4:]))

                if conf < self.conf:
                    continue

                # 还原到原图坐标
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
        """执行 bbox 检测并转换为二值 mask。"""
        self.ensure_loaded()
        h, w = image.shape[:2]

        blob, scale, pad = self._preprocess(image)
        outputs = self._model.run(self._output_names, {self._input_name: blob})
        detections = self._postprocess(outputs[0], h, w, scale, pad)

        if not detections:
            return self.empty_result(h, w)

        boxes = [(d[0], d[1], d[2], d[3]) for d in detections]
        confs = [d[4] for d in detections]
        mask = bbox_to_mask(h, w, boxes)

        return DetectionResult(
            part_name=self.part_name,
            mask=mask,
            confidence=sum(confs) / len(confs),
            count=len(detections),
        )
