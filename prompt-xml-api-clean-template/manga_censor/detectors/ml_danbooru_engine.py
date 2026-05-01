"""ML-Danbooru ONNX 推理引擎

基于 model_cache/ml_danbooru_onnx/model.onnx
输出所有标签的置信度字典，供性别分类等场景使用。
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model_cache")
MODEL_DIR = os.path.join(CACHE_DIR, "ml_danbooru_onnx")
MODEL_FILENAME = "model.onnx"
CLASSES_FILENAME = "classes.json"
IMAGE_SIZE = 448


class MLDanbooruEngine:
    """ML-Danbooru ONNX 本地推理引擎"""

    def __init__(self):
        self.session = None
        self.tags: List[str] = []
        self._initialized = False
        self.execution_provider = None

    def _load_classes(self):
        classes_path = os.path.join(MODEL_DIR, CLASSES_FILENAME)
        with open(classes_path, "r", encoding="utf-8") as f:
            self.tags = json.load(f)
        print(f"ML-Danbooru 标签已加载，共 {len(self.tags)} 个")

    def _init_onnx_runtime(self):
        import onnxruntime as ort

        onnx_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3

        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        self.execution_provider = self.session.get_providers()[0]
        print(f"ML-Danbooru ONNX Runtime 就绪: {self.execution_provider}")

    def initialize(self) -> bool:
        if self._initialized and self.session is not None:
            return True

        try:
            if not os.path.exists(os.path.join(MODEL_DIR, MODEL_FILENAME)):
                print(f"ML-Danbooru 模型不存在: {MODEL_DIR}")
                return False

            self._load_classes()
            self._init_onnx_runtime()
            self._initialized = True
            return True
        except Exception as e:
            print(f"ML-Danbooru 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        # 保持长宽比缩放，短边填充到 448
        image = image.convert("RGB")
        width, height = image.size
        scale = IMAGE_SIZE / max(width, height)
        new_w = int(width * scale)
        new_h = int(height * scale)

        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 创建画布并居中粘贴
        canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))
        paste_x = (IMAGE_SIZE - new_w) // 2
        paste_y = (IMAGE_SIZE - new_h) // 2
        canvas.paste(image, (paste_x, paste_y))

        img_array = np.array(canvas, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_array = (img_array - mean) / std

        img_array = np.expand_dims(img_array, axis=0)
        return img_array.astype(np.float32)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def predict(self, image_path_or_pil, return_all: bool = False) -> Optional[Dict[str, float]]:
        """推理单张图片，返回标签→置信度字典。

        Args:
            image_path_or_pil: 图片路径(str)或PIL.Image
            return_all: 是否返回所有标签（默认只返回阈值>0.35的）
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            if isinstance(image_path_or_pil, str):
                image = Image.open(image_path_or_pil)
            else:
                image = image_path_or_pil

            input_tensor = self._preprocess(image)

            start_time = time.time()
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            inference_time = time.time() - start_time

            logits = outputs[0][0]
            probs = self._sigmoid(logits)

            if return_all:
                result = {tag: float(probs[i]) for i, tag in enumerate(self.tags)}
            else:
                threshold = 0.35
                result = {
                    tag: float(probs[i])
                    for i, tag in enumerate(self.tags)
                    if probs[i] >= threshold
                }

            return {
                "_inference_time": inference_time,
                "_provider": self.execution_provider,
                **result,
            }

        except Exception as e:
            print(f"ML-Danbooru 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_gender_tags(self, image_path_or_pil) -> Optional[Dict]:
        """专门提取性别相关标签（返回所有相关标签的分数，不管阈值）。"""
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            if isinstance(image_path_or_pil, str):
                image = Image.open(image_path_or_pil)
            else:
                image = image_path_or_pil

            input_tensor = self._preprocess(image)

            start_time = time.time()
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            inference_time = time.time() - start_time

            logits = outputs[0][0]
            probs = self._sigmoid(logits)

            gender_related = [
                "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple_girls",
                "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple_boys",
                "male_focus", "female_focus",
                "male", "female",
                "boy", "girl",
                "shota", "loli",
                "bara", "muscular_male", "toned_male",
                "bishounen",
                "futanari", "otoko_no_ko",
                "crossdressing",
            ]

            result = {
                "_inference_time": inference_time,
                "_provider": self.execution_provider,
            }
            for tag in gender_related:
                if tag in self.tags:
                    idx = self.tags.index(tag)
                    result[tag] = float(probs[idx])

            return result

        except Exception as e:
            print(f"ML-Danbooru 性别标签推理失败: {e}")
            return None

    def is_model_ready(self) -> bool:
        return os.path.exists(os.path.join(MODEL_DIR, MODEL_FILENAME)) and \
               os.path.exists(os.path.join(MODEL_DIR, CLASSES_FILENAME))
