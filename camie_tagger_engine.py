# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false
# pyright: reportMissingTypeArgument=false, reportPrivateUsage=false, reportUnusedFunction=false
# pyright: reportUnusedImport=false, reportUnusedVariable=false, reportUnreachable=false
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportGeneralTypeIssues=false
# pyright: reportOptionalOperand=false, reportOperatorIssue=false, reportCallIssue=false, reportPossiblyUnboundVariable=false

"""
camie-tagger-v2 反推打标引擎
基于 Camais03/camie-tagger-v2 ONNX 模型，本地推理生成动漫图片标签
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

REPO_ID = "Camais03/camie-tagger-v2"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
CAMIE_MODEL_DIR = "camie_tagger_v2"
MODEL_FILENAME = "camie-tagger-v2.onnx"
METADATA_FILENAME = "camie-tagger-v2-metadata.json"
DEFAULT_IMAGE_SIZE = 512


@dataclass
class LabelData:
    names: List[Optional[str]]
    rating: np.ndarray
    general: np.ndarray
    artist: np.ndarray
    character: np.ndarray
    copyright: np.ndarray
    meta: np.ndarray
    quality: np.ndarray
    model: np.ndarray


def _normalize_category_name(category: str) -> str:
    value = (category or "").strip().lower()
    mapping = {
        "rating": "rating",
        "general": "general",
        "artist": "artist",
        "character": "character",
        "copyright": "copyright",
        "meta": "meta",
        "quality": "quality",
        "model": "model",
    }
    return mapping.get(value, "general")


def _load_metadata(metadata_path: str):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    dataset_info = metadata.get("dataset_info") or {}
    tag_mapping = dataset_info.get("tag_mapping") or {}

    idx_to_tag_raw = tag_mapping.get("idx_to_tag") or {}
    tag_to_category_raw = tag_mapping.get("tag_to_category") or {}

    if not idx_to_tag_raw or not tag_to_category_raw:
        raise ValueError("Invalid camie metadata: missing dataset_info.tag_mapping fields")

    idx_to_tag = {int(k): v for k, v in idx_to_tag_raw.items()}
    tag_to_category = {
        str(tag): _normalize_category_name(category)
        for tag, category in tag_to_category_raw.items()
    }

    names = [None] * (max(idx_to_tag.keys()) + 1)
    rating, general, artist, character, copyright_, meta, quality, model_name = [], [], [], [], [], [], [], []

    for idx, tag in idx_to_tag.items():
        if idx >= len(names):
            names.extend([None] * (idx - len(names) + 1))
        names[idx] = tag
        category = tag_to_category.get(tag, "general")

        if category == "rating":
            rating.append(idx)
        elif category == "general":
            general.append(idx)
        elif category == "artist":
            artist.append(idx)
        elif category == "character":
            character.append(idx)
        elif category == "copyright":
            copyright_.append(idx)
        elif category == "meta":
            meta.append(idx)
        elif category == "quality":
            quality.append(idx)
        elif category == "model":
            model_name.append(idx)

    labels = LabelData(
        names=names,
        rating=np.array(rating, dtype=np.int64),
        general=np.array(general, dtype=np.int64),
        artist=np.array(artist, dtype=np.int64),
        character=np.array(character, dtype=np.int64),
        copyright=np.array(copyright_, dtype=np.int64),
        meta=np.array(meta, dtype=np.int64),
        quality=np.array(quality, dtype=np.int64),
        model=np.array(model_name, dtype=np.int64),
    )

    image_size = int((metadata.get("model_info") or {}).get("img_size", DEFAULT_IMAGE_SIZE))
    return metadata, labels, idx_to_tag, tag_to_category, image_size


def _pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.split()[-1]
        background.paste(image.convert("RGBA"), mask=alpha)
        return background
    return image.convert("RGB")


def _preprocess_image(image: Image.Image, image_size: int) -> np.ndarray:
    """
    对齐官方 onnx_inference.py:
    - 保持纵横比缩放
    - 使用 ImageNet 均值对应颜色做 padding
    - 按 ImageNet mean/std 归一化
    """
    image = _pil_ensure_rgb(image)

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size")

    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = image_size
        new_height = max(1, int(new_width / aspect_ratio))
    else:
        new_height = image_size
        new_width = max(1, int(new_height * aspect_ratio))

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    pad_color = (124, 116, 104)
    canvas = Image.new("RGB", (image_size, image_size), pad_color)
    paste_x = (image_size - new_width) // 2
    paste_y = (image_size - new_height) // 2
    canvas.paste(image, (paste_x, paste_y))

    img_array = np.asarray(canvas, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img_array = (img_array - mean) / std

    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)


def _stable_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _get_tags(probs, labels: LabelData, gen_threshold: float, char_threshold: float) -> Dict:
    result = {
        "rating": [],
        "general": [],
        "character": [],
        "copyright": [],
        "artist": [],
        "meta": [],
        "quality": [],
        "model": [],
    }

    if len(labels.rating) > 0:
        valid = labels.rating[labels.rating < len(probs)]
        if len(valid) > 0:
            rating_probs = probs[valid]
            if len(rating_probs) > 0:
                idx_local = np.argmax(rating_probs)
                idx_global = valid[idx_local]
                if idx_global < len(labels.names) and labels.names[idx_global] is not None:
                    result["rating"].append((labels.names[idx_global], float(rating_probs[idx_local])))

    if len(labels.quality) > 0:
        valid = labels.quality[labels.quality < len(probs)]
        if len(valid) > 0:
            quality_probs = probs[valid]
            if len(quality_probs) > 0:
                idx_local = np.argmax(quality_probs)
                idx_global = valid[idx_local]
                if idx_global < len(labels.names) and labels.names[idx_global] is not None:
                    result["quality"].append((labels.names[idx_global], float(quality_probs[idx_local])))

    category_map = {
        "general": (labels.general, gen_threshold),
        "character": (labels.character, char_threshold),
        "copyright": (labels.copyright, char_threshold),
        "artist": (labels.artist, char_threshold),
        "meta": (labels.meta, gen_threshold),
        "model": (labels.model, gen_threshold),
    }

    for category, (indices, threshold) in category_map.items():
        if len(indices) == 0:
            continue
        valid = indices[indices < len(probs)]
        if len(valid) == 0:
            continue
        cat_probs = probs[valid]
        mask = cat_probs >= threshold
        selected_local = np.where(mask)[0]
        if len(selected_local) == 0:
            continue
        selected_global = valid[selected_local]
        selected_probs = cat_probs[selected_local]
        for idx_g, prob_val in zip(selected_global, selected_probs):
            if idx_g < len(labels.names) and labels.names[idx_g] is not None:
                result[category].append((labels.names[idx_g], float(prob_val)))

    for key in result:
        result[key] = sorted(result[key], key=lambda x: x[1], reverse=True)

    return result


class CamieTaggerEngine:
    """camie-tagger-v2 本地推理引擎"""

    def __init__(self, model_dir: str = CAMIE_MODEL_DIR):
        self.model_dir = model_dir
        self.model_path = os.path.join(CACHE_DIR, self.model_dir)
        self.session: Optional[Any] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.labels_data: Optional[LabelData] = None
        self.idx_to_tag: Optional[Dict[int, str]] = None
        self.tag_to_category: Optional[Dict[str, str]] = None
        self.image_size = DEFAULT_IMAGE_SIZE
        self.execution_provider: Optional[str] = None
        self._initialized = False

    def _get_model_paths(self) -> Tuple[str, str]:
        model_subdir = os.path.join(CACHE_DIR, self.model_dir)
        onnx_path = os.path.join(model_subdir, MODEL_FILENAME)
        metadata_path = os.path.join(model_subdir, METADATA_FILENAME)

        if os.path.exists(onnx_path) and os.path.exists(metadata_path):
            print(f"Camie Tagger 模型文件已存在于本地: {model_subdir}")
            return onnx_path, metadata_path

        print("本地未找到 Camie Tagger 模型文件，正在从 HuggingFace 下载...")
        print(f"仓库: {REPO_ID}")
        print(f"目标目录: {model_subdir}")

        from huggingface_hub import hf_hub_download

        os.makedirs(model_subdir, exist_ok=True)
        hf_token = os.environ.get("HF_TOKEN")

        onnx_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir=CACHE_DIR,
            token=hf_token,
            local_dir=model_subdir,
        )
        metadata_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=METADATA_FILENAME,
            cache_dir=CACHE_DIR,
            token=hf_token,
            local_dir=model_subdir,
        )

        actual_onnx = os.path.join(model_subdir, MODEL_FILENAME)
        actual_metadata = os.path.join(model_subdir, METADATA_FILENAME)
        if os.path.exists(actual_onnx) and os.path.exists(actual_metadata):
            return actual_onnx, actual_metadata

        return onnx_path, metadata_path

    def _init_onnx_runtime(self, onnx_path: str):
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        self.execution_provider = self.session.get_providers()[0]
        print(f"Camie Tagger ONNX Runtime 就绪: {self.execution_provider}")

    def initialize(self) -> bool:
        if self._initialized and self.session is not None:
            return True

        try:
            onnx_path, metadata_path = self._get_model_paths()

            print("加载 Camie Tagger metadata...")
            self.metadata, self.labels_data, self.idx_to_tag, self.tag_to_category, self.image_size = _load_metadata(metadata_path)
            print(f"Camie Tagger 标签已加载，共 {len(self.labels_data.names)} 个，输入尺寸 {self.image_size}")

            print("初始化 Camie Tagger 推理引擎...")
            self._init_onnx_runtime(onnx_path)

            self._initialized = True
            return True
        except Exception as e:
            print(f"Camie Tagger 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            return False

    def predict(self, image_path: str, gen_threshold: float = 0.5,
                char_threshold: float = 0.6) -> Optional[Dict]:
        if not self._initialized:
            if not self.initialize():
                return None

        if self.session is None or self.labels_data is None:
            print("Camie Tagger 尚未完成初始化")
            return None

        try:
            image = Image.open(image_path)
            input_tensor = _preprocess_image(image, self.image_size)

            start_time = time.time()
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            inference_time = time.time() - start_time
            print(f"Camie Tagger 推理耗时: {inference_time:.3f}s ({self.execution_provider})")

            if not outputs:
                raise RuntimeError("Model returned no outputs")

            if len(outputs) >= 2:
                main_logits = outputs[1]  # refined_predictions
                print(f"Camie Tagger 使用 refined_predictions 输出，shape={getattr(main_logits, 'shape', None)}")
            else:
                main_logits = outputs[0]
                print(f"Camie Tagger 使用 single output，shape={getattr(main_logits, 'shape', None)}")

            if np.isnan(main_logits).any() or np.isinf(main_logits).any():
                main_logits = np.nan_to_num(main_logits, nan=0.0, posinf=1.0, neginf=0.0)

            probs = _stable_sigmoid(main_logits[0])
            return _get_tags(probs, self.labels_data, gen_threshold, char_threshold)

        except Exception as e:
            print(f"Camie Tagger 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_to_text(self, image_path: str, gen_threshold: float = 0.5,
                        char_threshold: float = 0.6,
                        include_rating: bool = False,
                        include_quality: bool = False) -> Optional[str]:
        tags = self.predict(image_path, gen_threshold, char_threshold)
        if tags is None:
            return None

        parts = []

        if include_rating and tags.get("rating"):
            parts.extend([t[0] for t in tags["rating"]])

        if include_quality and tags.get("quality"):
            parts.extend([t[0] for t in tags["quality"]])

        if tags.get("character"):
            parts.extend([t[0] for t in tags["character"]])
        if tags.get("copyright"):
            parts.extend([t[0] for t in tags["copyright"]])
        if tags.get("artist"):
            parts.extend([t[0] for t in tags["artist"]])
        if tags.get("general"):
            parts.extend([t[0] for t in tags["general"]])

        return ", ".join(parts)

    def is_model_ready(self) -> bool:
        onnx_path = os.path.join(self.model_path, MODEL_FILENAME)
        metadata_path = os.path.join(self.model_path, METADATA_FILENAME)
        return os.path.exists(onnx_path) and os.path.exists(metadata_path)

    def get_status(self) -> Dict:
        onnx_exists = os.path.exists(os.path.join(self.model_path, MODEL_FILENAME))
        metadata_exists = os.path.exists(os.path.join(self.model_path, METADATA_FILENAME))
        return {
            "model_dir": self.model_dir,
            "cache_dir": self.model_path,
            "onnx_exists": onnx_exists,
            "metadata_exists": metadata_exists,
            "model_ready": onnx_exists and metadata_exists,
            "initialized": self._initialized,
            "execution_provider": self.execution_provider,
            "image_size": self.image_size,
        }
