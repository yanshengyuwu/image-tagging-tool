"""
cl_tagger 反推打标引擎
基于 cella110n/cl_tagger ONNX 模型，本地推理生成动漫图片标签
"""

import io
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

REPO_ID = "cella110n/cl_tagger"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
IMAGE_SIZE = 448

# 默认使用最新版本
DEFAULT_MODEL_DIR = "cl_tagger_1_02"


@dataclass
class LabelData:
    names: list
    rating: list
    general: list
    artist: list
    character: list
    copyright: list
    meta: list
    quality: list
    model: list


def _pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    return image


def _pil_pad_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    new_size = max(width, height)
    new_image = Image.new(image.mode, (new_size, new_size), (255, 255, 255))
    paste_position = ((new_size - width) // 2, (new_size - height) // 2)
    new_image.paste(image, paste_position)
    return new_image


def _load_tag_mapping(mapping_path: str):
    with open(mapping_path, "r", encoding="utf-8") as f:
        tag_mapping_data = json.load(f)

    if isinstance(tag_mapping_data, dict) and "idx_to_tag" in tag_mapping_data:
        idx_to_tag = {int(k): v for k, v in tag_mapping_data["idx_to_tag"].items()}
        tag_to_category = tag_mapping_data["tag_to_category"]
    elif isinstance(tag_mapping_data, dict):
        try:
            tag_mapping_data_int_keys = {int(k): v for k, v in tag_mapping_data.items()}
            idx_to_tag = {idx: data["tag"] for idx, data in tag_mapping_data_int_keys.items()}
            tag_to_category = {data["tag"]: data["category"] for data in tag_mapping_data_int_keys.values()}
        except (KeyError, ValueError):
            raise ValueError("Unsupported tag mapping format")
    else:
        raise ValueError("Unsupported tag mapping format")

    names = [None] * (max(idx_to_tag.keys()) + 1)
    rating, general, artist, character, copyright_, meta, quality, model_name = [], [], [], [], [], [], [], []

    for idx, tag in idx_to_tag.items():
        if idx >= len(names):
            names.extend([None] * (idx - len(names) + 1))
        names[idx] = tag
        category = tag_to_category.get(tag, "Unknown")
        idx_int = int(idx)
        if category == "Rating":
            rating.append(idx_int)
        elif category == "General":
            general.append(idx_int)
        elif category == "Artist":
            artist.append(idx_int)
        elif category == "Character":
            character.append(idx_int)
        elif category == "Copyright":
            copyright_.append(idx_int)
        elif category == "Meta":
            meta.append(idx_int)
        elif category == "Quality":
            quality.append(idx_int)
        elif category == "Model":
            model_name.append(idx_int)

    return (
        LabelData(
            names=names,
            rating=np.array(rating, dtype=np.int64),
            general=np.array(general, dtype=np.int64),
            artist=np.array(artist, dtype=np.int64),
            character=np.array(character, dtype=np.int64),
            copyright=np.array(copyright_, dtype=np.int64),
            meta=np.array(meta, dtype=np.int64),
            quality=np.array(quality, dtype=np.int64),
            model=np.array(model_name, dtype=np.int64),
        ),
        idx_to_tag,
        tag_to_category,
    )


def _preprocess_image(image: Image.Image, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    image = _pil_ensure_rgb(image)
    image = _pil_pad_square(image)
    image_resized = image.resize(target_size, Image.BICUBIC)
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    img_array = img_array[::-1, :, :]  # RGB -> BGR
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)


def _get_tags(probs, labels: LabelData, gen_threshold: float, char_threshold: float) -> Dict:
    result = {
        "rating": [], "general": [], "character": [],
        "copyright": [], "artist": [], "meta": [],
        "quality": [], "model": [],
    }

    # Rating (select max)
    if len(labels.rating) > 0:
        valid = labels.rating[labels.rating < len(probs)]
        if len(valid) > 0:
            rating_probs = probs[valid]
            if len(rating_probs) > 0:
                idx_local = np.argmax(rating_probs)
                idx_global = valid[idx_local]
                if idx_global < len(labels.names) and labels.names[idx_global] is not None:
                    result["rating"].append((labels.names[idx_global], float(rating_probs[idx_local])))

    # Quality (select max)
    if len(labels.quality) > 0:
        valid = labels.quality[labels.quality < len(probs)]
        if len(valid) > 0:
            quality_probs = probs[valid]
            if len(quality_probs) > 0:
                idx_local = np.argmax(quality_probs)
                idx_global = valid[idx_local]
                if idx_global < len(labels.names) and labels.names[idx_global] is not None:
                    result["quality"].append((labels.names[idx_global], float(quality_probs[idx_local])))

    # Threshold-based categories
    category_map = {
        "general": (labels.general, gen_threshold),
        "character": (labels.character, char_threshold),
        "copyright": (labels.copyright, char_threshold),
        "artist": (labels.artist, char_threshold),
        "meta": (labels.meta, gen_threshold),
        "model": (labels.model, gen_threshold),
    }

    for category, (indices, threshold) in category_map.items():
        if len(indices) > 0:
            valid = indices[indices < len(probs)]
            if len(valid) > 0:
                cat_probs = probs[valid]
                mask = cat_probs >= threshold
                selected_local = np.where(mask)[0]
                if len(selected_local) > 0:
                    selected_global = valid[selected_local]
                    selected_probs = cat_probs[selected_local]
                    for idx_g, prob_val in zip(selected_global, selected_probs):
                        if idx_g < len(labels.names) and labels.names[idx_g] is not None:
                            result[category].append((labels.names[idx_g], float(prob_val)))

    # Sort by probability descending
    for k in result:
        result[k] = sorted(result[k], key=lambda x: x[1], reverse=True)

    return result


class CLTaggerEngine:
    """cl_tagger 本地推理引擎"""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self.session = None
        self.labels_data = None
        self.idx_to_tag = None
        self.tag_to_category = None
        self.use_openvino = False
        self.execution_provider = None
        self._initialized = False

    def _get_model_paths(self) -> Tuple[str, str]:
        """获取模型文件路径，如果本地不存在则从 HuggingFace 下载"""
        model_subdir = os.path.join(CACHE_DIR, self.model_dir)
        onnx_path = os.path.join(model_subdir, "model_optimized.onnx")
        mapping_path = os.path.join(model_subdir, "tag_mapping.json")

        if os.path.exists(onnx_path) and os.path.exists(mapping_path):
            print(f"模型文件已存在于本地: {model_subdir}")
            return onnx_path, mapping_path

        # 需要下载
        print(f"本地未找到模型文件，正在从 HuggingFace 下载...")
        print(f"仓库: {REPO_ID}")
        print(f"目标目录: {model_subdir}")

        from huggingface_hub import hf_hub_download

        os.makedirs(model_subdir, exist_ok=True)
        hf_token = os.environ.get("HF_TOKEN")

        onnx_filename = f"{self.model_dir}/model_optimized.onnx"
        mapping_filename = f"{self.model_dir}/tag_mapping.json"

        print(f"下载 model_optimized.onnx (~1.43GB)，请耐心等待...")
        onnx_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=onnx_filename,
            cache_dir=CACHE_DIR,
            token=hf_token,
            force_filename="model_optimized.onnx",
            local_dir=os.path.join(CACHE_DIR),
        )
        print(f"ONNX 模型下载完成: {onnx_path}")

        print(f"下载 tag_mapping.json...")
        mapping_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=mapping_filename,
            cache_dir=CACHE_DIR,
            token=hf_token,
            force_filename="tag_mapping.json",
            local_dir=os.path.join(CACHE_DIR),
        )
        print(f"标签映射下载完成: {mapping_path}")

        # hf_hub_download 的 local_dir 模式会保留子目录结构
        # 确认文件实际位置
        actual_onnx = os.path.join(CACHE_DIR, self.model_dir, "model_optimized.onnx")
        actual_mapping = os.path.join(CACHE_DIR, self.model_dir, "tag_mapping.json")

        if os.path.exists(actual_onnx) and os.path.exists(actual_mapping):
            return actual_onnx, actual_mapping

        # 回退：直接返回 hf_hub_download 返回的路径
        return onnx_path, mapping_path

    def initialize(self) -> bool:
        """初始化模型（下载+加载），返回是否成功"""
        if self._initialized and self.session is not None:
            return True

        try:
            onnx_path, mapping_path = self._get_model_paths()

            print("加载标签映射...")
            self.labels_data, self.idx_to_tag, self.tag_to_category = _load_tag_mapping(mapping_path)
            print(f"标签已加载，共 {len(self.labels_data.names)} 个")

            # 尝试 OpenVINO，回退到 ONNX Runtime
            print("初始化推理引擎...")
            try:
                import openvino as ov
                core = ov.Core()
                model = core.read_model(onnx_path)
                self.session = core.compile_model(model, "CPU")
                self.use_openvino = True
                self.execution_provider = "CPU - OpenVINO"
                print("使用 OpenVINO 推理引擎")
            except ImportError:
                print("OpenVINO 不可用，使用 ONNX Runtime")
                self._init_onnx_runtime(onnx_path)
            except Exception as e:
                print(f"OpenVINO 初始化失败: {e}，回退到 ONNX Runtime")
                self._init_onnx_runtime(onnx_path)

            self._initialized = True
            return True

        except Exception as e:
            print(f"cl_tagger 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            return False

    def _init_onnx_runtime(self, onnx_path: str):
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        # 优先使用 CUDA GPU 加速，没有 GPU 则自动回退 CPU
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        self.use_openvino = False
        self.execution_provider = self.session.get_providers()[0]
        print(f"ONNX Runtime 就绪: {self.execution_provider}")

    def predict(self, image_path: str, gen_threshold: float = 0.55,
                char_threshold: float = 0.6) -> Optional[Dict]:
        """
        对单张图片进行反推打标
        返回: {rating: [...], general: [...], character: [...], ...}
        每个类别是 (tag_name, confidence) 的列表
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            image = Image.open(image_path)
            input_tensor = _preprocess_image(image)

            start_time = time.time()

            if self.use_openvino:
                results = self.session(input_tensor)
                outputs = list(results.values())[0]
            else:
                input_name = self.session.get_inputs()[0].name
                output_name = self.session.get_outputs()[0].name
                outputs = self.session.run([output_name], {input_name: input_tensor})[0]

            inference_time = time.time() - start_time
            print(f"推理耗时: {inference_time:.3f}s ({self.execution_provider})")

            # 处理异常值
            if np.isnan(outputs).any() or np.isinf(outputs).any():
                outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)

            # Sigmoid
            def stable_sigmoid(x):
                return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

            probs = stable_sigmoid(outputs[0])

            tags = _get_tags(probs, self.labels_data, gen_threshold, char_threshold)
            return tags

        except Exception as e:
            print(f"推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_to_text(self, image_path: str, gen_threshold: float = 0.55,
                        char_threshold: float = 0.6,
                        include_rating: bool = False,
                        include_quality: bool = False) -> Optional[str]:
        """
        对单张图片反推打标，返回逗号分隔的标签文本
        """
        tags = self.predict(image_path, gen_threshold, char_threshold)
        if tags is None:
            return None

        parts = []

        if include_rating and tags.get("rating"):
            parts.extend([t[0] for t in tags["rating"]])

        if include_quality and tags.get("quality"):
            parts.extend([t[0] for t in tags["quality"]])

        # character 和 copyright 放前面
        if tags.get("character"):
            parts.extend([t[0] for t in tags["character"]])
        if tags.get("copyright"):
            parts.extend([t[0] for t in tags["copyright"]])

        # general tags
        if tags.get("general"):
            parts.extend([t[0] for t in tags["general"]])

        return ", ".join(parts)

    def is_model_ready(self) -> bool:
        """检查模型文件是否已存在于本地"""
        model_subdir = os.path.join(CACHE_DIR, self.model_dir)
        onnx_path = os.path.join(model_subdir, "model_optimized.onnx")
        mapping_path = os.path.join(model_subdir, "tag_mapping.json")
        return os.path.exists(onnx_path) and os.path.exists(mapping_path)

    def get_status(self) -> Dict:
        """获取引擎状态"""
        model_subdir = os.path.join(CACHE_DIR, self.model_dir)
        onnx_exists = os.path.exists(os.path.join(model_subdir, "model_optimized.onnx"))
        mapping_exists = os.path.exists(os.path.join(model_subdir, "tag_mapping.json"))
        return {
            "model_dir": self.model_dir,
            "cache_dir": model_subdir,
            "onnx_exists": onnx_exists,
            "mapping_exists": mapping_exists,
            "model_ready": onnx_exists and mapping_exists,
            "initialized": self._initialized,
            "execution_provider": self.execution_provider,
        }
