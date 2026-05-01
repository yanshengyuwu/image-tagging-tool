"""
PixAI Tagger 反推打标引擎
基于 pixai-labs/pixai-tagger-v0.9 模型，本地推理生成动漫图片标签
"""

import importlib.util
import os
import sys
import time
from typing import Dict, Optional

from PIL import Image

PIXAI_REPO_ID = "pixai-labs/pixai-tagger-v0.9"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
PIXAI_MODEL_DIR = "pixai_tagger_v09"


class PixAITaggerEngine:
    """PixAI Tagger 本地推理引擎"""

    def __init__(self):
        self.handler = None
        self.model_path = os.path.join(CACHE_DIR, PIXAI_MODEL_DIR)
        self._initialized = False
        self.device = None

    def _download_model(self):
        """从 HuggingFace 下载完整模型仓库"""
        if self._check_model_files():
            print(f"PixAI Tagger 模型已存在于本地: {self.model_path}")
            return True

        print(f"本地未找到 PixAI Tagger 模型，正在从 HuggingFace 下载...")
        print(f"仓库: {PIXAI_REPO_ID}")
        print(f"目标目录: {self.model_path}")

        try:
            from huggingface_hub import snapshot_download

            hf_token = os.environ.get("HF_TOKEN")
            snapshot_download(
                repo_id=PIXAI_REPO_ID,
                local_dir=self.model_path,
                token=hf_token,
            )
            print(f"PixAI Tagger 模型下载完成")
            return True
        except Exception as e:
            print(f"PixAI Tagger 模型下载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _check_model_files(self) -> bool:
        """检查模型关键文件是否存在"""
        handler_path = os.path.join(self.model_path, "handler.py")
        # 检查 handler.py 和至少一个模型权重文件
        if not os.path.exists(handler_path):
            return False
        # 检查是否有 .safetensors 或 .bin 或 .pth 模型文件
        for f in os.listdir(self.model_path):
            if f.endswith(('.safetensors', '.bin', '.pth', '.pt')):
                return True
        # 也可能模型文件在子目录中
        for root, dirs, files in os.walk(self.model_path):
            for f in files:
                if f.endswith(('.safetensors', '.bin', '.pth', '.pt')):
                    return True
        return False

    def _load_handler(self):
        """动态加载 handler 模块并初始化"""
        handler_path = os.path.join(self.model_path, "handler.py")
        if not os.path.exists(handler_path):
            raise FileNotFoundError(f"handler.py 不存在: {handler_path}")

        # 将模型目录加入 sys.path 以便 handler 内部的相对导入
        if self.model_path not in sys.path:
            sys.path.insert(0, self.model_path)

        # 动态导入 handler 模块
        spec = importlib.util.spec_from_file_location("pixai_handler", handler_path)
        handler_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(handler_module)

        # 尝试使用 handler_wrapper 中的 CPUEndpointHandler（支持 GPU/CPU 自动选择）
        wrapper_path = os.path.join(self.model_path, "handler_wrapper.py")
        if os.path.exists(wrapper_path):
            try:
                wrapper_spec = importlib.util.spec_from_file_location("pixai_wrapper", wrapper_path)
                wrapper_module = importlib.util.module_from_spec(wrapper_spec)
                wrapper_spec.loader.exec_module(wrapper_module)
                if hasattr(wrapper_module, 'CPUEndpointHandler'):
                    self.handler = wrapper_module.CPUEndpointHandler(path=self.model_path)
                    self.device = getattr(self.handler, 'device', 'unknown')
                    print(f"PixAI Tagger 使用 CPUEndpointHandler (device: {self.device})")
                    return
            except Exception as e:
                print(f"加载 handler_wrapper 失败: {e}，回退到原始 handler")

        # 回退：使用原始 EndpointHandler
        if hasattr(handler_module, 'EndpointHandler'):
            self.handler = handler_module.EndpointHandler(path=self.model_path)
            self.device = getattr(self.handler, 'device', 'cpu')
            print(f"PixAI Tagger 使用 EndpointHandler (device: {self.device})")
        else:
            raise AttributeError("handler.py 中未找到 EndpointHandler 类")

    def initialize(self) -> bool:
        """初始化模型（下载+加载），返回是否成功"""
        if self._initialized and self.handler is not None:
            return True

        try:
            if not self._download_model():
                return False

            print("加载 PixAI Tagger 模型...")
            self._load_handler()
            self._initialized = True
            print("PixAI Tagger 初始化完成")
            return True

        except Exception as e:
            print(f"PixAI Tagger 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            return False

    def predict(self, image_path: str, gen_threshold: float = 0.35,
                char_threshold: float = 0.85) -> Optional[Dict]:
        """
        对单张图片进行反推打标
        返回: {general: [...], character: [...], copyright: [...]}
        每个类别是 (tag_name, confidence) 的列表
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            image = Image.open(image_path)

            start_time = time.time()

            result = self.handler({
                "inputs": image,
                "parameters": {
                    "general_threshold": gen_threshold,
                    "character_threshold": char_threshold,
                }
            })

            inference_time = time.time() - start_time
            print(f"PixAI 推理耗时: {inference_time:.3f}s")

            # 转换为统一格式：(tag_name, confidence) 列表
            # pixai handler 返回的是纯标签列表（无置信度），统一赋 1.0
            tags = {
                "rating": [],
                "general": [(t, 1.0) for t in result.get("feature", [])],
                "character": [(t, 1.0) for t in result.get("character", [])],
                "copyright": [(t, 1.0) for t in result.get("ip", [])],
                "artist": [],
                "meta": [],
                "quality": [],
                "model": [],
            }
            return tags

        except Exception as e:
            print(f"PixAI 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_to_text(self, image_path: str, gen_threshold: float = 0.35,
                        char_threshold: float = 0.85,
                        include_rating: bool = False,
                        include_quality: bool = False) -> Optional[str]:
        """对单张图片反推打标，返回逗号分隔的标签文本"""
        tags = self.predict(image_path, gen_threshold, char_threshold)
        if tags is None:
            return None

        parts = []

        if tags.get("character"):
            parts.extend([t[0] for t in tags["character"]])
        if tags.get("copyright"):
            parts.extend([t[0] for t in tags["copyright"]])
        if tags.get("general"):
            parts.extend([t[0] for t in tags["general"]])

        return ", ".join(parts)

    def is_model_ready(self) -> bool:
        """检查模型文件是否已存在于本地"""
        return self._check_model_files()

    def get_status(self) -> Dict:
        """获取引擎状态"""
        model_ready = self._check_model_files()
        return {
            "model_dir": PIXAI_MODEL_DIR,
            "cache_dir": self.model_path,
            "model_ready": model_ready,
            "initialized": self._initialized,
            "execution_provider": f"PyTorch ({self.device})" if self.device else None,
        }
