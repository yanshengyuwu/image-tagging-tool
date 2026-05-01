"""
修复后的 handler wrapper，智能选择GPU或CPU
"""
import sys
sys.path.append(r'D:\pixai-tagger')

import os
import time
import base64
import io
from typing import Any

import torch
import requests
from PIL import Image

from handler import EndpointHandler as OriginalHandler

class CPUEndpointHandler(OriginalHandler):
    """智能选择GPU/CPU的handler"""
    def __init__(self, path: str):
        super().__init__(path)
        
        # 智能选择设备
        if torch.cuda.is_available():
            self.device = 'cuda'
            if hasattr(self, 'model'):
                self.model = self.model.cuda()
            print(f"✅ 使用GPU模式 (device: {self.device}, GPU: {torch.cuda.get_device_name(0)})")
        else:
            self.device = 'cpu'
            if hasattr(self, 'model'):
                self.model = self.model.cpu()
            print(f"✅ 使用CPU模式 (未检测到可用GPU)")
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """重写__call__方法，兼容GPU和CPU"""
        inputs = data.pop("inputs", data)

        fetch_start_time = time.time()
        if isinstance(inputs, Image.Image):
            image = inputs
        elif image_url := inputs.pop("url", None):
            with requests.get(
                image_url, stream=True, timeout=self.fetch_image_timeout
            ) as res:
                res.raise_for_status()
                image = Image.open(res.raw)
        elif image_base64_encoded := inputs.pop("image", None):
            image = Image.open(io.BytesIO(base64.b64decode(image_base64_encoded)))
        else:
            raise ValueError(f"No image or url provided: {data}")
        
        # 导入pil_to_rgb函数
        from handler import pil_to_rgb
        image = pil_to_rgb(image)
        fetch_time = time.time() - fetch_start_time

        parameters = data.pop("parameters", {})
        general_threshold = parameters.pop(
            "general_threshold", self.default_general_threshold
        )
        character_threshold = parameters.pop(
            "character_threshold", self.default_character_threshold
        )

        inference_start_time = time.time()
        with torch.inference_mode():
            # 预处理图像
            image_tensor = self.transform(image).unsqueeze(0)
            
            # 根据设备选择是否使用pin_memory
            if self.device == 'cuda' and torch.cuda.is_available():
                # GPU模式：使用pin_memory和异步传输
                image_tensor = image_tensor.pin_memory()
                image_tensor = image_tensor.to(self.device, non_blocking=True)
            else:
                # CPU模式：直接传输
                image_tensor = image_tensor.to(self.device)
            
            # 运行模型
            probs = self.model(image_tensor)[0]
            
            # 阈值处理
            general_mask = probs[: self.gen_tag_count] > general_threshold
            character_mask = probs[self.gen_tag_count :] > character_threshold
            
            # 获取正标签的索引
            general_indices = general_mask.nonzero(as_tuple=True)[0]
            character_indices = (
                character_mask.nonzero(as_tuple=True)[0] + self.gen_tag_count
            )
            
            # 合并索引并移到CPU
            combined_indices = torch.cat((general_indices, character_indices)).cpu()

        inference_time = time.time() - inference_start_time

        post_process_start_time = time.time()

        cur_gen_tags = []
        cur_char_tags = []

        # 使用预计算的映射进行查找
        for i in combined_indices:
            idx = i.item()
            tag = self.index_to_tag_map[idx]
            if idx < self.gen_tag_count:
                cur_gen_tags.append(tag)
            else:
                cur_char_tags.append(tag)

        ip_tags = []
        for tag in cur_char_tags:
            if tag in self.character_ip_mapping:
                ip_tags.extend(self.character_ip_mapping[tag])
        ip_tags = sorted(set(ip_tags))
        post_process_time = time.time() - post_process_start_time

        print(
            f"⏱️  Timing - Fetch: {fetch_time:.3f}s, Inference: {inference_time:.3f}s, Post-process: {post_process_time:.3f}s"
        )

        return {
            "feature": cur_gen_tags,
            "character": cur_char_tags,
            "ip": ip_tags,
        }

