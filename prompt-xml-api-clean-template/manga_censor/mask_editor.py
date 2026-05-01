"""分层 Mask 编辑器 - 后端逻辑

支持三层 mask 系统：
- auto: 自动检测层（只读）
- manual: 手动添加层（可编辑）
- inverse: 反向保留层（可编辑）
"""

import base64
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MaskPathManager:
    """Mask 路径管理器"""
    
    def __init__(self, mask_output_dir: str = "output/masks"):
        self.mask_output_dir = Path(mask_output_dir)
        self.mask_output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_mask_path(self, image_path: str, mask_type: str) -> Path:
        """
        根据图片路径生成对应的 mask 路径
        
        Args:
            image_path: 原图路径
            mask_type: mask 类型 ("auto", "manual", "inverse", "final", "preview")
        
        Returns:
            mask 文件路径
        """
        filename = Path(image_path).stem
        
        # final mask 直接使用原图名.png，其他类型添加后缀
        if mask_type == "final":
            return self.mask_output_dir / f"{filename}.png"
        else:
            return self.mask_output_dir / f"{filename}_{mask_type}.png"
    
    def get_all_mask_paths(self, image_path: str) -> Dict[str, Path]:
        """获取所有类型的 mask 路径"""
        return {
            "auto": self.get_mask_path(image_path, "auto"),
            "manual": self.get_mask_path(image_path, "manual"),
            "inverse": self.get_mask_path(image_path, "inverse"),
            "final": self.get_mask_path(image_path, "final"),
            "preview": self.get_mask_path(image_path, "preview")
        }
    
    def mask_exists(self, image_path: str, mask_type: str) -> bool:
        """检查 mask 是否存在"""
        return self.get_mask_path(image_path, mask_type).exists()


class LayeredMaskMerger:
    """分层 Mask 合并器"""
    
    @staticmethod
    def merge_layered_masks(
        auto_mask: np.ndarray,
        manual_mask: np.ndarray,
        inverse_mask: np.ndarray,
        mode: str = "standard"
    ) -> np.ndarray:
        """
        合并三层 mask
        
        Args:
            auto_mask: 自动检测 mask (0=保留, 255=遮盖)
            manual_mask: 手动添加 mask (0=保留, 255=遮盖)
            inverse_mask: 反向保留 mask (0=正常, 255=强制保留)
            mode: 合并模式
                - "standard": 标准合并（反向优先）
                - "additive": 仅添加（不反向）
                - "replace": 手动替换自动
        
        Returns:
            final_mask: 最终 mask
        """
        # 确保所有 mask 尺寸一致
        h, w = auto_mask.shape[:2]
        if manual_mask.shape[:2] != (h, w):
            manual_mask = cv2.resize(manual_mask, (w, h))
        if inverse_mask.shape[:2] != (h, w):
            inverse_mask = cv2.resize(inverse_mask, (w, h))
        
        # 转换为单通道灰度图
        if len(auto_mask.shape) == 3:
            auto_mask = cv2.cvtColor(auto_mask, cv2.COLOR_BGR2GRAY)
        if len(manual_mask.shape) == 3:
            manual_mask = cv2.cvtColor(manual_mask, cv2.COLOR_BGR2GRAY)
        if len(inverse_mask.shape) == 3:
            inverse_mask = cv2.cvtColor(inverse_mask, cv2.COLOR_BGR2GRAY)
        
        if mode == "standard":
            # 保留 AI 自动检测
            final = auto_mask.copy()
            # 手动添加层先应用反向保留（只清除手动添加的部分）
            manual_cleaned = manual_mask.copy()
            manual_cleaned[inverse_mask > 127] = 0
            # 合并清理后的手动层
            final = np.maximum(final, manual_cleaned)
            
        elif mode == "additive":
            # 仅添加，不反向
            final = np.maximum(auto_mask, manual_mask)
            
        elif mode == "replace":
            # 手动完全替换自动
            final = manual_mask.copy()
            final[inverse_mask > 127] = 0
        
        else:
            raise ValueError(f"Unknown merge mode: {mode}")
        
        return final
    
    @staticmethod
    def create_preview(
        image: np.ndarray,
        final_mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        创建遮盖预览图
        
        Args:
            image: 原图 (BGR)
            final_mask: 最终 mask
            color: 遮盖颜色 (BGR)
        
        Returns:
            preview: 预览图
        """
        preview = image.copy()
        preview[final_mask > 127] = color
        return preview


class MaskIO:
    """Mask 输入输出工具"""
    
    @staticmethod
    def load_mask(path: Path, default_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        加载 mask 文件
        
        Args:
            path: mask 文件路径
            default_shape: 如果文件不存在，创建空 mask 的尺寸 (h, w)
        
        Returns:
            mask: 灰度 mask (0-255)
        """
        if path.exists():
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                logger.info(f"加载 mask: {path.name}")
                return mask
        
        # 创建空 mask
        if default_shape is not None:
            h, w = default_shape
            logger.info(f"创建空 mask: {path.name} ({w}x{h})")
            return np.zeros((h, w), dtype=np.uint8)
        
        raise FileNotFoundError(f"Mask 文件不存在且未提供默认尺寸: {path}")
    
    @staticmethod
    def save_mask(mask: np.ndarray, path: Path) -> None:
        """保存 mask 文件"""
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), mask)
        logger.info(f"保存 mask: {path.name}")
    
    @staticmethod
    def mask_to_base64(mask: np.ndarray) -> str:
        """将 mask 转换为 base64 字符串"""
        _, buffer = cv2.imencode('.png', mask)
        return base64.b64encode(buffer).decode('utf-8')
    
    @staticmethod
    def base64_to_mask(base64_str: str) -> np.ndarray:
        """将 base64 字符串转换为 mask"""
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return mask


class MaskEditor:
    """Mask 编辑器主类"""
    
    def __init__(self, mask_output_dir: str = "output/masks"):
        self.path_manager = MaskPathManager(mask_output_dir)
        self.merger = LayeredMaskMerger()
        self.io = MaskIO()
    
    def load_editor_data(
        self,
        image_path: str,
        auto_mask: Optional[np.ndarray] = None
    ) -> Dict[str, str]:
        """
        加载编辑器数据
        
        Args:
            image_path: 图片路径
            auto_mask: 自动检测的 mask（如果提供，会保存为 auto mask）
        
        Returns:
            包含所有 mask 的 base64 数据
        """
        # 加载图片获取尺寸
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        
        h, w = image.shape[:2]
        
        # 获取所有 mask 路径
        paths = self.path_manager.get_all_mask_paths(image_path)
        
        # 如果提供了 auto_mask，保存它
        if auto_mask is not None:
            self.io.save_mask(auto_mask, paths["auto"])
        
        # 加载所有 mask
        masks = {}
        for mask_type in ["auto", "manual", "inverse"]:
            mask = self.io.load_mask(paths[mask_type], default_shape=(h, w))
            masks[mask_type] = self.io.mask_to_base64(mask)
        
        return masks
    
    def save_layer(
        self,
        image_path: str,
        layer_type: str,
        mask_base64: str
    ) -> str:
        """
        保存单个图层
        
        Args:
            image_path: 图片路径
            layer_type: 图层类型 ("manual" or "inverse")
            mask_base64: mask 的 base64 数据
        
        Returns:
            保存的文件路径
        """
        if layer_type not in ["manual", "inverse"]:
            raise ValueError(f"只能保存 manual 或 inverse 图层，不能保存: {layer_type}")
        
        mask = self.io.base64_to_mask(mask_base64)
        path = self.path_manager.get_mask_path(image_path, layer_type)
        self.io.save_mask(mask, path)
        
        return str(path)
    
    def merge_and_preview(
        self,
        image_path: str,
        mode: str = "standard"
    ) -> Dict[str, str]:
        """
        合并所有图层并生成预览
        
        Args:
            image_path: 图片路径
            mode: 合并模式
        
        Returns:
            包含 final_mask 和 preview 的 base64 数据
        """
        # 加载图片
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        
        h, w = image.shape[:2]
        
        # 获取所有 mask 路径
        paths = self.path_manager.get_all_mask_paths(image_path)
        
        # 加载所有图层
        auto_mask = self.io.load_mask(paths["auto"], default_shape=(h, w))
        manual_mask = self.io.load_mask(paths["manual"], default_shape=(h, w))
        inverse_mask = self.io.load_mask(paths["inverse"], default_shape=(h, w))
        
        # 合并
        final_mask = self.merger.merge_layered_masks(
            auto_mask, manual_mask, inverse_mask, mode=mode
        )
        
        # 保存最终 mask
        self.io.save_mask(final_mask, paths["final"])
        
        # 生成预览
        preview = self.merger.create_preview(image, final_mask)
        self.io.save_mask(preview, paths["preview"])
        
        return {
            "final_mask": self.io.mask_to_base64(final_mask),
            "preview": self.io.mask_to_base64(preview),
            "final_mask_path": str(paths["final"]),
            "preview_path": str(paths["preview"])
        }
