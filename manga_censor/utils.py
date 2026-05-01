"""Mask 工具函数。"""

import cv2
import numpy as np
from pathlib import Path


def cv2_imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Unicode 安全的图像读取，替代 cv2.imread。

    cv2.imread 在 Windows 上不支持中文路径，使用 np.fromfile + cv2.imdecode 绕过。
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def cv2_imwrite(path: str | Path, img: np.ndarray, params=None) -> bool:
    """Unicode 安全的图像写入，替代 cv2.imwrite。

    cv2.imwrite 在 Windows 上不支持中文路径，使用 cv2.imencode + ndarray.tofile 绕过。
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix if path.suffix else '.png'
        if params:
            success, buf = cv2.imencode(ext, img, params)
        else:
            success, buf = cv2.imencode(ext, img)
        if success:
            buf.tofile(str(path))
            return True
        return False
    except Exception:
        return False


def get_onnx_providers() -> list[str]:
    """返回当前环境可用的 ONNX Runtime 执行提供器列表。

    优先使用 CUDA，不可用时自动回退到 CPU，避免 UserWarning。
    """
    import onnxruntime as ort
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in preferred if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    return providers


def bbox_to_mask(h: int, w: int, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """将 bbox 列表转换为二值 mask。

    Args:
        h, w: 原图尺寸
        boxes: [(x1, y1, x2, y2), ...] 整数坐标

    Returns:
        uint8 mask (H, W), 0 or 255
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = 255
    return mask


def shrink_bboxes(boxes: list[tuple[int, int, int, int]], ratio: float) -> list[tuple[int, int, int, int]]:
    """将 bbox 向中心收缩，缩小遮盖范围。

    Args:
        boxes: [(x1, y1, x2, y2), ...] 原始 bbox 列表
        ratio: 收缩比例 (0.0-1.0)，例如 0.65 表示缩小到原尺寸的 65%

    Returns:
        收缩后的 bbox 列表
    """
    if ratio is None or ratio >= 1.0:
        return boxes

    shrunk = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half_w = (x2 - x1) * ratio / 2.0
        half_h = (y2 - y1) * ratio / 2.0

        nx1 = int(round(cx - half_w))
        ny1 = int(round(cy - half_h))
        nx2 = int(round(cx + half_w))
        ny2 = int(round(cy + half_h))

        # 确保不翻转
        if nx2 <= nx1:
            nx2 = nx1 + 1
        if ny2 <= ny1:
            ny2 = ny1 + 1

        shrunk.append((nx1, ny1, nx2, ny2))

    return shrunk


def seg_to_mask(h: int, w: int, seg_data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """将分割模型输出 resize 到原图尺寸并二值化。

    Args:
        h, w: 原图尺寸
        seg_data: 模型输出的 mask（任意尺寸，float）
        threshold: 二值化阈值

    Returns:
        uint8 mask (H, W), 0 or 255
    """
    if seg_data.shape[:2] != (h, w):
        seg_resized = cv2.resize(seg_data.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        seg_resized = seg_data.astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[seg_resized > threshold] = 255
    return mask


def merge_masks(masks: list[np.ndarray]) -> np.ndarray:
    """将多个 mask 合并为一个（OR 操作）。"""
    if not masks:
        raise ValueError("masks 列表为空")
    result = masks[0].copy()
    for m in masks[1:]:
        result = np.maximum(result, m)
    return result


def save_mask(mask: np.ndarray, path: str | Path):
    """保存单通道 mask 为 PNG（Unicode 安全）。"""
    cv2_imwrite(path, mask)


def ensure_model_dir(model_dir: str | Path) -> Path:
    """确保模型目录存在。"""
    p = Path(model_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def extract_connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """从二值 mask 中提取连通区域的 bbox 列表。

    Args:
        mask: uint8 mask (H, W), 0 or 255

    Returns:
        [(x1, y1, x2, y2), ...] 每个连通区域的外接矩形
    """
    binary = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    for i in range(1, num_labels):  # 跳过背景 (label=0)
        x, y, w, h, area = stats[i]
        if w > 2 and h > 2:  # 过滤极小区域
            boxes.append((x, y, x + w, y + h))

    return boxes


def extract_component_masks(mask: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """从二值 mask 中提取每个连通区域的独立 mask 和 bbox。

    Args:
        mask: uint8 mask (H, W), 0 or 255

    Returns:
        [(component_mask, (x1, y1, x2, y2)), ...] 每个连通区域的 mask 和 bbox
    """
    binary = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    components = []
    for i in range(1, num_labels):  # 跳过背景
        x, y, w, h, area = stats[i]
        if w > 2 and h > 2:
            component_mask = np.zeros_like(mask)
            component_mask[labels == i] = 255
            components.append((component_mask, (x, y, x + w, y + h)))

    return components


def smooth_mask_edges(mask: np.ndarray, blur_radius: int = 5) -> np.ndarray:
    """平滑 mask 边缘：高斯模糊 + 重新二值化。

    Args:
        mask: uint8 mask (H, W), 0 or 255
        blur_radius: 高斯模糊核大小（奇数）

    Returns:
        平滑后的 uint8 mask
    """
    if blur_radius < 3:
        return mask
    blurred = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
    _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return result


def filter_mask_components(
    mask: np.ndarray,
    min_area: int = 0,
    max_area: int | None = None,
    min_width: int = 0,
    min_height: int = 0,
    max_aspect_ratio: float | None = None,
) -> np.ndarray:
    """按连通域几何特征过滤 mask，去除噪点和离谱大块。"""
    binary = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(mask)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        if w < min_width or h < min_height:
            continue

        short_side = max(1, min(w, h))
        long_side = max(w, h)
        aspect_ratio = long_side / short_side
        if max_aspect_ratio is not None and aspect_ratio > max_aspect_ratio:
            continue

        filtered[labels == i] = 255

    return filtered


def morph_close_open(
    mask: np.ndarray,
    close_kernel: int = 0,
    open_kernel: int = 0,
) -> np.ndarray:
    """对 mask 做闭运算和开运算，填小洞并去小毛刺。"""
    result = mask.copy()
    if close_kernel and close_kernel >= 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    if open_kernel and open_kernel >= 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    return result


def dilate_mask(mask: np.ndarray, kernel_size: int = 0, iterations: int = 1) -> np.ndarray:
    """轻微膨胀 mask，避免边缘漏遮。"""
    if kernel_size < 2:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=iterations)
