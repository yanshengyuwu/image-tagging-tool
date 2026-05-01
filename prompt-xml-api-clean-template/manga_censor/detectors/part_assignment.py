"""部位归属算法 — 将检测到的部位分配给对应的人物实例。

多策略匹配：
1. IoU (Intersection over Union): 重叠度匹配
2. 中心点距离: 部位中心与人物中心的距离
3. 包含关系: 部位是否完全在人物 bbox 内

支持多人场景特殊处理：
- 自动降低阈值
- 允许部位被多人共享（按距离分配）
- 未分配部位作为公共部位处理
"""

import logging
from typing import Dict, Set, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def calculate_iou(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """计算 mask 与 bbox 的 IoU。
    
    Args:
        mask: 二值 mask (H, W)
        bbox: (x1, y1, x2, y2)
    
    Returns:
        IoU 值 [0, 1]
    """
    x1, y1, x2, y2 = bbox
    h, w = mask.shape[:2]
    
    # 确保 bbox 在图像范围内
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    
    # 创建 bbox mask
    bbox_mask = np.zeros_like(mask)
    bbox_mask[y1:y2, x1:x2] = 255
    
    # 计算交集和并集
    intersection = np.logical_and(mask > 0, bbox_mask > 0).sum()
    union = np.logical_or(mask > 0, bbox_mask > 0).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def calculate_mask_center(mask: np.ndarray) -> Tuple[float, float]:
    """计算 mask 的中心点（质心）。
    
    Args:
        mask: 二值 mask (H, W)
    
    Returns:
        (center_x, center_y)
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0.0, 0.0)
    
    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))
    return (center_x, center_y)


def calculate_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """计算 bbox 的中心点。
    
    Args:
        bbox: (x1, y1, x2, y2)
    
    Returns:
        (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return (center_x, center_y)


def calculate_center_distance(
    mask: np.ndarray, 
    bbox: Tuple[int, int, int, int]
) -> float:
    """计算 mask 中心与 bbox 中心的归一化距离。
    
    Args:
        mask: 二值 mask (H, W)
        bbox: (x1, y1, x2, y2)
    
    Returns:
        归一化距离 [0, 1+]，越小越近
    """
    mask_center = calculate_mask_center(mask)
    bbox_center = calculate_bbox_center(bbox)
    
    # 计算欧氏距离
    dx = mask_center[0] - bbox_center[0]
    dy = mask_center[1] - bbox_center[1]
    distance = np.sqrt(dx * dx + dy * dy)
    
    # 归一化：除以 bbox 对角线长度
    x1, y1, x2, y2 = bbox
    bbox_diag = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if bbox_diag == 0:
        return float('inf')
    
    return distance / bbox_diag


def check_containment(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """检查 mask 是否被 bbox 包含。
    
    Args:
        mask: 二值 mask (H, W)
        bbox: (x1, y1, x2, y2)
    
    Returns:
        包含比例 [0, 1]，1 表示完全包含
    """
    x1, y1, x2, y2 = bbox
    h, w = mask.shape[:2]
    
    # 确保 bbox 在图像范围内
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    
    # 计算 mask 在 bbox 内的像素数
    mask_in_bbox = mask[y1:y2, x1:x2]
    pixels_in_bbox = (mask_in_bbox > 0).sum()
    total_pixels = (mask > 0).sum()
    
    if total_pixels == 0:
        return 0.0
    
    return float(pixels_in_bbox) / float(total_pixels)


def calculate_match_score(
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """计算部位与人物的综合匹配分数。
    
    Args:
        mask: 部位 mask
        bbox: 人物 bbox
        weights: 各策略权重 {"iou": 0.4, "distance": 0.3, "containment": 0.3}
    
    Returns:
        综合匹配分数 [0, 1]，越高越匹配
    """
    if weights is None:
        weights = {"iou": 0.4, "distance": 0.3, "containment": 0.3}
    
    # 1. IoU 分数
    iou = calculate_iou(mask, bbox)
    
    # 2. 中心距离分数（距离越小分数越高）
    distance = calculate_center_distance(mask, bbox)
    distance_score = max(0.0, 1.0 - distance)  # 转换为分数
    
    # 3. 包含关系分数
    containment = check_containment(mask, bbox)
    
    # 加权综合
    score = (
        weights["iou"] * iou +
        weights["distance"] * distance_score +
        weights["containment"] * containment
    )
    
    return score


def assign_parts_to_persons(
    part_masks: Dict[str, np.ndarray],
    persons: List,
    score_threshold: float = 0.15,
    multi_person_mode: bool = None,
    allow_shared_parts: bool = False
) -> Dict[int, Dict[str, np.ndarray]]:
    """将部位 mask 分配给对应的人物（多策略匹配）。
    
    Args:
        part_masks: {part_name: mask} 所有检测到的部位 mask
        persons: PersonInstance 列表
        score_threshold: 匹配分数阈值，低于此值的部位不分配
        multi_person_mode: 是否为多人模式（None 时自动判断）
        allow_shared_parts: 是否允许部位被多人共享
    
    Returns:
        {person_id: {part_name: mask}} 每个人物的部位 mask
    """
    person_parts: Dict[int, Dict[str, np.ndarray]] = {
        person.person_id: {} for person in persons
    }
    
    # 自动判断是否为多人场景
    if multi_person_mode is None:
        multi_person_mode = len(persons) > 1
    
    # 多人场景：降低阈值，提高召回率
    if multi_person_mode:
        effective_threshold = score_threshold * 0.7
        logger.info(
            f"[part_assignment] 检测到 {len(persons)} 个人物，"
            f"降低阈值至 {effective_threshold:.3f}"
        )
    else:
        effective_threshold = score_threshold
    
    # 未分配的部位（公共部位）
    unassigned_parts = {}
    
    for part_name, mask in part_masks.items():
        if mask is None or mask.sum() == 0:
            continue
        
        # 计算与每个人物的匹配分数
        scores = []
        for person in persons:
            score = calculate_match_score(mask, person.bbox)
            scores.append((person.person_id, score))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if not scores:
            continue
        
        best_person_id, best_score = scores[0]
        
        # 分配策略
        if best_score >= effective_threshold:
            # 分配给最佳匹配的人物
            person_parts[best_person_id][part_name] = mask
            logger.debug(
                f"[part_assignment] {part_name} → person_{best_person_id} "
                f"(score={best_score:.3f})"
            )
            
            # 多人共享模式：如果其他人物分数也很高，也分配给他们
            if allow_shared_parts and multi_person_mode:
                for person_id, score in scores[1:]:
                    if score >= effective_threshold * 0.8:  # 次优阈值
                        person_parts[person_id][part_name] = mask
                        logger.debug(
                            f"[part_assignment] {part_name} → person_{person_id} "
                            f"(shared, score={score:.3f})"
                        )
        else:
            # 未达到阈值，标记为公共部位
            unassigned_parts[part_name] = mask
            logger.debug(
                f"[part_assignment] {part_name} 未分配 "
                f"(best_score={best_score:.3f} < {effective_threshold:.3f})"
            )
    
    # 记录统计信息
    assigned_count = sum(len(parts) for parts in person_parts.values())
    logger.info(
        f"[part_assignment] 分配完成: {assigned_count} 个部位已分配, "
        f"{len(unassigned_parts)} 个未分配"
    )
    
    # 为没有分配到任何部位的人物记录警告
    for person in persons:
        if not person_parts[person.person_id]:
            logger.warning(
                f"[part_assignment] person_{person.person_id} 未分配到任何部位"
            )
    
    return person_parts


def get_parts_by_person(
    person_parts: Dict[int, Dict[str, np.ndarray]]
) -> Dict[int, Set[str]]:
    """提取每个人物检测到的部位名称集合。
    
    Args:
        person_parts: {person_id: {part_name: mask}}
    
    Returns:
        {person_id: {part_names}}
    """
    return {
        person_id: set(parts.keys())
        for person_id, parts in person_parts.items()
    }


def merge_person_masks(
    person_parts: Dict[str, np.ndarray],
    part_names: List[str]
) -> np.ndarray:
    """合并指定部位的 mask。
    
    Args:
        person_parts: {part_name: mask} 某个人物的所有部位
        part_names: 要合并的部位名称列表
    
    Returns:
        合并后的 mask
    """
    if not person_parts:
        return None
    
    # 获取第一个 mask 的形状
    first_mask = next(iter(person_parts.values()))
    h, w = first_mask.shape[:2]
    merged = np.zeros((h, w), dtype=np.uint8)
    
    for part_name in part_names:
        if part_name in person_parts:
            mask = person_parts[part_name]
            merged = np.maximum(merged, mask)
    
    return merged
