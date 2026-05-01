#!/usr/bin/env python3
"""
x-anylabeling 双向转换模块

支持：
1. 正向转换：PNG Mask → x-anylabeling JSON (COCO格式)
2. 反向转换：x-anylabeling JSON → PNG Mask

x-anylabeling 使用 COCO/VOC 风格的 JSON 格式，包含多边形（polygon）标注。
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any


def natural_sort_key(filename: str) -> List:
    """自然排序：将数字按数值大小排序"""
    import re
    parts = []
    for part in re.split(r'(\d+)', filename):
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part.lower()))
    return parts


def mask_to_polygons(mask: np.ndarray, min_area: int = 50) -> List[np.ndarray]:
    """
    从二值 mask 提取多边形轮廓点集
    
    Args:
        mask: 二值图像 (uint8, 0/255)
        min_area: 最小轮廓面积阈值，过小的轮廓会被忽略
    
    Returns:
        轮廓点列表，每个轮廓是一个 (N, 1, 2) 的 numpy 数组
    """
    # 确保是二值图像
    if mask.dtype != np.uint8:
        mask = (mask > 127).astype(np.uint8) * 255
    
    # 查找轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            # 简化轮廓（减少点数）
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygons.append(approx)
    
    return polygons


def polygons_to_mask(polygons: List, height: int, width: int, fill_value: int = 255) -> np.ndarray:
    """
    将多边形列表绘制为二值 mask
    
    Args:
        polygons: 多边形列表，每个元素是 (N, 1, 2) 的 numpy 数组或点列表
        height: 图像高度
        width: 图像宽度
        fill_value: 填充值
    
    Returns:
        二值 mask 图像 (uint8)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for polygon in polygons:
        # 转换为 numpy 数组
        if isinstance(polygon, list):
            pts = np.array(polygon, dtype=np.int32)
        else:
            pts = polygon.astype(np.int32)
        
        # 确保是 (N, 2) 形状
        if len(pts.shape) == 3 and pts.shape[1] == 1:
            pts = pts.reshape(-1, 2)
        
        # 绘制填充多边形
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], fill_value)
        elif len(pts) == 2:
            # 线段
            cv2.line(mask, tuple(pts[0]), tuple(pts[1]), fill_value, 2)
    
    return mask


def mask_to_xanylabeling(mask_path: str, image_path: str, output_path: str = None,
                         label: str = "mask", min_area: int = 50,
                         simplify_tolerance: float = 2.0) -> Dict[str, Any]:
    """
    将 PNG Mask 转换为 x-anylabeling JSON 格式
    
    Args:
        mask_path: PNG mask 文件路径
        image_path: 对应的原始图片路径（用于获取尺寸）
        output_path: 输出 JSON 路径，默认为同名的 .json 文件
        label: 标注标签名称
        min_area: 最小轮廓面积
        simplify_tolerance: 轮廓简化容差
    
    Returns:
        包含转换结果的字典
    """
    # 读取 mask
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask 文件不存在: {mask_path}")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取 mask 文件: {mask_path}")
    
    # 读取原图获取尺寸
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            img_height, img_width = img.shape[:2]
        else:
            img_height, img_width = mask.shape[:2]
    else:
        img_height, img_width = mask.shape[:2]
    
    # 提取多边形
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # 简化轮廓
        epsilon = simplify_tolerance * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 转换为点列表 [[x, y], [x, y], ...]
        points = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]
        
        if len(points) >= 3:
            shape = {
                "label": label,
                "points": points,
                "group_id": i + 1,
                "shape_type": "polygon",
                "flags": {}
            }
            shapes.append(shape)
    
    # 获取图片文件名
    image_filename = os.path.basename(image_path) if image_path else os.path.basename(mask_path).replace('.png', '.jpg')
    
    # 构建 x-anylabeling JSON 结构
    xanylabeling_data = {
        "version": "1.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }
    
    # 保存 JSON
    if output_path is None:
        output_path = os.path.splitext(mask_path)[0] + '.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(xanylabeling_data, f, ensure_ascii=False, indent=2)
    
    return {
        'success': True,
        'mask_path': mask_path,
        'image_path': image_path,
        'json_path': output_path,
        'shapes_count': len(shapes),
        'label': label
    }


def xanylabeling_to_mask(json_path: str, output_path: str = None, 
                         fill_value: int = 255) -> Dict[str, Any]:
    """
    将 x-anylabeling JSON 转换为 PNG Mask
    
    Args:
        json_path: x-anylabeling JSON 文件路径
        output_path: 输出 PNG mask 路径，默认为同名的 _mask.png 文件
        fill_value: 填充值（255=白色遮盖区域）
    
    Returns:
        包含转换结果的字典
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")
    
    # 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取图像尺寸
    height = data.get('imageHeight', 0)
    width = data.get('imageWidth', 0)
    
    if height == 0 or width == 0:
        raise ValueError(f"JSON 中缺少有效的图像尺寸信息")
    
    # 提取多边形
    shapes = data.get('shapes', [])
    polygons = []
    labels = []
    
    for shape in shapes:
        if shape.get('shape_type') == 'polygon':
            points = shape.get('points', [])
            label = shape.get('label', 'unknown')
            
            if len(points) >= 3:
                polygons.append(points)
                labels.append(label)
    
    # 绘制 mask
    mask = polygons_to_mask(polygons, height, width, fill_value)
    
    # 保存
    if output_path is None:
        json_basename = os.path.splitext(os.path.basename(json_path))[0]
        output_path = os.path.join(os.path.dirname(json_path), f"{json_basename}_mask.png")
    
    cv2.imwrite(output_path, mask)
    
    return {
        'success': True,
        'json_path': json_path,
        'mask_path': output_path,
        'shapes_count': len(polygons),
        'labels': list(set(labels)) if labels else [],
        'image_size': (width, height)
    }


def batch_mask_to_xanylabeling(mask_dir: str, image_dir: str = None, 
                                output_dir: str = None, 
                                label: str = "mask",
                                min_area: int = 50,
                                file_pattern: str = "*.png") -> Dict[str, Any]:
    """
    批量将 mask 文件转换为 x-anylabeling JSON
    
    Args:
        mask_dir: mask 文件目录
        image_dir: 原始图片目录（可选，默认与 mask_dir 相同）
        output_dir: 输出目录（可选，默认覆盖 mask 目录）
        label: 标注标签
        min_area: 最小轮廓面积
        file_pattern: 文件匹配模式
    
    Returns:
        转换结果统计
    """
    from glob import glob
    
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask 目录不存在: {mask_dir}")
    
    if output_dir is None:
        output_dir = mask_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    if image_dir is None:
        image_dir = mask_dir
    
    # 查找所有 mask 文件
    mask_files = glob(os.path.join(mask_dir, file_pattern))
    mask_files = sorted(mask_files, key=natural_sort_key)
    
    results = {
        'total': len(mask_files),
        'success': 0,
        'failed': 0,
        'details': []
    }
    
    for mask_path in mask_files:
        try:
            mask_name = os.path.basename(mask_path)
            mask_base = os.path.splitext(mask_name)[0]
            
            # 查找对应的图片
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                candidate = os.path.join(image_dir, mask_base + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            
            # 如果没找到图片，使用 mask 自身的尺寸
            if image_path is None:
                image_path = mask_path
            
            # 输出路径
            output_path = os.path.join(output_dir, mask_base + '.json')
            
            # 转换
            result = mask_to_xanylabeling(
                mask_path, image_path, output_path,
                label=label, min_area=min_area
            )
            
            results['success'] += 1
            results['details'].append({
                'mask': mask_name,
                'status': 'success',
                'shapes': result['shapes_count']
            })
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append({
                'mask': mask_name,
                'status': 'failed',
                'error': str(e)
            })
    
    results['summary'] = f"转换完成: {results['success']}/{results['total']} 成功"
    return results


def batch_xanylabeling_to_mask(json_dir: str, output_dir: str = None,
                               fill_value: int = 255,
                               file_pattern: str = "*.json") -> Dict[str, Any]:
    """
    批量将 x-anylabeling JSON 转换为 mask PNG
    
    Args:
        json_dir: JSON 文件目录
        output_dir: 输出目录（可选，默认覆盖 JSON 目录）
        fill_value: 填充值
        file_pattern: 文件匹配模式
    
    Returns:
        转换结果统计
    """
    from glob import glob
    
    if not os.path.exists(json_dir):
        raise FileNotFoundError(f"JSON 目录不存在: {json_dir}")
    
    if output_dir is None:
        output_dir = json_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有 JSON 文件
    json_files = glob(os.path.join(json_dir, file_pattern))
    json_files = sorted(json_files, key=natural_sort_key)
    
    results = {
        'total': len(json_files),
        'success': 0,
        'failed': 0,
        'details': []
    }
    
    for json_path in json_files:
        try:
            json_name = os.path.basename(json_path)
            json_base = os.path.splitext(json_name)[0]
            
            # 输出路径
            output_path = os.path.join(output_dir, json_base + '_mask.png')
            
            # 转换
            result = xanylabeling_to_mask(json_path, output_path, fill_value)
            
            results['success'] += 1
            results['details'].append({
                'json': json_name,
                'status': 'success',
                'shapes': result['shapes_count'],
                'mask': os.path.basename(output_path)
            })
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append({
                'json': json_name,
                'status': 'failed',
                'error': str(e)
            })
    
    results['summary'] = f"转换完成: {results['success']}/{results['total']} 成功"
    return results


def merge_xanylabeling_with_mask(json_path: str, mask_path: str = None,
                                  output_path: str = None,
                                  operation: str = 'union') -> Dict[str, Any]:
    """
    合并 x-anylabeling JSON 标注与现有 mask
    
    Args:
        json_path: x-anylabeling JSON 路径
        mask_path: 现有 mask 路径（可选）
        output_path: 输出路径
        operation: 合并操作 ('union', 'intersection', 'json_only', 'mask_only')
    
    Returns:
        合并结果
    """
    # 先转换 JSON 为 mask
    temp_mask = xanylabeling_to_mask(json_path)
    json_mask = cv2.imread(temp_mask['mask_path'], cv2.IMREAD_GRAYSCALE)
    
    if mask_path and os.path.exists(mask_path):
        existing_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 确保尺寸一致
        if json_mask.shape != existing_mask.shape:
            json_mask = cv2.resize(json_mask, (existing_mask.shape[1], existing_mask.shape[0]))
        
        # 执行合并操作
        if operation == 'union':
            merged = cv2.bitwise_or(json_mask, existing_mask)
        elif operation == 'intersection':
            merged = cv2.bitwise_and(json_mask, existing_mask)
        elif operation == 'json_only':
            merged = json_mask
        elif operation == 'mask_only':
            merged = existing_mask
        else:
            merged = cv2.bitwise_or(json_mask, existing_mask)
    else:
        merged = json_mask
    
    # 保存
    if output_path is None:
        json_base = os.path.splitext(os.path.basename(json_path))[0]
        output_path = os.path.join(os.path.dirname(json_path), f"{json_base}_merged.png")
    
    cv2.imwrite(output_path, merged)
    
    return {
        'success': True,
        'json_path': json_path,
        'mask_path': mask_path,
        'output_path': output_path,
        'operation': operation
    }


# CLI 入口
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='x-anylabeling 双向转换工具')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # mask 转 x-anylabeling
    mask_parser = subparsers.add_parser('mask2json', help='将 mask PNG 转为 x-anylabeling JSON')
    mask_parser.add_argument('--mask', '-m', required=True, help='Mask 文件路径')
    mask_parser.add_argument('--image', '-i', required=True, help='对应图片路径')
    mask_parser.add_argument('--output', '-o', help='输出 JSON 路径')
    mask_parser.add_argument('--label', '-l', default='mask', help='标注标签')
    mask_parser.add_argument('--min-area', type=int, default=50, help='最小轮廓面积')
    
    # x-anylabeling 转 mask
    json_parser = subparsers.add_parser('json2mask', help='将 x-anylabeling JSON 转为 mask PNG')
    json_parser.add_argument('--json', '-j', required=True, help='JSON 文件路径')
    json_parser.add_argument('--output', '-o', help='输出 Mask 路径')
    json_parser.add_argument('--fill-value', type=int, default=255, help='填充值')
    
    # 批量转换
    batch_parser = subparsers.add_parser('batch-mask2json', help='批量 mask 转 JSON')
    batch_parser.add_argument('--mask-dir', '-m', required=True, help='Mask 目录')
    batch_parser.add_argument('--image-dir', '-i', help='图片目录')
    batch_parser.add_argument('--output-dir', '-o', help='输出目录')
    batch_parser.add_argument('--label', '-l', default='mask', help='标注标签')
    
    batch_json_parser = subparsers.add_parser('batch-json2mask', help='批量 JSON 转 mask')
    batch_json_parser.add_argument('--json-dir', '-j', required=True, help='JSON 目录')
    batch_json_parser.add_argument('--output-dir', '-o', help='输出目录')
    
    args = parser.parse_args()
    
    if args.command == 'mask2json':
        result = mask_to_xanylabeling(args.mask, args.image, args.output, args.label, args.min_area)
        print(f"✅ 转换成功: {result['json_path']}")
        print(f"   标注数量: {result['shapes_count']}")
        
    elif args.command == 'json2mask':
        result = xanylabeling_to_mask(args.json, args.output, args.fill_value)
        print(f"✅ 转换成功: {result['mask_path']}")
        print(f"   标注数量: {result['shapes_count']}")
        print(f"   标签: {result['labels']}")
        
    elif args.command == 'batch-mask2json':
        result = batch_mask_to_xanylabeling(args.mask_dir, args.image_dir, args.output_dir, args.label)
        print(f"\n{result['summary']}")
        for detail in result['details']:
            if detail['status'] == 'success':
                print(f"  ✅ {detail['mask']}: {detail['shapes']} shapes")
            else:
                print(f"  ❌ {detail['mask']}: {detail['error']}")
                
    elif args.command == 'batch-json2mask':
        result = batch_xanylabeling_to_mask(args.json_dir, args.output_dir)
        print(f"\n{result['summary']}")
        for detail in result['details']:
            if detail['status'] == 'success':
                print(f"  ✅ {detail['json']}: {detail['shapes']} shapes → {detail['mask']}")
            else:
                print(f"  ❌ {detail['json']}: {detail['error']}")
                
    else:
        parser.print_help()
