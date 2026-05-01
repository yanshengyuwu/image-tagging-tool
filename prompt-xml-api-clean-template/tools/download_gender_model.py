"""下载 deepghs/ml-danbooru-onnx 模型用于性别分类。

ML-Danbooru 是一个多标签分类模型，输出 Danbooru 标签概率。
我们可以从中提取性别相关标签（如 1boy, 1girl, male, female）的概率来判断人物性别。

用法：
    python tools/download_gender_model.py
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "model_cache" / "ml_danbooru_onnx"

# HuggingFace 模型信息
REPO_ID = "deepghs/ml-danbooru-onnx"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"

# 需要下载的文件（文件名从 HuggingFace API 获取）
FILES_TO_DOWNLOAD = [
    # 主模型文件（ONNX，TResnet-D-FLq 最小最快）
    {"path": "model.onnx", "url": f"{BASE_URL}/TResnet-D-FLq_ema_2-40000.onnx"},
    # 标签列表（类别名称）
    {"path": "classes.json", "url": f"{BASE_URL}/classes.json"},
]


def get_file_size(url: str) -> int:
    """获取远程文件大小。"""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return int(resp.headers.get("Content-Length", 0))
    except Exception:
        return 0


def download_file(url: str, dest_path: Path):
    """下载文件并显示进度。"""
    # 确保目录存在
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    file_size = get_file_size(url)
    
    print(f"  下载: {dest_path.name}")
    if file_size > 0:
        print(f"  大小: {file_size / 1024 / 1024:.1f} MB")
    
    urllib.request.urlretrieve(url, dest_path)
    
    actual_size = dest_path.stat().st_size
    print(f"  完成: {actual_size / 1024 / 1024:.1f} MB")


def main():
    print("=" * 60)
    print("ML-Danbooru ONNX 模型下载器")
    print(f"目标目录: {MODEL_DIR}")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for file_info in FILES_TO_DOWNLOAD:
        url = file_info["url"]
        path = MODEL_DIR / file_info["path"]

        if path.exists():
            existing_size = path.stat().st_size
            remote_size = get_file_size(url)
            
            if remote_size > 0 and existing_size == remote_size:
                print(f"  [跳过] 已存在: {file_info['path']} ({existing_size/1024/1024:.1f} MB)")
                continue
            else:
                print(f"  [重新下载] 文件不完整或远程大小未知: {file_info['path']}")
        
        try:
            download_file(url, path)
        except Exception as e:
            print(f"  [错误] 下载失败: {e}")
            # 清理可能损坏的文件
            if path.exists():
                path.unlink()
            sys.exit(1)

    print()
    print("所有文件下载完成!")
    print(f"   目录: {MODEL_DIR}")

    # 列出已下载的文件
    print("\n已下载的文件:")
    for f in MODEL_DIR.iterdir():
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
