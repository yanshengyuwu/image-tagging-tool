"""NSFW 模型下载辅助脚本。

ntd11_anime_nsfw_segm_v5 需从 Civitai 手动下载：
  https://civitai.com/models/1313556

EraX-Anti-NSFW-V1.1 可自动从 HuggingFace 下载。

本脚本辅助创建目录结构并下载 EraX 模型。
"""

import sys
from pathlib import Path

def download_erax(model_size: str = "all"):
    """从 HuggingFace 下载 EraX 模型。
    
    Args:
        model_size: 模型大小，可选 'n', 's', 'm', 'l', 'x', 'all'
    """
    from huggingface_hub import snapshot_download

    target = Path("model_cache/nsfw_detectors/erax")
    target.mkdir(parents=True, exist_ok=True)

    # 模型大小映射
    size_patterns = {
        "n": ["*yolo11n*.pt"],
        "s": ["*yolo11s*.pt"],
        "m": ["*yolo11m*.pt"],
        "l": ["*yolo11l*.pt"],
        "x": ["*yolo11x*.pt"],
        "all": ["*.pt"],
    }
    
    pattern = size_patterns.get(model_size.lower(), ["*.pt"])
    
    print(f"正在下载 EraX-Anti-NSFW-V1.1 ({model_size} 变体)...")
    snapshot_download(
        repo_id="erax-ai/EraX-Anti-NSFW-V1.1",
        local_dir=str(target),
        allow_patterns=pattern,
    )
    print(f"EraX 模型已下载到: {target}")
    
    # 显示已下载的模型
    downloaded = list(target.glob("*.pt"))
    if downloaded:
        print("\n已下载的模型:")
        for f in sorted(downloaded):
            size = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size:.1f} MB)")


def check_ntd11():
    """检查 ntd11 模型是否存在。"""
    ntd11_dir = Path("model_cache/nsfw_detectors/ntd11")
    ntd11_dir.mkdir(parents=True, exist_ok=True)

    found = False
    for f in ntd11_dir.glob("*.pt"):
        if "nsfw" in f.name.lower():
            print(f"✓ ntd11 模型已存在: {f}")
            found = True

    if not found:
        print("✗ ntd11 模型未找到！")
        print(f"  请从 https://civitai.com/models/1313556 手动下载")
        print(f"  放入: {ntd11_dir}/")
        return False
    return True


def download_all_mask_models():
    """下载所有 mask pipeline 需要的自动下载模型。"""
    print("=" * 60)
    print("Mask Pipeline 模型下载工具")
    print("=" * 60)

    # 1. Anzhc seg 模型（eyes, hair, breasts）
    print("\n[1/3] Anzhc YOLO-seg 模型...")
    from huggingface_hub import hf_hub_download
    anzhc_dir = Path("model_cache/anzhc_seg")
    anzhc_dir.mkdir(parents=True, exist_ok=True)

    anzhc_files = [
        "Anzhc Eyes -seg-hd.pt",
        "Anzhc HeadHair seg y8m.pt",
        "Anzhc Breasts Seg v1 1024m.pt",
    ]
    for fname in anzhc_files:
        local = anzhc_dir / fname
        if local.exists():
            print(f"  ✓ 已缓存: {fname}")
        else:
            print(f"  下载中: {fname}...")
            hf_hub_download(
                repo_id="Anzhc/Anzhcs_YOLOs",
                filename=fname,
                local_dir=str(anzhc_dir),
            )
            print(f"  ✓ 完成: {fname}")

    # 2. DeepGHS ONNX 模型（face, hand）
    print("\n[2/3] DeepGHS ONNX 模型...")
    deepghs_dir = Path("model_cache/yolov8_anime")
    deepghs_dir.mkdir(parents=True, exist_ok=True)

    deepghs_repos = {
        "deepghs/anime_face_detection": "anime_face_v1.4_s.onnx",
        "deepghs/anime_hand_detection": "hand_detect_v1.0_s.onnx",
    }
    for repo, fname in deepghs_repos.items():
        local = deepghs_dir / fname
        if local.exists():
            print(f"  ✓ 已缓存: {fname}")
        else:
            print(f"  下载中: {repo}/{fname}...")
            hf_hub_download(repo_id=repo, filename=fname, local_dir=str(deepghs_dir))
            print(f"  ✓ 完成: {fname}")

    # 3. NSFW 模型
    print("\n[3/3] NSFW 检测模型...")
    check_ntd11()
    download_erax()

    print("\n" + "=" * 60)
    print("下载完成！")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "erax":
            # 支持指定模型大小: python download_nsfw_models.py erax l
            size = sys.argv[2] if len(sys.argv) > 2 else "all"
            download_erax(size)
        elif cmd == "ntd11":
            check_ntd11()
        elif cmd == "all":
            download_all_mask_models()
        else:
            print(f"未知命令: {cmd}")
            print("用法:")
            print("  python download_nsfw_models.py all              # 下载所有模型")
            print("  python download_nsfw_models.py erax [n|s|m|l|x|all]  # 下载指定 EraX 模型")
            print("  python download_nsfw_models.py ntd11            # 检查 ntd11 模型")
    else:
        download_all_mask_models()
