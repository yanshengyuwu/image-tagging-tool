"""检查更多动漫人物实例分割模型 - 第二轮"""
import urllib.request
import json

# 第二轮: 检查实例分割和更多选项
models_to_check = [
    # 实例分割 (能分割出单个角色)
    "deepghs/anime_ins_seg_yolov8",
    "deepghs/anime_instance_segmentation",
    "Bingsu/anime-segmentation-isnet",
    "skytnt/anime-seg",
    
    # YOLO 系列分割模型
    "Anzhc/Anime-seg",
    "Anzhc/anime-person-seg",
    "anzhc/anime_person_seg",
    
    # 通用实例分割
    "facebook/mask2former-swin-large-coco-instance",
    "facebook/sam2-hiera-large",
    
    # 其他可能有用的
    "deepghs/anime_classification",
    "deepghs/anime_ch_detection",
]

print("=" * 60)
print("第二轮: 动漫人物实例分割模型检查")
print("=" * 60)

for model in models_to_check:
    url = f"https://huggingface.co/api/models/{model}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read().decode())
        
        pipeline_tag = data.get("pipeline_tag", "N/A")
        downloads = data.get("downloads", "N/A")
        tags = data.get("tags", [])
        
        print(f"\n✅ {model}")
        print(f"   Pipeline: {pipeline_tag}")
        print(f"   Downloads: {downloads}")
        print(f"   Tags: {', '.join(tags[:8])}")
        
        # 查看模型文件
        files_url = f"https://huggingface.co/api/models/{model}/tree/main"
        try:
            req2 = urllib.request.Request(files_url, headers={"User-Agent": "Mozilla/5.0"})
            resp2 = urllib.request.urlopen(req2, timeout=10)
            files = json.loads(resp2.read().decode())
            model_files = [f for f in files if isinstance(f, dict) and f.get("path", "").endswith((".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".bin"))]
            if model_files:
                print(f"   Model files:")
                for mf in model_files[:10]:
                    size_mb = mf.get("size", 0) / (1024*1024)
                    print(f"     - {mf['path']} ({size_mb:.1f} MB)")
            
            # 检查子目录
            dirs = [f for f in files if isinstance(f, dict) and f.get("type") == "directory"]
            if dirs:
                print(f"   Directories: {', '.join(d['path'] for d in dirs[:10])}")
                for d in dirs[:5]:
                    sub_url = f"https://huggingface.co/api/models/{model}/tree/main/{d['path']}"
                    try:
                        req3 = urllib.request.Request(sub_url, headers={"User-Agent": "Mozilla/5.0"})
                        resp3 = urllib.request.urlopen(req3, timeout=10)
                        sub_files = json.loads(resp3.read().decode())
                        sub_model_files = [f for f in sub_files if isinstance(f, dict) and f.get("path", "").endswith((".onnx", ".pt", ".pth"))]
                        for smf in sub_model_files[:3]:
                            size_mb = smf.get("size", 0) / (1024*1024)
                            print(f"     - {smf['path']} ({size_mb:.1f} MB)")
                    except:
                        pass
        except Exception as e:
            print(f"   (Could not list files: {e})")
            
    except urllib.error.HTTPError as e:
        print(f"\n❌ {model} - HTTP {e.code}")
    except Exception as e:
        print(f"\n❌ {model} - {e}")


# 检查 imgutils 是否有 segment 相关功能
print("\n" + "=" * 60)
print("检查 imgutils Python 库的分割功能")
print("=" * 60)

try:
    import importlib
    imgutils = importlib.import_module("imgutils")
    print(f"imgutils version: {imgutils.__version__ if hasattr(imgutils, '__version__') else 'unknown'}")
    
    # 检查 segment 模块
    try:
        from imgutils.segment import segment_person
        print("✅ imgutils.segment.segment_person 可用!")
    except ImportError as e:
        print(f"❌ imgutils.segment.segment_person: {e}")
    
    try:
        from imgutils.segment import isnetis
        print("✅ imgutils.segment.isnetis 可用!")
    except ImportError as e:
        print(f"❌ imgutils.segment.isnetis: {e}")
        
    # 列出所有 segment 相关
    try:
        from imgutils import segment
        attrs = [a for a in dir(segment) if not a.startswith('_')]
        print(f"imgutils.segment 中的功能: {attrs}")
    except ImportError as e:
        print(f"❌ imgutils.segment 不存在: {e}")
        
except ImportError:
    print("imgutils 未安装")


# 检查 SAM2 方案的可行性
print("\n" + "=" * 60)
print("已有模型的可行性分析")
print("=" * 60)
print("""
已有模型:
1. ✅ person_detect_v1.3_s.onnx - 人物检测 (bbox)
2. ✅ sam2.1_l.pt / sam2.1_s.pt - SAM2 通用分割
3. ✅ sapiens_0.3b - 人体分割 (针对真人)

可行的方案组合:
方案A: person_detector(bbox) → SAM2(精确分割) → 性别分类 → 遮盖
方案B: skytnt/anime-seg(前景分离) + person_detector(多人区域裁剪)
方案C: 直接用 YOLO-seg 实例分割模型 (如果有动漫专用的)
""")
