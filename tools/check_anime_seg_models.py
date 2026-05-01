"""检查动漫人物全身分割模型的可用性"""
import urllib.request
import json

models_to_check = [
    "skytnt/anime-seg",
    "deepghs/anime_person_segmentation", 
    "deepghs/anime_character_segment",
    "briaai/RMBG-2.0",
    "ZhengPeng7/BiRefNet",
    "ANE/anime-segmentation",
    "deepghs/imgutils-models",
]

print("=" * 60)
print("动漫人物全身分割模型可用性检查")
print("=" * 60)

for model in models_to_check:
    url = f"https://huggingface.co/api/models/{model}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read().decode())
        
        tags = data.get("tags", [])
        pipeline_tag = data.get("pipeline_tag", "N/A")
        downloads = data.get("downloads", "N/A")
        last_modified = data.get("lastModified", "N/A")
        
        print(f"\n✅ {model}")
        print(f"   Pipeline: {pipeline_tag}")
        print(f"   Downloads: {downloads}")
        print(f"   Last Modified: {last_modified[:10] if isinstance(last_modified, str) else 'N/A'}")
        print(f"   Tags: {', '.join(tags[:5])}")
        
        # 查看模型文件
        files_url = f"https://huggingface.co/api/models/{model}/tree/main"
        try:
            req2 = urllib.request.Request(files_url, headers={"User-Agent": "Mozilla/5.0"})
            resp2 = urllib.request.urlopen(req2, timeout=10)
            files = json.loads(resp2.read().decode())
            model_files = [f for f in files if isinstance(f, dict) and f.get("path", "").endswith((".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".bin"))]
            if model_files:
                print(f"   Model files:")
                for mf in model_files[:8]:
                    size_mb = mf.get("size", 0) / (1024*1024)
                    print(f"     - {mf['path']} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"   (Could not list files: {e})")
            
    except urllib.error.HTTPError as e:
        print(f"\n❌ {model} - HTTP {e.code}")
    except Exception as e:
        print(f"\n❌ {model} - {e}")

# 额外检查 deepghs 的 imgutils 库中是否有相关功能
print("\n" + "=" * 60)
print("检查 deepghs/imgutils 相关模型库")
print("=" * 60)

deepghs_models = [
    "deepghs/anime_person_detection",
    "deepghs/anime_face_detection", 
    "deepghs/anime_halfbody_detection",
    "deepghs/anime_head_detection",
    "deepghs/anime_ins_seg",
    "deepghs/anime_character_segment",
]

for model in deepghs_models:
    url = f"https://huggingface.co/api/models/{model}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read().decode())
        downloads = data.get("downloads", "N/A")
        print(f"  ✅ {model} (downloads: {downloads})")
        
        # 查看文件
        files_url = f"https://huggingface.co/api/models/{model}/tree/main"
        try:
            req2 = urllib.request.Request(files_url, headers={"User-Agent": "Mozilla/5.0"})
            resp2 = urllib.request.urlopen(req2, timeout=10)
            files = json.loads(resp2.read().decode())
            model_files = [f for f in files if isinstance(f, dict) and f.get("path", "").endswith((".onnx", ".pt", ".pth", ".ckpt", ".safetensors"))]
            if model_files:
                for mf in model_files[:5]:
                    size_mb = mf.get("size", 0) / (1024*1024)
                    print(f"       - {mf['path']} ({size_mb:.1f} MB)")
        except:
            pass
            
    except urllib.error.HTTPError as e:
        print(f"  ❌ {model} - HTTP {e.code}")
    except Exception as e:
        print(f"  ❌ {model} - {e}")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
对于「遮盖男性角色全身」的需求，需要的模型组合:
1. 人物检测 (Person Detection) → 找到所有人物的位置
2. 人物分割 (Person Segmentation) → 获取人物的精确轮廓 mask
3. 性别分类 (Gender Classification) → 判断哪些是男性
4. 根据性别选择性遮盖
""")
