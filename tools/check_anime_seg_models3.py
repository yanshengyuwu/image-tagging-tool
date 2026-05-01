"""第三轮: 搜索 HuggingFace 上的动漫分割模型"""
import urllib.request
import json

# 用 HuggingFace 搜索 API 搜索相关模型
search_queries = [
    "anime segmentation",
    "anime person segmentation", 
    "anime instance segmentation",
    "anime character segment",
    "anime foreground",
]

print("=" * 60)
print("HuggingFace 搜索: 动漫人物分割模型")
print("=" * 60)

seen = set()
for query in search_queries:
    url = f"https://huggingface.co/api/models?search={query.replace(' ', '+')}&limit=10&sort=downloads&direction=-1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        results = json.loads(resp.read().decode())
        
        print(f"\n--- 搜索: '{query}' (找到 {len(results)} 个) ---")
        for r in results:
            model_id = r.get("modelId", r.get("id", ""))
            if model_id in seen:
                continue
            seen.add(model_id)
            
            pipeline = r.get("pipeline_tag", "N/A")
            downloads = r.get("downloads", 0)
            tags = r.get("tags", [])
            
            # 只显示可能相关的
            relevant_tags = [t for t in tags if any(k in t.lower() for k in ["seg", "anime", "person", "character", "mask", "foreground"])]
            
            print(f"  {model_id}")
            print(f"    Pipeline: {pipeline} | Downloads: {downloads}")
            if relevant_tags:
                print(f"    Relevant tags: {', '.join(relevant_tags[:5])}")
                
    except Exception as e:
        print(f"  Error searching '{query}': {e}")

# 单独搜索 YOLO seg 动漫模型
print("\n" + "=" * 60)
print("搜索 YOLO 系列动漫分割模型")
print("=" * 60)

yolo_queries = ["yolo anime seg", "yolov8 anime segmentation", "yolo11 anime seg", "anime yolo segment"]
for query in yolo_queries:
    url = f"https://huggingface.co/api/models?search={query.replace(' ', '+')}&limit=5&sort=downloads&direction=-1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        results = json.loads(resp.read().decode())
        
        print(f"\n--- 搜索: '{query}' ---")
        for r in results:
            model_id = r.get("modelId", r.get("id", ""))
            if model_id in seen:
                continue
            seen.add(model_id)
            downloads = r.get("downloads", 0)
            pipeline = r.get("pipeline_tag", "N/A")
            print(f"  {model_id} (downloads: {downloads}, pipeline: {pipeline})")
    except Exception as e:
        print(f"  Error: {e}")

# 检查 Anzhc 的所有模型（这个作者有很多动漫分割模型）
print("\n" + "=" * 60)
print("检查 Anzhc 作者的所有模型")  
print("=" * 60)

url = "https://huggingface.co/api/models?author=Anzhc&limit=50&sort=downloads&direction=-1"
try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    results = json.loads(resp.read().decode())
    
    print(f"找到 {len(results)} 个 Anzhc 的模型:")
    for r in results:
        model_id = r.get("modelId", r.get("id", ""))
        downloads = r.get("downloads", 0)
        tags = r.get("tags", [])
        # 过滤出可能与 seg 相关的
        is_seg = any("seg" in t.lower() for t in tags) or "seg" in model_id.lower()
        marker = "🔵" if is_seg else "  "
        print(f"  {marker} {model_id} (downloads: {downloads})")
except Exception as e:
    print(f"  Error: {e}")

# 检查 deepghs 的所有模型
print("\n" + "=" * 60)
print("检查 deepghs 作者的相关模型")
print("=" * 60)

url = "https://huggingface.co/api/models?author=deepghs&limit=50&sort=downloads&direction=-1"
try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    results = json.loads(resp.read().decode())
    
    print(f"找到 {len(results)} 个 deepghs 的模型:")
    for r in results:
        model_id = r.get("modelId", r.get("id", ""))
        downloads = r.get("downloads", 0)
        # 只显示可能相关的
        relevant = any(k in model_id.lower() for k in ["seg", "person", "character", "detect", "gender", "body"])
        if relevant:
            print(f"  🔵 {model_id} (downloads: {downloads})")
except Exception as e:
    print(f"  Error: {e}")
