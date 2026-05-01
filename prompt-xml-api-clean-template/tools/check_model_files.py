import urllib.request
import json

url = "https://huggingface.co/api/models/deepghs/ml-danbooru-onnx"
resp = urllib.request.urlopen(url, timeout=15)
data = json.loads(resp.read())

for f in data.get("siblings", []):
    name = f["rfilename"]
    size_mb = f.get("size", 0) / 1024 / 1024
    print(f"{name}  ({size_mb:.1f}MB)")
