"""分析 ML-Danbooru 标签中性别相关标签的索引位置。"""

import json
from pathlib import Path

# 加载标签列表
classes_path = Path("model_cache/ml_danbooru_onnx/classes.json")
with open(classes_path, 'r', encoding='utf-8') as f:
    tags = json.load(f)

# 性别相关标签
gender_tags = [
    "1boy", "1girl", 
    "male_focus", "female_focus",
    "multiple_boys", "multiple_girls",
    "futanari", "otoko_no_ko",
    "boy", "girl",
    "shota", "loli",
    "adult_male", "adult_female",
    "ambiguous_gender",
    "crossdressing",
    "genderswap", "genderswap_(ftm)", "genderswap_(mtf)",
    "trap", "reverse_trap",
    "male", "female",
    "bara", "muscular_male", "toned_male",
    "bishounen",
]

print("=" * 60)
print("ML-Danbooru 性别相关标签索引")
print("=" * 60)

found_tags = {}
for tag in gender_tags:
    try:
        idx = tags.index(tag)
        found_tags[tag] = idx
        print(f"  [{idx:5d}] {tag}")
    except ValueError:
        print(f"  [  N/A] {tag} (不存在)")

print()
print("=" * 60)
print("找到的性别标签索引字典（可用于代码）：")
print("=" * 60)
print("GENDER_TAG_INDICES = {")
for tag, idx in sorted(found_tags.items(), key=lambda x: x[1]):
    print(f'    "{tag}": {idx},')
print("}")

# 同时检查一些辅助判断标签
print()
print("=" * 60)
print("其他可能辅助性别判断的标签：")
print("=" * 60)
auxiliary_tags = [
    "breasts", "penis", "pussy", 
    "long_hair", "short_hair",
    "skirt", "pants",
    "thighhighs", "pantyhose",
]
for tag in auxiliary_tags:
    try:
        idx = tags.index(tag)
        print(f"  [{idx:5d}] {tag}")
    except ValueError:
        print(f"  [  N/A] {tag}")
