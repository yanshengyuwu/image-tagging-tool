import sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 将 \" 替换为 "，将 \'\'\' 替换为 '''
content = content.replace('\\"', '"')
content = content.replace("\\'", "'")
content = content.replace("\\n", "\n")
content = content.replace("\\t", "\t")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed quotes in", path)
