# prompt-xml-api clean template

这是从 `A:\工具箱\prompt-xml-api` 复制出来的纯净项目范本。

生成时间：2026-05-01 15:20:49

## 已保留

- Flask 后端源码
- 前端模板与静态脚本
- `manga_censor/` 核心模块源码
- `tools/` 中的项目辅助源码
- `requirements.txt`
- `start.bat`
- `.gitignore`
- `danbooru_tags_full.csv` 等基础数据文件

## 已排除

- `model_cache/`：已下载模型缓存
- `profiles/`：个人配置方案
- `newbie-tags/`：独立 Eagle 插件目录
- `config.json`：当前用户 API / 路径配置
- `mask_config.yaml`：当前用户遮罩配置
- `yolov8n.pt`：模型权重
- Python / 测试 / 编辑器缓存
- 常见模型权重文件：`*.pt`, `*.pt2`, `*.onnx`, `*.pth`, `*.safetensors`, `*.bin`, `*.ckpt`, `*.engine`
- 日志和临时文件

## 使用方式

```bat
cd /d A:\工具箱\prompt-xml-api-clean-template
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

首次运行时请按项目需要重新创建配置、配置 API、下载模型。
