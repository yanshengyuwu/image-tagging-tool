# Image Tagging Tool

[English README](README.md) | 中文文档

Image Tagging Tool 是一个基于 Flask 的本地 Web 图片打标工具，同时提供提示词 / XML 处理、训练数据检查、Mask 生成与编辑，以及 x-anylabeling 标注格式转换等辅助能力。

本仓库是 Image Tagging Tool 的可复用发布包。运行缓存、个人配置、模型权重、用户配置方案等文件已被有意排除，便于安全复用、迁移或重新部署。

---

## 目录

- [功能概览](#功能概览)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [安装方式](#安装方式)
- [快速启动](#快速启动)
- [配置说明](#配置说明)
- [主要功能用法](#主要功能用法)
- [模型与缓存文件](#模型与缓存文件)
- [常见工作流](#常见工作流)
- [常见问题排查](#常见问题排查)
- [安全与隐私提醒](#安全与隐私提醒)

---

## 功能概览

### 提示词与 TXT 处理

- 通过 OpenAI 兼容接口或 Anthropic 兼容接口批量处理 `.txt` 文件。
- 将普通提示词转换为结构化 XML 或指定格式。
- 可选择将同名图片随 TXT 一起发送给 AI。
- 可选择查询 Danbooru 标签并作为参考信息发送给 AI。
- 检查 TXT 文件是否完整，例如是否以指定代码块开头和结尾。
- 检查单段文本或整个文件夹中是否包含指定关键词。
- 检测 AI 输出中常见的道歉、拒绝、政策限制类内容。
- 批量删除关键词前面或后面的内容。
- 将混合的 tag / caption 文本统一为固定格式。
- 批量插入固定标签。
- 将混合标注拆分为 tag 数据集与 caption 数据集。

### 图片打标

- 对文件夹图片或手动选择的图片进行批量打标。
- 为图片生成同名 `.txt` 标签文件。
- 支持 Danbooru 元数据查询。
- 支持固定标签插入。
- 支持 AI 二次整理标签，可发送纯文本或图片 + 文本。
- 支持 CL Tagger / PixAI Tagger / Camie Tagger 反推打标。
- 支持标签清单扫描，构建“标签 → 图片”的反向索引，并可显示中文翻译。

### 训练数据工具

- 对训练数据集进行完整质检。
- 检查相似图片。
- 将问题文件移动到指定子目录。
- 批量重命名图片及其同名 TXT 文件。

### Mask 生成与编辑

- 独立 Mask 生成 Pipeline。
- 普通 Mask 模式，可选择不同检测器与部位。
- 性别感知 Mask 模式。
- 自定义本地模型 Mask 生成。
- SAM2 点提示和矩形框提示细化接口。
- 手动 Mask 编辑器，支持多图层：
  - 自动层 auto
  - 手动添加层 manual
  - 反向扣除层 inverse
  - 最终合并层 final
- 高级 Mask 编辑器接口。
- 支持批量合并 Mask 和从 JSON 报告重新渲染 Mask。

### x-anylabeling 转换

- 将 PNG Mask 批量导出为 x-anylabeling JSON 标注。
- 将 x-anylabeling JSON 批量导入为 PNG Mask。
- 支持单个 Mask / JSON 文件转换。
- 支持将 x-anylabeling JSON 与已有 Mask 合并。
- 支持预览 JSON 标注对应的 Mask。

---

## 项目结构

```text
.
├── app.py                         # Flask 主程序，包含 Web 页面与 API 路由
├── start.bat                      # Windows 启动脚本
├── requirements.txt               # Python 依赖清单
├── CLEAN_TEMPLATE_README.md       # 历史模板复制说明
├── danbooru_tags_full.csv         # Danbooru 标签翻译 / 参考数据
├── image_tagger.py                # 图片打标封装
├── training_checker.py            # 训练数据检查工具
├── cl_tagger_engine.py            # CL Tagger 引擎
├── pixai_tagger_engine.py         # PixAI Tagger 引擎
├── camie_tagger_engine.py         # Camie Tagger 引擎
├── xanylabeling_converter.py      # x-anylabeling 导入导出转换器
├── handler_wrapper.py             # 辅助封装模块
├── download_nsfw_models.py        # 模型下载辅助脚本
├── manga_censor/                  # Mask 生成与检测器核心模块
│   ├── pipeline.py
│   ├── mask_editor.py
│   ├── utils.py
│   └── detectors/
├── static/                        # 前端 JavaScript 文件
├── templates/                     # Flask HTML 模板
└── tools/                         # 诊断、测试、模型检查、辅助脚本
```

---

## 环境要求

推荐环境：

- 操作系统：Windows 10 / Windows 11
- Python：Python 3.x
- GPU：推荐 NVIDIA 显卡，用于模型推理
- CUDA：当前依赖注释目标为 CUDA 12.8
- 浏览器：任意现代浏览器
- 网络：API 调用与首次下载模型时需要联网

项目作为本地 Web 应用运行，默认启动地址为：

```text
http://localhost:5000
```

---

## 安装方式

### 1. 进入项目目录

打开命令行并进入项目目录，例如：

```bat
cd /d path\to\Image-Tagging-Tool
```

### 2. 创建虚拟环境

```bat
python -m venv venv
```

### 3. 激活虚拟环境

```bat
venv\Scripts\activate
```

### 4. 安装 CUDA 12.8 版本 PyTorch

`requirements.txt` 中的注释目标为 CUDA 12.8。建议先从 PyTorch 官方 CUDA 源安装：

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

如果你需要严格使用 `requirements.txt` 中固定的版本，请确认 PyTorch 官方源中存在对应构建。

### 5. 安装其他依赖

```bat
pip install -r requirements.txt
```

> 注意：深度学习相关依赖体积较大，安装可能需要较长时间。

---

## 快速启动

### 方式一：手动启动

```bat
venv\Scripts\activate
python app.py
```

然后在浏览器中打开：

```text
http://localhost:5000
```

### 方式二：使用启动脚本

双击：

```text
start.bat
```

或在命令行中运行：

```bat
start.bat
```

该脚本会自动执行以下步骤：

1. 切换到项目目录。
2. 检查是否存在 `venv\Scripts\activate.bat`。
3. 激活虚拟环境。
4. 检查 Flask 是否已安装。
5. 如有需要，安装 PyTorch 和其他依赖。
6. 启动 `app.py`。

---

## 配置说明

运行时配置保存在本地，不包含在可复用项目发布包中。

### 主配置文件

```text
config.json
```

该文件通常在 Web 页面中保存配置后生成，可能包含：

- API URL
- API Key
- 模型名称
- 提示词
- 输入 / 输出文件夹路径
- Danbooru 账号与 API Key
- 图片打标选项

### 配置方案目录

```text
profiles/
```

用于保存多个配置预设。该目录不包含在可复用项目发布包中。

### Mask 配置文件

```text
mask_config.yaml
```

用于保存 Mask 生成、性别感知遮罩等配置。该文件可能由应用自动生成或更新。

### 模型缓存目录

```text
model_cache/
```

下载的模型文件和缓存建议存放在本地缓存目录中。该目录被可复用项目发布包排除。

---

## 主要功能用法

### 1. API 测试

在 Web 页面中填写 API 信息后，可以先使用 API 测试功能确认接口可用。

通常需要填写：

- API URL
- API Key
- 模型名称
- 可选测试短语
- 可选请求超时时间

后端会尝试自动补全常见接口后缀，例如：

- `/v1/chat/completions`
- `/v1/messages`
- `/chat/completions`

多个路由中兼容 OpenAI 风格和 Anthropic 风格的消息格式。

---

### 2. 批量 TXT 转换

当你有一个包含多个 `.txt` 文件的文件夹，并希望用 AI 批量改写、格式化或转换内容时，可以使用该功能。

常用输入：

- 包含 `.txt` 文件的输入文件夹
- 输出文件夹或覆盖模式
- API URL
- API Key
- 模型名称
- 系统提示词
- 请求超时时间
- 可选并发处理
- 可选同名图片发送
- 可选 Danbooru 标签查询

注意事项：

- 开启覆盖模式后，原始 TXT 文件会被直接重写。
- 开启同名图片发送后，程序会查找与 TXT 同名的图片。
- 支持的图片扩展名包括 `.jpg`、`.jpeg`、`.png`、`.webp`、`.bmp`、`.gif`。
- 并发处理可以提高速度，但可能触发 API 限流。

---

### 3. TXT 完整性检查

该工具会扫描文件夹中的 TXT 文件，并检查其格式是否符合预期。

可检查内容包括：

- 是否以指定代码块开头。
- 是否以指定代码块结尾。
- 是否包含指定关键词。

适合在批量 AI 转换后检查输出质量。

---

### 4. 道歉 / 拒绝内容检测

该功能用于检测 AI 输出中常见的道歉、拒绝或政策限制类文本。

会检查类似内容：

- `I cannot`
- `I'm sorry`
- `cannot provide`
- 中文的“抱歉 / 无法 / 不能”等表达
- policy / guidelines 等政策限制词汇

检测到的文件可用于人工复查或重新处理。

---

### 5. 按关键词裁剪文本

关键词删除工具支持批量删除：

- 关键词之后的内容
- 关键词之前的内容
- 是否包含关键词本身
- 是否大小写敏感

适合清理生成的 caption、提示词或异常输出。

---

### 6. 格式统一

格式统一工具会尝试将混合文本整理为：

```text
tag1, tag2, tag3

caption text here
```

适合处理同时包含逗号分隔标签和自然语言描述的训练数据集。

---

### 7. 图片打标

图片打标功能会处理图片并生成同名 `.txt` 标签文件。

支持输入：

- 图片文件夹
- 手动选择上传的图片

输出：

- 每张图片对应一个 TXT 标签文件

可选功能：

- Danbooru 元数据查询
- 固定标签插入
- AI 二次整理
- 将图片发送给 AI
- 并发处理

---

### 8. CL / PixAI / Camie 反推打标

反推打标功能支持多个引擎：

- CL Tagger
- PixAI Tagger
- Camie Tagger

常用参数：

- General 标签阈值
- Character 标签阈值
- 是否包含 rating 标签
- 是否包含 quality 标签
- 输出文件夹
- 是否覆盖原文件

模型可能会在首次使用时初始化或下载。

---

### 9. 标签清单

标签清单功能会扫描数据集中的 TXT 文件，并构建反向索引：

```text
tag -> related images
```

同时可读取 `danbooru_tags_full.csv` 显示中文翻译。

该功能适合用来查找包含指定标签的图片。

---

### 10. 训练数据质检

训练数据检查器可以检查数据集文件夹并报告潜在问题。

可用于：

- 数据集一致性检查
- 相似图片检测
- 查找问题文件
- 将问题文件移动到单独子文件夹

---

### 11. Caption / Tag 数据集拆分

该工具可以将混合标注拆分为：

- tag 数据集
- caption 数据集

也可以根据选项将原目录中的 TXT 改写为仅 tag 内容。

---

### 12. 批量生成 Mask

Mask Pipeline 支持对图片批量生成 Mask。

常用输入：

- 图片文件夹
- 输出文件夹
- 启用的身体 / 部位检测器
- 置信度覆盖设置
- 是否反相
- 是否合并为单个 Mask
- 是否保存各部位独立 Mask

生成文件会保存到指定输出目录。

---

### 13. 性别感知遮罩模式

性别感知模式会先检测人物，并根据性别应用不同遮罩策略。

支持配置：

- male 策略
- female 策略
- 自定义部位
- 全身策略
- 置信度覆盖
- 多人物处理方式
- 文字检测选项

配置可以保存到 `mask_config.yaml`。

---

### 14. 手动 Mask 编辑

手动 Mask 系统使用多图层结构：

- `auto`：自动生成的 Mask
- `manual`：用户手动添加区域
- `inverse`：用户手动扣除区域
- `final`：最终合并结果

标准合并逻辑：

```text
final = auto + manual - inverse
```

反向模式下：

```text
final = auto - manual + inverse
```

---

### 15. SAM2 提示细化

项目提供 SAM2 相关接口用于 Mask 细化：

- 点提示
- 矩形框提示
- 矩形框 + 点混合提示
- 可用模型列表查询

这些接口主要供编辑器前端调用。

---

### 16. 自定义本地模型 Mask 生成

你可以加载本地自定义模型并用于 Mask 生成。

常用参数：

- 模型路径
- 图片文件夹
- 输出文件夹
- 置信度阈值
- 目标类别列表
- 是否反相输出

模型文件不包含在可复用项目发布包中，需要用户自行准备。

---

### 17. x-anylabeling 导入导出

转换器支持：

- 批量 Mask 导出为 x-anylabeling JSON
- 批量 x-anylabeling JSON 导入为 Mask
- 单个 Mask 导出
- 单个 JSON 导入
- JSON 与已有 Mask 合并
- JSON 标注预览

典型用途：

- 将生成的 Mask 转为多边形标注。
- 将手动标注转换回栅格 Mask。
- 将人工修正的标注与自动生成 Mask 合并。

---

## 模型与缓存文件

可复用项目发布包有意排除了大型文件和用户私有文件，包括：

- `model_cache/`
- `profiles/`
- `config.json`
- `mask_config.yaml`
- `yolov8n.pt`
- 常见模型权重文件：
  - `.pt`
  - `.pt2`
  - `.onnx`
  - `.pth`
  - `.safetensors`
  - `.bin`
  - `.ckpt`
  - `.engine`

部分功能会在首次运行时下载或创建模型 / 缓存文件。

如果某个功能提示模型缺失，请检查：

1. 该模型是否应由程序自动下载。
2. 当前网络是否能访问模型托管服务。
3. 本地模型路径是否配置正确。
4. 模型文件是否已从这个可复用项目发布包中排除。

---

## 常见工作流

### 工作流 1：启动 Web 应用

```bat
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python app.py
```

打开浏览器访问：

```text
http://localhost:5000
```

---

### 工作流 2：使用 AI 转换 TXT 提示词

1. 启动 Web 应用。
2. 填写 API URL、API Key 和模型名称。
3. 点击 API 测试按钮。
4. 选择包含 TXT 文件的文件夹。
5. 输入转换提示词。
6. 选择覆盖模式或输出文件夹。
7. 执行转换。
8. 使用 TXT 完整性检查验证输出。

---

### 工作流 3：生成图片标签

1. 启动 Web 应用。
2. 打开图片打标区域。
3. 选择图片文件夹或上传图片。
4. 配置输出文件夹。
5. 可选配置 Danbooru API。
6. 可选启用 AI 二次整理。
7. 执行打标。
8. 检查生成的 TXT 文件。

---

### 工作流 4：准备训练数据集

1. 生成或导入图片 TXT 标注。
2. 统一提示词格式。
3. 按需插入固定标签。
4. 执行数据集质检。
5. 移动问题文件。
6. 按需拆分 tag / caption 数据集。
7. 按需批量重命名文件。

---

### 工作流 5：生成并编辑 Mask

1. 选择图片文件夹。
2. 初始化 Mask Pipeline。
3. 配置启用部位和置信度。
4. 执行批量 Mask 生成。
5. 打开手动或高级 Mask 编辑器。
6. 调整 manual 与 inverse 图层。
7. 合并最终 Mask。
8. 按需导出为 x-anylabeling JSON。

---

## 常见问题排查

### 找不到 `venv`

先创建虚拟环境：

```bat
python -m venv venv
```

然后重新运行 `start.bat`。

---

### Flask 未安装

激活虚拟环境并安装依赖：

```bat
venv\Scripts\activate
pip install -r requirements.txt
```

---

### PyTorch 安装失败

建议单独从官方 CUDA 源安装 PyTorch：

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

如果你的 CUDA 或 GPU 环境不同，请安装与系统匹配的 PyTorch 版本。

---

### 端口 5000 被占用

可能已有其他程序占用了 Flask 默认端口。

可以修改 `app.py` 最后一行附近的端口：

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

将 `5000` 改为其他端口，例如 `7860`。

---

### API 测试失败

请检查：

- API URL 是否正确
- API Key 是否正确
- 模型名称是否正确
- 接口是否为 OpenAI 兼容或 Anthropic 兼容
- 网络是否可用
- 请求超时时间是否过短
- 服务商是否限流
- 所选模型是否在服务商后台启用

---

### 模型初始化失败

请检查：

- 所需模型文件是否存在。
- 模型缓存是否已下载。
- CUDA / PyTorch / ONNX Runtime GPU 是否安装正确。
- 模型路径中是否包含不兼容字符。
- GPU 显存是否足够。

---

### 图片无法读取

请检查：

- 文件扩展名是否受支持。
- 文件路径是否正确。
- 图片文件是否损坏。
- 路径是否包含异常字符。
- OpenCV 或 Pillow 是否能打开该文件。

---

### 并发处理出现大量错误

可能原因：

- API 限流
- 服务商并发限制
- GPU 显存不足
- 线程数过多
- 网络超时

可以尝试降低并发数量。

---

## 安全与隐私提醒

本项目主要设计为本地使用。

不要提交或公开分享以下内容：

- `config.json`
- API Key
- 个人文件夹路径
- `profiles/`
- 包含私有路径的 `mask_config.yaml`
- 无权分发的模型权重
- 包含隐私信息的运行日志

项目中的 `.gitignore` 已配置排除许多生成文件、缓存文件和模型权重文件。

---

## 许可证

当前发布包未包含许可证文件。如需公开发布或重新分发 Image Tagging Tool，请先补充合适的 License。
