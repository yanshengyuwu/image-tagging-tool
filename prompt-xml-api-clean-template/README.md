# Image Tagging Tool

English | [中文文档](README.zh-CN.md)

Image Tagging Tool is a local Flask-based Web toolkit for prompt/XML processing, image tagging, training dataset inspection, mask generation/editing, and x-anylabeling conversion.

This repository is a reusable distribution package of Image Tagging Tool. Runtime caches, personal configuration files, model weights, and user profiles are intentionally excluded so the project can be reused safely.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Environment Requirements](#environment-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Main Usage Guide](#main-usage-guide)
- [Models and Cache Files](#models-and-cache-files)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)

---

## Features

### Prompt and TXT Processing

- Batch convert `.txt` prompt files through an OpenAI-compatible or Anthropic-compatible API.
- Convert plain text prompts into structured XML-like outputs.
- Optionally send same-name peer images together with text prompts.
- Optionally fetch Danbooru tags as extra reference information.
- Check whether TXT files are complete, for example whether they start and end with expected code fences.
- Check single text content or a full folder for specific keywords.
- Detect common apology/refusal outputs from AI models.
- Delete text before or after a keyword in batch.
- Normalize mixed tag/caption files into a consistent format.
- Insert fixed tags into batch TXT files.
- Split mixed tag/caption datasets into separate tag and caption datasets.

### Image Tagging

- Batch image tagging from a folder or selected image files.
- Generate TXT tag files for images.
- Use Danbooru metadata when available.
- Apply fixed tags.
- Optional AI post-processing with text-only or image+text prompts.
- CL Tagger / PixAI Tagger / Camie Tagger reverse tagging support.
- Tag inventory: scan TXT files and build a tag-to-image index with Chinese translations.

### Training Dataset Utilities

- Full training dataset quality check.
- Similar image checking.
- Move problematic training files into a subfolder.
- Batch rename images and same-name TXT files.

### Mask Generation and Editing

- Independent mask generation pipeline.
- Normal mask mode with selectable detectors.
- Gender-aware mask mode.
- Custom local model mask generation.
- SAM2 point prompt and bounding-box prompt refinement APIs.
- Manual mask editor with multiple mask layers:
  - auto layer
  - manual layer
  - inverse layer
  - final merged layer
- Advanced mask editor APIs.
- Batch mask merge and re-render from JSON reports.

### x-anylabeling Conversion

- Export PNG masks to x-anylabeling JSON annotations.
- Import x-anylabeling JSON annotations to PNG masks.
- Convert a single mask or JSON file.
- Merge x-anylabeling JSON annotations with an existing mask.
- Preview annotation JSON as a mask.

---

## Project Structure

```text
.
├── app.py                         # Main Flask application and API routes
├── start.bat                      # Windows startup script
├── requirements.txt               # Python dependency list
├── CLEAN_TEMPLATE_README.md       # Historical template-copy summary
├── danbooru_tags_full.csv         # Danbooru tag translation/reference data
├── image_tagger.py                # Image tagging wrapper
├── training_checker.py            # Training dataset inspection utilities
├── cl_tagger_engine.py            # CL Tagger engine
├── pixai_tagger_engine.py         # PixAI Tagger engine
├── camie_tagger_engine.py         # Camie Tagger engine
├── xanylabeling_converter.py      # x-anylabeling import/export converter
├── handler_wrapper.py             # Helper wrapper module
├── download_nsfw_models.py        # Model download helper
├── manga_censor/                  # Mask generation and detector modules
│   ├── pipeline.py
│   ├── mask_editor.py
│   ├── utils.py
│   └── detectors/
├── static/                        # Frontend JavaScript files
├── templates/                     # Flask HTML templates
└── tools/                         # Diagnostics, tests, model checks, helper scripts
```

---

## Environment Requirements

Recommended environment:

- OS: Windows 10/11
- Python: Python 3.x
- GPU: NVIDIA GPU recommended for model inference
- CUDA: The included dependency comments target CUDA 12.8
- Browser: Any modern browser
- Network access: Required for API calls and first-time model downloads

The project is designed as a local Web application. By default, it starts a Flask server at:

```text
http://localhost:5000
```

---

## Installation

### 1. Clone or copy the project

Open a terminal in the project directory.

Example path:

```bat
cd /d path\to\Image-Tagging-Tool
```

### 2. Create a virtual environment

```bat
python -m venv venv
```

### 3. Activate the virtual environment

```bat
venv\Scripts\activate
```

### 4. Install PyTorch for CUDA 12.8

The dependency file comments target CUDA 12.8. Install PyTorch from the official PyTorch CUDA index:

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If you need the exact pinned versions from `requirements.txt`, use the PyTorch CUDA index that provides those builds.

### 5. Install other dependencies

```bat
pip install -r requirements.txt
```

> Note: Deep learning dependencies are large. Installation can take a long time.

---

## Quick Start

### Option A: Start manually

```bat
venv\Scripts\activate
python app.py
```

Then open:

```text
http://localhost:5000
```

### Option B: Use the startup script

Double-click:

```text
start.bat
```

Or run it from Command Prompt:

```bat
start.bat
```

The script will:

1. Switch to the project directory.
2. Check whether `venv\Scripts\activate.bat` exists.
3. Activate the virtual environment.
4. Check whether Flask is installed.
5. Install PyTorch and other dependencies if needed.
6. Start `app.py`.

---

## Configuration

Runtime configuration is stored locally and is not included in the reusable project package.

### Main configuration file

```text
config.json
```

This file is generated after saving settings from the Web UI. It may contain:

- API URL
- API key
- model name
- prompt
- input/output folder paths
- Danbooru credentials
- image tagging options

### Profiles

```text
profiles/
```

Profiles are saved configuration presets. This folder is excluded from the reusable project package.

### Mask configuration

```text
mask_config.yaml
```

Used by mask generation and gender-aware mask settings. This file may be generated or updated by the application.

### Model cache

```text
model_cache/
```

Downloaded model files and caches should be stored locally. They are intentionally excluded from the reusable project package.

---

## Main Usage Guide

### 1. API Test

Use the API test area in the Web UI to verify your model endpoint.

You usually need:

- API URL
- API Key
- Model name
- Optional custom test phrase
- Optional request timeout

The backend can automatically complete common endpoint suffixes, such as:

- `/v1/chat/completions`
- `/v1/messages`
- `/chat/completions`

The app supports both OpenAI-style and Anthropic-style message formats in several routes.

---

### 2. Batch TXT Conversion

Use this when you have a folder of `.txt` files and want to process them with an AI model.

Typical input:

- Input folder containing `.txt` files
- Output folder or overwrite mode
- API URL
- API Key
- Model name
- System prompt
- Request timeout
- Optional parallel processing
- Optional peer image sending
- Optional Danbooru tag fetching

Important notes:

- When overwrite mode is enabled, the original TXT files are rewritten.
- When peer image sending is enabled, the app searches for same-name images.
- Supported image extensions include `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, and `.gif`.
- Parallel processing can improve speed but may hit API rate limits.

---

### 3. TXT Integrity Check

The TXT check tool scans a folder and checks whether TXT files match the expected format.

It can check:

- Whether the content starts with a specific code fence.
- Whether the content ends with a closing code fence.
- Whether specific keywords appear in the file.

This is useful after batch AI conversion.

---

### 4. Apology / Refusal Detection

The app can detect common apology or refusal phrases in generated TXT files.

It checks for patterns such as:

- "I cannot"
- "I'm sorry"
- "cannot provide"
- Chinese apology/refusal phrases
- policy or guideline related refusal wording

Detected files can be reviewed or moved for reprocessing.

---

### 5. Keyword-Based Text Trimming

The keyword deletion tool can batch remove content:

- after a keyword
- before a keyword
- with or without the keyword itself
- case-sensitive or case-insensitive

Use this for cleaning generated captions or prompt files.

---

### 6. Format Normalization

The normalization tool attempts to convert mixed prompt text into a consistent layout:

```text
tag1, tag2, tag3

caption text here
```

This is useful for preparing training datasets that contain both comma-separated tags and natural-language captions.

---

### 7. Image Tagging

The image tagging tools can process images and generate same-name `.txt` tag files.

Supported input:

- Image folder
- Uploaded selected images

Output:

- A TXT file for each image

Optional features:

- Danbooru metadata lookup
- Fixed tag insertion
- AI post-processing
- Image-to-AI sending
- Parallel processing

---

### 8. CL / PixAI / Camie Reverse Tagging

The reverse tagging routes support different tagger engines:

- CL Tagger
- PixAI Tagger
- Camie Tagger

Common parameters:

- General tag threshold
- Character tag threshold
- Include rating tags
- Include quality tags
- Output folder
- Overwrite mode

The model may be downloaded or initialized on first use.

---

### 9. Tag Inventory

The tag inventory feature scans TXT files in a dataset and builds a reverse index:

```text
tag -> related images
```

It can also use `danbooru_tags_full.csv` to show Chinese tag translations.

This helps identify which images contain specific tags.

---

### 10. Training Dataset Check

The training checker can inspect a dataset folder and report potential issues.

It can be used for:

- Dataset consistency checking
- Similar image detection
- Finding problematic files
- Moving problematic files into a separate subfolder

---

### 11. Caption / Tag Dataset Splitting

The app can split mixed annotations into separate datasets:

- tag dataset
- caption dataset

It can also rewrite the original TXT files as tag-only files depending on the selected option.

---

### 12. Mask Batch Generation

The mask pipeline supports batch mask generation for images.

Basic inputs:

- Image folder
- Output folder
- Enabled body/part detectors
- Confidence overrides
- Invert option
- Merge-to-single-mask option
- Save-individual-masks option

Generated files are saved to the selected output directory.

---

### 13. Gender-Aware Mask Mode

Gender-aware mode detects persons and applies different strategies depending on gender.

Supported options include:

- male strategy
- female strategy
- custom selected parts
- full-body strategy
- confidence overrides
- multi-person handling
- text detection options

The configuration can be saved to `mask_config.yaml`.

---

### 14. Manual Mask Editing

The manual mask system uses multiple layers:

- `auto`: automatically generated mask
- `manual`: user-added mask
- `inverse`: user-subtracted mask
- `final`: merged result

The final mask is calculated from these layers.

Standard merge logic:

```text
final = auto + manual - inverse
```

In inverted mode:

```text
final = auto - manual + inverse
```

---

### 15. SAM2 Prompt Refinement

SAM2 APIs are available for mask refinement:

- Point prompt
- Bounding-box prompt
- Bounding-box + point hybrid prompt
- Model list query

These APIs are mainly used by the editor frontend.

---

### 16. Custom Local Model Mask Generation

You can load a custom local model file and use it for mask generation.

Typical parameters:

- Model path
- Image folder
- Output folder
- Confidence threshold
- Target class list
- Invert output option

Model files are not included in the reusable project package.

---

### 17. x-anylabeling Import and Export

The converter supports:

- Batch mask to x-anylabeling JSON
- Batch x-anylabeling JSON to mask
- Single mask export
- Single JSON import
- JSON and existing mask merge
- JSON annotation preview

Typical use cases:

- Convert generated masks into polygon annotations.
- Convert manual annotations back into raster masks.
- Merge annotation edits with generated masks.

---

## Models and Cache Files

The reusable project package intentionally excludes large or user-specific files, including:

- `model_cache/`
- `profiles/`
- `config.json`
- `mask_config.yaml`
- `yolov8n.pt`
- common model weight files:
  - `.pt`
  - `.pt2`
  - `.onnx`
  - `.pth`
  - `.safetensors`
  - `.bin`
  - `.ckpt`
  - `.engine`

Some features will download or create model/cache files on first use.

If a feature reports that a model is missing, check:

1. Whether the model is expected to be downloaded automatically.
2. Whether the network can access the model hosting service.
3. Whether the required local model path is configured correctly.
4. Whether the file was excluded from this reusable project package.

---

## Common Workflows

### Workflow 1: Start the Web App

```bat
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python app.py
```

Open:

```text
http://localhost:5000
```

---

### Workflow 2: Convert TXT Prompts with AI

1. Start the Web app.
2. Fill in API URL, API Key, and model.
3. Click the API test button.
4. Select the folder containing TXT files.
5. Enter the conversion prompt.
6. Choose overwrite mode or output folder.
7. Run conversion.
8. Use TXT integrity check to verify the output.

---

### Workflow 3: Generate Image Tags

1. Start the Web app.
2. Open the image tagging section.
3. Select an image folder or upload images.
4. Configure output folder.
5. Optionally configure Danbooru API.
6. Optionally enable AI post-processing.
7. Run tagging.
8. Review generated TXT files.

---

### Workflow 4: Prepare Training Dataset

1. Generate or import image TXT annotations.
2. Normalize prompt format.
3. Insert fixed tags if needed.
4. Run dataset quality check.
5. Move problematic files.
6. Split tag/caption datasets if needed.
7. Batch rename files if needed.

---

### Workflow 5: Generate and Edit Masks

1. Select an image folder.
2. Initialize the mask pipeline.
3. Configure enabled parts and confidence values.
4. Run batch mask generation.
5. Open manual or advanced mask editor.
6. Adjust manual and inverse layers.
7. Merge final masks.
8. Export to x-anylabeling JSON if needed.

---

## Troubleshooting

### `venv` not found

Create a virtual environment first:

```bat
python -m venv venv
```

Then run `start.bat` again.

---

### Flask is not installed

Activate the virtual environment and install dependencies:

```bat
venv\Scripts\activate
pip install -r requirements.txt
```

---

### PyTorch installation fails

Install PyTorch separately from the official CUDA wheel index:

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If your CUDA or GPU environment is different, install the PyTorch build that matches your system.

---

### Port 5000 is already in use

Another application may already be using Flask's default port.

You can modify the last line in `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

Change `5000` to another port, for example `7860`.

---

### API test fails

Check:

- API URL
- API Key
- Model name
- Whether the endpoint is OpenAI-compatible or Anthropic-compatible
- Network availability
- Request timeout
- Provider rate limits
- Whether the selected model is enabled by the provider

---

### Model initialization fails

Check:

- Whether the required model file exists.
- Whether the model cache is downloaded.
- Whether CUDA / PyTorch / ONNX Runtime GPU are installed correctly.
- Whether the model path contains unsupported characters.
- Whether the GPU has enough VRAM.

---

### Image files cannot be read

Check:

- File extension
- File path
- Whether the file is corrupted
- Whether the path contains unusual characters
- Whether OpenCV or Pillow can open the file

---

### Parallel processing returns many errors

Possible causes:

- API rate limit
- Provider concurrency limit
- Insufficient GPU memory
- Too many worker threads
- Network timeout

Try lowering the parallel count.

---

## Security Notes

This project is designed for local use.

Do not commit or share:

- `config.json`
- API keys
- personal folder paths
- `profiles/`
- `mask_config.yaml` if it contains private paths
- model weights that you are not allowed to redistribute
- generated logs containing private data

The `.gitignore` file is configured to exclude many generated, cache, and model files.

---

## License

No license file is included in this package. Add a license before publishing or redistributing Image Tagging Tool.
