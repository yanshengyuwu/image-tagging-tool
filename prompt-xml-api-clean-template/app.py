# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false
# pyright: reportMissingTypeArgument=false, reportPrivateUsage=false, reportUnusedFunction=false
# pyright: reportUnusedImport=false, reportUnusedVariable=false, reportUnreachable=false
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportGeneralTypeIssues=false
# pyright: reportOptionalOperand=false, reportOperatorIssue=false, reportCallIssue=false, reportPossiblyUnboundVariable=false

from flask import Flask, request, render_template, jsonify, send_file
import os
import base64
import mimetypes
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import webbrowser
import threading
import tempfile
import shutil
import importlib
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from image_tagger import ImageTagger
from training_checker import TrainingDataChecker
from cl_tagger_engine import CLTaggerEngine
from pixai_tagger_engine import PixAITaggerEngine
from camie_tagger_engine import CamieTaggerEngine

# Mask / Detection imports
from manga_censor.detectors.sam2_refiner import SAM2Refiner
from manga_censor.detectors.bbox_sam2 import BboxSam2Detector
from manga_censor.detectors.anzhc_seg import AnzhcSegDetector
from manga_censor.detectors.nsfw_seg import NsfwSegDetector
from manga_censor.detectors.gender_aware_pipeline import GenderAwarePipeline
from manga_censor.detectors.custom_model import CustomModelDetector

import logging

# 配置日志输出 — 让 manga_censor 模块的诊断日志可见
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# manga_censor 子模块使用 INFO 级别
logging.getLogger("manga_censor").setLevel(logging.INFO)

app = Flask(__name__)

# 配置全局 requests session 支持连接池和重试
def create_session_with_retries():
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# 线程本地存储，每个线程有自己的 session
_thread_local = threading.local()
CONFIG_FILE = 'config.json'
PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profiles')
STOP_FLAGS = {
    'convert': False,
    'tag_images': False,
    'cl_tag': False,
    'body_mask': False
}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}


def _is_truthy(val):
    return str(val).lower() in ('1', 'true', 'yes', 'on', 'y', 'checked')


def natural_sort_key(filename):
    """自然排序：将数字按数值大小排序，而不是字母顺序"""
    import re
    # 将文件名分解为文本和数字的列表
    parts = []
    for part in re.split(r'(\d+)', filename):
        if part.isdigit():
            parts.append((0, int(part)))  # 数字：(0, 数值)
        else:
            parts.append((1, part.lower()))  # 文本：(1, 小写文本)
    return parts


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def ensure_full_api_endpoint(api_url, model=None):
    """根据模型名称自动补全常见聊天端点。"""
    if not api_url:
        return api_url

    raw_url = api_url
    url = api_url.rstrip('/')
    lower = url.lower()

    if any(seg in lower for seg in ['/chat/completions', '/messages', '/completions']):
        return raw_url

    model_name = (model or '').lower()
    if lower.endswith('/v1'):
        return url + ('/messages' if 'claude' in model_name else '/chat/completions')
    if lower.endswith('/v4') or '/api/paas/v4' in lower or 'bigmodel' in lower:
        # 智谱/BigModel v4 基础地址，默认补全到 chat/completions
        return url + '/chat/completions'

    return raw_url


def get_default_request_timeout(model=None, scene='general'):
    """根据模型特征和使用场景返回默认超时时间（秒）。"""
    model_name = (model or '').lower()
    is_thinking_model = 'reasoner' in model_name or 'thinking' in model_name

    if scene == 'test_api':
        return 120 if is_thinking_model else 60
    if scene == 'convert':
        return 180 if is_thinking_model else 120

    return 120 if is_thinking_model else 120


def parse_request_timeout(raw_value, model=None, scene='general'):
    """解析用户传入的超时时间，非法时回退到默认值。"""
    default_timeout = get_default_request_timeout(model, scene)

    if raw_value is None:
        return default_timeout

    raw_text = str(raw_value).strip()
    if not raw_text:
        return default_timeout

    try:
        timeout = int(float(raw_text))
    except (TypeError, ValueError):
        return default_timeout

    if timeout < 10:
        return 10
    if timeout > 3600:
        return 3600
    return timeout


def find_peer_image(folder, base_name):
    """查找与给定基名同名的图片文件，按已知图片扩展名匹配。"""
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(folder, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def reset_stop_flag(name):
    if name in STOP_FLAGS:
        STOP_FLAGS[name] = False


def request_stop(name):
    if name in STOP_FLAGS:
        STOP_FLAGS[name] = True


def is_stopped(name):
    return STOP_FLAGS.get(name, False)


def extract_text_from_response(response_data, raw_text=""):
    """提取模型回复文本，兼容 OpenAI/Claude 风格。"""
    if not isinstance(response_data, dict):
        return raw_text or ''

    if isinstance(response_data.get('choices'), list) and response_data['choices']:
        choice0 = response_data['choices'][0] or {}
        message = choice0.get('message') or {}
        content = message.get('content')

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get('type') == 'text' and isinstance(part.get('text'), str):
                    parts.append(part['text'])
                elif isinstance(part.get('text'), str):
                    parts.append(part['text'])
            if parts:
                return '\n'.join(parts)

        if isinstance(choice0.get('content'), str):
            return choice0['content']
        if isinstance(choice0.get('text'), str):
            return choice0['text']

    if isinstance(response_data.get('content'), list):
        parts = []
        for part in response_data['content']:
            if not isinstance(part, dict):
                continue
            if part.get('type') == 'text' and isinstance(part.get('text'), str):
                parts.append(part['text'])
        if parts:
            return '\n'.join(parts)

    if isinstance(response_data.get('text'), str):
        return response_data['text']
    if isinstance(response_data.get('message'), dict):
        msg_content = response_data['message'].get('content')
        if isinstance(msg_content, str):
            return msg_content

    return raw_text or json.dumps(response_data, ensure_ascii=False)


def _build_candidate_endpoints(raw_url, model):
    """根据原始URL和模型名构造一组候选端点供测试。"""
    if not raw_url:
        return []

    url = raw_url.strip()
    url_no_trailing = url.rstrip('/')
    lower = url_no_trailing.lower()
    model_name = (model or '').lower()

    candidates = [url]

    if any(seg in lower for seg in ['/chat/completions', '/messages', '/completions']):
        if url != url_no_trailing:
            candidates.append(url_no_trailing)
        return list(dict.fromkeys(candidates))

    idx = lower.find('/v1')
    base = url_no_trailing[: idx + 3] if idx != -1 else url_no_trailing

    if 'claude' in model_name:
        candidates.extend([base + '/messages', base + '/chat/completions'])
    else:
        candidates.extend([base + '/chat/completions', base + '/messages'])

    return list(dict.fromkeys(candidates))


@app.route('/test_api', methods=['POST'])
def test_api():
    api_url_raw = request.form.get('api_url', '')
    api_key = request.form.get('api_key', '')
    model = request.form.get('model') or 'gpt-3.5-turbo'
    test_phrase = request.form.get('test_phrase', '').strip()

    print('\n=== API测试开始 ===')
    print(f'API URL (raw): {api_url_raw}')
    print(f'Model: {model}')
    print(f"API Key: {api_key[:10]}..." if len(api_key) > 10 else 'API Key: [too short]')

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # 使用自定义测试短语，如果为空则使用默认
    test_message = test_phrase if test_phrase else 'Hello, this is a test message.'
    
    payload = {
        'model': model,
        'messages': [
            {'role': 'user', 'content': test_message}
        ]
    }

    timeout = parse_request_timeout(request.form.get('request_timeout'), model, 'test_api')
    print(f'测试短语: {test_message}')
    print(f'Request Payload: {json.dumps(payload, ensure_ascii=False)}')
    print(f'Timeout: {timeout}s')

    candidates = _build_candidate_endpoints(api_url_raw, model)
    print('候选端点列表：')
    for i, c in enumerate(candidates, 1):
        print(f'  {i}. {c}')

    last_error = None
    logs = []

    for idx, url in enumerate(candidates, 1):
        try:
            print(f'\n--> 尝试端点 {idx}/{len(candidates)}: {url}')
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            snippet = r.text[:300]
            logs.append(f'[{idx}] {url} -> {r.status_code}: {snippet}')
            print(f'Status: {r.status_code}')
            print(f'Body: {snippet}...')

            if r.status_code == 200:
                response_data = r.json()
                content = extract_text_from_response(response_data, r.text)
                print(f'Success on {url}! Parsed Content: {content}')
                log_text = '\n'.join(logs)
                return jsonify({
                    'success': True,
                    'response': f'✅ 连接成功！使用端点: {url}\n\n📝 测试短语: {test_message}\n\n🤖 模型响应:\n{content}\n\n调试信息:\n{log_text}'
                })

            if r.status_code == 503 and '无可用渠道' in r.text:
                last_error = f'模型当前在提供方无可用渠道，请更换模型或分组。\n端点: {url}\n返回: {r.text}'
                print(f'503 no channel: {last_error}')
                break

            last_error = f'状态码 {r.status_code}: {r.text}'
        except Exception as e:
            last_error = f'请求异常({url}): {e}'
            logs.append(f'[{idx}] {url} -> EXCEPTION: {e}')
            print(last_error)

    log_text = '\n'.join(logs)
    if last_error is None:
        last_error = '未能连接到任何候选端点，请检查URL和网络。'

    print(f'API测试失败: {last_error}')
    return jsonify({'success': False, 'error': f'{last_error}\n\n调试信息:\n{log_text}'})


@app.route('/convert', methods=['POST'])
def convert():
    folder = request.form.get('folder', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    prompt = request.form.get('prompt', '').strip()
    api_url_raw = request.form.get('api_url', '').strip()
    api_key = request.form.get('api_key', '').strip()
    model = request.form.get('model') or 'gpt-3.5-turbo'
    request_timeout_raw = request.form.get('request_timeout')
    send_peer_image = _is_truthy(request.form.get('convert_send_image', ''))
    fetch_danbooru = _is_truthy(request.form.get('convert_fetch_danbooru', ''))
    overwrite = _is_truthy(request.form.get('convert_overwrite', ''))
    prompt_strict = _is_truthy(request.form.get('convert_prompt_strict', ''))
    
    # 新增并发参数
    parallel_enable = _is_truthy(request.form.get('convert_parallel_enable', '0'))
    try:
        parallel_count = int(request.form.get('convert_parallel_count', '10'))
        if parallel_count < 1:
            parallel_count = 1
        if parallel_count > 50:
            parallel_count = 50
    except Exception:
        parallel_count = 10

    if not folder:
        return jsonify({'error': '缺少输入文件夹路径'}), 400
    if overwrite and not output_folder:
        # 当选择覆盖时，允许不填写输出文件夹，默认写回原文件夹
        output_folder = folder
    if not output_folder:
        return jsonify({'error': '缺少输出文件夹路径'}), 400
    if not api_url_raw or not api_key:
        return jsonify({'error': '缺少 API URL 或 API Key'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'输入文件夹不存在或不是文件夹：{folder}'}), 400

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    api_url = ensure_full_api_endpoint(api_url_raw, model)
    is_anthropic = 'anthropic' in api_url.lower() or 'claude' in (model or '').lower()
    print('\n=== 批量转换开始 ===')
    print(f'Input Folder: {folder}')
    print(f'Output Folder: {output_folder}')
    print(f'API URL: {api_url}')
    print(f'Model: {model}')
    print(f'并发模式: {"启用 (并发数: {})".format(parallel_count) if parallel_enable else "禁用"}')
    print(f'覆盖模式: {"是" if overwrite else "否"}')

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    reset_stop_flag('convert')
    results = {}

    # 请求节流：防止并发请求同时触发API频率限制
    import threading
    _request_lock = threading.Lock()
    _request_interval = 0.5  # 每个请求之间至少间隔0.5秒
    _last_request_time = [0.0]

    def throttled_post(session_post_func, *args, **kwargs):
        """带节流的请求发送，确保并发请求之间有间隔"""
        import time as _time
        with _request_lock:
            now = _time.time()
            elapsed = now - _last_request_time[0]
            if elapsed < _request_interval:
                _time.sleep(_request_interval - elapsed)
            _last_request_time[0] = _time.time()
        return session_post_func(*args, **kwargs)

    # 获取所有txt文件列表
    txt_files = sorted([fname for fname in os.listdir(folder) if fname.lower().endswith('.txt')])
    
    print(f'\n\n{"="*70}')
    print(f'🚀 开始批量处理')
    print(f'{"="*70}')
    print(f'📂 输入文件夹: {folder}')
    print(f'🔄 并发模式: {"启用 (最多{}个并发)".format(parallel_count) if parallel_enable else "禁用"}')
    print(f'💾 覆盖模式: 是')
    print(f'📊 待处理文件: {len(txt_files)}个')
    print(f'{"="*70}')
    
    def process_txt_file(fname):
        """处理单个TXT文件（带同名图片和danbooru标签）"""
        if is_stopped('convert'):
            return fname, '⏹️ 已停止'
        
        file_path = os.path.join(folder, fname)
        base_name = os.path.splitext(fname)[0]
        
        # 查找同名图片
        peer_image_path = None
        image_filename = None
        danbooru_tags = None
        group_name = base_name  # 组的名称，用于日志标识
        
        if send_peer_image:
            peer_image_path = find_peer_image(folder, base_name)
            if peer_image_path:
                image_filename = os.path.basename(peer_image_path)
                
                # 尝试获取danbooru标签（仅当开关启用时）
                if fetch_danbooru:
                    try:
                        # 获取danbooru配置
                        config = load_config()
                        danbooru_username = config.get('danbooru_username', '')
                        danbooru_api_key = config.get('danbooru_api_key', '')
                        
                        if danbooru_api_key:
                            print(f'  🔍 正在查询Danbooru标签...')
                            tagger = ImageTagger(
                                danbooru_api_key=danbooru_api_key,
                                danbooru_username=danbooru_username
                            )
                            md5_hash = tagger.get_image_md5(peer_image_path)
                            danbooru_tags = tagger.search_danbooru_by_md5(md5_hash, image_filename)
                            if danbooru_tags:
                                print(f'  ✅ Danbooru标签获取成功')
                            else:
                                print(f'  ℹ️  该图片不在Danbooru数据库中')
                        else:
                            print(f'  ℹ️  未配置Danbooru API，跳过标签查询')
                    except Exception as e:
                        print(f'  ⚠️ Danbooru标签获取失败: {e}')
                        danbooru_tags = None
        
        # 组合名称用于日志标识
        if image_filename:
            if danbooru_tags:
                group_display = f'{base_name} (txt+图片+danbooru)'
            else:
                group_display = f'{base_name} (txt+图片)'
        else:
            group_display = f'{base_name} (纯文本)'
        
        # 输出处理组信息
        print(f'\n{"="*70}')
        print(f'【处理组】{group_display}')
        print(f'  📄 文本: {fname}')
        if image_filename:
            print(f'  🖼️  图片: {image_filename}')
        if danbooru_tags:
            print(f'  🏷️  Danbooru: {len(danbooru_tags.get("general", []))} general, {len(danbooru_tags.get("character", []))} character')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                txt = f.read()
        except Exception as e:
            print(f'  ❌ 读取失败: {e}')
            return fname, f'❌ 读取失败'

        timeout = parse_request_timeout(request_timeout_raw, model, 'convert')

        base_messages = []
        if prompt:
            base_messages.append({'role': 'system', 'content': prompt})
        if send_peer_image and prompt_strict:
            base_messages.append({'role': 'system', 'content': '如果提供了图片和Danbooru标签，严格以用户文本提示为唯一权威，图片和标签仅作参考，若有冲突以文本为准，不要根据图片或标签增删改提示内容。'})
        
        # 如果有danbooru标签，添加标签信息到系统消息
        if danbooru_tags:
            tag_info_parts = []
            if danbooru_tags.get('general'):
                tag_info_parts.append(f"General标签: {', '.join(danbooru_tags['general'][:50])}")
            if danbooru_tags.get('character'):
                tag_info_parts.append(f"Character标签: {', '.join(danbooru_tags['character'])}")
            if danbooru_tags.get('copyright'):
                tag_info_parts.append(f"Copyright标签: {', '.join(danbooru_tags['copyright'])}")
            if danbooru_tags.get('artist'):
                tag_info_parts.append(f"Artist标签: {', '.join(danbooru_tags['artist'])}")
            
            if tag_info_parts:
                tag_info = "以下是从Danbooru数据库获取的该图片标签信息，可作为参考：\n" + "\n".join(tag_info_parts)
                base_messages.append({'role': 'system', 'content': tag_info})

        image_data_url = None
        anthropic_image_payload = None
        if send_peer_image and peer_image_path:
            try:
                if is_anthropic:
                    media_type, b64_data = encode_image_to_base64_payload(peer_image_path)
                    anthropic_image_payload = {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': media_type,
                            'data': b64_data
                        }
                    }
                else:
                    image_data_url = encode_image_to_data_url(peer_image_path)
                print(f'  ✓ 图片已加载: {image_filename}')
            except Exception as e:
                print(f'  ⚠️ 图片加载失败: {e}')
                image_data_url = None
                anthropic_image_payload = None

        user_message = {'role': 'user', 'content': txt}
        if is_anthropic and anthropic_image_payload:
            user_message = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': txt},
                    anthropic_image_payload
                ]
            }
        elif image_data_url:
            user_message = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': txt},
                    {'type': 'image_url', 'image_url': {'url': image_data_url}}
                ]
            }

        def _post_messages(msgs):
            # Anthropic API 需要特殊处理：system消息必须作为顶层参数
            request_headers = dict(headers)
            if is_anthropic or 'anthropic' in api_url.lower():
                request_headers['anthropic-version'] = '2023-06-01'
                
                # 提取system消息
                system_content = []
                filtered_msgs = []
                for msg in msgs:
                    if msg.get('role') == 'system':
                        system_content.append(msg.get('content', ''))
                    else:
                        filtered_msgs.append(msg)
                
                payload = {
                    'model': model,
                    'messages': filtered_msgs,
                    'max_tokens': 8192
                }

                # 如果有system消息，添加到顶层
                if system_content:
                    payload['system'] = '\n\n'.join(system_content)
            else:
                # 其他API（如OpenAI）保持原样
                payload = {
                    'model': model,
                    'messages': msgs
                }
            
            if not hasattr(_thread_local, 'session'):
                _thread_local.session = create_session_with_retries()
            return throttled_post(_thread_local.session.post, api_url, json=payload, headers=request_headers, timeout=timeout)

        # 添加重试机制：最多重试3次
        max_retries = 3
        retry_count = 0
        last_exception = None
        
        while retry_count < max_retries:
            try:
                if retry_count > 0:
                    print(f'  🔄 第{retry_count}次重试...')
                else:
                    print(f'  ⏳ 正在发送给AI处理...')
                
                r = _post_messages(base_messages + [user_message])

                if r.status_code != 200 and (image_data_url or anthropic_image_payload):
                    if r.status_code in [429, 500, 502, 503, 504]:
                        # 频率限制或服务端错误，不降级为纯文本，交给下面的重试逻辑处理
                        print(f'  ⚠️ 带图请求被限流 (HTTP {r.status_code})，将等待后重试...')
                    else:
                        # 其他错误(如400)说明API不支持图片，降级为纯文本
                        print(f'  ⚠️ 带图请求失败 (HTTP {r.status_code})，错误: {r.text[:200]}')
                        print(f'  ⚠️ 改用纯文本重试...')
                        r = _post_messages(base_messages + [{'role': 'user', 'content': txt}])

                if r.status_code == 200:
                    response_data = r.json()
                    content = extract_text_from_response(response_data, r.text)
                    
                    target_path = os.path.join(folder, fname)
                    with open(target_path, 'w', encoding='utf-8') as out_f:
                        out_f.write(content)
                    
                    source_info = f'{fname}' if not image_filename else f'{fname}+{image_filename}'
                    print(f'  ✅ 覆盖完成')
                    print(f'     目标文件: {fname}')
                    print(f'     来源组: {source_info}')
                    print(f'     数据大小: {len(content)}字符')
                    
                    # 返回更详细的成功信息
                    success_msg = f'✅ 成功 ({len(content)}字符)'
                    if retry_count > 0:
                        success_msg += f' [重试{retry_count}次后成功]'
                    return fname, success_msg
                else:
                    # 对于某些HTTP错误状态码，进行重试
                    should_retry = r.status_code in [429, 500, 502, 503, 504]
                    error_body = r.text[:500] if r.text else '(空响应)'
                    
                    if should_retry and retry_count < max_retries - 1:
                        retry_count += 1
                        print(f'  ⚠️ HTTP {r.status_code} 错误，将重试...')
                        print(f'  📋 错误详情: {error_body}')
                        import time, random
                        # 429用更长的等待时间 + 随机抖动，避免并发线程同时重试
                        if r.status_code == 429:
                            wait_time = retry_count * 5 + random.uniform(1, 5)
                        else:
                            wait_time = retry_count * 2 + random.uniform(0, 1)
                        print(f'  ⏱️  等待{wait_time:.1f}秒后重试...')
                        time.sleep(wait_time)
                        continue  # 继续重试循环
                    else:
                        # 不重试或已达最大重试次数
                        print(f'  ❌ AI返回错误: HTTP {r.status_code}')
                        print(f'  📋 错误详情: {error_body}')
                        try:
                            error_json = r.json()
                            if 'error' in error_json:
                                error_detail = error_json['error']
                                if isinstance(error_detail, dict):
                                    print(f'  🔍 错误类型: {error_detail.get("type", "未知")}')
                                    print(f'  💬 错误消息: {error_detail.get("message", "无")}')
                                else:
                                    print(f'  💬 错误消息: {error_detail}')
                        except:
                            pass
                        
                        error_msg = f'❌ HTTP {r.status_code}'
                        if retry_count > 0:
                            error_msg += f' [重试{retry_count}次后仍失败]'
                        return fname, error_msg
                    
            except Exception as e:
                last_exception = e
                retry_count += 1
                error_msg = str(e)[:100]
                print(f'  ⚠️ 请求异常: {error_msg}')
                
                if retry_count < max_retries:
                    import time
                    wait_time = retry_count * 2  # 递增等待时间：2秒、4秒
                    print(f'  ⏱️  等待{wait_time}秒后重试...')
                    time.sleep(wait_time)
                else:
                    print(f'  ❌ 已达最大重试次数({max_retries}次)')
        
        # 所有重试都失败
        return fname, f'❌ 异常: {str(last_exception)[:50]}'
    
    # 并发或串行处理
    if parallel_enable:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
            # 使用 lambda 固化 fname 值，避免闭包陷阱
            futures = [executor.submit(lambda f=fname: process_txt_file(f)) for fname in txt_files]
            for future in concurrent.futures.as_completed(futures):
                fname, msg = future.result()
                results[fname] = msg
    else:
        # 串行处理
        for fname in txt_files:
            fname, msg = process_txt_file(fname)
            results[fname] = msg

    target_dir = folder if overwrite else output_folder
    print('=' * 70)
    print('\n📊 处理完成统计:')
    print('=' * 70)
    successful = len([v for v in results.values() if isinstance(v, str) and v.startswith('✅')])
    print(f'✅ 成功处理: {successful}个文件')
    print(f'❌ 失败处理: {len(txt_files) - successful}个文件')
    print(f'📁 输出位置: {target_dir}')
    print('=' * 70 + '\n')
    results['_summary'] = f'处理完成 ({successful}/{len(txt_files)}成功)'
    return jsonify(results)


@app.route('/check_txt_folder', methods=['POST'])
def check_txt_folder():
    folder = request.form.get('output_folder') or request.form.get('folder')
    raw_keywords = request.form.get('keywords', '')
    keywords = _parse_keywords(raw_keywords)

    if not folder:
        return jsonify({'error': '未提供文件夹路径'}), 400
    if not os.path.exists(folder):
        return jsonify({'error': f'文件夹不存在：{folder}'}), 400
    if not os.path.isdir(folder):
        return jsonify({'error': f'路径不是文件夹：{folder}'}), 400

    print('\n=== TXT 完整性检查开始 ===')
    print(f'Check Folder: {folder}')

    result = {
        'folder': folder,
        'total_txt': 0,
        'ok': 0,
        'failed': 0,
        'details': {},
        'failed_paths': [],
        'keyword_hits': {},
        'hit_files': []
    }

    for fname in os.listdir(folder):
        if not fname.lower().endswith('.txt'):
            continue

        result['total_txt'] += 1
        file_path = os.path.join(folder, fname)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            msg = f'读取失败: {e}'
            result['failed'] += 1
            result['details'][fname] = msg
            print(f'  ❌ {fname}: {msg}')
            continue

        stripped = content.strip()
        ok = True
        reasons = []
        hits = []

        if keywords:
            lowered = content.lower()
            for kw in keywords:
                if kw.lower() in lowered:
                    hits.append(kw)
            if hits:
                result['keyword_hits'][fname] = hits
                result['hit_files'].append(file_path)

        if not stripped.startswith('```xml'):
            ok = False
            reasons.append('开头不是```xml')

        if not stripped.endswith('```'):
            ok = False
            reasons.append('结尾不是```')

        if ok:
            result['ok'] += 1
            suffix = ''
            if hits:
                suffix = f"，包含关键词: {', '.join(hits)}"
            result['details'][fname] = '✅ 完整（```xml 开头，``` 结尾）' + suffix
            print(f"  ✅ {fname}: OK{suffix}")
        else:
            result['failed'] += 1
            msg = '；'.join(reasons) if reasons else '格式不符合要求'
            if hits:
                msg += f"；包含关键词: {', '.join(hits)}"
            result['details'][fname] = '❌ ' + msg
            result['failed_paths'].append(file_path)
            print(f'  ❌ {fname}: {msg}')

    print(f"=== TXT 完整性检查结束: OK {result['ok']}/{result['total_txt']}，失败 {result['failed']} ===\n")
    return jsonify(result)


def _strip_code_fence(text):
    """提取去除 ```xml ... ``` 代码块后的内容，并返回标记。"""
    normalized = (text or '').strip()
    lines = normalized.splitlines()

    has_open = bool(lines) and lines[0].strip().lower().startswith('```xml')
    has_close = bool(lines) and lines[-1].strip() == '```'

    if has_open and has_close and len(lines) >= 2:
        body = '\n'.join(lines[1:-1]).strip()
    else:
        body = normalized

    return body, has_open, has_close


def _parse_keywords(raw):
    """将逗号/换行分隔的关键词文本拆分为去重列表。"""
    if not raw:
        return []
    normalized = raw.replace('\r', '\n').replace('\n', ',')
    tokens = [t.strip() for t in normalized.split(',')]
    # 保持原始大小写，但用lower做匹配
    seen_lower = set()
    keywords = []
    for tok in tokens:
        if not tok:
            continue
        low = tok.lower()
        if low in seen_lower:
            continue
        seen_lower.add(low)
        keywords.append(tok)
    return keywords


def detect_apology_content(text):
    """
    检测文本是否包含道歉或拒绝内容。
    返回: (is_apology: bool, confidence: str, matched_patterns: list, details: str)
    """
    if not text or not text.strip():
        return False, '低', [], '内容为空'
    
    content_lower = text.lower()
    
    # 道歉和拒绝的关键模式（按严重程度分级）
    high_confidence_patterns = [
        # 英文直接拒绝
        'i cannot', 'i can\'t', 'i apologize', 'i\'m sorry',
        'cannot provide', 'cannot assist', 'cannot help',
        'unable to', 'not able to', 'i decline',
        # 中文直接拒绝
        '无法提供', '不能提供', '无法协助', '不能协助',
        '抱歉', '对不起', '很抱歉', '十分抱歉',
        '我无法', '我不能', '不便提供',
    ]
    
    medium_confidence_patterns = [
        # 英文委婉拒绝
        'inappropriate', 'offensive', 'explicit', 'nsfw',
        'against policy', 'against guidelines', 'not appropriate',
        'uncomfortable', 'cannot comply',
        # 中文委婉拒绝
        '不合适', '不适当', '不恰当', '违反', '不符合',
        '敏感内容', '不便', '无法满足',
    ]
    
    low_confidence_patterns = [
        # 可能的解释性拒绝
        'policy', 'guidelines', 'terms', 'ethical',
        '政策', '准则', '规定', '伦理',
    ]
    
    matched_patterns = []
    confidence = '低'
    
    # 检查高置信度模式
    for pattern in high_confidence_patterns:
        if pattern in content_lower:
            matched_patterns.append(pattern)
            confidence = '高'
    
    # 如果没有高置信度，检查中等置信度
    if confidence != '高':
        for pattern in medium_confidence_patterns:
            if pattern in content_lower:
                matched_patterns.append(pattern)
                confidence = '中'
    
    # 如果还没有匹配，检查低置信度
    if confidence == '低':
        for pattern in low_confidence_patterns:
            if pattern in content_lower:
                matched_patterns.append(pattern)
    
    # 额外检查：如果内容很短且包含XML标签，可能是正常内容
    stripped = text.strip()
    has_xml_tags = '<' in stripped and '>' in stripped
    is_very_short = len(stripped) < 200
    
    is_apology = len(matched_patterns) > 0
    
    # 生成详细说明
    if is_apology:
        if confidence == '高':
            details = f'检测到明确的拒绝/道歉内容（匹配{len(matched_patterns)}个高风险模式）'
        elif confidence == '中':
            details = f'检测到可能的拒绝/道歉内容（匹配{len(matched_patterns)}个中风险模式）'
        else:
            details = f'检测到疑似拒绝/道歉内容（匹配{len(matched_patterns)}个低风险模式）'
    else:
        if has_xml_tags:
            details = '内容包含XML标签，似乎是正常的标签数据'
        else:
            details = '未检测到明显的拒绝/道歉模式'
    
    return is_apology, confidence, matched_patterns, details


@app.route('/check_txt_content', methods=['POST'])
def check_txt_content():
    raw_text = request.form.get('text', '')
    raw_keywords = request.form.get('keywords', '')
    normalized = (raw_text or '').replace('\r\n', '\n')

    if not normalized.strip():
        return jsonify({'error': '内容为空，请输入内容后再检查'}), 400

    xml_body, has_open, has_close = _strip_code_fence(normalized)
    keywords = _parse_keywords(raw_keywords)

    xml_ok = True
    xml_error = ''
    xml_root = None

    if xml_body:
        try:
            root = ET.fromstring(xml_body)
            xml_root = root.tag
        except Exception as e:
            xml_ok = False
            xml_error = str(e)
    else:
        xml_ok = False
        xml_error = '内容为空'

    suggestions = []
    if not has_open:
        suggestions.append('缺少 ```xml 开头')
    if not has_close:
        suggestions.append('缺少 ``` 结尾')
    if not xml_ok:
        suggestions.append('XML 解析失败，请检查标签闭合、编码或特殊字符')

    keyword_hits = []
    if keywords:
        lowered = normalized.lower()
        for kw in keywords:
            if kw.lower() in lowered:
                keyword_hits.append(kw)
        if keyword_hits:
            suggestions.append(f'检测到关键词: {", ".join(keyword_hits)}')
        else:
            suggestions.append('未检测到关键词匹配')

    return jsonify({
        'ok': xml_ok and has_open and has_close,
        'code_fence_ok': has_open and has_close,
        'xml_ok': xml_ok,
        'xml_error': xml_error,
        'xml_root': xml_root,
        'length': len(normalized),
        'lines': normalized.count('\n') + 1,
        'keywords': keywords,
        'keyword_hits': keyword_hits,
        'suggestions': suggestions,
        'preview': xml_body[:2000]
    })


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_config', methods=['GET'])
def get_config():
    return jsonify(load_config())


def _build_config_dict(form):
    return {
        'api_url': form.get('api_url'),
        'api_key': form.get('api_key'),
        'model': form.get('model'),
        'custom_model': form.get('custom_model'),
        'prompt': form.get('prompt'),
        'test_phrase': form.get('test_phrase'),
        'request_timeout': form.get('request_timeout'),
        'folder': form.get('folder'),
        'output_folder': form.get('output_folder'),
        'convert_overwrite': _is_truthy(form.get('convert_overwrite')),
        'convert_prompt_strict': _is_truthy(form.get('convert_prompt_strict')),
        'convert_fetch_danbooru': _is_truthy(form.get('convert_fetch_danbooru')),
        'danbooru_username': form.get('danbooru_username'),
        'danbooru_api_key': form.get('danbooru_api_key'),
        'image_folder': form.get('image_folder'),
        'image_output_folder': form.get('image_output_folder'),
        'fixed_tags': form.get('fixed_tags'),
        'image_disable_ai': _is_truthy(form.get('image_disable_ai')),
        'image_send_image': _is_truthy(form.get('image_send_image')),
        'convert_send_image': _is_truthy(form.get('convert_send_image')),
    }


@app.route('/save_config', methods=['POST'])
def save_config_route():
    config = _build_config_dict(request.form)
    save_config(config)
    return jsonify({'success': True})


@app.route('/list_profiles', methods=['GET'])
def list_profiles():
    os.makedirs(PROFILES_DIR, exist_ok=True)
    names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(PROFILES_DIR)
        if f.endswith('.json')
    )
    return jsonify({'profiles': names})


@app.route('/save_profile', methods=['POST'])
def save_profile():
    name = (request.form.get('profile_name') or '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'profile name is required'}), 400
    safe = secure_filename(name)
    if not safe:
        return jsonify({'success': False, 'error': 'invalid profile name'}), 400
    os.makedirs(PROFILES_DIR, exist_ok=True)
    config = _build_config_dict(request.form)
    path = os.path.join(PROFILES_DIR, safe + '.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return jsonify({'success': True, 'name': safe})


@app.route('/load_profile', methods=['GET'])
def load_profile():
    name = (request.args.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'name is required'}), 400
    safe = secure_filename(name)
    path = os.path.join(PROFILES_DIR, safe + '.json')
    if not os.path.isfile(path):
        return jsonify({'error': 'profile not found'}), 404
    with open(path, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))


@app.route('/delete_profile', methods=['POST'])
def delete_profile():
    name = (request.form.get('profile_name') or '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'name is required'}), 400
    safe = secure_filename(name)
    path = os.path.join(PROFILES_DIR, safe + '.json')
    if os.path.isfile(path):
        os.remove(path)
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'profile not found'}), 404


# ===== 标签预览接口 =====

@app.route('/preview_folder', methods=['POST'])
def preview_folder():
    if request.is_json:
        folder = (request.json.get('folder') or '').strip()
    else:
        folder = (request.form.get('folder') or '').strip()
    if not folder:
        return jsonify({'error': 'folder path is empty'}), 400
    if not os.path.exists(folder):
        return jsonify({'error': f'path does not exist: {folder}'}), 400
    if not os.path.isdir(folder):
        return jsonify({'error': f'path is not a directory: {folder}'}), 400
    images = []
    for fname in sorted(os.listdir(folder), key=natural_sort_key):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            base = os.path.splitext(fname)[0]
            txt_path = os.path.join(folder, base + '.txt')
            images.append({'filename': fname, 'has_txt': os.path.isfile(txt_path)})
    tagged = sum(1 for img in images if img['has_txt'])
    return jsonify({'folder': folder, 'images': images, 'total': len(images), 'tagged': tagged})


@app.route('/preview_image', methods=['GET'])
def preview_image():
    path = (request.args.get('path') or '').strip()
    if not path or not os.path.isfile(path):
        return jsonify({'error': 'file not found'}), 404
    ext = os.path.splitext(path)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        return jsonify({'error': 'not an image file'}), 400
    mime = mimetypes.guess_type(path)[0] or 'application/octet-stream'
    return send_file(path, mimetype=mime)


@app.route('/preview_tags', methods=['GET'])
def preview_tags():
    img_path = (request.args.get('path') or '').strip()
    if not img_path:
        return jsonify({'error': 'path is required'}), 400
    base = os.path.splitext(img_path)[0]
    txt_path = base + '.txt'
    if not os.path.isfile(txt_path):
        return jsonify({'content': '', 'exists': False})
    with open(txt_path, 'r', encoding='utf-8') as f:
        return jsonify({'content': f.read(), 'exists': True})


@app.route('/save_tags', methods=['POST'])
def save_tags():
    img_path = (request.form.get('image_path') or '').strip()
    tags = request.form.get('tags', '')
    if not img_path:
        return jsonify({'success': False, 'error': 'image_path is required'}), 400
    base = os.path.splitext(img_path)[0]
    txt_path = base + '.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(tags)
    return jsonify({'success': True})


@app.route('/delete_pair', methods=['POST'])
def delete_pair():
    img_path = (request.form.get('image_path') or '').strip()
    if not img_path:
        return jsonify({'success': False, 'error': 'image_path is required'}), 400
    results = {}
    if os.path.isfile(img_path):
        os.remove(img_path)
        results['image'] = 'deleted'
    else:
        results['image'] = 'not found'
    base = os.path.splitext(img_path)[0]
    txt_path = base + '.txt'
    if os.path.isfile(txt_path):
        os.remove(txt_path)
        results['txt'] = 'deleted'
    else:
        results['txt'] = 'not found'
    return jsonify({'success': True, 'results': results})


@app.route('/tag_images', methods=['POST'])
def tag_images():
    response, status = _tag_images_common(request.form, request.files)
    return response, status


@app.route('/tag_single_image', methods=['POST'])
def tag_single_image():
    response, status = _tag_images_common(request.form, request.files)
    return response, status


@app.route('/tag_selected_images', methods=['POST'])
def tag_selected_images():
    response, status = _tag_images_common(request.form, request.files)
    return response, status


def generate_xml_from_tags(tags_dict):
    xml_lines = ['<tags>']

    if tags_dict['general']:
        xml_lines.append(f"  <general>{', '.join(tags_dict['general'])}</general>")
    if tags_dict['character']:
        xml_lines.append(f"  <character>{', '.join(tags_dict['character'])}</character>")
    if tags_dict['copyright']:
        xml_lines.append(f"  <copyright>{', '.join(tags_dict['copyright'])}</copyright>")
    if tags_dict.get('artist'):
        xml_lines.append(f"  <artist>{', '.join(tags_dict['artist'])}</artist>")

    xml_lines.append('</tags>')
    return '\n'.join(xml_lines)


def parse_fixed_tags(fixed_tags_text):
    normalized = (fixed_tags_text or '').replace('\n', ',')
    tokens = [t.strip() for t in normalized.split(',')]
    return [t for t in tokens if t]


def apply_fixed_tags(tags_dict, fixed_tags_text):
    extra_tags = parse_fixed_tags(fixed_tags_text)
    if not extra_tags:
        return tags_dict

    # Work on a shallow copy to avoid mutating original dict unexpectedly.
    updated = dict(tags_dict)
    general_tags = list(updated.get('general', []))
    existing = set(general_tags)

    for tag in extra_tags:
        if tag not in existing:
            general_tags.append(tag)
            existing.add(tag)

    updated['general'] = general_tags
    return updated


def optimize_image_for_api(image_path, max_size=1024, quality=85, size_threshold_mb=3):
    """
    优化图片以适合API传输：
    - 仅当文件大小超过 size_threshold_mb（默认3MB）时才压缩
    - 限制最大边长（默认1024px）
    - 压缩质量（默认85%）
    - 转换为JPEG格式以减小体积
    
    返回优化后的图片字节数据
    """
    try:
        from PIL import Image
        import io

        original_size = os.path.getsize(image_path)
        size_threshold_bytes = size_threshold_mb * 1024 * 1024

        # 文件未超过阈值，直接返回原始数据
        if original_size <= size_threshold_bytes:
            print(f'  ✓ 图片大小 {original_size/1024:.1f}KB，无需压缩')
            with open(image_path, 'rb') as f:
                data = f.read()
            mime_type, _ = mimetypes.guess_type(image_path)
            return data, mime_type or 'image/jpeg'

        img = Image.open(image_path)
        
        # 转换RGBA到RGB（某些格式不支持透明度）
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 获取原始尺寸
        width, height = img.size
        
        # 如果图片较大，按比例缩小
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f'  📐 图片已缩放: {width}x{height} -> {new_width}x{new_height}')
        
        # 保存为JPEG格式到内存
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        optimized_data = buffer.getvalue()
        
        # 显示压缩效果
        optimized_size = len(optimized_data)
        compression_ratio = (1 - optimized_size / original_size) * 100 if original_size > 0 else 0
        print(f'  💾 图片已优化: {original_size/1024:.1f}KB -> {optimized_size/1024:.1f}KB (压缩{compression_ratio:.1f}%)')
        
        return optimized_data, 'image/jpeg'
        
    except ImportError:
        print('  ⚠️ PIL/Pillow未安装，使用原始图片')
        with open(image_path, 'rb') as f:
            data = f.read()
        mime_type, _ = mimetypes.guess_type(image_path)
        return data, mime_type or 'image/png'
    except Exception as e:
        print(f'  ⚠️ 图片优化失败: {e}，使用原始图片')
        with open(image_path, 'rb') as f:
            data = f.read()
        mime_type, _ = mimetypes.guess_type(image_path)
        return data, mime_type or 'image/png'


def encode_image_to_data_url(image_path, optimize=True):
    """将图片编码为data URL，供支持图片输入的聊天接口使用。"""
    if optimize:
        image_data, mime_type = optimize_image_for_api(image_path)
        encoded = base64.b64encode(image_data).decode('ascii')
    else:
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or 'image/png'
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('ascii')

    return f'data:{mime_type};base64,{encoded}'


def encode_image_to_base64_payload(image_path, optimize=True):
    """读取图片为Anthropic消息需要的base64结构。"""
    if optimize:
        image_data, mime_type = optimize_image_for_api(image_path)
        encoded = base64.b64encode(image_data).decode('ascii')
    else:
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or 'image/png'
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('ascii')
    
    return mime_type, encoded


def prepare_image_sources(folder, uploaded_files):
    sources = []
    temp_dir = None

    if folder:
        if not os.path.exists(folder):
            raise ValueError(f'文件夹不存在：{folder}')
        if not os.path.isdir(folder):
            raise ValueError(f'路径不是文件夹：{folder}')

        # 自然排序：1, 2, 3, 10, 100 而不是 1, 10, 100, 2, 3
        fnames = sorted([fname for fname in os.listdir(folder) if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS], 
                       key=natural_sort_key)
        
        for fname in fnames:
            sources.append((fname, os.path.join(folder, fname), False))

    if uploaded_files:
        temp_dir = tempfile.mkdtemp(prefix='selected_images_')
        for file in uploaded_files:
            filename = secure_filename(file.filename)
            if not filename:
                continue
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in IMAGE_EXTENSIONS:
                sources.append((filename, None, False))
                continue
            save_path = os.path.join(temp_dir, filename)
            file.save(save_path)
            sources.append((filename, save_path, True))

    # 对最终的sources列表进行自然排序
    sources = sorted(sources, key=lambda x: natural_sort_key(x[0]))
    
    return sources, temp_dir


def process_and_save_image(image_path, output_folder, tagger, fixed_tags_text, custom_prompt,
                           disable_ai, send_image_to_ai, api_url, api_key, model, request_timeout_raw=None):
    merged_tags = tagger.process_image(image_path)
    merged_tags = apply_fixed_tags(merged_tags, fixed_tags_text)

    use_ai = (not disable_ai) and api_url and api_key and (custom_prompt or send_image_to_ai)
    if use_ai:
        print('使用AI处理自定义提示词...')
        tag_summary = (
            "请基于下述标签与图片，产出最合理的XML标签，仅返回XML内容。\n\n"
            f"General Tags (部分)：{', '.join(merged_tags['general'][:50])}\n"
            f"Character Tags：{', '.join(merged_tags['character'])}\n"
            f"Copyright Tags：{', '.join(merged_tags['copyright'])}"
        )

        user_content_parts = [{'type': 'text', 'text': tag_summary}]

        if custom_prompt:
            user_content_parts.append({'type': 'text', 'text': f'用户自定义要求：{custom_prompt}'})

        data_url = None
        if send_image_to_ai:
            try:
                data_url = encode_image_to_data_url(image_path)
                user_content_parts.append({'type': 'image_url', 'image_url': {'url': data_url}})
            except Exception as e:
                print(f'⚠️ 图片编码失败，将仅发送标签文本：{e}')
                data_url = None

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        system_prompt = '你是图片标签整理助手，结合图片内容与给定标签，输出精简、有序的XML标签列表。'

        def _post_with_messages(messages):
            payload = {
                'model': model,
                'messages': messages
            }
            # 使用线程本地 session 来支持并发
            if not hasattr(_thread_local, 'session'):
                _thread_local.session = create_session_with_retries()
            timeout = parse_request_timeout(request_timeout_raw, model, 'general')
            return _thread_local.session.post(api_url, json=payload, headers=headers, timeout=timeout)

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content_parts if (send_image_to_ai and data_url) else '\n\n'.join(p['text'] for p in user_content_parts if p.get('text'))}
        ]

        # 添加重试机制：最多重试3次
        max_retries = 3
        retry_count = 0
        last_exception = None
        ai_output = None
        
        while retry_count < max_retries and ai_output is None:
            try:
                if retry_count > 0:
                    print(f'🔄 第{retry_count}次重试AI请求...')
                
                r = _post_with_messages(messages)

                if r.status_code != 200 and send_image_to_ai and data_url:
                    print('⚠️ 图像+文本请求失败，尝试仅文本降级...')
                    fallback_messages = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': '\n\n'.join(p['text'] for p in user_content_parts if p.get('text'))}
                    ]
                    r = _post_with_messages(fallback_messages)

                if r.status_code == 200:
                    response_data = r.json()
                    ai_output = extract_text_from_response(response_data, r.text)
                else:
                    # HTTP错误不重试，直接返回
                    ai_output = f'API错误 {r.status_code}: {r.text[:200]}'
                    
            except Exception as e:
                last_exception = e
                retry_count += 1
                error_msg = str(e)[:100]
                print(f'⚠️ AI请求异常: {error_msg}')
                
                if retry_count < max_retries:
                    import time
                    wait_time = retry_count * 2  # 递增等待时间：2秒、4秒
                    print(f'⏱️  等待{wait_time}秒后重试...')
                    time.sleep(wait_time)
                else:
                    print(f'❌ 已达最大重试次数({max_retries}次)')
        
        # 如果所有重试都失败，使用异常信息
        if ai_output is None:
            ai_output = f'AI处理失败: {last_exception}'
    else:
        ai_output = generate_xml_from_tags(merged_tags)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_filename = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    output_path = os.path.join(output_folder, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ai_output)

    print(f'✅ 保存成功: {output_filename}')

    return {
        'filename': os.path.basename(image_path),
        'output_file': output_path,
        'ai_output': ai_output,
        'message': f'✅ 成功保存到: {output_filename}'
    }


def _tag_images_common(form, files):
    folder = (form.get('image_folder') or '').strip()
    uploaded_files = files.getlist('image_files') if files else []

    output_folder = (form.get('image_output_folder') or '').strip()
    if not output_folder:
        if folder:
            output_folder = folder + '_tagged'
        else:
            output_folder = os.path.join(tempfile.gettempdir(), 'image_tags')

    danbooru_username = form.get('danbooru_username', '')
    danbooru_api_key = form.get('danbooru_api_key', '')
    custom_prompt = form.get('prompt', '')
    fixed_tags_text = form.get('fixed_tags', '')
    disable_ai = _is_truthy(form.get('image_disable_ai', ''))
    send_image_to_ai = _is_truthy(form.get('image_send_image', ''))

    api_url_raw = form.get('api_url', '')
    api_key = form.get('api_key', '')
    model = form.get('model', 'gpt-3.5-turbo')
    request_timeout_raw = form.get('request_timeout')
    api_url = ensure_full_api_endpoint(api_url_raw, model) if api_url_raw else ''

    try:
        sources, temp_dir = prepare_image_sources(folder, uploaded_files)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    if not sources:
        return jsonify({'error': '请提供图片文件夹或直接选择图片文件'}), 400

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reset_stop_flag('tag_images')

    print('\n=== 图片标注开始 ===')
    if folder:
        print(f'Input Folder: {folder}')
    if uploaded_files:
        print(f'Uploaded Files: {len(uploaded_files)}')
    print(f'Output Folder: {output_folder}')

    # 新增并发参数
    parallel_enable = _is_truthy(form.get('parallel_enable', '0'))
    try:
        parallel_count = int(form.get('parallel_count', '10'))
        if parallel_count < 1:
            parallel_count = 1
        if parallel_count > 50:
            parallel_count = 50
    except Exception:
        parallel_count = 10

    try:
        tagger = ImageTagger(
            danbooru_api_key=danbooru_api_key if danbooru_api_key else None,
            danbooru_username=danbooru_username if danbooru_username else None
        )
    except Exception as e:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({'error': f'ImageTagger初始化失败: {e}'}), 500

    import concurrent.futures
    results = {}
    
    print(f'\n\n{"="*70}')
    print(f'🚀 开始图片标注处理')
    print(f'{"="*70}')
    print(f'📂 输入位置: {output_folder}')
    print(f'🔄 并发模式: {"启用 (最多{}个并发)".format(parallel_count) if parallel_enable else "禁用"}')
    print(f'📊 待处理图片: {len(sources)}张')
    print(f'{"="*70}')

    def tag_worker(fname, image_path, is_temp):
        if not image_path:
            return fname, '❌ 不支持的图片格式'
        if is_stopped('tag_images'):
            return fname, '⏹️ 已停止'
        
        print(f'\n{"="*70}')
        print(f'【处理组】{fname}')
        print(f'  📄 图片文件: {fname}')
        print(f'  📁 源位置: {image_path}')
        
        try:
            print(f'  ⏳ 正在标注...')
            outcome = process_and_save_image(
                image_path=image_path,
                output_folder=output_folder,
                tagger=tagger,
                fixed_tags_text=fixed_tags_text,
                custom_prompt=custom_prompt,
                disable_ai=disable_ai,
                send_image_to_ai=send_image_to_ai,
                api_url=api_url,
                api_key=api_key,
                model=model,
                request_timeout_raw=request_timeout_raw
            )
            print(f'  ✅ 标注完成')
            print(f'     输出文件: {os.path.basename(outcome["output_file"])}')
            print(f'     数据大小: {len(outcome["ai_output"])}字符')
            return fname, outcome['message']
        except Exception as e:
            print(f'  ❌ 标注失败: {str(e)[:100]}')
            return fname, f'❌ 处理失败: {e}'

    if parallel_enable:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
            # 使用 lambda 固化参数，避免闭包陷阱
            futures = [executor.submit(lambda f=fname, p=image_path, t=is_temp: tag_worker(f, p, t)) 
                      for fname, image_path, is_temp in sources]
            for future in concurrent.futures.as_completed(futures):
                fname, msg = future.result()
                results[fname] = msg
    else:
        # 串行处理
        for fname, image_path, is_temp in sources:
            fname, msg = tag_worker(fname, image_path, is_temp)
            results[fname] = msg

    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    processed_count = len([k for k in results.keys() if not k.startswith('_')])
    success_count = len([v for v in results.values() if isinstance(v, str) and v.startswith('✅')])
    
    print('=' * 70)
    print('\n📊 处理完成统计:')
    print('=' * 70)
    print(f'✅ 成功标注: {success_count}张图片')
    print(f'❌ 失败标注: {processed_count - success_count}张图片')
    print(f'📁 输出位置: {output_folder}')
    print('=' * 70 + '\n')
    
    results['_summary'] = f'图片标注完成 ({success_count}/{processed_count}成功)'
    return jsonify(results), 200


def open_browser():
    webbrowser.open('http://localhost:5000')


@app.route('/stop_convert', methods=['POST'])
def stop_convert_route():
    request_stop('convert')
    return jsonify({'success': True, 'message': '已请求停止文本转换'})


@app.route('/stop_tagging', methods=['POST'])
def stop_tagging_route():
    request_stop('tag_images')
    return jsonify({'success': True, 'message': '已请求停止图片标注'})


@app.route('/check_apology', methods=['POST'])
def check_apology():
    """批量检测文件夹中的txt文件是否包含道歉/拒绝内容"""
    folder = request.form.get('folder', '').strip()
    
    if not folder:
        return jsonify({'error': '未提供文件夹路径'}), 400
    if not os.path.exists(folder):
        return jsonify({'error': f'文件夹不存在：{folder}'}), 400
    if not os.path.isdir(folder):
        return jsonify({'error': f'路径不是文件夹：{folder}'}), 400
    
    print('\n=== 道歉内容检测开始 ===')
    print(f'检测文件夹: {folder}')
    
    result = {
        'folder': folder,
        'total_txt': 0,
        'apology_count': 0,
        'normal_count': 0,
        'details': {},
        'apology_files': [],
        'high_confidence': [],
        'medium_confidence': [],
        'low_confidence': []
    }
    
    # 使用自然排序
    txt_files = sorted([fname for fname in os.listdir(folder) if fname.lower().endswith('.txt')],
                      key=natural_sort_key)
    
    for fname in txt_files:
        result['total_txt'] += 1
        file_path = os.path.join(folder, fname)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            msg = f'❌ 读取失败: {e}'
            result['details'][fname] = msg
            print(f'  {msg}')
            continue
        
        # 检测道歉内容
        is_apology, confidence, patterns, details = detect_apology_content(content)
        
        if is_apology:
            result['apology_count'] += 1
            result['apology_files'].append(file_path)
            
            # 按置信度分类
            if confidence == '高':
                result['high_confidence'].append(fname)
                icon = '🚨'
            elif confidence == '中':
                result['medium_confidence'].append(fname)
                icon = '⚠️'
            else:
                result['low_confidence'].append(fname)
                icon = 'ℹ️'
            
            # 生成详细信息
            pattern_info = f"（匹配模式: {', '.join(patterns[:3])}{'...' if len(patterns) > 3 else ''}）"
            msg = f'{icon} 道歉内容 [{confidence}置信度] {pattern_info}'
            result['details'][fname] = msg
            print(f'  {fname}: {msg}')
        else:
            result['normal_count'] += 1
            msg = f'✅ 正常内容'
            result['details'][fname] = msg
            print(f'  {fname}: {msg}')
    
    # 生成统计摘要
    summary_lines = [
        f'总文件数: {result["total_txt"]}',
        f'✅ 正常内容: {result["normal_count"]}个',
        f'⚠️ 疑似道歉: {result["apology_count"]}个'
    ]
    
    if result['high_confidence']:
        summary_lines.append(f'  🚨 高置信度: {len(result["high_confidence"])}个')
    if result['medium_confidence']:
        summary_lines.append(f'  ⚠️ 中置信度: {len(result["medium_confidence"])}个')
    if result['low_confidence']:
        summary_lines.append(f'  ℹ️ 低置信度: {len(result["low_confidence"])}个')
    
    result['_summary'] = '\n'.join(summary_lines)
    
    print('=' * 70)
    print('📊 道歉内容检测完成:')
    print('=' * 70)
    for line in summary_lines:
        print(line)
    print('=' * 70 + '\n')
    
    return jsonify(result)


@app.route('/check_single_apology', methods=['POST'])
def check_single_apology():
    """检测单个文本内容是否包含道歉/拒绝内容"""
    text = request.form.get('text', '')
    
    if not text or not text.strip():
        return jsonify({'error': '内容为空，请输入内容后再检查'}), 400
    
    is_apology, confidence, patterns, details = detect_apology_content(text)
    
    result = {
        'is_apology': is_apology,
        'confidence': confidence,
        'matched_patterns': patterns,
        'details': details,
        'text_length': len(text),
        'text_preview': text[:500]
    }
    
    if is_apology:
        result['suggestion'] = f'检测到{confidence}置信度的道歉/拒绝内容，建议检查并重新处理该文件'
    else:
        result['suggestion'] = '内容正常，未检测到明显的道歉/拒绝模式'
    
    return jsonify(result)


@app.route('/delete_by_keyword', methods=['POST'])
def delete_by_keyword():
    """删除txt文件中指定关键词前面或后面的所有内容"""
    folder = request.form.get('folder', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    keyword = request.form.get('keyword', '').strip()
    delete_direction = request.form.get('delete_direction', 'after')  # 'after' 或 'before'
    include_keyword = _is_truthy(request.form.get('include_keyword', ''))
    overwrite = _is_truthy(request.form.get('delete_overwrite', ''))
    case_sensitive = _is_truthy(request.form.get('case_sensitive', ''))
    
    if not folder:
        return jsonify({'error': '缺少输入文件夹路径'}), 400
    if not keyword:
        return jsonify({'error': '请输入要搜索的关键词'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'输入文件夹不存在或不是文件夹：{folder}'}), 400
    
    if overwrite:
        output_folder = folder
    elif not output_folder:
        return jsonify({'error': '请指定输出文件夹路径，或勾选覆盖原文件'}), 400
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print('\n=== 关键词删除开始 ===')
    print(f'输入文件夹: {folder}')
    print(f'输出文件夹: {output_folder}')
    print(f'关键词: {keyword}')
    print(f'删除方向: {"删除关键词之后的内容" if delete_direction == "after" else "删除关键词之前的内容"}')
    print(f'包含关键词: {"是" if include_keyword else "否"}')
    print(f'大小写敏感: {"是" if case_sensitive else "否"}')
    print(f'覆盖模式: {"是" if overwrite else "否"}')
    
    results = {}
    txt_files = sorted([fname for fname in os.listdir(folder) if fname.lower().endswith('.txt')],
                      key=natural_sort_key)
    
    processed = 0
    success = 0
    skipped = 0
    
    for fname in txt_files:
        file_path = os.path.join(folder, fname)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            results[fname] = f'❌ 读取失败: {e}'
            print(f'  ❌ {fname}: 读取失败')
            processed += 1
            continue
        
        # 查找关键词位置
        if case_sensitive:
            keyword_pos = content.find(keyword)
        else:
            keyword_pos = content.lower().find(keyword.lower())
        
        if keyword_pos == -1:
            results[fname] = 'ℹ️ 未找到关键词，跳过'
            print(f'  ℹ️ {fname}: 未找到关键词')
            skipped += 1
            processed += 1
            continue
        
        # 根据方向和是否包含关键词来裁剪内容
        if delete_direction == 'after':
            # 删除关键词之后的内容
            if include_keyword:
                new_content = content[:keyword_pos]
            else:
                new_content = content[:keyword_pos + len(keyword)]
        else:  # before
            # 删除关键词之前的内容
            if include_keyword:
                new_content = content[keyword_pos + len(keyword):]
            else:
                new_content = content[keyword_pos:]
        
        # 保存处理后的内容
        output_path = os.path.join(output_folder, fname)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            original_len = len(content)
            new_len = len(new_content)
            deleted_len = original_len - new_len
            
            results[fname] = f'✅ 成功处理 (原{original_len}字符 → {new_len}字符，删除{deleted_len}字符)'
            print(f'  ✅ {fname}: 成功处理 (删除{deleted_len}字符)')
            success += 1
        except Exception as e:
            results[fname] = f'❌ 保存失败: {e}'
            print(f'  ❌ {fname}: 保存失败')
        
        processed += 1
    
    print('=' * 70)
    print('\n📊 关键词删除完成统计:')
    print('=' * 70)
    print(f'✅ 成功处理: {success}个文件')
    print(f'ℹ️ 跳过（未找到关键词）: {skipped}个文件')
    print(f'❌ 失败处理: {processed - success - skipped}个文件')
    print(f'📁 输出位置: {output_folder}')
    print('=' * 70 + '\n')
    
    results['_summary'] = f'关键词删除完成 ({success}个成功, {skipped}个跳过, {processed - success - skipped}个失败)'
    return jsonify(results)


@app.route('/normalize_format', methods=['POST'])
def normalize_format():
    """统一提示词格式：tags一行 + 空行 + caption一行"""
    folder = request.form.get('folder', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    overwrite = _is_truthy(request.form.get('delete_overwrite', ''))

    if not folder:
        return jsonify({'error': '缺少输入文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'输入文件夹不存在或不是文件夹：{folder}'}), 400

    if overwrite:
        output_folder = folder
    elif not output_folder:
        return jsonify({'error': '请指定输出文件夹路径，或勾选覆盖原文件'}), 400

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print('\n=== 格式统一开始 ===')
    print(f'输入文件夹: {folder}')
    print(f'输出文件夹: {output_folder}')
    print(f'覆盖模式: {"是" if overwrite else "否"}')

    results = {}
    txt_files = sorted([fname for fname in os.listdir(folder) if fname.lower().endswith('.txt')],
                      key=natural_sort_key)

    processed = 0
    success = 0
    skipped = 0

    for fname in txt_files:
        file_path = os.path.join(folder, fname)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            results[fname] = f'❌ 读取失败: {e}'
            print(f'  ❌ {fname}: 读取失败')
            processed += 1
            continue

        # 统一换行符，合并全文为一行以统一处理
        normalized = content.replace('\r\n', '\n').replace('\r', '\n')
        merged = ' '.join(normalized.split())

        tags_line = ''
        desc_line = ''

        # 方法1：查找 "caption:" 前缀
        caption_match = re.search(r'caption\s*:', merged, re.IGNORECASE)
        if caption_match:
            tags_line = merged[:caption_match.start()].strip()
            desc_line = merged[caption_match.start():].strip()
        else:
            # 方法2：按逗号分段，通过词数和大写字母定位 tags→描述 的边界
            # booru tags 特征：全小写、逗号分隔、每段 1~4 个词
            # 自然语言特征：句子以大写字母开头、段落较长
            segments = merged.split(', ')
            tag_segments = []
            found = False

            for i, seg in enumerate(segments):
                words = seg.split()
                if len(words) > 5:
                    # 这一段太长，可能是 tags 和描述粘在一起
                    # 在段内查找第一个大写字母开头的词作为描述起点
                    for j in range(1, len(words)):
                        if words[j] and words[j][0].isupper():
                            # 确认后续内容足够长（是句子而非偶然大写的tag）
                            if len(words) - j > 3:
                                if j > 0:
                                    tag_segments.append(' '.join(words[:j]))
                                desc_text = ' '.join(words[j:])
                                remaining = segments[i + 1:]
                                if remaining:
                                    desc_text += ', ' + ', '.join(remaining)
                                tags_line = ', '.join(tag_segments)
                                desc_line = desc_text
                                found = True
                                break
                    if not found:
                        # 段内未找到明确分界，整段视为描述起点
                        tags_line = ', '.join(tag_segments) if tag_segments else ''
                        desc_line = ', '.join(segments[i:])
                        found = True
                    break
                else:
                    tag_segments.append(seg)

            if not found:
                # 全部为 tags，无描述
                tags_line = ', '.join(tag_segments)
                desc_line = ''

        # 组装最终内容
        if tags_line and desc_line:
            new_content = tags_line + '\n\n' + desc_line
        elif tags_line:
            new_content = tags_line
        else:
            new_content = desc_line

        if new_content == content:
            results[fname] = 'ℹ️ 格式已统一，跳过'
            print(f'  ℹ️ {fname}: 已是统一格式')
            skipped += 1
            processed += 1
            continue

        output_path = os.path.join(output_folder, fname)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            results[fname] = f'✅ 格式已统一'
            print(f'  ✅ {fname}: 格式已统一')
            success += 1
        except Exception as e:
            results[fname] = f'❌ 保存失败: {e}'
            print(f'  ❌ {fname}: 保存失败')

        processed += 1

    print('=' * 70)
    print('\n📊 格式统一完成统计:')
    print('=' * 70)
    print(f'✅ 成功处理: {success}个文件')
    print(f'ℹ️ 跳过（已统一）: {skipped}个文件')
    print(f'❌ 失败处理: {processed - success - skipped}个文件')
    print(f'📁 输出位置: {output_folder}')
    print('=' * 70 + '\n')

    results['_summary'] = f'格式统一完成 ({success}个成功, {skipped}个跳过, {processed - success - skipped}个失败)'
    return jsonify(results)


@app.route('/move_apology_files', methods=['POST'])
def move_apology_files():
    """将道歉文件的同名图片剪切到子文件夹，并删除对应TXT文件"""
    import json as _json
    folder = request.form.get('folder', '').strip()
    subfolder = request.form.get('subfolder', 'again').strip() or 'again'
    files_json = request.form.get('files', '[]')

    try:
        files = _json.loads(files_json)
    except Exception:
        files = []

    if not folder:
        return jsonify({'error': '未提供文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在：{folder}'}), 400
    if not files:
        return jsonify({'error': '没有要处理的文件'}), 400

    again_folder = os.path.join(folder, subfolder)
    os.makedirs(again_folder, exist_ok=True)

    print('\n=== 道歉文件处理开始 ===')
    print(f'源文件夹: {folder}')
    print(f'图片目标文件夹: {again_folder}')
    print(f'待处理TXT数: {len(files)}')

    results = {}
    txt_deleted = 0
    img_moved = 0
    failed = 0

    for fname in files:
        base_name = os.path.splitext(fname)[0]

        # 删除 TXT 文件
        txt_path = os.path.join(folder, fname)
        txt_status = ''
        if os.path.exists(txt_path):
            try:
                os.remove(txt_path)
                txt_status = '🗑️ TXT已删除'
                txt_deleted += 1
                print(f'  🗑️ 删除: {fname}')
            except Exception as e:
                txt_status = f'❌ TXT删除失败: {e}'
                failed += 1
                print(f'  ❌ {fname}: {e}')
        else:
            txt_status = 'ℹ️ TXT不存在（已处理过？）'

        # 剪切同名图片文件
        img_statuses = []
        for ext in IMAGE_EXTENSIONS:
            img_fname = base_name + ext
            img_src = os.path.join(folder, img_fname)
            if os.path.exists(img_src):
                try:
                    shutil.move(img_src, os.path.join(again_folder, img_fname))
                    img_statuses.append(f'✂️ {img_fname}→{subfolder}/')
                    img_moved += 1
                    print(f'  ✂️ 剪切: {img_fname} → {subfolder}/')
                except Exception as e:
                    img_statuses.append(f'❌ {img_fname}移动失败: {e}')
                    failed += 1
                    print(f'  ❌ {img_fname}: {e}')

        if img_statuses:
            results[fname] = txt_status + ' | ' + ', '.join(img_statuses)
        else:
            results[fname] = txt_status + ' | 无同名图片'

    print(f'=== 处理完成: TXT删除{txt_deleted}个，图片移动{img_moved}个，失败{failed}个 ===\n')
    results['_summary'] = (
        f'处理完成\n'
        f'🗑️ TXT已删除：{txt_deleted}个\n'
        f'✂️ 图片已剪切到 {subfolder}/：{img_moved}个\n'
        f'❌ 失败：{failed}个'
    )
    return jsonify(results)


# ===== 训练数据质检接口 =====

@app.route('/check_training_dataset', methods=['POST'])
def check_training_dataset():
    """对训练数据集执行完整质检"""
    folder = request.form.get('folder', '').strip()
    if not folder:
        return jsonify({'error': '未提供文件夹路径'}), 400
    if not os.path.exists(folder):
        return jsonify({'error': f'文件夹不存在：{folder}'}), 400
    if not os.path.isdir(folder):
        return jsonify({'error': f'路径不是文件夹：{folder}'}), 400

    print('\n=== 训练数据质检开始 ===')
    print(f'检测文件夹: {folder}')

    # 相似图检测参数
    try:
        sim_threshold = int(request.form.get('similar_threshold', '10'))
    except (ValueError, TypeError):
        sim_threshold = 10
    try:
        sim_keep_ratio = float(request.form.get('similar_keep_ratio', '0.3'))
    except (ValueError, TypeError):
        sim_keep_ratio = 0.3
    try:
        sim_min_keep = int(request.form.get('similar_min_keep', '2'))
    except (ValueError, TypeError):
        sim_min_keep = 2

    try:
        checker = TrainingDataChecker(folder)
        result = checker.full_check(
            similar_threshold=sim_threshold,
            similar_keep_ratio=sim_keep_ratio,
            similar_min_keep=sim_min_keep,
        )
        print(f'=== 质检完成 ===\n')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'质检失败: {str(e)}'}), 500


@app.route('/move_training_files', methods=['POST'])
def move_training_files():
    """移动训练数据中的问题文件到子文件夹"""
    folder = request.form.get('folder', '').strip()
    subfolder = request.form.get('subfolder', 'problematic').strip() or 'problematic'
    files_json = request.form.get('files', '[]')

    try:
        file_list = json.loads(files_json)
    except Exception:
        file_list = []

    if not folder:
        return jsonify({'error': '未提供文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在：{folder}'}), 400
    if not file_list:
        return jsonify({'error': '没有要处理的文件'}), 400

    print(f'\n=== 移动问题文件 ===')
    print(f'源文件夹: {folder}')
    print(f'目标子文件夹: {subfolder}')
    print(f'待移动文件数: {len(file_list)}')

    try:
        checker = TrainingDataChecker(folder)
        result = checker.move_files(file_list, subfolder)
        print(f'=== 移动完成: {result["moved_count"]} 个文件已移动, {result["failed_count"]} 个失败 ===\n')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'移动失败: {str(e)}'}), 500


@app.route('/insert_fixed_tags', methods=['POST'])
def insert_fixed_tags():
    """批量往txt文件的tags行插入固定标签"""
    folder = request.form.get('folder', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    fixed_tags_raw = request.form.get('fixed_tags', '').strip()
    position = request.form.get('position', 'prepend')
    overwrite = _is_truthy(request.form.get('overwrite', ''))

    if not folder:
        return jsonify({'error': '缺少输入文件夹路径'}), 400
    if not fixed_tags_raw:
        return jsonify({'error': '请输入要插入的固定标签'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'输入文件夹不存在或不是文件夹：{folder}'}), 400

    if overwrite:
        output_folder = folder
    elif not output_folder:
        return jsonify({'error': '请指定输出文件夹路径，或勾选覆盖原文件'}), 400

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fixed_tags = parse_fixed_tags(fixed_tags_raw)
    if not fixed_tags:
        return jsonify({'error': '解析后无有效标签'}), 400

    print('\n=== 批量插入固定标签开始 ===')
    print(f'输入文件夹: {folder}')
    print(f'输出文件夹: {output_folder}')
    print(f'插入位置: {"tags行开头" if position == "prepend" else "tags行末尾"}')
    print(f'固定标签: {", ".join(fixed_tags)}')

    results = {}
    txt_files = sorted(
        [fname for fname in os.listdir(folder) if fname.lower().endswith('.txt')],
        key=natural_sort_key
    )

    processed = 0
    success = 0
    skipped = 0

    def _normalize_tag(t):
        return t.strip().lower().replace(' ', '_')

    for fname in txt_files:
        file_path = os.path.join(folder, fname)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            results[fname] = f'❌ 读取失败: {e}'
            processed += 1
            continue

        normalized = content.replace('\r\n', '\n').replace('\r', '\n')
        parts = normalized.split('\n\n', 1)
        tags_line = parts[0].strip()
        caption_part = parts[1] if len(parts) > 1 else ''

        existing_tags = [t.strip() for t in tags_line.split(',') if t.strip()]
        existing_normalized = {_normalize_tag(t) for t in existing_tags}

        new_tags = [t for t in fixed_tags if _normalize_tag(t) not in existing_normalized]

        if not new_tags:
            results[fname] = 'ℹ️ 所有标签已存在，跳过'
            skipped += 1
            processed += 1
            continue

        if position == 'prepend':
            merged_tags = new_tags + existing_tags
        else:
            merged_tags = existing_tags + new_tags

        new_tags_line = ', '.join(merged_tags)

        if caption_part:
            new_content = new_tags_line + '\n\n' + caption_part
        else:
            new_content = new_tags_line

        output_path = os.path.join(output_folder, fname)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            results[fname] = f'✅ 插入 {len(new_tags)} 个标签 ({", ".join(new_tags)})'
            print(f'  ✅ {fname}: +{len(new_tags)} tags')
            success += 1
        except Exception as e:
            results[fname] = f'❌ 保存失败: {e}'

        processed += 1

    print('=' * 70)
    print(f'📊 批量插入固定标签完成: ✅ {success}个成功, ℹ️ {skipped}个跳过, ❌ {processed - success - skipped}个失败')
    print('=' * 70 + '\n')

    results['_summary'] = f'批量插入完成 ({success}个成功, {skipped}个跳过, {processed - success - skipped}个失败)'
    return jsonify(results)


@app.route('/split_caption_dataset', methods=['POST'])
def split_caption_dataset():
    """拆分混合标注为 tag 数据集与 caption 数据集，并将原目录 txt 改写为仅 tag。"""
    folder = request.form.get('folder', '').strip()
    rewrite_original_txt = _is_truthy(request.form.get('rewrite_original_txt', '1'))

    if not folder:
        return jsonify({'error': '未提供文件夹路径'}), 400
    if not os.path.exists(folder):
        return jsonify({'error': f'文件夹不存在：{folder}'}), 400
    if not os.path.isdir(folder):
        return jsonify({'error': f'路径不是文件夹：{folder}'}), 400

    print('\n=== 混合标注拆分开始 ===')
    print(f'源文件夹: {folder}')
    print(f'改写原目录TXT: {"是" if rewrite_original_txt else "否"}')

    try:
        checker = TrainingDataChecker(folder)
        result = checker.export_split_caption_datasets(
            rewrite_original_txt=rewrite_original_txt
        )
        print(f'标签目录: {result["tag_folder"]}')
        print(f'自然语言目录: {result["caption_folder"]}')
        print(f'=== 拆分完成: {result["summary"]} ===\n')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'拆分失败: {str(e)}'}), 500


# ===== 标签中文翻译字典 =====

_tag_zh_dict = None
_tag_zh_lock = threading.Lock()

DANBOORU_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'danbooru_tags_full.csv')


def _load_tag_zh_dict():
    """加载 danbooru_tags_full.csv 构建 tag→中文翻译 字典"""
    global _tag_zh_dict
    if _tag_zh_dict is not None:
        return _tag_zh_dict
    with _tag_zh_lock:
        if _tag_zh_dict is not None:
            return _tag_zh_dict
        d = {}
        if os.path.exists(DANBOORU_CSV_PATH):
            import csv
            try:
                with open(DANBOORU_CSV_PATH, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        tag = row.get('tag', '').strip()
                        alias = row.get('alias', '').strip()
                        if tag and alias:
                            # alias 可能是 "中文,日文,..." 取第一个
                            first = alias.split(',')[0].strip().strip('"')
                            if first:
                                d[tag] = first
                print(f'标签翻译字典已加载: {len(d)} 条')
            except Exception as e:
                print(f'加载标签翻译字典失败: {e}')
        else:
            print(f'标签翻译CSV不存在: {DANBOORU_CSV_PATH}')
        _tag_zh_dict = d
        return _tag_zh_dict


# ===== cl_tagger 反推打标接口 =====

_cl_tagger_engine = None
_cl_tagger_lock = threading.Lock()
_pixai_tagger_engine = None
_pixai_tagger_lock = threading.Lock()
_camie_tagger_engine = None
_camie_tagger_lock = threading.Lock()
def _get_cl_tagger():
    global _cl_tagger_engine
    if _cl_tagger_engine is None:
        with _cl_tagger_lock:
            if _cl_tagger_engine is None:
                _cl_tagger_engine = CLTaggerEngine()
    return _cl_tagger_engine


def _get_pixai_tagger():
    global _pixai_tagger_engine
    if _pixai_tagger_engine is None:
        with _pixai_tagger_lock:
            if _pixai_tagger_engine is None:
                _pixai_tagger_engine = PixAITaggerEngine()
    return _pixai_tagger_engine


def _get_camie_tagger():
    global _camie_tagger_engine
    if _camie_tagger_engine is None:
        with _camie_tagger_lock:
            if _camie_tagger_engine is None:
                _camie_tagger_engine = CamieTaggerEngine()
    return _camie_tagger_engine


@app.route('/cl_tagger_status', methods=['GET'])
def cl_tagger_status():
    """获取反推模型状态"""
    tagger_model = request.args.get('tagger_model', 'cl_tagger')
    if tagger_model == 'pixai_tagger':
        engine = _get_pixai_tagger()
    elif tagger_model == 'camie_tagger':
        engine = _get_camie_tagger()
    else:
        engine = _get_cl_tagger()
    return jsonify(engine.get_status())


@app.route('/cl_tag_batch', methods=['POST'])
def cl_tag_batch():
    """批量反推打标：对文件夹中的图片生成标签 txt"""
    folder = request.form.get('folder', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    overwrite = _is_truthy(request.form.get('overwrite', ''))
    tagger_model = request.form.get('tagger_model', 'cl_tagger')
    gen_threshold = float(request.form.get('gen_threshold', '0.55'))
    char_threshold = float(request.form.get('char_threshold', '0.6'))
    include_rating = _is_truthy(request.form.get('include_rating', ''))
    include_quality = _is_truthy(request.form.get('include_quality', ''))

    if not folder:
        return jsonify({'error': '缺少图片文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在或不是文件夹：{folder}'}), 400

    if overwrite:
        output_folder = folder
    elif not output_folder:
        output_folder = folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reset_stop_flag('cl_tag')

    # 收集图片
    image_files = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key
    )

    if not image_files:
        return jsonify({'error': '文件夹中没有找到图片文件'}), 400

    if tagger_model == 'pixai_tagger':
        model_display = 'PixAI Tagger v0.9'
    elif tagger_model == 'camie_tagger':
        model_display = 'camie-tagger-v2'
    else:
        model_display = 'cl_tagger'

    print(f'\n=== {model_display} 批量反推打标 ===')
    print(f'文件夹: {folder}')
    print(f'输出: {output_folder}')
    print(f'模型: {model_display}')
    print(f'阈值: general={gen_threshold}, character={char_threshold}')
    print(f'图片数: {len(image_files)}')

    if tagger_model == 'pixai_tagger':
        engine = _get_pixai_tagger()
    elif tagger_model == 'camie_tagger':
        engine = _get_camie_tagger()
    else:
        engine = _get_cl_tagger()

    # 初始化模型（首次会下载）
    try:
        if not engine.initialize():
            return jsonify({'error': f'{model_display} 模型初始化失败，请检查日志'}), 500
    except Exception as e:
        return jsonify({'error': f'模型初始化失败: {str(e)}'}), 500

    results = {}
    success = 0

    for fname in image_files:
        if is_stopped('cl_tag'):
            results[fname] = '⏹️ 已停止'
            continue

        image_path = os.path.join(folder, fname)
        base_name = os.path.splitext(fname)[0]
        txt_name = base_name + '.txt'

        try:
            tags = engine.predict(image_path, gen_threshold, char_threshold)
            if tags is None:
                results[fname] = '❌ 推理失败'
                continue

            # 组装文本
            parts = []
            if include_rating and tags.get('rating'):
                parts.extend([t[0] for t in tags['rating']])
            if include_quality and tags.get('quality'):
                parts.extend([t[0] for t in tags['quality']])
            if tags.get('character'):
                parts.extend([t[0] for t in tags['character']])
            if tags.get('copyright'):
                parts.extend([t[0] for t in tags['copyright']])
            if tags.get('general'):
                parts.extend([t[0] for t in tags['general']])

            tag_text = ', '.join(parts)

            out_path = os.path.join(output_folder, txt_name)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(tag_text)

            tag_count = len(parts)
            results[fname] = f'✅ {tag_count} 个标签'
            success += 1
            print(f'  ✅ {fname}: {tag_count} tags')

        except Exception as e:
            results[fname] = f'❌ {str(e)[:80]}'
            print(f'  ❌ {fname}: {e}')

    results['_summary'] = f'反推打标完成 ({success}/{len(image_files)} 成功)'
    print(f'=== 完成: {success}/{len(image_files)} ===\n')
    return jsonify(results)


@app.route('/cl_tag_inventory', methods=['POST'])
def cl_tag_inventory():
    """扫描数据集txt文件，构建标签→图片反向索引，附带中文翻译"""
    folder = request.form.get('folder', '').strip()
    if not folder:
        return jsonify({'error': '缺少文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在或不是文件夹：{folder}'}), 400

    zh_dict = _load_tag_zh_dict()

    # 扫描所有txt文件，构建 tag → [图片文件名] 的反向索引
    tag_to_images = {}  # tag -> set of image filenames
    total_images = 0
    total_txt = 0

    txt_files = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith('.txt')],
        key=natural_sort_key
    )

    for fname in txt_files:
        base_name = os.path.splitext(fname)[0]
        # 查找同名图片
        peer_image = None
        for ext in IMAGE_EXTENSIONS:
            candidate = base_name + ext
            if os.path.exists(os.path.join(folder, candidate)):
                peer_image = candidate
                break

        if not peer_image:
            continue  # 没有同名图片的txt跳过

        total_txt += 1
        total_images += 1

        txt_path = os.path.join(folder, fname)
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue

        # 解析标签：取第一行（或空行前的部分）作为tags行
        lines = content.replace('\r\n', '\n').split('\n\n', 1)
        tags_line = lines[0].strip()
        tags = [t.strip() for t in tags_line.split(',') if t.strip()]

        for tag in tags:
            normalized = tag.strip().lower().replace(' ', '_')
            if normalized not in tag_to_images:
                tag_to_images[normalized] = {'display': tag, 'images': []}
            if peer_image not in tag_to_images[normalized]['images']:
                tag_to_images[normalized]['images'].append(peer_image)

    # 构建结果列表，按出现次数降序
    tag_list = []
    for normalized, info in tag_to_images.items():
        zh = zh_dict.get(normalized, '')
        tag_list.append({
            'tag': info['display'],
            'tag_normalized': normalized,
            'zh': zh,
            'count': len(info['images']),
            'images': info['images']
        })

    tag_list.sort(key=lambda x: x['count'], reverse=True)

    return jsonify({
        'folder': folder,
        'total_images': total_images,
        'total_txt': total_txt,
        'total_tags': len(tag_list),
        'tags': tag_list
    })


@app.route('/move_tag_images', methods=['POST'])
def move_tag_images():
    """将指定标签关联的图片（及同名txt）移动到子文件夹"""
    folder = request.form.get('folder', '').strip()
    subfolder = request.form.get('subfolder', '').strip()
    files_json = request.form.get('files', '[]')

    if not folder:
        return jsonify({'error': '未提供文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在：{folder}'}), 400
    if not subfolder:
        return jsonify({'error': '未提供子文件夹名称'}), 400

    try:
        file_list = json.loads(files_json)
    except Exception:
        file_list = []

    if not file_list:
        return jsonify({'error': '没有要移动的文件'}), 400

    target_dir = os.path.join(folder, subfolder)
    os.makedirs(target_dir, exist_ok=True)

    moved = 0
    failed = 0
    results = {}

    for fname in file_list:
        src = os.path.join(folder, fname)
        if not os.path.isfile(src):
            results[fname] = 'ℹ️ 文件不存在，跳过'
            continue

        base = os.path.splitext(fname)[0]
        statuses = []

        # 移动图片
        try:
            shutil.move(src, os.path.join(target_dir, fname))
            statuses.append(f'✅ 图片已移动')
            moved += 1
        except Exception as e:
            statuses.append(f'❌ 图片移动失败: {e}')
            failed += 1

        # 移动同名txt
        txt_name = base + '.txt'
        txt_src = os.path.join(folder, txt_name)
        if os.path.isfile(txt_src):
            try:
                shutil.move(txt_src, os.path.join(target_dir, txt_name))
                statuses.append('✅ txt已移动')
            except Exception as e:
                statuses.append(f'❌ txt移动失败: {e}')
                failed += 1

        results[fname] = ' | '.join(statuses)

    results['_summary'] = f'移动完成: {moved} 个图片已移动到 {subfolder}/, {failed} 个失败'
    return jsonify(results)


@app.route('/stop_cl_tag', methods=['POST'])
def stop_cl_tag():
    request_stop('cl_tag')
    return jsonify({'success': True, 'message': '已请求停止反推打标'})


# ===== 独立 Mask 生成 Pipeline =====

_mask_pipeline = None
_mask_pipeline_lock = threading.Lock()


def _get_mask_pipeline():
    global _mask_pipeline
    if _mask_pipeline is None:
        with _mask_pipeline_lock:
            if _mask_pipeline is None:
                import yaml
                config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mask_config.yaml')
                config = {}
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f) or {}
                    except Exception as e:
                        print(f'加载 mask_config.yaml 失败: {e}')
                from manga_censor.pipeline import IndependentMaskPipeline
                _mask_pipeline = IndependentMaskPipeline(config)
    return _mask_pipeline


@app.route('/mask_pipeline_status', methods=['GET'])
def mask_pipeline_status():
    pipeline = _get_mask_pipeline()
    return jsonify(pipeline.get_status())


@app.route('/mask_pipeline_init', methods=['POST'])
def mask_pipeline_init():
    import json as _json
    pipeline = _get_mask_pipeline()
    enabled_parts = None
    confidence_overrides = None
    parts_json = request.form.get('enabled_parts', '')
    conf_json = request.form.get('confidence_overrides', '')
    if parts_json:
        try:
            enabled_parts = _json.loads(parts_json)
        except Exception:
            pass
    if conf_json:
        try:
            confidence_overrides = _json.loads(conf_json)
        except Exception:
            pass
    try:
        pipeline.initialize(enabled_parts=enabled_parts, confidence_overrides=confidence_overrides)
        return jsonify({'success': True, 'status': pipeline.get_status()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/mask_batch', methods=['POST'])
def mask_batch():
    import json as _json
    import cv2

    # 解析配置
    config = {}
    config_json = request.form.get('config', '').strip()
    if config_json:
        try:
            config = _json.loads(config_json)
        except Exception:
            return jsonify({'error': 'config JSON 解析失败'}), 400

    # 基础参数
    mode = (request.form.get('mode') or config.get('mode') or 'normal').strip()
    folder = (request.form.get('folder') or config.get('folder') or '').strip()
    output_folder = (request.form.get('output_folder') or config.get('output_folder') or '').strip()
    
    # 统一的输出选项
    invert = _is_truthy(request.form.get('invert', str(config.get('invert', '0'))))
    merge_single = _is_truthy(request.form.get('merge_single', str(config.get('merge_single', '0'))))
    save_individual = _is_truthy(request.form.get('save_individual', str(config.get('save_individual', '0'))))

    if not folder:
        return jsonify({'error': '缺少图片文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在：{folder}'}), 400

    reset_stop_flag('body_mask')

    image_files = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key
    )
    if not image_files:
        return jsonify({'error': '文件夹中没有找到图片文件'}), 400

    # 性别感知模式
    if mode == 'gender_aware':
        # 解析性别配置
        gender_config_json = request.form.get('gender_config', config.get('gender_config', '{}'))
        if isinstance(gender_config_json, str):
            try:
                gender_config = _json.loads(gender_config_json)
            except Exception:
                return jsonify({'error': '性别配置 JSON 解析失败'}), 400
        else:
            gender_config = gender_config_json

        if not output_folder:
            output_folder = folder + '_gender_masks'
        os.makedirs(output_folder, exist_ok=True)

        pipeline = _get_gender_aware_pipeline()
        try:
            if not pipeline._initialized:
                pipeline.initialize()
            # 更新策略引擎的配置（只支持 male / female）
            if gender_config:
                print(f'[mask_batch] 应用前端性别配置: {gender_config}')
                from manga_censor.detectors.strategy_engine import StrategyConfig

                for gender_name in ('male', 'female'):
                    key = f'{gender_name}_strategy'
                    if key not in gender_config:
                        continue
                    sdata = gender_config[key] or {}
                    s_mode = sdata.get('mode', 'custom')
                    merge = sdata.get('merge_to_single_mask', True)
                    parts = list(sdata.get('custom_parts', []) or [])

                    if s_mode == 'full_body':
                        # full_body 模式：扩展为所有可用部位
                        parts = list(pipeline.detectors.keys())
                        pipeline.strategy_engine.strategies[gender_name] = StrategyConfig(
                            mode=s_mode,
                            custom_parts=parts,
                            merge_to_single_mask=merge,
                        )
                        print(f'[mask_batch] 更新 {gender_name} 策略: mode=full_body, parts={parts}')
                    elif s_mode == 'custom' and not parts:
                        # custom 模式但 parts 为空 → 明确标记为 __skip__，表示该性别不遮盖
                        pipeline.strategy_engine.strategies[gender_name] = StrategyConfig(
                            mode='custom',
                            custom_parts=['__skip__'],
                            merge_to_single_mask=merge,
                        )
                        print(f'[mask_batch] {gender_name} 策略 custom 且未选部位 → 标记为 __skip__（不遮盖）')
                    else:
                        pipeline.strategy_engine.strategies[gender_name] = StrategyConfig(
                            mode=s_mode,
                            custom_parts=parts,
                            merge_to_single_mask=merge,
                        )
                        print(f'[mask_batch] 更新 {gender_name} 策略: mode={s_mode}, parts={parts}')

                # 应用置信度覆盖（无论策略是否被覆盖，都尝试应用）
                for gender_name in ('male', 'female'):
                    key = f'{gender_name}_strategy'
                    confs = (gender_config.get(key) or {}).get('confidence_overrides') or {}
                    for part, conf in confs.items():
                        det = pipeline.detectors.get(part)
                        if det is not None and hasattr(det, 'conf'):
                            try:
                                det.conf = float(conf)
                                print(f'[mask_batch] 置信度覆盖: {gender_name}/{part} = {conf}')
                            except Exception:
                                pass
        except Exception as e:
            import traceback
            print(f'[mask_batch] Pipeline 初始化失败: {e}')
            print(traceback.format_exc())
            return jsonify({'error': f'Pipeline 初始化失败: {str(e)}'}), 500

        results = {}
        success_count = 0
        failed_count = 0

        print(f'\n=== 性别感知遮盖批量处理 ===')
        print(f'输入文件夹: {folder}')
        print(f'输出文件夹: {output_folder}')
        print(f'多人物模式: {gender_config.get("multi_person_mode", "per_person")}')
        print(f'文字检测: {"启用" if gender_config.get("text_enabled") else "禁用"}')
        print(f'图片数: {len(image_files)}')

        for fname in image_files:
            if is_stopped('body_mask'):
                results[fname] = '⏹️ 已停止'
                continue

            image_path = os.path.join(folder, fname)
            try:
                from manga_censor.utils import cv2_imread
                image = cv2_imread(image_path)
                if image is None:
                    results[fname] = '❌ 无法读取图片'
                    failed_count += 1
                    continue

                # 处理图片
                result = pipeline.process(image)
                
                # 保存遮罩（应用输出选项）
                saved_files = pipeline.save_masks(
                    result, 
                    output_folder, 
                    prefix=os.path.splitext(fname)[0],
                    invert=invert,
                    merge_single=merge_single,
                    save_individual=save_individual
                )

                # 统计信息
                gender_stats = {}
                for person_result in result.person_results:
                    gender = person_result.gender
                    gender_stats[gender] = gender_stats.get(gender, 0) + 1

                gender_str = ', '.join([f'{k}:{v}' for k, v in gender_stats.items()]) if gender_stats else '无人物'
                total_parts = sum(len(pr.masks) for pr in result.person_results)

                success_count += 1
                results[fname] = f'✅ {len(result.persons)}人 [{gender_str}] | {total_parts}部位 | {len(saved_files)}文件'
                print(f'  ✅ {fname}: {len(result.persons)}人, {total_parts}部位')
            except Exception as e:
                failed_count += 1
                results[fname] = f'❌ {str(e)[:100]}'
                print(f'  ❌ {fname}: {e}')

        results['_summary'] = f'性别感知遮盖完成 | 总数 {len(image_files)} | 成功 {success_count} | 失败 {failed_count} | 输出: {output_folder}'
        print(f'=== 完成: {success_count}/{len(image_files)} ===\n')
        return jsonify(results)

    # 普通模式
    if not output_folder:
        output_folder = folder + '_masks'
    os.makedirs(output_folder, exist_ok=True)

    # 解析部位配置
    enabled_parts = config.get('enabled_parts')
    confidence_overrides = config.get('confidence_overrides')

    parts_json = request.form.get('enabled_parts', '')
    conf_json = request.form.get('confidence_overrides', '')
    if parts_json:
        try:
            enabled_parts = _json.loads(parts_json)
        except Exception:
            pass
    if conf_json:
        try:
            confidence_overrides = _json.loads(conf_json)
        except Exception:
            pass

    pipeline = _get_mask_pipeline()
    
    try:
        pipeline.initialize(enabled_parts=enabled_parts, confidence_overrides=confidence_overrides)
    except Exception as e:
        import traceback
        print(f'❌ Pipeline 初始化失败: {e}')
        print(f'错误详情:\n{traceback.format_exc()}')
        return jsonify({'error': f'Pipeline 初始化失败: {str(e)}'}), 500

    loaded_detectors = list(pipeline.detectors.keys())
    print(f'\n=== 普通遮盖批量处理 ===')
    print(f'输入文件夹: {folder}')
    print(f'输出文件夹: {output_folder}')
    print(f'启用部位: {enabled_parts}')
    print(f'实际加载的检测器: {loaded_detectors} ({len(loaded_detectors)}个)')
    if enabled_parts and len(loaded_detectors) == 0:
        print(f'  ⚠️ 警告: 请求了 {len(enabled_parts)} 个部位但没有任何检测器成功加载！')
        print(f'  ⚠️ 可能原因: 部位名称不匹配或检测器加载失败')
        return jsonify({'error': f'没有任何检测器成功加载！请检查部位名称是否正确。\n请求的部位: {enabled_parts}\n可用部位: {list(pipeline.get_status()["available_parts"].keys())}'}), 500
    print(f'图片数: {len(image_files)}')

    image_paths = [os.path.join(folder, f) for f in image_files]
    results = {}
    success_count = 0
    failed_count = 0
    
    for img_path in image_paths:
        if is_stopped('body_mask'):
            results[os.path.basename(img_path)] = '⏹️ 已停止'
            continue
        try:
            report = pipeline.process_image(
                img_path, 
                output_folder, 
                invert=invert, 
                merge_single=merge_single,
                save_individual=save_individual
            )
            parts_ok = sum(1 for p in report.get('parts', {}).values() if p.get('has_detection'))
            total_parts = len(report.get('parts', {}))
            timing = report.get('timing', {}).get('total', 0)
            success_count += 1
            inv_tag = ' [反相]' if invert else ''
            results[os.path.basename(img_path)] = f'✅ {parts_ok}/{total_parts} 部位检出{inv_tag} | {timing:.1f}s'
            print(f'  ✅ {os.path.basename(img_path)}: {parts_ok}/{total_parts} 部位')
        except Exception as e:
            failed_count += 1
            results[os.path.basename(img_path)] = f'❌ {str(e)[:100]}'
            print(f'  ❌ {os.path.basename(img_path)}: {e}')
    
    inv_note = '（反相模式）' if invert else ''
    results['_summary'] = f'Mask 生成完成{inv_note} | 总数 {len(image_files)} | 成功 {success_count} | 失败 {failed_count} | 输出: {output_folder}'
    print(f'=== 完成: {success_count}/{len(image_files)} ===\n')
    return jsonify(results)


@app.route('/mask_preview', methods=['POST'])
def mask_preview():
    import base64 as b64
    from io import BytesIO
    from PIL import Image
    file = request.files.get('file')
    image_path = request.form.get('image_path', '').strip()
    if not file and not image_path:
        return jsonify({'error': '请上传图片或提供路径'}), 400
    pipeline = _get_mask_pipeline()
    if not pipeline._initialized:
        try:
            pipeline.initialize()
        except Exception as e:
            return jsonify({'error': f'初始化失败: {str(e)}'}), 500
    try:
        if file:
            img = Image.open(file.stream).convert('RGB')
            import numpy as np
            image = np.array(img)[:, :, ::-1]
        else:
            import cv2
            from manga_censor.utils import cv2_imread
            image = cv2_imread(image_path)
            if image is None:
                return jsonify({'error': f'无法读取: {image_path}'}), 400
        h, w = image.shape[:2]
        parts_result = {}
        for name, detector in pipeline.detectors.items():
            result = detector.detect(image)
            mask_bytes = cv2.imencode('.png', result.mask)[1].tobytes()
            parts_result[name] = {
                'mask_base64': b64.b64encode(mask_bytes).decode('ascii'),
                'count': result.count,
                'confidence': round(result.confidence, 4),
                'has_detection': result.count > 0,
            }
        return jsonify({'image_size': [w, h], 'parts': parts_result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/mask_merge', methods=['POST'])
def mask_merge():
    """合并多个 mask 文件"""
    import json as _json
    
    mask_files_json = request.form.get('mask_files', '[]')
    output_path = request.form.get('output_path', '').strip()
    invert = _is_truthy(request.form.get('invert', '0'))
    operation = request.form.get('operation', 'union').strip()
    
    try:
        mask_files = _json.loads(mask_files_json)
    except Exception:
        return jsonify({'error': 'mask_files JSON 解析失败'}), 400
    
    if not mask_files:
        return jsonify({'error': '请提供要合并的 mask 文件列表'}), 400
    if not output_path:
        return jsonify({'error': '请提供输出路径'}), 400
    
    pipeline = _get_mask_pipeline()
    
    try:
        report = pipeline.merge_masks(mask_files, output_path, invert, operation)
        return jsonify({'success': True, 'report': report})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/mask_render_from_json', methods=['POST'])
def mask_render_from_json():
    """从 JSON 报告重新渲染 mask"""
    import json as _json
    
    json_path = request.form.get('json_path', '').strip()
    output_path = request.form.get('output_path', '').strip()
    parts_filter_json = request.form.get('parts_filter', '')
    invert = _is_truthy(request.form.get('invert', '0'))
    operation = request.form.get('operation', 'union').strip()
    
    if not json_path:
        return jsonify({'error': '请提供 JSON 文件路径'}), 400
    if not output_path:
        return jsonify({'error': '请提供输出路径'}), 400
    
    parts_filter = None
    if parts_filter_json:
        try:
            parts_filter = _json.loads(parts_filter_json)
        except Exception:
            pass
    
    pipeline = _get_mask_pipeline()
    
    try:
        report = pipeline.render_from_json(json_path, output_path, parts_filter, invert, operation)
        return jsonify({'success': True, 'report': report})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


_anime_face_detector = None
_anime_face_lock = threading.Lock()


def _get_anime_face_detector():
    global _anime_face_detector
    if _anime_face_detector is None:
        with _anime_face_lock:
            if _anime_face_detector is None:
                try:
                    anime_face_module = importlib.import_module('anime_face_detector')
                    detector_cls = getattr(anime_face_module, 'AnimeFaceDetector')
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        'anime_face_detector 模块不存在。'
                        '如果需要使用二次元面部遮罩功能，请恢复/安装该模块；'
                        '如果不需要，可以忽略此功能。'
                    ) from exc
                _anime_face_detector = detector_cls()
    return _anime_face_detector


@app.route('/anime_face_status', methods=['GET'])
def anime_face_status():
    detector = _get_anime_face_detector()
    return jsonify(detector.get_status())


@app.route('/anime_face_batch', methods=['POST'])
def anime_face_batch():
    """批量生成二次元面部遮罩。"""

    folder = request.form.get('folder', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    save_json = _is_truthy(request.form.get('save_json', '1'))

    try:
        confidence_threshold = float(request.form.get('confidence_threshold', '0.5'))
    except Exception:
        confidence_threshold = 0.5
    try:
        dilate_px = int(request.form.get('dilate_px', '10'))
    except Exception:
        dilate_px = 10
    try:
        expand_ratio = float(request.form.get('expand_ratio', '0.15'))
    except Exception:
        expand_ratio = 0.15

    if not folder:
        return jsonify({'error': '缺少图片文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在或不是文件夹：{folder}'}), 400

    if not output_folder:
        output_folder = folder + '_anime_face_masks'

    os.makedirs(output_folder, exist_ok=True)
    reset_stop_flag('body_mask')

    image_files = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key
    )
    if not image_files:
        return jsonify({'error': '文件夹中没有找到图片文件'}), 400

    print(f'\n=== AnimeFace 二次元面部遮罩批量生成 ===')
    print(f'输入文件夹: {folder}')
    print(f'输出文件夹: {output_folder}')
    print(f'置信度阈值: {confidence_threshold}')
    print(f'膨胀像素: {dilate_px}')
    print(f'图片数: {len(image_files)}')

    detector = _get_anime_face_detector()
    try:
        if not detector.initialize():
            return jsonify({'error': f'AnimeFace 初始化失败: {detector._error or "未知错误"}'}), 500
    except Exception as e:
        return jsonify({'error': f'AnimeFace 初始化失败: {str(e)}'}), 500

    results = {}
    success_count = 0
    failed_count = 0

    for fname in image_files:
        if is_stopped('body_mask'):
            results[fname] = '⏹️ 已停止'
            continue

        image_path = os.path.join(folder, fname)
        base_name = os.path.splitext(fname)[0]
        mask_path = os.path.join(output_folder, base_name + '_mask.png')
        json_path = os.path.join(output_folder, base_name + '_mask.json') if save_json else None

        try:
            image = Image.open(image_path).convert('RGB')
            mask, debug = detector.build_mask(
                image=image,
                confidence_threshold=confidence_threshold,
                dilate_px=dilate_px,
                expand_ratio=expand_ratio,
            )
            mask.save(mask_path)

            result = {
                'filename': fname,
                'output_mask': mask_path,
                'faces_detected': debug.get('faces_detected', 0),
                'mask_ratio': debug.get('mask_ratio', 0),
                'confidence_threshold': confidence_threshold,
                'dilate_px': dilate_px,
                'expand_ratio': expand_ratio,
                'engine': 'anime_face_detector',
                'details': debug,
            }

            if save_json and json_path:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

            success_count += 1
            faces_count = debug.get('faces_detected', 0)
            results[fname] = f'✅ 检测到 {faces_count} 个面部 | mask={debug.get("mask_ratio", 0):.3%}'
        except Exception as e:
            failed_count += 1
            results[fname] = f'❌ 处理失败: {str(e)[:160]}'

    results['_summary'] = f'AnimeFace 面部遮罩生成完成 | 总数 {len(image_files)} | 成功 {success_count} | 失败 {failed_count}'
    return jsonify(results)


def _get_manga_censor_pipeline():
    """旧漫画审查 Pipeline 已废弃，保留占位以避免历史代码静态检查报未定义。"""
    raise RuntimeError('旧漫画审查 Pipeline 已废弃，请使用独立 Mask Pipeline 相关接口')


def _get_body_mask_engine():
    """旧 body mask engine 已废弃，保留占位以避免历史代码静态检查报未定义。"""
    raise RuntimeError('旧 body mask engine 已废弃，请使用 /mask_batch 接口')


def _get_text_mask_engine():
    """旧 text mask engine 已废弃，保留占位以避免历史代码静态检查报未定义。"""
    raise RuntimeError('旧 text mask engine 已废弃，请使用独立 Mask Pipeline 或文字遮罩新接口')


# [已移除] sapiens_mask_batch — 旧 Sapiens 遮罩路由已废弃
def _deprecated_sapiens_mask_batch():
    """批量生成 Sapiens 高精度人体部位遮罩。"""
    from body_part_mask_engine import SAPIENS_LABEL_MAP, ensure_binary
    from PIL import Image

    folder = request.form.get('folder', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    labels_raw = (request.form.get('labels') or '').strip()
    variant = (request.form.get('variant') or 'sapiens_0.3b').strip()
    save_json = _is_truthy(request.form.get('save_json', '1'))

    labels = [x.strip() for x in labels_raw.split(',') if x.strip()]

    if not folder:
        return jsonify({'error': '缺少图片文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在或不是文件夹：{folder}'}), 400
    if not labels:
        return jsonify({'error': '请至少选择一个 Sapiens 部位标签'}), 400

    if not output_folder:
        output_folder = folder + '_sapiens_masks'

    os.makedirs(output_folder, exist_ok=True)
    reset_stop_flag('body_mask')

    image_files = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key
    )
    if not image_files:
        return jsonify({'error': '文件夹中没有找到图片文件'}), 400

    print(f'\n=== Sapiens 高精度部位遮罩批量生成 ===')
    print(f'输入文件夹: {folder}')
    print(f'输出文件夹: {output_folder}')
    print(f'Sapiens 变体: {variant}')
    print(f'标签: {labels}')
    print(f'图片数: {len(image_files)}')

    engine = _get_body_mask_engine()
    sapiens = engine._sapiens

    try:
        if not sapiens.initialize(variant=variant):
            return jsonify({'error': f'Sapiens 模型初始化失败: {sapiens._error or "未知错误"}'}), 500
    except Exception as e:
        return jsonify({'error': f'Sapiens 初始化失败: {str(e)}'}), 500

    results = {}
    success_count = 0
    failed_count = 0

    for fname in image_files:
        if is_stopped('body_mask'):
            results[fname] = '⏹️ 已停止'
            continue

        image_path = os.path.join(folder, fname)
        base_name = os.path.splitext(fname)[0]
        mask_path = os.path.join(output_folder, base_name + '_mask.png')
        json_path = os.path.join(output_folder, base_name + '_mask.json') if save_json else None

        try:
            image = Image.open(image_path).convert('RGB')
            mask, debug = sapiens.build_mask(image=image, selected_labels=labels)
            mask = ensure_binary(mask, threshold=127)
            mask.save(mask_path)

            import numpy as np
            mask_ratio = float((np.array(mask, dtype=np.uint8) > 127).mean())

            result = {
                'filename': fname,
                'output_mask': mask_path,
                'selected_labels': labels,
                'mask_ratio': round(mask_ratio, 6),
                'engine': 'sapiens',
                'variant': variant,
                'details': debug,
            }

            if save_json and json_path:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

            success_count += 1
            results[fname] = f'✅ 已生成遮罩 | mask={mask_ratio:.3%} | engine=sapiens/{variant}'
        except Exception as e:
            failed_count += 1
            results[fname] = f'❌ 处理失败: {str(e)[:160]}'

    results['_summary'] = f'Sapiens 部位遮罩生成完成 | 总数 {len(image_files)} | 成功 {success_count} | 失败 {failed_count}'
    return jsonify(results)


# [已移除] text_mask_batch — 旧文字遮罩路由已废弃
def _deprecated_text_mask_batch():
    """批量生成文字 / 水印 / 字幕区域遮罩图。支持 OCR 与 DiffPipeForge 训练语义导出。"""
    folder = request.form.get('folder', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    save_json = _is_truthy(request.form.get('save_json', '1'))
    text_engine_name = (request.form.get('text_mask_engine') or 'auto').strip()
    ocr_lang = (request.form.get('text_mask_ocr_lang') or 'ch').strip()
    include_text = _is_truthy(request.form.get('text_mask_include_text', '1'))
    include_watermark = _is_truthy(request.form.get('text_mask_include_watermark', '1'))
    text_region_mode = (request.form.get('text_mask_text_region_mode') or 'full').strip()
    watermark_region_mode = (request.form.get('text_mask_watermark_region_mode') or 'edge').strip()
    invert_mask = _is_truthy(request.form.get('text_mask_invert_mask', '0'))

    try:
        confidence_threshold = float(request.form.get('text_mask_confidence_threshold', '0.5'))
    except Exception:
        confidence_threshold = 0.5
    try:
        dilate_pixels = int(request.form.get('text_mask_dilate_pixels', '4'))
    except Exception:
        dilate_pixels = 4
    try:
        min_area_ratio = float(request.form.get('text_mask_min_area_ratio', '0.0'))
    except Exception:
        min_area_ratio = 0.0
    try:
        max_area_ratio = float(request.form.get('text_mask_max_area_ratio', '0.2'))
    except Exception:
        max_area_ratio = 0.2

    if not folder:
        return jsonify({'error': '缺少图片文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在或不是文件夹：{folder}'}), 400
    if not include_text and not include_watermark:
        return jsonify({'error': '请至少启用一种 OCR 来源：文字或水印'}), 400

    if not output_folder:
        output_folder = folder + ('_diffpipe_text_masks' if invert_mask else '_text_masks')

    os.makedirs(output_folder, exist_ok=True)
    reset_stop_flag('body_mask')

    image_files = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key
    )
    if not image_files:
        return jsonify({'error': '文件夹中没有找到图片文件'}), 400

    print('\n=== 文字 / 水印遮罩批量生成 ===')
    print(f'输入文件夹: {folder}')
    print(f'输出文件夹: {output_folder}')
    print(f'引擎: {text_engine_name}')
    print(f'OCR语言: {ocr_lang}')
    print(f'包含文字: {"是" if include_text else "否"}')
    print(f'包含水印: {"是" if include_watermark else "否"}')
    print(f'文字区域模式: {text_region_mode}')
    print(f'水印区域模式: {watermark_region_mode}')
    print(f'DiffPipeForge训练语义(反相): {"是" if invert_mask else "否"}')
    print(f'输出 JSON: {"是" if save_json else "否"}')
    print(f'图片数: {len(image_files)}')

    try:
        engine = _get_text_mask_engine()
        if not engine.initialize(preferred_engine=text_engine_name, ocr_lang=ocr_lang):
            return jsonify({'error': f'文字遮罩引擎初始化失败: {engine.get_status().get("ocr_error") or "未知错误"}'}), 500

        results = engine.batch_process(
            folder=folder,
            output_folder=output_folder,
            save_json=save_json,
            stop_check=lambda: is_stopped('body_mask'),
            engine=text_engine_name,
            ocr_lang=ocr_lang,
            confidence_threshold=confidence_threshold,
            dilate_pixels=dilate_pixels,
            include_text=include_text,
            include_watermark=include_watermark,
            text_region_mode=text_region_mode,
            watermark_region_mode=watermark_region_mode,
            invert_mask=invert_mask,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'文字遮罩生成失败: {str(e)}'}), 500


# [已移除] combined_mask_batch — 旧合并遮罩路由已废弃
def _deprecated_combined_mask_batch():
    """批量生成多来源合并遮罩。支持：身体/服饰部位 + OCR文字 + OCR水印，并可反相导出训练器语义。"""
    import numpy as np
    from PIL import Image, ImageOps

    folder = request.form.get('folder', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    labels_raw = (request.form.get('labels') or '').strip()
    include_body = _is_truthy(request.form.get('include_body', '1'))
    include_text = _is_truthy(request.form.get('include_text', '0'))
    save_json = _is_truthy(request.form.get('save_json', '1'))
    invert_mask = _is_truthy(request.form.get('text_mask_invert_mask', '0'))

    text_engine_name = (request.form.get('text_mask_engine') or 'auto').strip()
    ocr_lang = (request.form.get('text_mask_ocr_lang') or 'ch').strip()
    include_ocr_text = _is_truthy(request.form.get('text_mask_include_text', '1'))
    include_ocr_watermark = _is_truthy(request.form.get('text_mask_include_watermark', '1'))
    text_region_mode = (request.form.get('text_mask_text_region_mode') or 'full').strip()
    watermark_region_mode = (request.form.get('text_mask_watermark_region_mode') or 'edge').strip()

    detail_method = (request.form.get('detail_method') or 'None').strip()
    try:
        detail_erode = int(request.form.get('detail_erode', '12'))
    except Exception:
        detail_erode = 12
    try:
        detail_dilate = int(request.form.get('detail_dilate', '6'))
    except Exception:
        detail_dilate = 6
    try:
        black_point = float(request.form.get('black_point', '0.15'))
    except Exception:
        black_point = 0.15
    try:
        white_point = float(request.form.get('white_point', '0.99'))
    except Exception:
        white_point = 0.99
    try:
        output_white_level = float(request.form.get('output_white_level', '1.0'))
    except Exception:
        output_white_level = 1.0
    try:
        output_black_level = float(request.form.get('output_black_level', '0.0'))
    except Exception:
        output_black_level = 0.0
    try:
        confidence_threshold = float(request.form.get('text_mask_confidence_threshold', '0.5'))
    except Exception:
        confidence_threshold = 0.5
    try:
        dilate_pixels = int(request.form.get('text_mask_dilate_pixels', '4'))
    except Exception:
        dilate_pixels = 4
    try:
        min_area_ratio = float(request.form.get('text_mask_min_area_ratio', '0.0'))
    except Exception:
        min_area_ratio = 0.0
    try:
        max_area_ratio = float(request.form.get('text_mask_max_area_ratio', '0.2'))
    except Exception:
        max_area_ratio = 0.2

    labels = [x.strip() for x in labels_raw.split(',') if x.strip()]

    if not folder:
        return jsonify({'error': '缺少图片文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在或不是文件夹：{folder}'}), 400
    if include_body and not labels:
        return jsonify({'error': '启用身体/部位遮罩时，请至少选择一个部位标签'}), 400
    if include_text and not include_ocr_text and not include_ocr_watermark:
        return jsonify({'error': '启用 OCR 来源时，请至少勾选文字或水印'}), 400
    if not include_body and not include_text:
        return jsonify({'error': '请至少启用一种遮罩来源'}), 400

    if not output_folder:
        output_folder = folder + ('_diffpipe_combined_masks' if invert_mask else '_combined_masks')

    os.makedirs(output_folder, exist_ok=True)
    reset_stop_flag('body_mask')

    image_files = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key
    )
    if not image_files:
        return jsonify({'error': '文件夹中没有找到图片文件'}), 400

    body_engine = _get_body_mask_engine() if include_body else None
    text_engine = _get_text_mask_engine() if include_text else None

    if body_engine and not body_engine.initialize():
        return jsonify({'error': f'部位遮罩引擎初始化失败: {body_engine.last_error or ""}'}), 500
    if text_engine and not text_engine.initialize(preferred_engine=text_engine_name, ocr_lang=ocr_lang):
        return jsonify({'error': f'文字遮罩引擎初始化失败: {text_engine.get_status().get("ocr_error") or ""}'}), 500

    results = {}
    success_count = 0
    failed_count = 0

    for fname in image_files:
        if is_stopped('body_mask'):
            results[fname] = '⏹️ 已停止'
            continue

        image_path = os.path.join(folder, fname)
        base_name = os.path.splitext(fname)[0]
        suffix = '_diffpipe_combined_mask' if invert_mask else '_combined_mask'
        mask_path = os.path.join(output_folder, base_name + suffix + '.png')
        json_path = os.path.join(output_folder, base_name + suffix + '.json') if save_json else None

        try:
            image = Image.open(image_path).convert('RGB')
            merged_arr = np.zeros((image.height, image.width), dtype=np.uint8)
            details = {
                'sources': [],
                'image_size': [image.width, image.height],
                'invert_mask': invert_mask,
            }

            if body_engine:
                body_mask, body_debug = body_engine._build_mask(
                    image=image,
                    selected_labels=labels,
                    detail_method=detail_method,
                    detail_erode=detail_erode,
                    detail_dilate=detail_dilate,
                    black_point=black_point,
                    white_point=white_point,
                    output_white_level=output_white_level,
                    output_black_level=output_black_level,
                )
                merged_arr = np.maximum(merged_arr, (np.array(body_mask, dtype=np.uint8) > 127).astype(np.uint8))
                details['sources'].append({'type': 'body_part', 'labels': labels, 'details': body_debug})

            if text_engine:
                text_mask, text_debug = text_engine._build_text_mask(
                    image=image,
                    engine=text_engine_name,
                    ocr_lang=ocr_lang,
                    confidence_threshold=confidence_threshold,
                    dilate_pixels=dilate_pixels,
                    include_text=include_ocr_text,
                    include_watermark=include_ocr_watermark,
                    text_region_mode=text_region_mode,
                    watermark_region_mode=watermark_region_mode,
                    invert_mask=False,
                    min_area_ratio=min_area_ratio,
                    max_area_ratio=max_area_ratio,
                )
                merged_arr = np.maximum(merged_arr, (np.array(text_mask, dtype=np.uint8) > 127).astype(np.uint8))
                details['sources'].append({'type': 'text_ocr', 'details': text_debug})

            final_mask = Image.fromarray((merged_arr * 255).astype(np.uint8), 'L')
            if invert_mask:
                final_mask = ImageOps.invert(final_mask)

            mask_ratio = float((np.array(final_mask, dtype=np.uint8) > 127).mean())
            final_mask.save(mask_path)

            result = {
                'filename': fname,
                'output_mask': mask_path,
                'mask_ratio': round(mask_ratio, 6),
                'sources': [s['type'] for s in details['sources']],
                'details': details,
            }

            if save_json and json_path:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

            success_count += 1
            mode = 'DiffPipeForge训练语义' if invert_mask else '普通遮罩语义'
            results[fname] = f"✅ 已生成合并遮罩 | {mode} | mask={mask_ratio:.3%} | sources={','.join(result['sources'])}"
        except Exception as e:
            failed_count += 1
            results[fname] = f'❌ 处理失败: {str(e)[:160]}'

    results['_summary'] = f'合并遮罩生成完成 | 总数 {len(image_files)} | 成功 {success_count} | 失败 {failed_count}'
    return jsonify(results)


# [已移除] body_mask_batch — 旧部位遮罩路由已废弃
def _deprecated_body_mask_batch():
    """批量生成按部位选择的遮罩图。支持 Segformer 部位分割 + NudeNet 精密区域检测。"""
    folder = request.form.get('folder', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    labels_raw = (request.form.get('labels') or '').strip()
    nudenet_labels_raw = (request.form.get('nudenet_labels') or '').strip()
    detail_method = (request.form.get('detail_method') or 'None').strip()
    save_json = _is_truthy(request.form.get('save_json', '1'))

    try:
        detail_erode = int(request.form.get('detail_erode', '12'))
    except Exception:
        detail_erode = 12
    try:
        detail_dilate = int(request.form.get('detail_dilate', '6'))
    except Exception:
        detail_dilate = 6
    try:
        black_point = float(request.form.get('black_point', '0.15'))
    except Exception:
        black_point = 0.15
    try:
        white_point = float(request.form.get('white_point', '0.99'))
    except Exception:
        white_point = 0.99
    try:
        output_white_level = float(request.form.get('output_white_level', '1.0'))
    except Exception:
        output_white_level = 1.0
    try:
        output_black_level = float(request.form.get('output_black_level', '0.0'))
    except Exception:
        output_black_level = 0.0
    try:
        nudenet_confidence = float(request.form.get('nudenet_confidence', '0.25'))
    except Exception:
        nudenet_confidence = 0.25
    try:
        nudenet_expand_pixels = int(request.form.get('nudenet_expand_pixels', '8'))
    except Exception:
        nudenet_expand_pixels = 8
    nudenet_use_ellipse = _is_truthy(request.form.get('nudenet_use_ellipse', '1'))

    labels = [x.strip() for x in labels_raw.split(',') if x.strip()]
    nudenet_labels = [x.strip() for x in nudenet_labels_raw.split(',') if x.strip()]

    if not folder:
        return jsonify({'error': '缺少图片文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在或不是文件夹：{folder}'}), 400
    if not labels and not nudenet_labels:
        return jsonify({'error': '请至少选择一个 Segformer 部位标签或 NudeNet 精密区域标签'}), 400

    if not output_folder:
        output_folder = folder + '_masks'

    os.makedirs(output_folder, exist_ok=True)
    reset_stop_flag('body_mask')

    image_files = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key
    )
    if not image_files:
        return jsonify({'error': '文件夹中没有找到图片文件'}), 400

    print('\n=== 部位遮罩批量生成 ===')
    print(f'输入文件夹: {folder}')
    print(f'输出文件夹: {output_folder}')
    print(f'Segformer 标签: {labels}')
    print(f'NudeNet 标签: {nudenet_labels}')
    print(f'细化方式: {detail_method}')
    print(f'输出 JSON: {"是" if save_json else "否"}')
    print(f'图片数: {len(image_files)}')

    try:
        engine = _get_body_mask_engine()
        results = engine.batch_process(
            folder=folder,
            output_folder=output_folder,
            selected_labels=labels if labels else None,
            nudenet_labels=nudenet_labels if nudenet_labels else None,
            nudenet_confidence=nudenet_confidence,
            nudenet_expand_pixels=nudenet_expand_pixels,
            nudenet_use_ellipse=nudenet_use_ellipse,
            save_json=save_json,
            detail_method=detail_method,
            detail_erode=detail_erode,
            detail_dilate=detail_dilate,
            black_point=black_point,
            white_point=white_point,
            output_white_level=output_white_level,
            output_black_level=output_black_level,
            stop_check=lambda: is_stopped('body_mask')
        )
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'部位遮罩生成失败: {str(e)}'}), 500


@app.route('/stop_body_mask', methods=['POST'])
def stop_body_mask():
    request_stop('body_mask')
    return jsonify({'success': True, 'message': '已请求停止部位遮罩生成'})


@app.route('/batch_rename', methods=['POST'])
def batch_rename():
    """批量重命名：按顺序给图片（及同名txt）重命名为 前缀+编号"""
    folder = request.form.get('folder', '').strip()
    prefix = request.form.get('prefix', '').strip()
    try:
        start_num = int(request.form.get('start_num', '1'))
    except (ValueError, TypeError):
        start_num = 1
    try:
        padding = int(request.form.get('padding', '0'))
    except (ValueError, TypeError):
        padding = 0

    if not folder:
        return jsonify({'error': '缺少文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在或不是文件夹：{folder}'}), 400
    if not prefix:
        return jsonify({'error': '请输入文件名前缀'}), 400

    # 收集图片文件，自然排序
    image_files = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key
    )

    if not image_files:
        return jsonify({'error': '文件夹中没有找到图片文件'}), 400

    print(f'\n=== 批量重命名 ===')
    print(f'文件夹: {folder}')
    print(f'前缀: {prefix}, 起始编号: {start_num}, 补零位数: {padding}')
    print(f'图片数: {len(image_files)}')

    # 第一步：将所有文件重命名为临时名称，避免命名冲突
    temp_mapping = []  # [(原图片名, 原txt名或None, 临时图片名, 临时txt名或None)]
    for i, fname in enumerate(image_files):
        ext = os.path.splitext(fname)[1]
        base = os.path.splitext(fname)[0]
        temp_img = f'__rename_temp_{i}_{ext}'
        temp_txt = None
        txt_path = os.path.join(folder, base + '.txt')
        has_txt = os.path.isfile(txt_path)
        if has_txt:
            temp_txt = f'__rename_temp_{i}_.txt'

        temp_mapping.append((fname, base + '.txt' if has_txt else None, temp_img, temp_txt))

    # 执行临时重命名
    try:
        for orig_img, orig_txt, temp_img, temp_txt in temp_mapping:
            os.rename(os.path.join(folder, orig_img), os.path.join(folder, temp_img))
            if orig_txt and temp_txt:
                os.rename(os.path.join(folder, orig_txt), os.path.join(folder, temp_txt))
    except Exception as e:
        return jsonify({'error': f'临时重命名阶段失败: {e}'}), 500

    # 第二步：从临时名称重命名为最终名称
    results = {}
    success = 0
    num = start_num

    for orig_img, orig_txt, temp_img, temp_txt in temp_mapping:
        ext = os.path.splitext(orig_img)[1]
        if padding > 0:
            num_str = str(num).zfill(padding)
        else:
            num_str = str(num)
        new_base = f'{prefix}{num_str}'
        new_img = new_base + ext
        new_txt = new_base + '.txt'

        try:
            os.rename(os.path.join(folder, temp_img), os.path.join(folder, new_img))
            txt_info = ''
            if temp_txt:
                os.rename(os.path.join(folder, temp_txt), os.path.join(folder, new_txt))
                txt_info = f' + {orig_txt} → {new_txt}'
            results[orig_img] = f'✅ {orig_img} → {new_img}{txt_info}'
            print(f'  ✅ {orig_img} → {new_img}{txt_info}')
            success += 1
        except Exception as e:
            results[orig_img] = f'❌ 重命名失败: {e}'
            print(f'  ❌ {orig_img}: {e}')

        num += 1

    results['_summary'] = f'批量重命名完成 ({success}/{len(image_files)} 成功)'
    print(f'=== 完成: {success}/{len(image_files)} ===\n')
    return jsonify(results)


# ===== 漫画审查 Pipeline（已废弃，由独立 Mask Pipeline 替代） =====

def _deprecated_manga_censor_status():
    """获取漫画审查 Pipeline 状态"""
    pipeline = _get_manga_censor_pipeline()
    return jsonify(pipeline.get_status())


def _deprecated_manga_censor_init():
    """初始化漫画审查 Pipeline（加载所有检测器模型）"""
    pipeline = _get_manga_censor_pipeline()
    status = pipeline.initialize()
    return jsonify({
        'success': all(status.values()),
        'detector_status': status
    })


def _deprecated_manga_censor_preview():
    """预览检测 - 只检测不遮盖，返回所有检测区域供用户选择"""
    import base64
    from io import BytesIO
    
    # 支持文件上传或路径
    file = request.files.get('file')
    image_path = request.form.get('image_path', '').strip()
    
    if not file and not image_path:
        return jsonify({'error': '请上传图片或提供图片路径'}), 400
    
    pipeline = _get_manga_censor_pipeline()
    
    # 初始化
    try:
        pipeline.initialize()
    except Exception as e:
        return jsonify({'error': f'Pipeline 初始化失败: {str(e)}'}), 500
    
    # 读取图片
    try:
        if file:
            image = Image.open(file.stream).convert('RGB')
        else:
            if not os.path.exists(image_path):
                return jsonify({'error': f'图片不存在：{image_path}'}), 400
            image = Image.open(image_path).convert('RGB')
        image_rgb = np.array(image)
    except Exception as e:
        return jsonify({'error': f'读取图片失败: {str(e)}'}), 400
    
    # 执行检测
    try:
        regions = pipeline.preview_detect(image_rgb)
    except Exception as e:
        return jsonify({'error': f'检测失败: {str(e)}'}), 500
    
    # 返回 base64 编码的原图
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('ascii')
    
    return jsonify({
        'image_size': [image.width, image.height],
        'image_base64': img_base64,
        'regions': regions,
        'region_count': len(regions)
    })


def _deprecated_manga_censor_apply():
    """交互式遮盖 - 根据用户选择的区域执行遮盖"""
    import base64
    import json as _json
    from io import BytesIO
    
    file = request.files.get('file')
    image_path = request.form.get('image_path', '').strip()
    selected_ids_json = request.form.get('selected_ids', '[]')
    all_regions_json = request.form.get('all_regions', '[]')
    output_path = request.form.get('output_path', '').strip()
    
    try:
        selected_ids = _json.loads(selected_ids_json)
        all_regions = _json.loads(all_regions_json)
    except Exception as e:
        return jsonify({'error': f'解析参数失败: {str(e)}'}), 400
    
    if not file and not image_path:
        return jsonify({'error': '请上传图片或提供图片路径'}), 400
    
    pipeline = _get_manga_censor_pipeline()
    
    # 读取图片
    try:
        if file:
            image = Image.open(file.stream).convert('RGB')
        else:
            if not os.path.exists(image_path):
                return jsonify({'error': f'图片不存在：{image_path}'}), 400
            image = Image.open(image_path).convert('RGB')
        image_rgb = np.array(image)
    except Exception as e:
        return jsonify({'error': f'读取图片失败: {str(e)}'}), 400
    
    # 执行遮盖
    try:
        output = pipeline.apply_selected_regions(image_rgb, selected_ids, all_regions)
    except Exception as e:
        return jsonify({'error': f'遮盖失败: {str(e)}'}), 500
    
    output_image = Image.fromarray(output)
    
    # 保存或返回 base64
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        output_image.save(output_path)
        return jsonify({
            'success': True,
            'output_path': output_path,
            'selected_count': len(selected_ids)
        })
    else:
        buffer = BytesIO()
        output_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('ascii')
        return jsonify({
            'success': True,
            'image_base64': img_base64,
            'selected_count': len(selected_ids)
        })


def _deprecated_manga_censor_batch():
    """批量审查漫画图片"""
    folder = request.form.get('folder', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    save_mask = _is_truthy(request.form.get('save_mask', '1'))
    save_report = _is_truthy(request.form.get('save_report', '1'))

    if not folder:
        return jsonify({'error': '缺少图片文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在：{folder}'}), 400

    if not output_folder:
        output_folder = folder + '_censored'

    os.makedirs(output_folder, exist_ok=True)
    reset_stop_flag('body_mask')

    image_files = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key
    )
    if not image_files:
        return jsonify({'error': '文件夹中没有找到图片文件'}), 400

    print(f'\n=== 漫画审查批量处理 ===')
    print(f'输入文件夹: {folder}')
    print(f'输出文件夹: {output_folder}')
    print(f'图片数: {len(image_files)}')

    pipeline = _get_manga_censor_pipeline()

    # 首次初始化
    try:
        init_status = pipeline.initialize()
        print(f'检测器状态: {init_status}')
    except Exception as e:
        return jsonify({'error': f'Pipeline 初始化失败: {str(e)}'}), 500

    results = {}
    success_count = 0
    failed_count = 0

    for fname in image_files:
        if is_stopped('body_mask'):
            results[fname] = '⏹️ 已停止'
            continue

        image_path = os.path.join(folder, fname)

        try:
            result = pipeline.process_image(
                image_path=image_path,
                output_dir=output_folder,
                save_mask=save_mask,
                save_report=save_report
            )

            region_count = len(result.get('regions', []))
            mask_ratio = 0
            if result.get('mask_path') and os.path.exists(result['mask_path']):
                from PIL import Image
                import numpy as np
                mask = np.array(Image.open(result['mask_path']))
                mask_ratio = float((mask > 127).mean())

            success_count += 1
            results[fname] = f'✅ 检测到 {region_count} 个区域 | mask={mask_ratio:.2%} | 耗时 {result.get("timing", {}).get("total", 0):.2f}s'
            print(f'  ✅ {fname}: {region_count} regions, {mask_ratio:.2%} mask')
        except Exception as e:
            failed_count += 1
            results[fname] = f'❌ 处理失败: {str(e)[:100]}'
            print(f'  ❌ {fname}: {e}')

    results['_summary'] = f'漫画审查完成 | 总数 {len(image_files)} | 成功 {success_count} | 失败 {failed_count}'
    print(f'=== 完成: {success_count}/{len(image_files)} ===\n')
    return jsonify(results)




# ===== 性别感知遮盖 Pipeline =====

_gender_aware_pipeline = None
_gender_pipeline_lock = threading.Lock()


def _get_gender_aware_pipeline():
    global _gender_aware_pipeline
    if _gender_aware_pipeline is None:
        with _gender_pipeline_lock:
            if _gender_aware_pipeline is None:
                import yaml
                config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mask_config.yaml')
                config = {}
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f) or {}
                    except Exception as e:
                        print(f'加载 mask_config.yaml 失败: {e}')
                
                # 加载所有可用的检测器
                detectors = {
                    'face': BboxSam2Detector('face', conf=0.25),
                    'eyes': AnzhcSegDetector('eyes', conf=0.35),
                    'hair': AnzhcSegDetector('hair', conf=0.5),
                    'hand': BboxSam2Detector('hand', conf=0.4),
                    'breasts': AnzhcSegDetector('breasts', conf=0.5),
                    'nsfw': NsfwSegDetector(conf=0.3, use_erax=True),
                }
                
                # 使用 DeepGHS 人物检测器（替代旧版 ONNX）
                # conf=0.5 减少假阳性，level=m 平衡速度/精度，max_det=15 限制检测数量
                _gender_aware_pipeline = GenderAwarePipeline(
                    detectors,
                    config_path,
                    person_detector_type='deepghs',
                    person_conf=0.5,
                    person_level='m',
                    person_version='v1.1',
                    person_iou=0.5,
                    person_max_det=15,
                )
    
    return _gender_aware_pipeline


@app.route('/gender_aware_pipeline_status', methods=['GET'])
def gender_aware_pipeline_status():
    pipeline = _get_gender_aware_pipeline()
    return jsonify({
        'initialized': pipeline._initialized,
        'detector_count': len(pipeline.detectors)
    })


@app.route('/gender_aware_pipeline_init', methods=['POST'])
def gender_aware_pipeline_init():
    pipeline = _get_gender_aware_pipeline()
    try:
        pipeline.initialize()
        return jsonify({'success': True, 'status': {
            'initialized': pipeline._initialized,
            'detector_count': len(pipeline.detectors)
        }})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/gender_aware_preview', methods=['POST'])
def gender_aware_preview():
    import base64 as b64
    from PIL import Image
    import numpy as np
    import cv2
    
    file = request.files.get('file')
    image_path = request.form.get('image_path', '').strip()
    
    if not file and not image_path:
        return jsonify({'error': '请上传图片或提供路径'}), 400
    
    pipeline = _get_gender_aware_pipeline()
    
    try:
        if not pipeline._initialized:
            pipeline.initialize()
    except Exception as e:
        return jsonify({'error': f'初始化失败: {str(e)}'}), 500
    
    try:
        if file:
            img = Image.open(file.stream).convert('RGB')
            image = np.array(img)[:, :, ::-1]
        else:
            from manga_censor.utils import cv2_imread
            image = cv2_imread(image_path)
            if image is None:
                return jsonify({'error': f'无法读取: {image_path}'}), 400
        
        h, w = image.shape[:2]
        result = pipeline.process(image)
        
        response = {
            'image_size': [w, h],
            'person_count': len(result.persons),
            'persons': [],
            'all_detected_parts': {},
            'final_mask_count': len(result.final_masks)
        }
        
        for person_result in result.person_results:
            response['persons'].append({
                'person_id': person_result.person_id,
                'gender': person_result.gender,
                'gender_confidence': person_result.gender_confidence,
                'parts': list(person_result.masks.keys()),
                'part_count': len(person_result.masks)
            })
        
        for part_name, detection in result.all_parts.items():
            mask_bytes = cv2.imencode('.png', detection.mask)[1].tobytes()
            response['all_detected_parts'][part_name] = {
                'mask_base64': b64.b64encode(mask_bytes).decode('ascii'),
                'count': detection.count,
                'confidence': round(detection.confidence, 4),
                'has_detection': detection.count > 0
            }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/gender_mask_batch', methods=['POST'])
def gender_mask_batch():
    return mask_batch()


@app.route('/save_gender_config', methods=['POST'])
def save_gender_config():
    """保存性别遮盖配置到 mask_config.yaml"""
    import yaml
    config_json = request.form.get('gender_config', '{}')
    try:
        gender_config = json.loads(config_json)
    except Exception:
        return jsonify({'success': False, 'error': '无效的 JSON 配置'}), 400
    
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mask_config.yaml')
    try:
        existing = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                existing = yaml.safe_load(f) or {}
        
        existing['gender_mask'] = gender_config
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(existing, f, allow_unicode=True, default_flow_style=False)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get_gender_config', methods=['GET'])
def get_gender_config():
    """读取性别遮盖配置"""
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mask_config.yaml')
    try:
        if not os.path.exists(config_path):
            return jsonify({'error': '配置文件不存在'})
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        return jsonify(config.get('gender_mask', {}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== 手动遮盖 API（三层 mask 系统） =====

def _get_manual_mask_dir(image_folder, output_folder=None):
    """获取手动遮盖的输出目录"""
    if output_folder:
        return output_folder
    return image_folder + '_manual_masks'


def _find_auto_mask(image_path, auto_mask_dir):
    """从 auto_mask_dir 查找对应的 auto mask 文件"""
    if not auto_mask_dir or not os.path.isdir(auto_mask_dir):
        return None
    base = os.path.splitext(os.path.basename(image_path))[0]
    # 尝试多种命名格式
    candidates = [
        os.path.join(auto_mask_dir, base + '.png'),
        os.path.join(auto_mask_dir, base + '_mask.png'),
        os.path.join(auto_mask_dir, base + '_merged.png'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


@app.route('/manual_mask/list', methods=['POST'])
def manual_mask_list():
    """列出文件夹中的图片及三层 mask 状态"""
    folder = request.form.get('folder', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    auto_mask_dir = request.form.get('auto_mask_dir', '').strip()

    if not folder or not os.path.isdir(folder):
        return jsonify({'error': '文件夹不存在'}), 400

    exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    images = []
    for fname in sorted(os.listdir(folder), key=natural_sort_key):
        if os.path.splitext(fname)[1].lower() in exts:
            images.append(fname)

    mask_dir = _get_manual_mask_dir(folder, output_folder)
    has_mask = {}
    layer_status = {}

    for fname in images:
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(folder, fname)

        # 检查三层 mask 状态
        layers = {}
        # auto 层：优先从 auto_mask_dir 查找，其次从 mask_dir
        auto_path = _find_auto_mask(img_path, auto_mask_dir)
        if not auto_path:
            auto_path_default = os.path.join(mask_dir, base + '_auto.png')
            if os.path.isfile(auto_path_default):
                auto_path = auto_path_default
        layers['auto'] = auto_path is not None

        # manual 层
        manual_path = os.path.join(mask_dir, base + '_manual.png')
        layers['manual'] = os.path.isfile(manual_path)

        # inverse 层
        inverse_path = os.path.join(mask_dir, base + '_inverse.png')
        layers['inverse'] = os.path.isfile(inverse_path)

        # final 层
        final_path = os.path.join(mask_dir, base + '_final.png')
        layers['final'] = os.path.isfile(final_path)

        layer_status[fname] = layers
        # 兼容旧接口
        has_mask[fname] = layers['manual'] or layers['auto'] or layers['final']

    return jsonify({
        'folder': folder,
        'images': images,
        'has_mask': has_mask,
        'layer_status': layer_status,
        'mask_dir': mask_dir,
        'auto_mask_dir': auto_mask_dir
    })


@app.route('/manual_mask/image', methods=['GET'])
def manual_mask_image():
    """返回原图"""
    path = request.args.get('path', '').strip()
    if not path or not os.path.isfile(path):
        return jsonify({'error': '文件不存在'}), 404
    ext = os.path.splitext(path)[1].lower()
    if ext not in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}:
        return jsonify({'error': '不是图片文件'}), 400
    mime = mimetypes.guess_type(path)[0] or 'application/octet-stream'
    return send_file(path, mimetype=mime)


@app.route('/manual_mask/load', methods=['GET'])
def manual_mask_load():
    """返回指定层的 mask（支持 layer 参数：auto/manual/inverse/final）"""
    path = request.args.get('path', '').strip()
    layer = request.args.get('layer', 'manual').strip()
    output_folder = request.args.get('output_folder', '').strip()
    auto_mask_dir = request.args.get('auto_mask_dir', '').strip()

    if not path:
        return jsonify({'error': '路径为空'}), 400

    fname = os.path.basename(path)
    base = os.path.splitext(fname)[0]
    folder = os.path.dirname(path)
    mask_dir = _get_manual_mask_dir(folder, output_folder)

    mask_path = None

    if layer == 'auto':
        # 优先从 auto_mask_dir 查找
        mask_path = _find_auto_mask(path, auto_mask_dir)
        if not mask_path:
            mask_path = os.path.join(mask_dir, base + '_auto.png')
    elif layer == 'manual':
        mask_path = os.path.join(mask_dir, base + '_manual.png')
    elif layer == 'inverse':
        mask_path = os.path.join(mask_dir, base + '_inverse.png')
    elif layer == 'final':
        mask_path = os.path.join(mask_dir, base + '_final.png')
    else:
        return jsonify({'error': f'无效的层类型: {layer}'}), 400

    if mask_path and os.path.isfile(mask_path):
        return send_file(mask_path, mimetype='image/png')
    return jsonify({'error': f'无 {layer} mask'}), 404


@app.route('/manual_mask/save', methods=['POST'])
def manual_mask_save():
    """保存指定层的 mask（支持 layer_type 参数：manual/inverse）"""
    image_path = request.form.get('image_path', '').strip()
    mask_b64 = request.form.get('mask_base64', '').strip()
    layer_type = request.form.get('layer_type', '').strip()
    inverted = _is_truthy(request.form.get('inverted', '0'))
    output_folder = request.form.get('output_folder', '').strip()

    if not image_path or not mask_b64:
        return jsonify({'error': '缺少参数'}), 400

    import base64 as b64_mod
    from PIL import Image
    import io

    # 解码 mask PNG
    try:
        raw = mask_b64.split(',')[1] if ',' in mask_b64 else mask_b64
        mask_data = b64_mod.b64decode(raw)
        mask_img = Image.open(io.BytesIO(mask_data)).convert('L')
    except Exception as e:
        return jsonify({'error': f'Mask 解码失败: {e}'}), 400

    fname = os.path.basename(image_path)
    base = os.path.splitext(fname)[0]
    folder = os.path.dirname(image_path)
    mask_dir = _get_manual_mask_dir(folder, output_folder)
    os.makedirs(mask_dir, exist_ok=True)

    # 新三层系统：根据 layer_type 保存
    if layer_type in ('manual', 'inverse'):
        save_path = os.path.join(mask_dir, f'{base}_{layer_type}.png')
        mask_img.save(save_path)
        return jsonify({'success': True, 'saved': save_path, 'mode': layer_type})

    # 兼容旧接口：inverted 参数
    if inverted:
        inv = Image.eval(mask_img, lambda p: 255 - p)
        save_path = os.path.join(mask_dir, base + '_manual.png')
        inv.save(save_path)
        return jsonify({'success': True, 'saved': save_path, 'mode': 'inverted'})
    else:
        save_path = os.path.join(mask_dir, base + '_manual.png')
        mask_img.save(save_path)
        return jsonify({'success': True, 'saved': save_path, 'mode': 'normal'})


@app.route('/manual_mask/merge', methods=['POST'])
def manual_mask_merge():
    """合并三层 mask 生成 final mask: final = auto + manual - inverse
    
    支持两种模式：
    1. 从磁盘读取图层文件（旧模式，用于合并预览按钮）
    2. 直接接收 base64 数据（新模式，用于 Ctrl+S/X 快捷键合并保存，不产生中间文件）
    """
    image_path = request.form.get('image_path', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    auto_mask_dir = request.form.get('auto_mask_dir', '').strip()
    invert = _is_truthy(request.form.get('invert', '0'))
    # 新增：直接接收 base64 图层数据（可选）
    manual_base64 = request.form.get('manual_base64', '').strip()
    inverse_base64 = request.form.get('inverse_base64', '').strip()

    if not image_path:
        return jsonify({'error': '缺少图片路径'}), 400

    import numpy as np
    from PIL import Image
    import io as _io
    import base64 as b64_mod

    fname = os.path.basename(image_path)
    base = os.path.splitext(fname)[0]
    folder = os.path.dirname(image_path)
    mask_dir = _get_manual_mask_dir(folder, output_folder)

    # 获取图片尺寸
    try:
        orig_img = Image.open(image_path)
        w, h = orig_img.size
    except Exception as e:
        return jsonify({'error': f'无法读取图片: {e}'}), 400

    def decode_base64_mask(b64_str, target_w, target_h):
        """从 base64 解码 mask 图层"""
        raw = b64_str.split(',')[1] if ',' in b64_str else b64_str
        data = b64_mod.b64decode(raw)
        img = Image.open(_io.BytesIO(data)).convert('L').resize((target_w, target_h), Image.NEAREST)
        return np.array(img, dtype=np.float32) / 255.0

    # 加载三层
    def load_layer(layer_name):
        if layer_name == 'auto':
            p = _find_auto_mask(image_path, auto_mask_dir)
            if not p:
                p = os.path.join(mask_dir, base + '_auto.png')
        else:
            p = os.path.join(mask_dir, f'{base}_{layer_name}.png')
        if p and os.path.isfile(p):
            img = Image.open(p).convert('L').resize((w, h), Image.NEAREST)
            return np.array(img, dtype=np.float32) / 255.0
        return np.zeros((h, w), dtype=np.float32)

    auto = load_layer('auto')
    # 如果前端直接传了 base64 数据，优先使用（不读磁盘文件）
    manual = decode_base64_mask(manual_base64, w, h) if manual_base64 else load_layer('manual')
    inverse = decode_base64_mask(inverse_base64, w, h) if inverse_base64 else load_layer('inverse')

    # 合并: 正常 = auto + manual - inverse, 反向 = auto - manual + inverse
    if invert:
        final = np.clip(auto - manual + inverse, 0.0, 1.0)
    else:
        final = np.clip(auto + manual - inverse, 0.0, 1.0)

    final_uint8 = (final * 255).astype(np.uint8)

    # 保存 final mask — 只输出一个 {原名}.png
    os.makedirs(mask_dir, exist_ok=True)
    final_path = os.path.join(mask_dir, base + '.png')
    Image.fromarray(final_uint8, 'L').save(final_path)

    mask_ratio = float(np.mean(final_uint8 > 127))

    return jsonify({
        'success': True,
        'final_path': final_path,
        'mask_ratio': round(mask_ratio, 4),
        'inverted': invert
    })


# ===== 分层 Mask 编辑器 API =====

_mask_editor = None
_mask_editor_lock = threading.Lock()


def _get_mask_editor():
    global _mask_editor
    if _mask_editor is None:
        with _mask_editor_lock:
            if _mask_editor is None:
                from manga_censor.mask_editor import MaskEditor
                _mask_editor = MaskEditor()
    return _mask_editor


@app.route('/editor/load', methods=['POST'])
def editor_load():
    """加载编辑器数据（图片 + 自动检测 mask）"""
    data = request.json or request.form
    image_path = data.get('image_path', '').strip()
    
    if not image_path:
        return jsonify({'error': '缺少图片路径'}), 400
    if not os.path.exists(image_path):
        return jsonify({'error': f'图片不存在: {image_path}'}), 404
    
    editor = _get_mask_editor()
    
    try:
        # 检查是否需要运行自动检测
        auto_mask = None
        if not editor.path_manager.mask_exists(image_path, 'auto'):
            # 运行自动检测
            pipeline = _get_mask_pipeline()
            if not pipeline._initialized:
                pipeline.initialize()
            
            import cv2
            from manga_censor.utils import cv2_imread
            image = cv2_imread(image_path)
            if image is None:
                return jsonify({'error': f'无法读取图片: {image_path}'}), 400
            
            # 执行检测并合并所有部位
            import numpy as np
            h, w = image.shape[:2]
            auto_mask = np.zeros((h, w), dtype=np.uint8)
            
            for name, detector in pipeline.detectors.items():
                result = detector.detect(image)
                if result.count > 0:
                    auto_mask = np.maximum(auto_mask, result.mask)
        
        # 加载所有 mask
        masks = editor.load_editor_data(image_path, auto_mask)
        
        # 生成图片 URL
        image_filename = os.path.basename(image_path)
        
        return jsonify({
            'success': True,
            'image_path': image_path,
            'image_filename': image_filename,
            'masks': masks
        })
    
    except Exception as e:
        import traceback
        print(f'编辑器加载失败: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/editor/save_layer', methods=['POST'])
def editor_save_layer():
    """保存单个图层"""
    data = request.json or request.form
    image_path = data.get('image_path', '').strip()
    layer_type = data.get('layer_type', '').strip()
    mask_data = data.get('mask_data', '').strip()
    
    if not image_path or not layer_type or not mask_data:
        return jsonify({'error': '缺少必要参数'}), 400
    
    if layer_type not in ['manual', 'inverse']:
        return jsonify({'error': f'无效的图层类型: {layer_type}'}), 400
    
    editor = _get_mask_editor()
    
    try:
        path = editor.save_layer(image_path, layer_type, mask_data)
        return jsonify({
            'success': True,
            'path': path,
            'layer_type': layer_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/editor/merge', methods=['POST'])
def editor_merge():
    """合并所有图层并生成最终 mask"""
    data = request.json or request.form
    image_path = data.get('image_path', '').strip()
    mode = data.get('mode', 'standard').strip()
    
    if not image_path:
        return jsonify({'error': '缺少图片路径'}), 400
    
    editor = _get_mask_editor()
    
    try:
        result = editor.merge_and_preview(image_path, mode)
        return jsonify({
            'success': True,
            **result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/editor/get_image', methods=['GET'])
def editor_get_image():
    """获取图片文件"""
    path = request.args.get('path', '').strip()
    if not path or not os.path.isfile(path):
        return jsonify({'error': '文件不存在'}), 404
    
    mime = mimetypes.guess_type(path)[0] or 'image/png'
    return send_file(path, mimetype=mime)


@app.route('/editor/get_mask', methods=['GET'])
def editor_get_mask():
    """获取 mask 文件"""
    path = request.args.get('path', '').strip()
    if not path or not os.path.isfile(path):
        return jsonify({'error': 'Mask 文件不存在'}), 404
    
    return send_file(path, mimetype='image/png')


# ===== SAM2 Point Prompt API =====

_sam2_refiner = None
_sam2_lock = threading.Lock()


def _get_sam2_refiner():
    global _sam2_refiner
    if _sam2_refiner is None:
        with _sam2_lock:
            if _sam2_refiner is None:
                _sam2_refiner = SAM2Refiner()
    return _sam2_refiner


@app.route('/editor/list_images', methods=['POST'])
def editor_list_images():
    """列出文件夹中的所有图片，支持自然排序"""
    data = request.json or request.form
    folder = data.get('folder', '').strip()
    
    if not folder:
        return jsonify({'error': '缺少文件夹路径'}), 400
    if not os.path.exists(folder):
        return jsonify({'error': f'文件夹不存在: {folder}'}), 404
    if not os.path.isdir(folder):
        return jsonify({'error': f'路径不是文件夹: {folder}'}), 400
    
    exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
    images = []
    
    for fname in sorted(os.listdir(folder), key=natural_sort_key):
        ext = os.path.splitext(fname)[1].lower()
        if ext in exts:
            # 检查是否有对应的 txt 文件
            base = os.path.splitext(fname)[0]
            has_txt = os.path.isfile(os.path.join(folder, base + '.txt'))
            images.append({
                'filename': fname,
                'has_txt': has_txt
            })
    
    return jsonify({
        'success': True,
        'folder': folder,
        'images': images,
        'total': len(images)
    })


@app.route('/editor/sam2_point', methods=['POST'])
def editor_sam2_point():
    """SAM2 点提示生成 mask"""
    data = request.json or request.form
    image_path = data.get('image_path', '').strip()
    points_raw = data.get('points', '[]')
    labels_raw = data.get('labels', '')
    model_name = data.get('model_name', '').strip() or None
    
    if not image_path:
        return jsonify({'error': '缺少图片路径'}), 400
    if not os.path.exists(image_path):
        return jsonify({'error': f'图片不存在: {image_path}'}), 404
    
    # 解析 points
    try:
        if isinstance(points_raw, str):
            points = json.loads(points_raw)
        else:
            points = points_raw
        if not isinstance(points, list) or len(points) == 0:
            return jsonify({'error': 'points 必须是非空数组'}), 400
        # 验证格式
        for p in points:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                return jsonify({'error': '每个 point 必须是 [x, y] 格式'}), 400
    except Exception as e:
        return jsonify({'error': f'points 解析失败: {e}'}), 400
    
    # 解析 labels
    labels = None
    if labels_raw:
        try:
            if isinstance(labels_raw, str):
                labels = json.loads(labels_raw)
            else:
                labels = labels_raw
            if not isinstance(labels, list):
                labels = None
        except Exception:
            labels = None
    
    import cv2
    from manga_censor.utils import cv2_imread
    import numpy as np
    import base64 as b64_mod
    from PIL import Image
    import io
    
    image = cv2_imread(image_path)
    if image is None:
        return jsonify({'error': f'无法读取图片: {image_path}'}), 400
    
    try:
        refiner = _get_sam2_refiner()
        mask = refiner.predict_from_points(image, points, labels, model_name)
        
        # 编码为 base64 PNG
        _, buf = cv2.imencode('.png', mask)
        mask_b64 = b64_mod.b64encode(buf.tobytes()).decode('ascii')
        
        return jsonify({
            'success': True,
            'mask_base64': f'data:image/png;base64,{mask_b64}',
            'model_used': model_name or refiner.model_name,
            'points_count': len(points)
        })
    except Exception as e:
        import traceback
        print(f'SAM2 point 推理失败: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/editor/sam2_bbox', methods=['POST'])
def editor_sam2_bbox():
    """SAM2 矩形框提示生成 mask
    
    支持两种模式:
    1. 仅 bbox: {"bbox": [x1, y1, x2, y2]}
    2. bbox + points 混合: {"bbox": [x1, y1, x2, y2], "points": [...], "labels": [...]}
    """
    data = request.json or request.form
    image_path = data.get('image_path', '').strip()
    bbox_raw = data.get('bbox', '')
    points_raw = data.get('points', '[]')
    labels_raw = data.get('labels', '')
    model_name = data.get('model_name', '').strip() or None
    
    if not image_path:
        return jsonify({'error': '缺少图片路径'}), 400
    if not os.path.exists(image_path):
        return jsonify({'error': f'图片不存在: {image_path}'}), 404
    
    # 解析 bbox
    try:
        if isinstance(bbox_raw, str):
            bbox = json.loads(bbox_raw)
        else:
            bbox = bbox_raw
        
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return jsonify({'error': 'bbox 必须是 [x1, y1, x2, y2] 格式'}), 400
        
        # 确保坐标是整数
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        if x2 <= x1 or y2 <= y1:
            return jsonify({'error': f'无效的 bbox: [{x1}, {y1}, {x2}, {y2}]'}), 400
            
    except Exception as e:
        return jsonify({'error': f'bbox 解析失败: {e}'}), 400
    
    # 解析 points（可选，混合模式）
    points = None
    labels = None
    if points_raw:
        try:
            if isinstance(points_raw, str):
                points = json.loads(points_raw)
            else:
                points = points_raw
            if isinstance(points, list) and len(points) > 0:
                for p in points:
                    if not isinstance(p, (list, tuple)) or len(p) != 2:
                        return jsonify({'error': '每个 point 必须是 [x, y] 格式'}), 400
        except Exception as e:
            return jsonify({'error': f'points 解析失败: {e}'}), 400
        
        # 解析 labels
        if labels_raw:
            try:
                if isinstance(labels_raw, str):
                    labels = json.loads(labels_raw)
                else:
                    labels = labels_raw
                if not isinstance(labels, list):
                    labels = None
            except Exception:
                labels = None
    
    import cv2
    from manga_censor.utils import cv2_imread
    import numpy as np
    import base64 as b64_mod
    
    image = cv2_imread(image_path)
    if image is None:
        return jsonify({'error': f'无法读取图片: {image_path}'}), 400
    
    try:
        refiner = _get_sam2_refiner()
        
        # 根据是否有 points 选择不同的方法
        if points and len(points) > 0:
            # 混合模式: bbox + points
            mask = refiner.predict_from_bbox_and_points(
                image, 
                (x1, y1, x2, y2), 
                points, 
                labels, 
                model_name
            )
            mode = 'bbox+points'
        else:
            # 仅 bbox 模式
            mask = refiner.predict_from_bbox(image, (x1, y1, x2, y2), model_name)
            mode = 'bbox'
        
        # 编码为 base64 PNG
        _, buf = cv2.imencode('.png', mask)
        mask_b64 = b64_mod.b64encode(buf.tobytes()).decode('ascii')
        
        return jsonify({
            'success': True,
            'mask_base64': f'data:image/png;base64,{mask_b64}',
            'model_used': model_name or refiner.model_name,
            'mode': mode,
            'bbox': [x1, y1, x2, y2],
            'points_count': len(points) if points else 0
        })
    except Exception as e:
        import traceback
        print(f'SAM2 bbox 推理失败: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/editor/sam2_models', methods=['GET'])
def editor_sam2_models():
    """获取可用的 SAM2 模型列表"""
    try:
        refiner = _get_sam2_refiner()
        models = refiner.get_available_models()
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== 高级 Mask 编辑器 API =====

@app.route('/advanced_mask_editor')
def advanced_mask_editor():
    """高级 Mask 编辑器页面"""
    return render_template('advanced_mask_editor.html')


@app.route('/api/load_mask', methods=['POST'])
def api_load_mask():
    """加载现有 mask"""
    data = request.json
    image_path = data.get('image_path', '').strip()
    mask_type = data.get('mask_type', 'manual').strip()
    
    if not image_path:
        return jsonify({'error': '缺少图片路径'}), 400
    
    # 构造 mask 路径
    folder = os.path.dirname(image_path) or '.'
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(folder + '_masks', base_name + f'_{mask_type}.png')
    
    if not os.path.exists(mask_path):
        return jsonify({'mask_base64': None})
    
    try:
        import base64
        with open(mask_path, 'rb') as f:
            mask_data = f.read()
        mask_base64 = base64.b64encode(mask_data).decode('ascii')
        return jsonify({'mask_base64': mask_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/save_mask', methods=['POST'])
def api_save_mask():
    """保存 mask"""
    data = request.json
    image_path = data.get('image_path', '').strip()
    mask_type = data.get('mask_type', 'manual').strip()
    mask_base64 = data.get('mask_base64', '').strip()
    
    if not image_path or not mask_base64:
        return jsonify({'error': '缺少必要参数'}), 400
    
    try:
        import base64
        from PIL import Image
        import io
        
        # 解码 base64
        mask_data = base64.b64decode(mask_base64)
        mask_img = Image.open(io.BytesIO(mask_data)).convert('L')
        
        # 构造保存路径
        folder = os.path.dirname(image_path) or '.'
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = folder + '_masks'
        os.makedirs(output_dir, exist_ok=True)
        
        mask_path = os.path.join(output_dir, base_name + f'_{mask_type}.png')
        mask_img.save(mask_path)
        
        return jsonify({'success': True, 'path': mask_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== x-anylabeling 双向转换 API =====

@app.route('/xanylabeling/export', methods=['POST'])
def xanylabeling_export():
    """
    批量将 PNG Mask 导出为 x-anylabeling JSON 格式
    
    参数:
        mask_dir: Mask 文件目录
        image_dir: 原始图片目录（可选，默认与 mask_dir 相同）
        output_dir: 输出目录（可选，默认覆盖 mask 目录）
        label: 标注标签（默认 "mask"）
        min_area: 最小轮廓面积（默认 50）
        overwrite: 是否覆盖已有 JSON（默认 false）
    """
    from xanylabeling_converter import batch_mask_to_xanylabeling
    
    mask_dir = request.form.get('mask_dir', '').strip()
    image_dir = request.form.get('image_dir', '').strip()
    output_dir = request.form.get('output_dir', '').strip()
    label = request.form.get('label', 'mask').strip()
    min_area = int(request.form.get('min_area', '50'))
    overwrite = _is_truthy(request.form.get('overwrite', '0'))
    
    if not mask_dir:
        return jsonify({'error': '缺少 mask 目录'}), 400
    if not os.path.exists(mask_dir):
        return jsonify({'error': f'Mask 目录不存在: {mask_dir}'}), 400
    if not os.path.isdir(mask_dir):
        return jsonify({'error': f'路径不是目录: {mask_dir}'}), 400
    
    if image_dir and not os.path.exists(image_dir):
        return jsonify({'error': f'图片目录不存在: {image_dir}'}), 400
    
    if not output_dir:
        output_dir = mask_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否覆盖已有文件
    if not overwrite:
        from glob import glob
        existing = glob(os.path.join(output_dir, '*.json'))
        if existing:
            return jsonify({
                'error': f'输出目录中已有 {len(existing)} 个 JSON 文件',
                'hint': '勾选"覆盖已有文件"选项以跳过此检查'
            }), 400
    
    print('\n=== x-anylabeling 批量导出 ===')
    print(f'Mask 目录: {mask_dir}')
    print(f'图片目录: {image_dir or mask_dir}')
    print(f'输出目录: {output_dir}')
    print(f'标注标签: {label}')
    print(f'最小轮廓面积: {min_area}')
    
    try:
        result = batch_mask_to_xanylabeling(
            mask_dir=mask_dir,
            image_dir=image_dir if image_dir else None,
            output_dir=output_dir,
            label=label,
            min_area=min_area
        )
        print(f'\n{result["summary"]}')
        return jsonify(result)
    except Exception as e:
        import traceback
        print(f'导出失败: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/xanylabeling/import', methods=['POST'])
def xanylabeling_import():
    """
    批量将 x-anylabeling JSON 导入为 PNG Mask
    
    参数:
        json_dir: JSON 文件目录
        output_dir: 输出目录（可选，默认覆盖 JSON 目录）
        fill_value: 填充值（默认 255）
        overwrite: 是否覆盖已有 Mask（默认 false）
    """
    from xanylabeling_converter import batch_xanylabeling_to_mask
    
    json_dir = request.form.get('json_dir', '').strip()
    output_dir = request.form.get('output_dir', '').strip()
    fill_value = int(request.form.get('fill_value', '255'))
    overwrite = _is_truthy(request.form.get('overwrite', '0'))
    
    if not json_dir:
        return jsonify({'error': '缺少 JSON 目录'}), 400
    if not os.path.exists(json_dir):
        return jsonify({'error': f'JSON 目录不存在: {json_dir}'}), 400
    if not os.path.isdir(json_dir):
        return jsonify({'error': f'路径不是目录: {json_dir}'}), 400
    
    if not output_dir:
        output_dir = json_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否覆盖已有文件
    if not overwrite:
        from glob import glob
        existing = glob(os.path.join(output_dir, '*_mask.png'))
        if existing:
            return jsonify({
                'error': f'输出目录中已有 {len(existing)} 个 Mask 文件',
                'hint': '勾选"覆盖已有文件"选项以跳过此检查'
            }), 400
    
    print('\n=== x-anylabeling 批量导入 ===')
    print(f'JSON 目录: {json_dir}')
    print(f'输出目录: {output_dir}')
    print(f'填充值: {fill_value}')
    
    try:
        result = batch_xanylabeling_to_mask(
            json_dir=json_dir,
            output_dir=output_dir,
            fill_value=fill_value
        )
        print(f'\n{result["summary"]}')
        return jsonify(result)
    except Exception as e:
        import traceback
        print(f'导入失败: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/xanylabeling/single_export', methods=['POST'])
def xanylabeling_single_export():
    """
    单个 mask 文件转换为 x-anylabeling JSON
    
    参数:
        mask_path: Mask 文件路径
        image_path: 对应图片路径
        output_path: 输出 JSON 路径（可选）
        label: 标注标签（默认 "mask"）
        min_area: 最小轮廓面积（默认 50）
    """
    from xanylabeling_converter import mask_to_xanylabeling
    
    mask_path = request.form.get('mask_path', '').strip()
    image_path = request.form.get('image_path', '').strip()
    output_path = request.form.get('output_path', '').strip() or None
    label = request.form.get('label', 'mask').strip()
    min_area = int(request.form.get('min_area', '50'))
    
    if not mask_path:
        return jsonify({'error': '缺少 mask 路径'}), 400
    if not os.path.exists(mask_path):
        return jsonify({'error': f'Mask 文件不存在: {mask_path}'}), 400
    
    print(f'\n=== x-anylabeling 单个导出 ===')
    print(f'Mask: {mask_path}')
    print(f'图片: {image_path or "使用 mask 尺寸"}')
    print(f'标签: {label}')
    
    try:
        result = mask_to_xanylabeling(
            mask_path=mask_path,
            image_path=image_path if image_path else mask_path,
            output_path=output_path,
            label=label,
            min_area=min_area
        )
        print(f'✅ 导出成功: {result["json_path"]}')
        print(f'   标注数量: {result["shapes_count"]}')
        return jsonify(result)
    except Exception as e:
        import traceback
        print(f'导出失败: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/xanylabeling/single_import', methods=['POST'])
def xanylabeling_single_import():
    """
    单个 x-anylabeling JSON 转换为 mask PNG
    
    参数:
        json_path: JSON 文件路径
        output_path: 输出 Mask 路径（可选）
        fill_value: 填充值（默认 255）
    """
    from xanylabeling_converter import xanylabeling_to_mask
    
    json_path = request.form.get('json_path', '').strip()
    output_path = request.form.get('output_path', '').strip() or None
    fill_value = int(request.form.get('fill_value', '255'))
    
    if not json_path:
        return jsonify({'error': '缺少 JSON 路径'}), 400
    if not os.path.exists(json_path):
        return jsonify({'error': f'JSON 文件不存在: {json_path}'}), 400
    
    print(f'\n=== x-anylabeling 单个导入 ===')
    print(f'JSON: {json_path}')
    print(f'输出: {output_path or "默认位置"}')
    print(f'填充值: {fill_value}')
    
    try:
        result = xanylabeling_to_mask(
            json_path=json_path,
            output_path=output_path,
            fill_value=fill_value
        )
        print(f'✅ 导入成功: {result["mask_path"]}')
        print(f'   标注数量: {result["shapes_count"]}')
        print(f'   标签: {result["labels"]}')
        return jsonify(result)
    except Exception as e:
        import traceback
        print(f'导入失败: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/xanylabeling/merge', methods=['POST'])
def xanylabeling_merge():
    """
    合并 x-anylabeling JSON 标注与现有 mask
    
    参数:
        json_path: x-anylabeling JSON 路径
        mask_path: 现有 mask 路径（可选）
        output_path: 输出路径
        operation: 合并操作 ('union', 'intersection', 'json_only', 'mask_only')
    """
    from xanylabeling_converter import merge_xanylabeling_with_mask
    
    json_path = request.form.get('json_path', '').strip()
    mask_path = request.form.get('mask_path', '').strip() or None
    output_path = request.form.get('output_path', '').strip()
    operation = request.form.get('operation', 'union').strip()
    
    if not json_path:
        return jsonify({'error': '缺少 JSON 路径'}), 400
    if not os.path.exists(json_path):
        return jsonify({'error': f'JSON 文件不存在: {json_path}'}), 400
    if not output_path:
        return jsonify({'error': '缺少输出路径'}), 400
    
    print(f'\n=== x-anylabeling 合并 ===')
    print(f'JSON: {json_path}')
    print(f'现有 Mask: {mask_path or "无"}')
    print(f'输出: {output_path}')
    print(f'操作: {operation}')
    
    try:
        result = merge_xanylabeling_with_mask(
            json_path=json_path,
            mask_path=mask_path,
            output_path=output_path,
            operation=operation
        )
        print(f'✅ 合并成功: {result["output_path"]}')
        return jsonify(result)
    except Exception as e:
        import traceback
        print(f'合并失败: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/xanylabeling/preview', methods=['POST'])
def xanylabeling_preview():
    """
    预览 x-anylabeling JSON 标注（返回标注信息，不生成文件）
    
    参数:
        json_path: JSON 文件路径
    """
    import base64 as b64_mod
    import cv2
    from io import BytesIO
    
    json_path = request.form.get('json_path', '').strip()
    
    if not json_path:
        return jsonify({'error': '缺少 JSON 路径'}), 400
    if not os.path.exists(json_path):
        return jsonify({'error': f'JSON 文件不存在: {json_path}'}), 400
    
    try:
        # 读取 JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        shapes = data.get('shapes', [])
        height = data.get('imageHeight', 0)
        width = data.get('imageWidth', 0)
        image_path = data.get('imagePath', '')
        
        # 提取标注信息
        shape_info = []
        for i, shape in enumerate(shapes):
            shape_info.append({
                'index': i,
                'label': shape.get('label', 'unknown'),
                'shape_type': shape.get('shape_type', 'unknown'),
                'points_count': len(shape.get('points', [])),
                'group_id': shape.get('group_id')
            })
        
        # 生成预览 mask
        from xanylabeling_converter import xanylabeling_to_mask
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_mask_path = tmp.name
        
        result = xanylabeling_to_mask(json_path, temp_mask_path)
        mask = cv2.imread(temp_mask_path, cv2.IMREAD_GRAYSCALE)
        os.unlink(temp_mask_path)
        
        # 编码为 base64
        _, buf = cv2.imencode('.png', mask)
        mask_b64 = b64_mod.b64encode(buf.tobytes()).decode('ascii')
        
        # 统计信息
        unique_labels = list(set(s.get('label', 'unknown') for s in shapes))
        
        return jsonify({
            'success': True,
            'json_path': json_path,
            'image_path': image_path,
            'image_size': {'width': width, 'height': height},
            'shapes_count': len(shapes),
            'unique_labels': unique_labels,
            'shape_info': shape_info,
            'mask_preview': f'data:image/png;base64,{mask_b64}',
            'mask_ratio': round(float((mask > 127).mean()), 4)
        })
        
    except Exception as e:
        import traceback
        print(f'预览失败: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ===== 自定义本地模型 Mask 生成 =====

_custom_model_cache = {}  # model_path -> CustomModelDetector


def _get_custom_model_detector(model_path: str, conf: float = 0.25, target_classes: list | None = None):
    """获取或创建自定义模型检测器（带缓存）。"""
    cache_key = f"{model_path}|{conf}|{tuple(target_classes or [])}"
    if cache_key in _custom_model_cache:
        return _custom_model_cache[cache_key]

    detector = CustomModelDetector(
        model_path=model_path,
        conf=conf,
        target_classes=target_classes,
        part_name="custom"
    )
    detector.load_model()
    _custom_model_cache[cache_key] = detector
    return detector


@app.route('/custom_model/load', methods=['POST'])
def custom_model_load():
    """加载自定义模型并返回类别信息。"""
    model_path = request.form.get('model_path', '').strip()
    conf_str = request.form.get('conf', '0.25').strip()

    if not model_path:
        return jsonify({'error': '缺少模型路径'}), 400
    if not os.path.isfile(model_path):
        return jsonify({'error': f'模型文件不存在: {model_path}'}), 400

    try:
        conf = float(conf_str)
    except Exception:
        conf = 0.25

    try:
        detector = _get_custom_model_detector(model_path, conf)
        class_names = detector.class_names or []
        return jsonify({
            'success': True,
            'model_path': model_path,
            'class_names': class_names,
            'class_count': len(class_names),
        })
    except Exception as e:
        import traceback
        print(f'自定义模型加载失败: {e}')
        print(traceback.format_exc())
        return jsonify({'error': f'模型加载失败: {str(e)}'}), 500


@app.route('/custom_model/batch', methods=['POST'])
def custom_model_batch():
    """使用自定义模型批量生成 Mask。"""
    import cv2
    import numpy as np

    model_path = request.form.get('model_path', '').strip()
    folder = request.form.get('folder', '').strip()
    output_folder = request.form.get('output_folder', '').strip()
    conf_str = request.form.get('conf', '0.25').strip()
    target_classes_json = request.form.get('target_classes', '[]').strip()
    invert = _is_truthy(request.form.get('invert', '0'))

    if not model_path:
        return jsonify({'error': '缺少模型路径'}), 400
    if not os.path.isfile(model_path):
        return jsonify({'error': f'模型文件不存在: {model_path}'}), 400
    if not folder:
        return jsonify({'error': '缺少图片文件夹路径'}), 400
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify({'error': f'文件夹不存在：{folder}'}), 400

    try:
        conf = float(conf_str)
    except Exception:
        conf = 0.25

    target_classes = None
    try:
        target_classes = json.loads(target_classes_json)
        if not isinstance(target_classes, list):
            target_classes = None
        else:
            target_classes = [int(c) for c in target_classes]
    except Exception:
        target_classes = None

    if not output_folder:
        output_folder = folder + '_custom_masks'

    os.makedirs(output_folder, exist_ok=True)
    reset_stop_flag('body_mask')

    image_files = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key
    )
    if not image_files:
        return jsonify({'error': '文件夹中没有找到图片文件'}), 400

    print(f'\n=== 自定义模型 Mask 批量生成 ===')
    print(f'模型: {model_path}')
    print(f'输入: {folder}')
    print(f'输出: {output_folder}')
    print(f'置信度: {conf}')
    print(f'目标类别: {target_classes}')
    print(f'图片数: {len(image_files)}')

    try:
        detector = _get_custom_model_detector(model_path, conf, target_classes)
    except Exception as e:
        return jsonify({'error': f'模型加载失败: {str(e)}'}), 500

    results = {}
    success_count = 0
    failed_count = 0

    for fname in image_files:
        if is_stopped('body_mask'):
            results[fname] = '⏹️ 已停止'
            continue

        image_path = os.path.join(folder, fname)
        base_name = os.path.splitext(fname)[0]
        mask_path = os.path.join(output_folder, base_name + '_mask.png')

        try:
            image = cv2.imread(image_path)
            if image is None:
                results[fname] = '❌ 无法读取图片'
                failed_count += 1
                continue

            result = detector.detect(image)
            mask = result.mask

            if invert:
                mask = 255 - mask

            cv2.imwrite(mask_path, mask)

            mask_ratio = float(np.mean(mask > 127))
            success_count += 1
            inv_tag = ' [反相]' if invert else ''
            results[fname] = f'✅ {result.count} 个目标{inv_tag} | mask={mask_ratio:.3%}'
            print(f'  ✅ {fname}: {result.count} 个目标')
        except Exception as e:
            failed_count += 1
            results[fname] = f'❌ {str(e)[:100]}'
            print(f'  ❌ {fname}: {e}')

    inv_note = '（反相模式）' if invert else ''
    results['_summary'] = f'自定义模型 Mask 生成完成{inv_note} | 总数 {len(image_files)} | 成功 {success_count} | 失败 {failed_count} | 输出: {output_folder}'
    print(f'=== 完成: {success_count}/{len(image_files)} ===\n')
    return jsonify(results)


if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        threading.Timer(1.5, open_browser).start()

    app.run(debug=True, host='0.0.0.0', port=5000)
