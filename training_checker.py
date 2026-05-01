"""
训练数据质检模块
检测可能导致LoRA训练质量下降的数据问题
"""

import os
import re
import shutil
import hashlib
import random
from collections import defaultdict
from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

DANGEROUS_TAGS = {
    'multiple views', 'multiple_views', 'character sheet', 'character_sheet',
    'reference sheet', 'reference_sheet', 'comic', '4koma', 'comic strip',
    'sequence', 'turnaround', 'expression chart', 'expression_chart',
    'variations', 'panel layout',
}

MULTI_PERSON_TAGS = {
    '2girls': 2, '2boys': 2, '2others': 2,
    '3girls': 3, '3boys': 3, '3others': 3,
    '4girls': 4, '4boys': 4, '4others': 4,
    '5girls': 5, '5boys': 5, '5others': 5,
    '6+girls': 6, '6+boys': 6,
    'multiple girls': 2, 'multiple_girls': 2,
    'multiple boys': 2, 'multiple_boys': 2,
}

WATERMARK_TAGS = {
    'watermark', 'text', 'username', 'web address', 'web_address',
    'patreon username', 'patreon_username', 'twitter username', 'twitter_username',
    'pixiv id', 'pixiv_id', 'artist name', 'artist_name',
    'signature', 'copyright notice', 'copyright_notice',
    'logo', 'english text', 'english_text', 'url', 'sample watermark',
}

SINGLE_PERSON_TAGS = {'1girl', '1boy', '1other'}

NSFW_TAGS = {
    'nude', 'naked', 'nipples', 'pussy', 'penis', 'sex', 'explicit',
    'nsfw', 'cum', 'vagina', 'genitals',
}

COMPOSITION_TAGS = {
    'full body': 'full_body', 'full_body': 'full_body',
    'upper body': 'upper_body', 'upper_body': 'upper_body',
    'lower body': 'lower_body', 'lower_body': 'lower_body',
    'cowboy shot': 'cowboy_shot', 'cowboy_shot': 'cowboy_shot',
    'portrait': 'portrait', 'close-up': 'close_up', 'close up': 'close_up',
    'face focus': 'face_focus', 'face_focus': 'face_focus',
    'bust': 'bust', 'half body': 'upper_body',
}

APOLOGY_PATTERNS = [
    'i cannot', "i can't", 'i apologize', "i'm sorry",
    'cannot provide', 'unable to',
    '无法提供', '不能提供', '抱歉', '对不起', '我无法', '我不能',
]

_CAPTION_STARTERS = re.compile(
    r'(?<=[,\s])'
    r'(?:'
    r'Shot\b|Seen\b|Framed\b|Depicted\b|Viewed\b|Rendered\b|'
    r'A\b|An\b|The\b|She\b|He\b|It\b|This\b|That\b|Her\b|His\b|'
    r'In\b|From\b|Against\b|Atop\b|Beneath\b|Under\b|Above\b|'
    r'Standing\b|Sitting\b|Kneeling\b|Wearing\b|Holding\b|Gripping\b|'
    r'Looking\b|Gazing\b|Staring\b'
    r')'
)

_CAPTION_VERBS = re.compile(
    r'\b(?:'
    r'depicts?|shows?|features?|portrays?|stands?|standing|sits?|sitting|'
    r'kneels?|kneeling|looks?|looking|wears?|wearing|holds?|holding|'
    r'grips?|gripping|smiles?|smiling|glows?|glowing|gazes?|gazing|'
    r'stares?|staring|billows?|billowing|floats?|floating|'
    r'is|are|was|were'
    r')\b',
    re.IGNORECASE,
)


def _match_tag(tag, token_set):
    return tag.lower() in token_set


def _tokenize(content):
    return {t.strip().lower() for t in content.split(',') if t.strip()}


def _read_caption(folder, txt_file):
    try:
        with open(os.path.join(folder, txt_file), 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ''


def _is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text or ''))


def _clean_caption_prefix(text):
    text = (text or '').strip()
    if not text:
        return ''
    return re.sub(r'^caption\s*:\s*', '', text, flags=re.IGNORECASE).strip()


def _find_caption_boundary(text):
    """
    在整段合并文本里找 caption 起点的字符索引。
    返回 (index, confidence) 或 None。
    """
    t = (text or '').strip()
    if not t:
        return None

    if _is_chinese(t):
        for m in re.finditer(r'[。！？]', t):
            before = t[:m.start()]
            last_comma = max(before.rfind('，'), before.rfind(','), before.rfind(' '))
            if last_comma > 0:
                candidate = t[last_comma + 1:].strip()
                if len(candidate.split()) >= 5 or len(candidate) >= 20:
                    return last_comma + 1, 'high'
        return None

    first_period = t.find('.')
    if first_period > 0:
        before_period = t[:first_period]
        for m in reversed(list(_CAPTION_STARTERS.finditer(before_period))):
            idx = m.start()
            tail = t[idx:].strip()
            head = t[:idx].strip()
            if len(tail.split()) >= 8 and head:
                return idx, 'high'

    for m in _CAPTION_STARTERS.finditer(t):
        idx = m.start()
        tail = t[idx:].strip()
        head = t[:idx].strip()
        if not head or not tail:
            continue
        if len(tail.split()) >= 12 and _CAPTION_VERBS.search(tail):
            return idx, 'medium'

    return None


def _split_mixed_caption(content):
    """
    将混合标注拆分为 tags_text 与 caption_text。
    核心策略：先在整段文本里找 caption 边界，再 split tag 区域。
    """
    if not (content or '').strip():
        return {
            'tags_text': '',
            'caption_text': '',
            'mode': 'empty',
            'confidence': 'high',
            'warnings': [],
        }

    normalized = content.replace('\r\n', '\n').replace('\r', '\n')

    caption_match = re.search(r'caption\s*:', normalized, re.IGNORECASE)
    if caption_match:
        tags_text = normalized[:caption_match.start()].strip().strip(',')
        caption_text = _clean_caption_prefix(normalized[caption_match.start():].strip())
        return {
            'tags_text': tags_text,
            'caption_text': caption_text,
            'mode': 'explicit_prefix',
            'confidence': 'high',
            'warnings': [],
        }

    has_double_newline = '\n\n' in normalized
    merged = ' '.join(normalized.split())

    result = _find_caption_boundary(merged)
    if result is not None:
        idx, confidence = result
        tags_raw = merged[:idx].strip().strip(',')
        caption_raw = merged[idx:].strip()

        last_comma = tags_raw.rfind(',')
        if last_comma > 0:
            last_tag = tags_raw[last_comma + 1:].strip()
            if len(last_tag.split()) <= 6:
                tags_text = tags_raw
            else:
                tags_text = tags_raw[:last_comma].strip().strip(',')
                caption_raw = last_tag + ' ' + caption_raw
        else:
            tags_text = tags_raw

        return {
            'tags_text': tags_text,
            'caption_text': _clean_caption_prefix(caption_raw),
            'mode': 'boundary_split',
            'confidence': confidence,
            'warnings': [],
        }

    if has_double_newline:
        parts = [p.strip() for p in re.split(r'\n\s*\n', normalized) if p.strip()]
        if len(parts) >= 2:
            tags_candidate = ' '.join(parts[:-1]).strip()
            caption_candidate = parts[-1].strip()
            if _is_chinese(caption_candidate) or re.search(r'[.!?。！？]', caption_candidate):
                return {
                    'tags_text': tags_candidate,
                    'caption_text': _clean_caption_prefix(caption_candidate),
                    'mode': 'newline_split',
                    'confidence': 'medium',
                    'warnings': ['split by double newline, last block has sentence punctuation'],
                }

    return {
        'tags_text': merged,
        'caption_text': '',
        'mode': 'tags_only',
        'confidence': 'low',
        'warnings': ['no caption boundary found, treated as tags-only'],
    }


def _compute_dhash(image_path, hash_size=8):
    """计算图片的差异哈希(dHash)，返回整数哈希值。
    dHash: 缩放到(hash_size+1, hash_size)灰度图，比较相邻像素。
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            width = hash_size + 1
            diff = []
            for row in range(hash_size):
                for col in range(hash_size):
                    idx = row * width + col
                    diff.append(1 if pixels[idx] > pixels[idx + 1] else 0)
            return int(''.join(str(b) for b in diff), 2)
    except Exception:
        return None


def _hamming_distance(h1, h2):
    """计算两个整数哈希之间的汉明距离。"""
    return bin(h1 ^ h2).count('1')


def _group_similar_by_hash(hash_map, threshold=10):
    """将哈希值相近的图片分组（Union-Find）。
    hash_map: {filename: hash_int}
    返回: [[filename, ...], ...]  每组至少2张
    """
    files = list(hash_map.keys())
    n = len(files)
    if n < 2:
        return []

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    hashes = [hash_map[f] for f in files]
    for i in range(n):
        if hashes[i] is None:
            continue
        for j in range(i + 1, n):
            if hashes[j] is None:
                continue
            if _hamming_distance(hashes[i], hashes[j]) <= threshold:
                union(i, j)

    groups_map = defaultdict(list)
    for i in range(n):
        if hashes[i] is not None:
            groups_map[find(i)].append(files[i])

    return [g for g in groups_map.values() if len(g) >= 2]


class TrainingDataChecker:
    """训练数据质检器"""

    def __init__(self, folder):
        if not os.path.exists(folder) or not os.path.isdir(folder):
            raise ValueError(f'文件夹不存在或不是文件夹：{folder}')
        self.folder = folder
        self.image_files = sorted(
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )
        self.txt_files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith('.txt')
        )
        self._image_cache = None

    def _load_image_cache(self):
        if self._image_cache is not None:
            return
        self._image_cache = {}
        for fname in self.image_files:
            try:
                with Image.open(os.path.join(self.folder, fname)) as img:
                    img.verify()
                with Image.open(os.path.join(self.folder, fname)) as img:
                    self._image_cache[fname] = (img.width, img.height, None)
            except Exception as e:
                self._image_cache[fname] = (0, 0, str(e)[:80])

    def _find_peer_image(self, base_name):
        for ext in IMAGE_EXTENSIONS:
            candidate = base_name + ext
            if os.path.exists(os.path.join(self.folder, candidate)):
                return candidate
        return None

    def get_dataset_stats(self):
        image_bases = {os.path.splitext(f)[0] for f in self.image_files}
        txt_bases = {os.path.splitext(f)[0] for f in self.txt_files}
        fmt_dist = defaultdict(int)
        for f in self.image_files:
            fmt_dist[os.path.splitext(f)[1].lower()] += 1
        return {
            'image_count': len(self.image_files),
            'txt_count': len(self.txt_files),
            'paired_count': len(image_bases & txt_bases),
            'images_without_txt': sorted(image_bases - txt_bases),
            'txts_without_image': sorted(txt_bases - image_bases),
            'format_distribution': dict(fmt_dist),
        }

    def check_image_integrity(self):
        self._load_image_cache()
        hits = [
            {'image_file': f, 'error': self._image_cache[f][2]}
            for f in self.image_files if self._image_cache[f][2]
        ]
        return _result('image_integrity', 'critical', '图片完整性检测',
                       '损坏的图片必须移除，否则训练时会报错。',
                       len(self.image_files), hits)

    def check_duplicates(self):
        hash_map = defaultdict(list)
        for fname in self.image_files:
            try:
                with open(os.path.join(self.folder, fname), 'rb') as f:
                    h = hashlib.md5(f.read()).hexdigest()
                hash_map[h].append(fname)
            except Exception:
                pass
        hits = [{'files': files, 'hash': h} for h, files in hash_map.items() if len(files) > 1]
        return _result('duplicates', 'high', '重复图片检测',
                       '完全相同的图片会让模型过拟合该样本，建议只保留一张。',
                       len(self.image_files), hits)

    def check_resolution(self, min_short_side=768):
        self._load_image_cache()
        hits = []
        for fname in self.image_files:
            w, h, err = self._image_cache[fname]
            if err:
                continue
            if min(w, h) < min_short_side:
                hits.append({'image_file': fname, 'width': w, 'height': h, 'short_side': min(w, h)})
        return _result('resolution', 'medium', '低分辨率检测',
                       f'短边<{min_short_side}px的图片放大到训练分辨率后会模糊，建议移除或替换。',
                       len(self.image_files), hits)

    def check_aspect_ratio(self, min_ar=0.4, max_ar=2.5):
        self._load_image_cache()
        hits = []
        for fname in self.image_files:
            w, h, err = self._image_cache[fname]
            if err or h == 0:
                continue
            ar = round(w / h, 3)
            if ar < min_ar or ar > max_ar:
                hits.append({'image_file': fname, 'width': w, 'height': h,
                             'aspect_ratio': ar, 'reason': '过窄' if ar < min_ar else '过宽'})
        return _result('aspect_ratio', 'low', '极端宽高比检测',
                       f'AR<{min_ar}或>{max_ar}的图片裁切后构图可能异常。',
                       len(self.image_files), hits)

    def check_caption_integrity(self):
        empty, short, apology = [], [], []
        for txt_file in self.txt_files:
            content = _read_caption(self.folder, txt_file)
            base = os.path.splitext(txt_file)[0]
            peer = self._find_peer_image(base)
            item = {'txt_file': txt_file, 'image_file': peer}
            stripped = content.strip()
            if not stripped:
                empty.append(item)
                continue
            if len(stripped) < 20:
                short.append({**item, 'length': len(stripped)})
            cl = content.lower()
            matched = [p for p in APOLOGY_PATTERNS if p in cl]
            if matched:
                apology.append({**item, 'matched_patterns': matched})
        hits = (
            [{'type': 'empty', **i} for i in empty] +
            [{'type': 'short', **i} for i in short] +
            [{'type': 'apology', **i} for i in apology]
        )
        r = _result('caption_integrity', 'critical', '标注完整性检测',
                    '空标注和AI拒绝内容必须修复，否则训练数据无效。',
                    len(self.txt_files), hits)
        r['empty_count'] = len(empty)
        r['short_count'] = len(short)
        r['apology_count'] = len(apology)
        return r

    def check_dangerous_tags(self):
        view_hits, person_hits = [], []
        for txt_file in self.txt_files:
            content = _read_caption(self.folder, txt_file)
            tokens = _tokenize(content)
            base = os.path.splitext(txt_file)[0]
            peer = self._find_peer_image(base)
            item = {'txt_file': txt_file, 'image_file': peer}
            matched_view = [t for t in DANGEROUS_TAGS if _match_tag(t, tokens)]
            if matched_view:
                view_hits.append({**item, 'matched_tags': matched_view})
            max_persons, matched_tag = 0, None
            for tag, count in MULTI_PERSON_TAGS.items():
                if _match_tag(tag, tokens) and count > max_persons:
                    max_persons, matched_tag = count, tag
            if max_persons > 0:
                person_hits.append({**item, 'person_count': max_persons,
                                    'matched_tag': matched_tag,
                                    'risk': 'high' if max_persons >= 3 else 'medium'})
        hits = (
            [{'type': 'multiple_views', **i} for i in view_hits] +
            [{'type': 'multi_person', **i} for i in person_hits]
        )
        r = _result('dangerous_tags', 'high', '危险标签检测',
                    'multiple_views类图片是肢体崩坏最大原因，建议全部移除；3人以上高风险。',
                    len(self.txt_files), hits)
        r['multiple_views_count'] = len(view_hits)
        r['multi_person_count'] = len(person_hits)
        r['high_risk_person_count'] = sum(1 for h in person_hits if h['risk'] == 'high')
        return r

    def check_watermark_tags(self):
        hits = []
        for txt_file in self.txt_files:
            content = _read_caption(self.folder, txt_file)
            tokens = _tokenize(content)
            matched = [t for t in WATERMARK_TAGS if _match_tag(t, tokens)]
            if matched:
                base = os.path.splitext(txt_file)[0]
                hits.append({'txt_file': txt_file,
                             'image_file': self._find_peer_image(base),
                             'matched_tags': matched})
        return _result('watermark_tags', 'medium', '水印标签检测',
                       '水印标签会让模型学会在生成图中添加文字水印。',
                       len(self.txt_files), hits)

    def check_composition(self):
        dist = defaultdict(int)
        untagged = 0
        for txt_file in self.txt_files:
            tokens = _tokenize(_read_caption(self.folder, txt_file))
            found = False
            for tag, label in COMPOSITION_TAGS.items():
                if _match_tag(tag, tokens):
                    dist[label] += 1
                    found = True
                    break
            if not found:
                untagged += 1
        dist['untagged'] = untagged
        return {
            'check_name': 'composition',
            'severity': 'info',
            'description': '构图分布统计',
            'total_checked': len(self.txt_files),
            'distribution': dict(dist),
            'suggestion': '建议全身图占比30%以上，避免全部为特写导致模型不会画全身。',
        }

    def check_similar_images(self, threshold=10, keep_ratio=0.3, min_keep=2):
        """检测视觉相似的图片组（差分检测）。
        threshold: dHash汉明距离阈值，越小越严格（默认10）
        keep_ratio: 每组保留比例（默认30%）
        min_keep: 每组最少保留张数（默认2）
        返回包含相似组信息和建议移除列表的结果。
        """
        print('  🔍 正在计算图片感知哈希...')
        hash_map = {}
        for fname in self.image_files:
            h = _compute_dhash(os.path.join(self.folder, fname))
            if h is not None:
                hash_map[fname] = h

        print(f'  📊 成功计算 {len(hash_map)}/{len(self.image_files)} 张图片的哈希')
        print(f'  🔗 正在分组（汉明距离阈值={threshold}）...')
        groups = _group_similar_by_hash(hash_map, threshold)

        # 按组大小降序排列
        groups.sort(key=len, reverse=True)

        total_redundant = 0
        group_details = []
        suggest_remove = []

        for i, group in enumerate(groups):
            group_size = len(group)
            keep_count = max(min_keep, int(round(group_size * keep_ratio)))
            keep_count = min(keep_count, group_size)  # 不能超过组大小
            remove_count = group_size - keep_count

            # 随机选择保留哪些
            shuffled = list(group)
            random.shuffle(shuffled)
            keep_files = shuffled[:keep_count]
            remove_files = shuffled[keep_count:]

            total_redundant += remove_count
            suggest_remove.extend(remove_files)

            group_details.append({
                'group_id': i + 1,
                'group_size': group_size,
                'keep_count': keep_count,
                'remove_count': remove_count,
                'keep_files': sorted(keep_files),
                'remove_files': sorted(remove_files),
                'all_files': sorted(group),
            })

        # 构建 hits 列表（建议移除的文件）
        hits = []
        for gd in group_details:
            for fname in gd['remove_files']:
                base = os.path.splitext(fname)[0]
                hits.append({
                    'image_file': fname,
                    'txt_file': base + '.txt' if os.path.exists(
                        os.path.join(self.folder, base + '.txt')) else None,
                    'group_id': gd['group_id'],
                    'group_size': gd['group_size'],
                })

        r = _result('similar_images', 'medium', '相似图/差分检测',
                    f'同质化差分图过多会让模型过拟合特定构图。每组保留{int(keep_ratio*100)}%（最少{min_keep}张），其余建议移除。',
                    len(self.image_files), hits)
        r['group_count'] = len(groups)
        r['total_in_groups'] = sum(len(g) for g in groups)
        r['total_redundant'] = total_redundant
        r['groups'] = group_details
        r['threshold'] = threshold
        r['keep_ratio'] = keep_ratio
        r['min_keep'] = min_keep
        return r

    def check_nsfw_ratio(self):
        nsfw_files = []
        for txt_file in self.txt_files:
            tokens = _tokenize(_read_caption(self.folder, txt_file))
            matched = [t for t in NSFW_TAGS if _match_tag(t, tokens)]
            if matched:
                base = os.path.splitext(txt_file)[0]
                nsfw_files.append({'txt_file': txt_file,
                                   'image_file': self._find_peer_image(base),
                                   'matched_tags': matched})
        total = max(len(self.txt_files), 1)
        return {
            'check_name': 'nsfw_ratio',
            'severity': 'info',
            'description': 'NSFW内容比例统计',
            'total_checked': len(self.txt_files),
            'nsfw_count': len(nsfw_files),
            'nsfw_pct': round(len(nsfw_files) / total * 100, 1),
            'items': nsfw_files,
            'suggestion': '仅供参考，不影响评分。',
        }

    def split_caption_file(self, txt_file):
        content = _read_caption(self.folder, txt_file)
        result = _split_mixed_caption(content)
        base = os.path.splitext(txt_file)[0]
        result.update({
            'txt_file': txt_file,
            'image_file': self._find_peer_image(base),
        })
        return result

    def export_split_caption_datasets(self, rewrite_original_txt=True):
        parent_dir = os.path.dirname(self.folder.rstrip(r'\/'))
        folder_name = os.path.basename(self.folder.rstrip(r'\/'))
        tag_dir = os.path.join(parent_dir, f'{folder_name}-tag')
        caption_dir = os.path.join(parent_dir, f'{folder_name}-caption')

        os.makedirs(tag_dir, exist_ok=True)
        os.makedirs(caption_dir, exist_ok=True)

        summary = {
            'source_folder': self.folder,
            'tag_folder': tag_dir,
            'caption_folder': caption_dir,
            'rewrite_original_txt': rewrite_original_txt,
            'processed_txt_count': 0,
            'split_success_count': 0,
            'tags_only_count': 0,
            'caption_only_count': 0,
            'empty_count': 0,
            'unclassified_count': 0,
            'original_rewritten_count': 0,
            'tag_dataset_written_count': 0,
            'caption_dataset_written_count': 0,
            'items': [],
        }

        for txt_file in self.txt_files:
            base = os.path.splitext(txt_file)[0]
            image_file = self._find_peer_image(base)
            split = self.split_caption_file(txt_file)

            tags_text = (split.get('tags_text') or '').strip()
            caption_text = (split.get('caption_text') or '').strip()
            mode = split.get('mode') or 'unknown'
            confidence = split.get('confidence') or 'low'
            warnings = split.get('warnings') or []

            item = {
                'txt_file': txt_file,
                'image_file': image_file,
                'mode': mode,
                'confidence': confidence,
                'warnings': warnings,
                'has_tags': bool(tags_text),
                'has_caption': bool(caption_text),
            }

            summary['processed_txt_count'] += 1

            if tags_text and caption_text:
                summary['split_success_count'] += 1
            elif tags_text:
                summary['tags_only_count'] += 1
            elif caption_text:
                summary['caption_only_count'] += 1
            elif mode == 'empty':
                summary['empty_count'] += 1
            else:
                summary['unclassified_count'] += 1

            if image_file:
                src_image = os.path.join(self.folder, image_file)
                if tags_text:
                    shutil.copy2(src_image, os.path.join(tag_dir, image_file))
                    summary['tag_dataset_written_count'] += 1
                if caption_text:
                    shutil.copy2(src_image, os.path.join(caption_dir, image_file))
                    summary['caption_dataset_written_count'] += 1

            if tags_text:
                with open(os.path.join(tag_dir, txt_file), 'w', encoding='utf-8') as f:
                    f.write(tags_text)
            if caption_text:
                with open(os.path.join(caption_dir, txt_file), 'w', encoding='utf-8') as f:
                    f.write(caption_text)

            if rewrite_original_txt and tags_text:
                with open(os.path.join(self.folder, txt_file), 'w', encoding='utf-8') as f:
                    f.write(tags_text)
                summary['original_rewritten_count'] += 1

            summary['items'].append(item)

        summary['summary'] = (
            f'已处理 {summary["processed_txt_count"]} 个txt；'
            f'混合拆分 {summary["split_success_count"]} 个，'
            f'tag-only {summary["tags_only_count"]} 个，'
            f'caption-only {summary["caption_only_count"]} 个，'
            f'空内容 {summary["empty_count"]} 个；'
            f'原目录改写 {summary["original_rewritten_count"]} 个。'
        )
        return summary

    def score(self, checks_result):
        weights = {'critical': 40, 'high': 30, 'medium': 20, 'low': 10}
        deductions = 0.0
        for check in checks_result.values():
            if not isinstance(check, dict) or 'severity' not in check:
                continue
            sev = check.get('severity')
            if sev not in weights or sev == 'info':
                continue
            total = check.get('total_checked', 0)
            count = check.get('count', 0)
            if total == 0:
                continue
            ratio = min(count / total, 1.0)
            deductions += weights[sev] * ratio
        return max(0, round(100 - deductions))

    def full_check(self, checks=None, similar_threshold=10,
                   similar_keep_ratio=0.3, similar_min_keep=2):
        enabled = {
            'image_integrity': True, 'duplicates': True, 'resolution': True,
            'aspect_ratio': True, 'caption_integrity': True, 'dangerous_tags': True,
            'watermark_tags': True, 'composition': True, 'nsfw_ratio': True,
            'similar_images': True,
        }
        if checks:
            enabled.update(checks)

        dispatch = {
            'image_integrity': self.check_image_integrity,
            'duplicates': self.check_duplicates,
            'resolution': self.check_resolution,
            'aspect_ratio': self.check_aspect_ratio,
            'caption_integrity': self.check_caption_integrity,
            'dangerous_tags': self.check_dangerous_tags,
            'watermark_tags': self.check_watermark_tags,
            'composition': self.check_composition,
            'nsfw_ratio': self.check_nsfw_ratio,
        }

        raw = {}
        for name, fn in dispatch.items():
            if enabled.get(name, True):
                raw[name] = fn()

        # 相似图检测单独调用（需要额外参数）
        if enabled.get('similar_images', True):
            raw['similar_images'] = self.check_similar_images(
                threshold=similar_threshold,
                keep_ratio=similar_keep_ratio,
                min_keep=similar_min_keep,
            )

        checks = {}
        if 'dangerous_tags' in raw:
            dt = raw['dangerous_tags']
            view_items = [i for i in dt['items'] if i.get('type') == 'multiple_views']
            person_items = [i for i in dt['items'] if i.get('type') == 'multi_person']
            checks['multiple_views'] = {
                'check_name': 'multiple_views', 'severity': 'high',
                'description': '多视角/拼图检测',
                'suggestion': 'multiple_views类图片是肢体崩坏最大原因，建议全部移除。',
                'total_checked': dt['total_checked'],
                'count': len(view_items),
                'pct': round(len(view_items) / max(dt['total_checked'], 1) * 100, 1),
                'items': view_items,
            }
            checks['multi_person'] = {
                'check_name': 'multi_person', 'severity': 'high',
                'description': '多人体检测',
                'suggestion': '多人体图片会导致模型混淆人物特征，3人以上高风险。',
                'total_checked': dt['total_checked'],
                'count': len(person_items),
                'pct': round(len(person_items) / max(dt['total_checked'], 1) * 100, 1),
                'items': person_items,
            }

        key_map = {
            'resolution': 'low_resolution',
            'aspect_ratio': 'extreme_ar',
            'caption_integrity': 'caption_quality',
            'image_integrity': 'image_integrity',
            'duplicates': 'duplicates',
            'watermark_tags': 'watermark_tags',
            'composition': 'composition',
            'nsfw_ratio': 'nsfw_ratio',
            'similar_images': 'similar_images',
        }
        for old_key, new_key in key_map.items():
            if old_key in raw:
                r = dict(raw[old_key])
                r['check_name'] = new_key
                checks[new_key] = r

        if 'multi_person' in checks:
            extra = []
            for txt_file in self.txt_files:
                content_txt = _read_caption(self.folder, txt_file)
                tokens = _tokenize(content_txt)
                if '1girl' in tokens and '1boy' in tokens:
                    base = os.path.splitext(txt_file)[0]
                    extra.append({
                        'type': 'multi_person',
                        'txt_file': txt_file,
                        'image_file': self._find_peer_image(base),
                        'person_count': 2,
                        'matched_tag': '1girl+1boy',
                        'risk': 'medium',
                    })
            existing_txts = {i['txt_file'] for i in checks['multi_person']['items']}
            new_items = [i for i in extra if i['txt_file'] not in existing_txts]
            checks['multi_person']['items'].extend(new_items)
            checks['multi_person']['count'] = len(checks['multi_person']['items'])
            checks['multi_person']['pct'] = round(
                checks['multi_person']['count'] / max(checks['multi_person']['total_checked'], 1) * 100, 1
            )
        health_score = self.score(checks)
        total_issues = sum(
            c.get('count', 0) for c in checks.values()
            if isinstance(c, dict) and c.get('severity') in ('high', 'critical')
        )

        for _, v in checks.items():
            if not isinstance(v, dict):
                continue
            v['hit_count'] = v.get('count', 0)
            v['hit_pct'] = v.get('pct', 0)
            v['hits'] = v.get('items', [])

        if 'multi_person' in checks:
            mp = checks['multi_person']
            mp['high_risk_count'] = sum(1 for i in mp['hits'] if i.get('risk') == 'high')
            mp['medium_risk_count'] = sum(1 for i in mp['hits'] if i.get('risk') != 'high')

        if 'caption_quality' in checks:
            cq = checks['caption_quality']
            cq['empty_count'] = cq.get('empty_count', 0)
            cq['short_count'] = cq.get('short_count', 0)
            cq['apology_count'] = cq.get('apology_count', 0)
            cq['no_person_count'] = sum(1 for i in cq['hits'] if i.get('type') == 'no_person')
            cq['watermark_count'] = sum(1 for i in cq['hits'] if i.get('type') == 'watermark')

        return {
            'folder': self.folder,
            'stats': self.get_dataset_stats(),
            'checks': checks,
            'health_score': health_score,
            'total_issues': total_issues,
            'summary': f'健康评分：{health_score}/100，高优先级问题：{total_issues}个',
        }

    def move_files(self, file_list, subfolder='problematic'):
        target_dir = os.path.join(self.folder, subfolder)
        os.makedirs(target_dir, exist_ok=True)
        moved, failed = 0, 0
        results = {}
        for fname in file_list:
            base = os.path.splitext(fname)[0]
            to_move = []
            peer = self._find_peer_image(base)
            if peer:
                to_move.append(peer)
            txt = base + '.txt'
            if os.path.exists(os.path.join(self.folder, txt)):
                to_move.append(txt)
            if not to_move:
                results[fname] = '⚠️ 未找到对应文件'
                continue
            for f in to_move:
                try:
                    shutil.move(os.path.join(self.folder, f),
                                os.path.join(target_dir, f))
                    moved += 1
                except Exception as e:
                    results[f] = f'❌ {e}'
                    failed += 1
            results[fname] = f'✅ 已移动到 {subfolder}/'
        return {'moved_count': moved, 'failed_count': failed,
                'target_dir': target_dir, 'results': results}


def _result(check_name, severity, description, suggestion, total_checked, hits):
    return {
        'check_name': check_name,
        'severity': severity,
        'description': description,
        'suggestion': suggestion,
        'total_checked': total_checked,
        'count': len(hits),
        'pct': round(len(hits) / max(total_checked, 1) * 100, 1),
        'items': hits,
    }
