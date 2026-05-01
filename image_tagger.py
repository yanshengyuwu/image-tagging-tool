"""
图片标注模块
整合pixai-tagger和Danbooru API，生成XML格式标签
"""

import os
import json
import hashlib
import requests
from pathlib import Path
from PIL import Image

from handler_wrapper import CPUEndpointHandler


class ImageTagger:
    def __init__(self, danbooru_api_key=None, danbooru_username=None):
        """
        初始化图片标注器
        :param danbooru_api_key: Danbooru API密钥
        :param danbooru_username: Danbooru用户名
        """
        self.danbooru_api_key = danbooru_api_key
        self.danbooru_username = danbooru_username
        
        # 初始化pixai-tagger（智能选择GPU/CPU）
        try:
            self.pixai_handler = CPUEndpointHandler(path=r'D:\pixai-tagger')
        except Exception as e:
            print(f"❌ PixAI Tagger加载失败: {e}")
            self.pixai_handler = None
    
    def get_image_md5(self, image_path):
        """计算图片的MD5值"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def extract_post_id_from_filename(self, filename):
        """
        从文件名中提取Danbooru post ID
        支持格式：danbooru_12345.jpg, __danbooru_12345.png, 12345_p0.jpg等
        :param filename: 文件名
        :return: post ID或None
        """
        import re
        
        # 移除扩展名
        name = os.path.splitext(filename)[0]
        
        # 尝试多种常见格式
        patterns = [
            r'danbooru[_-](\d+)',  # danbooru_12345 或 danbooru-12345
            r'__danbooru[_-](\d+)',  # __danbooru_12345
            r'^(\d{6,})(?:[_-]|$)',  # 以6位以上数字开头
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                post_id = match.group(1)
                print(f"  📝 从文件名提取到Post ID: {post_id}")
                return post_id
        
        return None
    
    def search_danbooru_by_post_id(self, post_id):
        """
        通过Post ID在Danbooru搜索图片
        :param post_id: Danbooru post ID
        :return: 标签字典或None
        """
        if not self.danbooru_api_key:
            return None
        
        try:
            params = {
                'login': self.danbooru_username,
                'api_key': self.danbooru_api_key
            }
            
            url = f"https://danbooru.donmai.us/posts/{post_id}.json"
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                post = response.json()
                tags = {
                    'general': post.get('tag_string_general', '').split(),
                    'character': post.get('tag_string_character', '').split(),
                    'copyright': post.get('tag_string_copyright', '').split(),
                    'artist': post.get('tag_string_artist', '').split(),
                    'meta': post.get('tag_string_meta', '').split()
                }
                print(f"✅ 通过Post ID找到标签: {len(tags['general'])} general, {len(tags['character'])} character")
                return tags
            else:
                return None
                
        except Exception as e:
            print(f"  ⚠️ Post ID查询失败: {e}")
            return None
    
    def search_danbooru_by_md5(self, md5_hash, filename=None):
        """
        通过MD5在Danbooru搜索图片，如果失败则尝试从文件名提取Post ID
        :param md5_hash: 图片MD5值
        :param filename: 文件名（可选，用于备用查询）
        :return: 标签列表或None
        """
        if not self.danbooru_api_key:
            print("ℹ️  未配置Danbooru API Key，跳过Danbooru搜索")
            return None
        
        try:
            # 方法1: 通过MD5查询
            params = {
                'md5': md5_hash,
                'login': self.danbooru_username,
                'api_key': self.danbooru_api_key
            }
            
            url = "https://danbooru.donmai.us/posts.json"
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                posts = response.json()
                if posts and len(posts) > 0:
                    post = posts[0]
                    tags = {
                        'general': post.get('tag_string_general', '').split(),
                        'character': post.get('tag_string_character', '').split(),
                        'copyright': post.get('tag_string_copyright', '').split(),
                        'artist': post.get('tag_string_artist', '').split(),
                        'meta': post.get('tag_string_meta', '').split()
                    }
                    print(f"✅ 通过MD5找到标签: {len(tags['general'])} general, {len(tags['character'])} character")
                    return tags
            
            # 方法2: 如果MD5查询失败，尝试从文件名提取Post ID
            if filename:
                print("  ℹ️  MD5未找到匹配，尝试从文件名查询...")
                post_id = self.extract_post_id_from_filename(filename)
                if post_id:
                    tags = self.search_danbooru_by_post_id(post_id)
                    if tags:
                        return tags
            
            print("ℹ️  该图片不在Danbooru数据库中（正常情况）")
            return None
            
        except Exception as e:
            print(f"❌ Danbooru搜索失败: {e}")
            return None
    
    def tag_image_with_pixai(self, image_path, general_threshold=0.30, character_threshold=0.75):
        """
        使用pixai-tagger标注图片
        :param image_path: 图片路径
        :param general_threshold: 通用标签阈值
        :param character_threshold: 角色标签阈值
        :return: 标签字典
        """
        if not self.pixai_handler:
            print("❌ PixAI Tagger未加载")
            return None
        
        try:
            image = Image.open(image_path)
            
            result = self.pixai_handler({
                "inputs": image,
                "parameters": {
                    "general_threshold": general_threshold,
                    "character_threshold": character_threshold
                }
            })
            
            tags = {
                'general': result.get('feature', []),
                'character': result.get('character', []),
                'copyright': result.get('ip', [])
            }
            
            print(f"✅ PixAI标注完成: {len(tags['general'])} general, {len(tags['character'])} character, {len(tags['copyright'])} copyright")
            return tags
            
        except Exception as e:
            print(f"❌ PixAI标注失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def merge_tags(self, danbooru_tags, pixai_tags):
        """
        合并Danbooru和PixAI的标签（去重）
        :param danbooru_tags: Danbooru标签字典
        :param pixai_tags: PixAI标签字典
        :return: 合并后的标签字典
        """
        merged = {
            'general': [],
            'character': [],
            'copyright': [],
            'artist': [],
            'meta': []
        }
        
        # 合并每个类别的标签
        for category in ['general', 'character', 'copyright']:
            tags_set = set()
            
            if danbooru_tags and category in danbooru_tags:
                tags_set.update(danbooru_tags[category])
            
            if pixai_tags and category in pixai_tags:
                tags_set.update(pixai_tags[category])
            
            merged[category] = sorted(list(tags_set))
        
        # Danbooru独有的类别
        if danbooru_tags:
            if 'artist' in danbooru_tags:
                merged['artist'] = danbooru_tags['artist']
            if 'meta' in danbooru_tags:
                merged['meta'] = danbooru_tags['meta']
        
        return merged
    
    def process_image(self, image_path, general_threshold=0.30, character_threshold=0.75):
        """
        完整处理单张图片：Danbooru搜索 + PixAI标注 + 合并
        :param image_path: 图片路径
        :param general_threshold: PixAI通用标签阈值
        :param character_threshold: PixAI角色标签阈值
        :return: 合并后的标签字典
        """
        filename = os.path.basename(image_path)
        print(f"\n🖼️  处理图片: {filename}")
        
        # 1. 计算MD5并搜索Danbooru（传递文件名用于备用查询）
        md5_hash = self.get_image_md5(image_path)
        print(f"MD5: {md5_hash}")
        danbooru_tags = self.search_danbooru_by_md5(md5_hash, filename)
        
        # 2. 使用PixAI标注
        pixai_tags = self.tag_image_with_pixai(image_path, general_threshold, character_threshold)
        
        # 3. 合并标签
        merged_tags = self.merge_tags(danbooru_tags, pixai_tags)
        
        print(f"📊 合并结果: {len(merged_tags['general'])} general, {len(merged_tags['character'])} character, {len(merged_tags['copyright'])} copyright")
        
        return merged_tags
