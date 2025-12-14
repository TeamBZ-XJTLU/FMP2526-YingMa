#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import re
import uuid
import time

def analyze_comment_length(content):
    """分析评论长度"""
    if pd.isna(content):
        return 0, 'short'
    
    content = str(content)
    length = len(content)
    
    # 根据长度分类
    if length <= 20:
        category = 'short'
    elif length <= 100:
        category = 'medium'
    else:
        category = 'long'
    
    return length, category

def clean_text(text):
    """
    清洗文本内容，去除首尾空格、HTML实体和编码乱码
    
    Args:
        text: 输入文本
    
    Returns:
        清洗后的文本
    """
    if pd.isna(text):
        return ''
    
    # 转换为字符串并去除首尾空格
    text = str(text).strip()
    
    # 检测并处理编码乱码（连续三个或更多的�字符）
    if '���' in text:
        # 尝试几种常见的编码方式解码
        for encoding in ['utf-8', 'gbk', 'latin-1']:
            try:
                # 对于已经损坏的文本，尝试重新编码和解码
                text = text.encode('utf-8', errors='replace').decode(encoding, errors='replace')
                break
            except:
                continue
    
    return text

def safe_read_csv(file_path, encoding_list=['utf-8-sig', 'gbk', 'utf-8', 'gb2312', 'ansi', 'latin1']):
    """安全读取CSV文件，尝试多种编码 - 兼容版"""
    print(f"[INFO] 正在读取文件: {os.path.basename(file_path)}")
    
    # 检测文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 检测文件大小
    file_size = os.path.getsize(file_path)
    print(f"[INFO] 文件大小: {file_size} 字节")
    
    # 尝试不同的读取方式
    for encoding in encoding_list:
        try:
            print(f"[DEBUG] 尝试使用编码: {encoding}")
            
            # 尝试基本读取方式，只使用error_bad_lines参数（兼容旧版pandas）
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    error_bad_lines=False,  # 旧版本pandas使用
                    low_memory=False
                )
                print(f"[INFO] 使用{encoding}编码读取成功，{len(df)}行数据")
                return df
            except TypeError:
                # 如果出现参数错误，尝试不使用可能不兼容的参数
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    low_memory=False
                )
                print(f"[INFO] 使用简化参数，{encoding}编码读取成功，{len(df)}行数据")
                return df
                
        except UnicodeDecodeError:
            print(f"[DEBUG] 编码 {encoding} 读取失败，尝试下一个编码...")
            continue
        except Exception as e:
            print(f"[DEBUG] 使用编码 {encoding} 读取文件时出错: {str(e)}")
            continue
    
    # 如果所有编码都尝试失败，尝试使用二进制模式和chardet检测编码
    try:
        print(f"[DEBUG] 尝试使用chardet检测文件编码...")
        # 读取文件头部进行编码检测
        with open(file_path, 'rb') as f:
            raw_sample = f.read(10000)
            
        # 使用chardet检测编码（如果可用）
        try:
            import chardet
            result = chardet.detect(raw_sample)
            detected_encoding = result['encoding']
            confidence = result['confidence']
            print(f"[DEBUG] chardet检测到编码: {detected_encoding} (置信度: {confidence:.2f})")
            
            if detected_encoding and confidence > 0.5:
                df = pd.read_csv(
                    file_path,
                    encoding=detected_encoding,
                    error_bad_lines=False,
                    low_memory=False
                )
                print(f"[INFO] 使用chardet检测的编码 {detected_encoding} 读取成功，{len(df)}行数据")
                return df
        except ImportError:
            print(f"[DEBUG] chardet不可用，跳过编码检测")
            pass
    except Exception as e:
        print(f"[DEBUG] 编码检测失败: {str(e)}")
    
    # 最后尝试，使用'ignore'错误处理方式
    print(f"[DEBUG] 尝试使用ignore错误处理方式读取...")
    try:
        df = pd.read_csv(
            file_path,
            encoding='utf-8',
            encoding_errors='ignore',  # 较新版本pandas支持
            low_memory=False
        )
        print(f"[INFO] 使用ignore错误处理读取成功，{len(df)}行数据")
        return df
    except Exception as e:
        print(f"[DEBUG] ignore错误处理方式失败: {str(e)}")
    
    # 如果所有尝试都失败，返回错误信息
    raise ValueError(f"无法读取文件 {file_path}。尝试过的编码: {', '.join(encoding_list)}")

def convert_timestamp(timestamp, is_milliseconds=None):
   
    if pd.isna(timestamp) or timestamp is None:
        return None
    
    # 转换为数字类型
    try:
        ts_num = float(timestamp)
    except (ValueError, TypeError):
        # 如果不是数字，可能是字符串格式，尝试转换
        try:
            # 尝试多种日期时间格式
            timestamp_str = str(timestamp).strip()
            # 处理常见的日期时间字符串格式
            if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', timestamp_str):
                dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            elif re.match(r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}', timestamp_str):
                dt = datetime.strptime(timestamp_str, '%Y/%m/%d %H:%M:%S')
            elif re.match(r'\d{13}', timestamp_str):  # 13位数字字符串
                return int(timestamp_str)
            elif re.match(r'\d{10}', timestamp_str):  # 10位数字字符串
                return int(timestamp_str) * 1000
            else:
                # 通用解析
                dt = pd.to_datetime(timestamp)
            
            # 确保时间有效（1970年之后）
            if dt.year < 1970:
                return None
                
            return int(dt.timestamp() * 1000)
        except Exception as e:
            print(f"[DEBUG] 无法解析时间字符串: {timestamp}, 错误: {e}")
            return None
    
    
    # 确保数字有效
    if not np.isfinite(ts_num):
        return None
        
    if is_milliseconds is None:
        # 如果大于1e12，视为毫秒级
        if ts_num > 1e12:
            # 毫秒级，确保是合理范围（1970-2100年）
            if 31536000000 <= ts_num <= 4102444800000:  # 1971-01-01到2099-12-31
                return int(ts_num)
            else:
                print(f"[DEBUG] 毫秒级时间戳超出合理范围: {ts_num}")
                return None
        elif ts_num > 1e9:  # 10位数，秒级，1970年后
            # 秒级，转换为毫秒
            return int(ts_num * 1000)
        else:
            print(f"[DEBUG] 时间戳过小，可能无效: {ts_num}")
            return None
    elif is_milliseconds:
        return int(ts_num)
    else:
        return int(ts_num * 1000)

def parse_chinese_number(num_str):
    """解析包含中文数字格式的字符串（如"1万"）为整数"""
    if pd.isna(num_str) or num_str is None:
        return 0
    
    if isinstance(num_str, (int, float)):
        return int(num_str)
    
    num_str = str(num_str).strip()
    if not num_str:
        return 0
    
    # 处理包含"万"的情况
    if '万' in num_str:
        parts = num_str.split('万')
        if len(parts) > 0 and parts[0]:
            try:
                return int(float(parts[0]) * 10000)
            except ValueError:
                return 0
    # 处理包含"千"的情况
    elif '千' in num_str:
        parts = num_str.split('千')
        if len(parts) > 0 and parts[0]:
            try:
                return int(float(parts[0]) * 1000)
            except ValueError:
                return 0
    # 尝试直接转换为整数
    else:
        # 移除非数字字符
        import re
        clean_num = re.sub(r'[^0-9.]', '', num_str)
        try:
            return int(float(clean_num))
        except ValueError:
            return 0
    
    return 0

def timestamp_to_datetime(milliseconds):
    """将毫秒级时间戳转换为UTC datetime，过滤1970之前的无效时间 - 优化版"""
    if pd.isna(milliseconds) or milliseconds is None or milliseconds < 0:
        return pd.NaT
    
    # 过滤1970之前的无效时间戳和2100年后的不合理时间
    if milliseconds < 31536000000:  # 1971-01-01 00:00:00
        print(f"[DEBUG] 时间戳早于1971年: {milliseconds}")
        return pd.NaT
    if milliseconds > 4102444800000:  # 2099-12-31 00:00:00
        print(f"[DEBUG] 时间戳晚于2099年: {milliseconds}")
        return pd.NaT
    
    try:
        # 转换为datetime并设置为UTC时区
        dt = datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc)
        return dt
    except (ValueError, OverflowError) as e:
        print(f"[DEBUG] 时间戳转换错误: {milliseconds}, 错误: {e}")
        return pd.NaT

def to_local_time(milliseconds):
    """将毫秒级时间戳转换为本地时间 - 新函数"""
    if pd.isna(milliseconds) or milliseconds is None or milliseconds < 0:
        return pd.NaT
    
    try:
        # 先转换为UTC时间，然后转换为本地时间
        utc_dt = timestamp_to_datetime(milliseconds)
        if utc_dt is pd.NaT:
            return pd.NaT
        # 转换为本地时间
        local_dt = utc_dt.astimezone()
        return local_dt
    except Exception as e:
        print(f"[DEBUG] 时间戳转换为本地时间错误: {milliseconds}, 错误: {e}")
        return pd.NaT

def normalize_data(df, platform, kol_id=None, post_time_dict=None, post_info_dict=None):
    """标准化数据，根据平台特性进行字段映射，正确区分帖子和评论发布时间"""
    # 复制原始数据
    normalized_df = df.copy()
    
    # 添加平台标识
    normalized_df['platform'] = platform
    
    # 生成全局唯一ID
    normalized_df['global_id'] = [str(uuid.uuid4()) for _ in range(len(normalized_df))]
    
    # 修复ID字段格式问题 - 确保ID以字符串形式保存
    id_columns = ['comment_id', 'aweme_id', 'post_id', 'user_id', 'note_id', 'parent_comment_id']
    for col in id_columns:
        if col in normalized_df.columns:
            # 处理可能的科学计数法问题
            normalized_df[col] = normalized_df[col].apply(
                lambda x: str(int(float(x))) if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit()) else str(x) if pd.notna(x) else ''
            )
    
    # 映射post_id
    if platform == 'douyin':
        # 确保aweme_id存在且不为空
        if 'aweme_id' in normalized_df.columns:
            # 转换为字符串以避免科学计数法问题
            normalized_df['post_id'] = normalized_df['aweme_id']
        else:
            # 如果aweme_id不存在，使用comment_id作为备用
            normalized_df['post_id'] = normalized_df['comment_id']
    else:  # xhs
        # 小红书使用note_id
        if 'note_id' in normalized_df.columns:
            normalized_df['post_id'] = normalized_df['note_id']
        else:
            normalized_df['post_id'] = normalized_df['comment_id']
    
    # 补充post_user_id - 确保使用KOL的ID而不是评论发布者的ID
    # 如果提供了kol_id，优先使用
    if kol_id:
        normalized_df['post_user_id'] = kol_id
    # 否则尝试从user_id_content填充（这通常是内容发布者ID）
    elif 'user_id_content' in normalized_df.columns:
        # 优先使用非空的user_id_content
        normalized_df['post_user_id'] = normalized_df['user_id_content'].apply(
            lambda x: str(x) if pd.notna(x) else 'unknown_kol'
        )
    # 如果没有合适的KOL ID，设置为unknown_kol并记录警告
    else:
        normalized_df['post_user_id'] = 'unknown_kol'
        print(f"[WARNING] 平台 {platform} 数据中未找到KOL ID信息")
    
    # 统一时间处理 - 正确区分帖子和评论发布时间
    # 根据用户提供的映射关系处理时间字段
    if platform == 'douyin':
        # 抖音平台
        # 处理comment_time_ts - 评论的发布时间
        if 'create_time_comment' in normalized_df.columns:
            normalized_df['comment_time_ts_ms'] = normalized_df['create_time_comment'].apply(convert_timestamp)
        elif 'create_time' in normalized_df.columns:
            normalized_df['comment_time_ts_ms'] = normalized_df['create_time'].apply(convert_timestamp)
        else:
            normalized_df['comment_time_ts_ms'] = None
            
        # 处理post_time_ts - 帖子的发布时间
        if 'create_time_content' in normalized_df.columns:
            normalized_df['post_time_ts_ms'] = normalized_df['create_time_content'].apply(convert_timestamp)
        else:
            # 如果没有直接的帖子时间，尝试从字典获取
            normalized_df['post_time_ts_ms'] = None
            if post_time_dict and 'post_id' in normalized_df.columns:
                normalized_df['post_time_ts_ms'] = normalized_df['post_id'].map(lambda x: post_time_dict.get(x) if pd.notna(x) else None)
    else:  # xhs
        # 小红书平台
        # 处理comment_time_ts - 评论的发布时间
        if 'create_time' in normalized_df.columns:
            normalized_df['comment_time_ts_ms'] = normalized_df['create_time'].apply(convert_timestamp)
        else:
            normalized_df['comment_time_ts_ms'] = None
            
        # 处理post_time_ts - 帖子的发布时间
        if 'time' in normalized_df.columns:
            normalized_df['post_time_ts_ms'] = normalized_df['time'].apply(convert_timestamp)
        else:
            # 如果没有直接的帖子时间，尝试从字典获取
            normalized_df['post_time_ts_ms'] = None
            if post_time_dict and 'post_id' in normalized_df.columns:
                normalized_df['post_time_ts_ms'] = normalized_df['post_id'].map(lambda x: post_time_dict.get(x) if pd.notna(x) else None)
    
    # 如果提供了帖子信息字典，添加互动数据和内容信息
    if post_info_dict and 'post_id' in normalized_df.columns:
        # 创建映射函数提取帖子信息
        def get_post_info(post_id, field):
            if pd.notna(post_id) and post_id in post_info_dict:
                return post_info_dict[post_id].get(field, 0)
            return 0
        
        # 添加帖子互动数据
        normalized_df['post_likes'] = normalized_df['post_id'].map(lambda x: get_post_info(x, 'liked_count'))
        normalized_df['post_comments'] = normalized_df['post_id'].map(lambda x: get_post_info(x, 'comment_count'))
        normalized_df['post_shares'] = normalized_df['post_id'].map(lambda x: get_post_info(x, 'share_count'))
        
        # 添加收藏数（小红书特有，但也为抖音预留）
        normalized_df['post_collects'] = normalized_df['post_id'].map(lambda x: get_post_info(x, 'collected_count'))
        
        # 添加帖子内容信息
        normalized_df['post_title'] = normalized_df['post_id'].map(lambda x: post_info_dict[x].get('title', '') if pd.notna(x) and x in post_info_dict else '')
        normalized_df['post_desc'] = normalized_df['post_id'].map(lambda x: post_info_dict[x].get('desc', '') if pd.notna(x) and x in post_info_dict else '')
        

    
    # 保留原始字段名用于向后兼容
    normalized_df['post_time_ts'] = normalized_df['post_time_ts_ms']
    normalized_df['comment_time_ts'] = normalized_df['comment_time_ts_ms']
    

    
    normalized_df['post_dt_local'] = normalized_df['post_time_ts_ms'].apply(to_local_time)
    normalized_df['comment_dt_local'] = normalized_df['comment_time_ts_ms'].apply(to_local_time)
    
    # 添加UTC时间列 - 新代码
    normalized_df['post_dt_utc'] = normalized_df['post_time_ts_ms'].apply(timestamp_to_datetime)
    normalized_df['comment_dt_utc'] = normalized_df['comment_time_ts_ms'].apply(timestamp_to_datetime)
    
    # 保留原字段名用于向后兼容
    normalized_df['post_dt'] = normalized_df['post_dt_utc']
    normalized_df['comment_dt'] = normalized_df['comment_dt_utc']
    
    # 映射评论用户ID - 优先使用user_id_comment（根据用户提供的映射关系）
    if 'user_id_comment' in normalized_df.columns:
        normalized_df['comment_user_id'] = normalized_df['user_id_comment'].fillna('')
    # 然后尝试其他可能的列名
    else:
        possible_user_id_columns = ['user_id', 'comment_user_id', 'user']
        found_column = None
        for col in possible_user_id_columns:
            if col in normalized_df.columns:
                found_column = col
                break
        
        if found_column:
            normalized_df['comment_user_id'] = normalized_df[found_column].fillna('')
        else:
            normalized_df['comment_user_id'] = ''  # 如果没有找到，设置为空字符串
    
    # 清洗文本内容
    if 'content' in normalized_df.columns:
        normalized_df['comment_text'] = normalized_df['content'].apply(clean_text)
    else:
        normalized_df['comment_text'] = ''
    
    # 新增：生成comment_text_clean字段（HTML解码、去URL、去冗余空格）
    def clean_text_for_nlp(text):
        if pd.isna(text) or text is None:
            return ''
        # 移除URL（替换为<URL>）
        import re
        text = re.sub(r'https?://\\S+|www\\.\\S+', '<URL>', str(text))
        # 去冗余空格
        text = re.sub(r'\\s+', ' ', text)
        # 移除控制字符
        text = ''.join(char for char in text if ord(char) >= 32)
        return text.strip()
    
    normalized_df['comment_text_clean'] = normalized_df['comment_text'].apply(clean_text_for_nlp)
    
    # 修正数值字段类型为整数
    numeric_columns = ['comment_like_count', 'sub_comment_count', 'reply_count', 'like_count']
    for col in numeric_columns:
        if col in normalized_df.columns:
            normalized_df[col] = pd.to_numeric(normalized_df[col], errors='coerce').fillna(0).astype(int)
    
    # 确保comment_like_count存在且为整数
    if 'comment_like_count' not in normalized_df.columns and 'like_count' in normalized_df.columns:
        normalized_df['comment_like_count'] = normalized_df['like_count']
    
    # 保留parent_comment_id
    if 'parent_comment_id' in normalized_df.columns:
        # 处理抖音的parent_comment_id为0的情况
        if platform == 'douyin':
            normalized_df['parent_comment_id'] = normalized_df['parent_comment_id'].apply(
                lambda x: None if pd.isna(x) or str(x) == '0' or str(x) == '0.0' or str(x) == '' else str(x)
            )
        else:
            normalized_df['parent_comment_id'] = normalized_df['parent_comment_id'].apply(
                lambda x: None if pd.isna(x) or str(x) == '0' or str(x) == '' else str(x)
            )
    else:
        normalized_df['parent_comment_id'] = None
    
    # 添加is_top_level标识
    normalized_df['is_top_level'] = normalized_df['parent_comment_id'].isna()
    
    # 计算hours_since_post和time_bucket
    def calculate_hours_since_post(comment_ts, post_ts):
        if pd.isna(comment_ts) or pd.isna(post_ts):
            return None
        return (comment_ts - post_ts) / 3600000.0  # 转换为小时
    
    def get_time_bucket(hours):
        if pd.isna(hours) or hours is None:
            return 'unknown'
        if hours < 24:
            return '0-24h'
        elif hours < 120:  # 5天
            return '1-5d'
        else:
            return '>5d'
    
    normalized_df['hours_since_post'] = normalized_df.apply(
        lambda x: calculate_hours_since_post(x['comment_time_ts'], x['post_time_ts']), axis=1
    )
    normalized_df['time_bucket'] = normalized_df['hours_since_post'].apply(get_time_bucket)
    
    # 确保抽样元信息被保留
    if 'is_sampled' not in normalized_df.columns:
        normalized_df['is_sampled'] = False
    if 'sample_method' not in normalized_df.columns:
        normalized_df['sample_method'] = 'full_data'
    
    # 计算评论长度
    normalized_df[['comment_length', 'comment_length_category']] = \
        pd.DataFrame(normalized_df['comment_text'].apply(analyze_comment_length).tolist(), 
                     index=normalized_df.index)
    
    # 处理子评论计数 - 优先使用sub_comment_count（根据用户提供的映射关系）
    if 'sub_comment_count' in normalized_df.columns:
        normalized_df['sub_comment_count'] = normalized_df['sub_comment_count'].apply(parse_chinese_number)
        # 同时设置reply_count用于向后兼容
        normalized_df['reply_count'] = normalized_df['sub_comment_count']
    elif 'reply_count' in normalized_df.columns:
        normalized_df['reply_count'] = normalized_df['reply_count'].apply(parse_chinese_number)
        normalized_df['sub_comment_count'] = normalized_df['reply_count']
    else:
        normalized_df['sub_comment_count'] = 0
        normalized_df['reply_count'] = 0
    
    # 添加平台标识（兼容原有设计）
    normalized_df['platform_name'] = '抖音' if platform == 'douyin' else '小红书'
    normalized_df['platform_code'] = 'dy' if platform == 'douyin' else 'xhs'
    
    # 尝试添加帖子相关信息（如果存在）
    # 帖子内容字段映射 - 确保使用正确的字段名
    post_content_fields = {
        'title': ['title', 'desc'],  # 可能的标题字段
        'content': ['content', 'desc', 'post_content']  # 可能的内容字段
    }
    
    # 为帖子信息初始化字段
    for field in ['post_title', 'post_desc', 'post_liked_count', 'post_collected_count', 'post_comment_count', 'post_share_count']:
        if field not in normalized_df.columns:
            normalized_df[field] = 0  # 使用0而不是None作为默认值，方便后续计算
    
    # 直接从DataFrame中获取帖子互动数据（根据用户提供的映射关系）
    if 'liked_count' in normalized_df.columns:
        normalized_df['post_liked_count'] = normalized_df['liked_count'].apply(parse_chinese_number)
    if 'collected_count' in normalized_df.columns:
        normalized_df['post_collected_count'] = normalized_df['collected_count'].apply(parse_chinese_number)
    if 'comment_count' in normalized_df.columns:
        normalized_df['post_comment_count'] = normalized_df['comment_count'].apply(parse_chinese_number)
    if 'share_count' in normalized_df.columns:
        normalized_df['post_share_count'] = normalized_df['share_count'].apply(parse_chinese_number)
    
    # 尝试从原始列中提取帖子标题和描述
    if 'title' in normalized_df.columns:
        normalized_df['post_title'] = normalized_df['title'].apply(clean_text)
    if 'desc' in normalized_df.columns:
        normalized_df['post_desc'] = normalized_df['desc'].apply(clean_text)
    
    # 添加帖子昵称信息
    if 'nickname_content' in normalized_df.columns:
        normalized_df['post_nickname'] = normalized_df['nickname_content'].fillna('')
    
    # 如果提供了帖子信息字典，使用它来填充帖子相关字段
    if post_info_dict and 'post_id' in normalized_df.columns:
        def fill_post_info(row):
            post_id = row['post_id']
            if post_id in post_info_dict:
                info = post_info_dict[post_id]
                # 填充帖子互动数据
                row['post_liked_count'] = info.get('liked_count', 0)
                row['post_collected_count'] = info.get('collected_count', 0)
                row['post_comment_count'] = info.get('comment_count', 0)
                row['post_share_count'] = info.get('share_count', 0)
                # 填充帖子内容信息（如果在字典中且当前为空）
                if not row.get('post_title') and 'title' in info:
                    row['post_title'] = clean_text(info['title'])
                if not row.get('post_desc') and 'desc' in info:
                    row['post_desc'] = clean_text(info['desc'])
            return row
        
        normalized_df = normalized_df.apply(fill_post_info, axis=1)
    
    
    # 添加昵称相关字段
    if 'nickname' in normalized_df.columns:
        normalized_df['nickname'] = normalized_df['nickname'].fillna('')
    if 'kol_name' not in normalized_df.columns:
        normalized_df['kol_name'] = kol_id if kol_id else 'unknown'
    
    # 添加IP位置信息 - 优先使用ip_location_comment（根据用户提供的映射关系）
    if 'ip_location_comment' in normalized_df.columns:
        normalized_df['ip_location_comment'] = normalized_df['ip_location_comment'].fillna('')
    elif 'ip_location' in normalized_df.columns:
        normalized_df['ip_location_comment'] = normalized_df['ip_location'].fillna('')
    else:
        normalized_df['ip_location_comment'] = ''
    
    return normalized_df

def merge_platform_data(douyin_file, xhs_file, output_dir, kol_name):
    """合并两个平台的数据并进行标准化处理"""
    print(f"[INFO] 开始处理KOL: {kol_name} 的数据文件: {os.path.basename(douyin_file)} 和 {os.path.basename(xhs_file)}")
    
    start_time = time.time()
    
    # 尝试从对应的content文件中获取KOL ID、帖子发布时间、互动数据和内容信息
    douyin_kol_id = None
    xhs_kol_id = None
    douyin_post_times = {}
    xhs_post_times = {}
    
    # 存储帖子的互动数据和内容信息
    douyin_post_info = {}
    xhs_post_info = {}
    
    # 构建content文件路径
    douyin_content_file = douyin_file.replace('comments', 'contents')
    xhs_content_file = xhs_file.replace('comments', 'contents')
    
    # 读取抖音内容文件以获取KOL ID、帖子发布时间、互动数据和内容信息
    if os.path.exists(douyin_content_file):
        try:
            douyin_content_df = safe_read_csv(douyin_content_file)
            # 提取抖音KOL ID
            if 'user_id' in douyin_content_df.columns and not douyin_content_df['user_id'].isna().all():
                # 获取非空的第一个user_id作为KOL ID
                douyin_kol_id = str(douyin_content_df['user_id'].dropna().iloc[0])
                print(f"[INFO] 从抖音内容文件获取KOL ID: {douyin_kol_id}")
            
            # 提取抖音帖子数据（发布时间、互动数据、标题、描述）
            if 'aweme_id' in douyin_content_df.columns:
                for _, row in douyin_content_df.iterrows():
                    if pd.notna(row['aweme_id']):
                        post_id = str(row['aweme_id']).split('.')[0] if '.' in str(row['aweme_id']) else str(row['aweme_id'])
                        
                        # 提取帖子发布时间
                        post_time = None
                        if 'create_time' in row:
                            post_time = convert_timestamp(row['create_time'])
                        
                        # 提取帖子互动数据
                        liked_count = int(row['liked_count']) if 'liked_count' in row and pd.notna(row['liked_count']) else 0
                        collected_count = int(row['collected_count']) if 'collected_count' in row and pd.notna(row['collected_count']) else 0
                        comment_count = int(row['comment_count']) if 'comment_count' in row and pd.notna(row['comment_count']) else 0
                        share_count = int(row['share_count']) if 'share_count' in row and pd.notna(row['share_count']) else 0
                        
                        # 提取帖子标题和描述
                        title = str(row['title']) if 'title' in row and pd.notna(row['title']) else ''
                        desc = str(row['desc']) if 'desc' in row and pd.notna(row['desc']) else ''
                        
                        # 保存帖子发布时间
                        douyin_post_times[post_id] = post_time
                        
                        # 保存帖子互动数据和内容信息
                        douyin_post_info[post_id] = {
                            'liked_count': liked_count,
                            'collected_count': collected_count,
                            'comment_count': comment_count,
                            'share_count': share_count,
                            'title': title,
                            'desc': desc
                        }
                print(f"[INFO] 从抖音内容文件提取了 {len(douyin_post_times)} 条帖子发布时间和互动数据")
        except Exception as e:
            print(f"[WARNING] 读取抖音内容文件时出错: {e}")
    
    # 读取小红书内容文件以获取KOL ID、帖子发布时间、互动数据和内容信息
    if os.path.exists(xhs_content_file):
        try:
            xhs_content_df = safe_read_csv(xhs_content_file)
            # 提取小红书KOL ID
            if 'user_id' in xhs_content_df.columns and not xhs_content_df['user_id'].isna().all():
                # 获取非空的第一个user_id作为KOL ID
                xhs_kol_id = str(xhs_content_df['user_id'].dropna().iloc[0])
                print(f"[INFO] 从小红书内容文件获取KOL ID: {xhs_kol_id}")
            
            # 提取小红书帖子发布时间和互动数据
            if 'note_id' in xhs_content_df.columns:
                for _, row in xhs_content_df.iterrows():
                    if pd.notna(row['note_id']):
                        post_id = str(row['note_id'])
                        
                        # 提取帖子发布时间
                        post_time = None
                        if 'time' in row:
                            post_time = convert_timestamp(row['time'])
                        elif 'create_time' in row:
                            post_time = convert_timestamp(row['create_time'])
                        
                        # 提取帖子互动数据 - 注意小红书的字段名可能不同
                        liked_count = int(row['liked_count']) if 'liked_count' in row and pd.notna(row['liked_count']) else 0
                        collected_count = int(row['collected_count']) if 'collected_count' in row and pd.notna(row['collected_count']) else 0
                        comment_count = int(row['comment_count']) if 'comment_count' in row and pd.notna(row['comment_count']) else 0
                        share_count = int(row['share_count']) if 'share_count' in row and pd.notna(row['share_count']) else 0
                        
                        # 提取帖子标题和描述
                        title = str(row['title']) if 'title' in row and pd.notna(row['title']) else ''
                        desc = str(row['desc']) if 'desc' in row and pd.notna(row['desc']) else ''
                        
                        if post_time is not None:
                            # 保存帖子发布时间
                            xhs_post_times[post_id] = post_time
                            
                            # 保存帖子互动数据和内容信息
                            xhs_post_info[post_id] = {
                                'liked_count': liked_count,
                                'collected_count': collected_count,
                                'comment_count': comment_count,
                                'share_count': share_count,
                                'title': title,
                                'desc': desc
                            }
                print(f"[INFO] 从小红书内容文件提取了 {len(xhs_post_times)} 条帖子发布时间和互动数据")
        except Exception as e:
            print(f"[WARNING] 读取小红书内容文件时出错: {e}")
    
    # 读取评论数据
    douyin_df = safe_read_csv(douyin_file)
    xhs_df = safe_read_csv(xhs_file)
    
    print(f"[INFO] 抖音数据读取完成，共 {len(douyin_df)} 条记录")
    print(f"[INFO] 小红书数据读取完成，共 {len(xhs_df)} 条记录")
    
    # 为原始数据添加平台标识
    douyin_df['platform'] = 'douyin'
    xhs_df['platform'] = 'xhs'
    douyin_df['platform_code'] = 'dy'
    xhs_df['platform_code'] = 'xhs'
    
    # 垂直合并两表（保留所有原始列）
    merged_raw = pd.concat([douyin_df, xhs_df], ignore_index=True)
    print(f"[INFO] 原始数据合并完成，共 {len(merged_raw)} 条记录")
    
    # 保存原始合并数据 - 新需求
    merged_raw_path = os.path.join(output_dir, 'merged_raw.csv')
    merged_raw.to_csv(merged_raw_path, index=False, encoding='utf-8-sig')
    print(f"[INFO] 已保存原始合并数据到 {merged_raw_path}")
    
    # 标准化数据
    print("[INFO] 开始标准化数据...")
    # 增加详细的日志输出
    print(f"[DEBUG] 抖音数据标准化前形状: {douyin_df.shape}, 列: {list(douyin_df.columns)[:5]}...")
    print(f"[DEBUG] 小红书数据标准化前形状: {xhs_df.shape}, 列: {list(xhs_df.columns)[:5]}...")
    
    # 传入KOL ID和帖子数据字典进行标准化
    douyin_normalized = normalize_data(douyin_df, 'douyin', douyin_kol_id, douyin_post_times, douyin_post_info)
    xhs_normalized = normalize_data(xhs_df, 'xhs', xhs_kol_id, xhs_post_times, xhs_post_info)
    
    # 检查标准化后的数据
    print(f"[DEBUG] 抖音数据标准化后形状: {douyin_normalized.shape}")
    print(f"[DEBUG] 小红书数据标准化后形状: {xhs_normalized.shape}")
    
    # 合并标准化数据
    merged_normalized = pd.concat([douyin_normalized, xhs_normalized], ignore_index=True)
    print(f"[DEBUG] 合并后标准化数据形状: {merged_normalized.shape}")
    print(f"[DEBUG] 合并后标准化数据平台分布: {merged_normalized['platform'].value_counts() if 'platform' in merged_normalized.columns else 'No platform column'}")
    
    # 添加KOL名称
    merged_normalized['kol_name'] = kol_name
    
    # 按时间排序，过滤无效时间
    if 'comment_dt_utc' in merged_normalized.columns:
        valid_time_df = merged_normalized.dropna(subset=['comment_dt_utc'])
        invalid_time_df = merged_normalized[merged_normalized['comment_dt_utc'].isna()]
        
        # 对有效时间记录排序
        if not valid_time_df.empty:
            valid_time_df = valid_time_df.sort_values('comment_dt_utc', ascending=False)
        
        # 合并排序后的有效记录和无效时间记录
        merged_normalized = pd.concat([valid_time_df, invalid_time_df], ignore_index=True)
    
    print(f"[INFO] 标准化数据合并完成，共 {len(merged_normalized)} 条记录")
    

    

    
    # 定义需要保留的字段 - 根据用户需求
    required_fields = [
        # 必须保留的字段
        'platform', 'post_id', 'post_user_id', 'post_time_ts', 'post_dt',
        'comment_id', 'comment_user_id', 'comment_time_ts', 'comment_dt',
        'comment_text', 'comment_like_count', 'parent_comment_id', 'sub_comment_count',
        # 建议保留的字段
        'nickname', 'kol_name', 'platform_name', 'platform_code', 'interaction_score',
        # 可选的地域字段
        'ip_location', 'ip_location_comment',
        # 新增的派生字段
        'is_top_level', 'parent_missing', 'hours_since_post', 'time_bucket',
        'comment_text_clean', 'comment_length', 'comment_length_category',
        # 帖子相关信息
        'post_title', 'post_desc', 'post_liked_count', 'post_collected_count', 'post_comment_count', 'post_share_count'
    ]
    
    # 过滤出数据中存在的字段
    available_fields = [field for field in required_fields if field in merged_normalized.columns]
    filtered_normalized = merged_normalized[available_fields].copy()
    
    # 保存精简的标准化数据 - 根据用户需求
    merged_normalized_path = os.path.join(output_dir, 'merged_normalized.csv')
    filtered_normalized.to_csv(merged_normalized_path, index=False, encoding='utf-8-sig')
    print(f"[INFO] 已保存精简的标准化数据到 {merged_normalized_path}")
    print(f"[INFO] 精简后的字段数量: {len(filtered_normalized.columns)}")
    print(f"[INFO] 保留的字段列表: {', '.join(available_fields)}")
    

    
    # 保存扩展格式（兼容原有设计）
    # 完整版本
    full_path = os.path.join(output_dir, f"merged_cross_platform_{kol_name}_full.csv")
    merged_normalized.to_csv(full_path, index=False, encoding='utf-8-sig')
    
    # 简化版本 - 更新字段名以匹配最新的标准化数据
    core_fields = ['global_id', 'kol_name', 'platform_name', 'platform_code', 'comment_dt_utc', 'comment_text', 
                   'comment_user_id', 'comment_like_count', 'reply_count', 'interaction_score', 'post_id', 
                   'is_top_level', 'is_sampled', 'sample_method']
    available_core_fields = [field for field in core_fields if field in merged_normalized.columns]
    simple_df = merged_normalized[available_core_fields]
    simple_path = os.path.join(output_dir, f"merged_cross_platform_{kol_name}_simple.csv")
    simple_df.to_csv(simple_path, index=False, encoding='utf-8-sig')
    
    # 分平台保存
    for platform, name in [('douyin', '抖音'), ('xhs', '小红书')]:
        platform_df = merged_normalized[merged_normalized['platform'] == platform]
        if not platform_df.empty:
            platform_path = os.path.join(output_dir, f"merged_{name}_{kol_name}.csv")
            platform_df.to_csv(platform_path, index=False, encoding='utf-8-sig')
    
    end_time = time.time()
    print(f"\n[INFO] 数据处理完成！耗时: {end_time - start_time:.2f} 秒")
    print(f"[INFO] 原始数据: {merged_raw_path}")
    print(f"[INFO] 精简标准化数据: {merged_normalized_path}")


    print(f"[INFO] 完整版本: {full_path}")
    print(f"[INFO] 简化版本: {simple_path}")
    
    return merged_raw, filtered_normalized

def main():
    """主函数 - 结合原有设计与新需求"""
    print("=" * 80)
    print("增强版平台数据融合工具")
    print("结合原有merge_platform_data.py的设计与新需求的数据处理流程")
    print("=" * 80)
    
    # 直接定义所有需要的文件路径
    print("[INFO] 使用预设的文件路径配置...")
    kols_files = {
        'doudou': {
            'douyin': 'c:\\Users\\联想\\Desktop\\shuju\\合并数据\\outputs_dy_doudou\\merged_master_clean_dy_doudou.csv',
            'xhs': 'c:\\Users\\联想\\Desktop\\shuju\\合并数据\\outputs_xhs_all\\merged_master_clean_xhs_doudou.csv'
        },
        'zky': {
            'douyin': 'c:\\Users\\联想\\Desktop\\shuju\\合并数据\\outputs_dy_zky\\merged_master_clean_dy_zky.csv',
            'xhs': 'c:\\Users\\联想\\Desktop\\shuju\\合并数据\\outputs_xhs_all\\merged_master_clean_xhs_zky.csv'
        },
        'xls': {
            'douyin': 'c:\\Users\\联想\\Desktop\\shuju\\合并数据\\outputs_dy_xls\\merged_master_clean_dy_xls.csv',
            'xhs': 'c:\\Users\\联想\\Desktop\\shuju\\合并数据\\outputs_xhs_all\\merged_master_clean_xhs_xls.csv'
        }
    }
    
    # 验证文件是否存在
    valid_kols = {}
    for kol_name, files in kols_files.items():
        if os.path.exists(files['douyin']) and os.path.exists(files['xhs']):
            valid_kols[kol_name] = files
            print(f"[INFO] 已确认KOL {kol_name} 的文件存在")
        else:
            if not os.path.exists(files['douyin']):
                print(f"[WARNING] KOL {kol_name} 的抖音文件不存在: {files['douyin']}")
            if not os.path.exists(files['xhs']):
                print(f"[WARNING] KOL {kol_name} 的小红书文件不存在: {files['xhs']}")
    
    kols_files = valid_kols
    
    if not kols_files:
        print("[ERROR] 未找到符合要求的抖音和小红书CSV文件")
        return
    
    print(f"[INFO] 找到 {len(kols_files)} 个KOL的数据文件对:")
    for kol in kols_files.keys():
        print(f"  - {kol}")
    print()
    
    # 创建输出目录
    base_output_dir = os.path.join(os.getcwd(), 'cross_platform_merged_enhanced')
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"[INFO] 输出目录: {base_output_dir}")
    
    # 保存所有KOL的数据用于汇总分析
    all_kol_data = []
    all_metrics = {}
    
    # 处理每个KOL的数据
    for kol_name, files in kols_files.items():
        print(f"\n{'=' * 60}")
        print(f"[INFO] 开始处理KOL: {kol_name}")
        print(f"{'=' * 60}")
        
        # 创建KOL专属输出目录
        kol_output_dir = os.path.join(base_output_dir, f'kol_{kol_name}')
        os.makedirs(kol_output_dir, exist_ok=True)
        
        try:
            # 合并数据
            douyin_file = files['douyin']
            xhs_file = files['xhs']
            merged_raw, filtered_normalized = merge_platform_data(
                douyin_file, xhs_file, kol_output_dir, kol_name
            )
            
            # 添加到汇总数据（使用过滤后的标准化数据）
            all_kol_data.append(filtered_normalized)
            
        except Exception as e:
            print(f"[ERROR] 处理KOL {kol_name} 过程中出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 合并所有KOL的数据（如果有）
    if all_kol_data:
        # 合并所有KOL的过滤后数据
        all_merged_df = pd.concat(all_kol_data, ignore_index=True)
        
        # 定义汇总数据需要保留的字段
        summary_required_fields = [
            # 必须保留的字段
            'platform', 'post_id', 'post_user_id', 'post_time_ts', 'post_dt',
            'comment_id', 'comment_user_id', 'comment_time_ts', 'comment_dt',
            'comment_text', 'comment_like_count', 'parent_comment_id', 'sub_comment_count',
            # 建议保留的字段
            'nickname', 'kol_name', 'platform_name', 'platform_code', 'interaction_score',
            # 可选的地域字段
            'ip_location', 'ip_location_comment',
            # 新增的派生字段
            'is_top_level', 'parent_missing', 'hours_since_post', 'time_bucket',
            'comment_text_clean', 'comment_length', 'comment_length_category'
        ]
        
        # 过滤出数据中存在的字段
        summary_available_fields = [field for field in summary_required_fields if field in all_merged_df.columns]
        filtered_summary_df = all_merged_df[summary_available_fields].copy()
        
        # 添加全局ID
        filtered_summary_df['global_id_all'] = [f"all_{i}" for i in range(len(filtered_summary_df))]
        
        # 保存汇总数据
        all_merged_path = os.path.join(base_output_dir, "merged_cross_platform_all_kol_enhanced.csv")
        filtered_summary_df.to_csv(all_merged_path, index=False, encoding='utf-8-sig')
        print(f"\n[INFO] 所有KOL的跨平台数据已合并并保存至: {all_merged_path}")
        print(f"[INFO] 汇总数据保留字段数量: {len(filtered_summary_df.columns)}")
        print(f"[INFO] 汇总数据字段列表: {', '.join(filtered_summary_df.columns)}")
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)