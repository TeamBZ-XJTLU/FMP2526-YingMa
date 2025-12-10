import os
import json
import time
import logging
import traceback
import requests
import shutil
import pandas as pd
import signal
from tqdm import tqdm
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

# 全局变量：跟踪API服务状态
API_SERVICE_AVAILABLE = True  # 默认为可用状态

# 键盘中断处理函数
def signal_handler(sig, frame):
    global API_SERVICE_AVAILABLE
    logger.info("收到键盘中断信号 (SIGINT/SIGTERM)，立即保存数据并强制退出...")
    API_SERVICE_AVAILABLE = False  # 通知程序停止处理新数据
    # 直接抛出SystemExit异常强制退出程序
    raise SystemExit("程序被用户中断，正在退出...")

# 注册信号处理
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def save_progress(df, output_file):
    """
    保存当前处理进度的辅助函数
    
    Args:
        df: 包含处理结果的数据框
        output_file: 输出文件路径
    """
    temp_output_file = output_file + '.temp'
    try:
        df.to_csv(temp_output_file, index=False, encoding='utf-8-sig')
        # 原子性重命名
        if os.path.exists(temp_output_file):
            os.replace(temp_output_file, output_file)
        logger.info(f"已安全保存当前进度到 {output_file}")
    except Exception as save_error:
        logger.error(f"保存进度失败: {str(save_error)}")
        # 备用保存策略 - 使用固定名称覆盖之前的备份
        backup_file = os.path.join(OUTPUT_DIR, "backup_progress.csv")
        try:
            df.to_csv(backup_file, index=False, encoding='utf-8-sig')
            logger.warning(f"已保存备用进度到 {backup_file}")
        except Exception as e2:
            logger.error(f"备用保存也失败: {e2}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ark API 配置
LLM_API_KEY = "1c6ec227-dc74-47d8-a951-b4a0bcf0ab83"
LLM_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
LLM_MODEL = "deepseek-v3-1-terminus"

# 确保输出目录存在
OUTPUT_DIR = "sentiment_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def batch_analyze_sentiment_with_llm(texts: list, max_retries: int = 3) -> list:
    """
    批量使用LLM API分析多条文本的情感
    
    Args:
        texts: 文本列表
        max_retries: 最大重试次数
        
    Returns:
        结果列表，每个元素为 (sentiment_score, needs_review) 元组
    """
    global API_SERVICE_AVAILABLE
    
    # 过滤空文本
    filtered_texts = [text.strip() if text and text.strip() else "" for text in texts]
    
    # 构建批量请求的提示
    batch_content = "请对以下每条评论文本进行情感分析，每条都输出一行JSON。\n\n"
    for i, text in enumerate(filtered_texts):
        batch_content += f"[{i+1}] 文本：{text}\n"
    
    # 构建简化的请求体，减少prompt长度
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个专门用于中国社交媒体评论的情感评分助手。你只返回一行合法 JSON，且只能包含两个键：{\"sentiment_score\":0.0-1.0,\"confidence\":0.0-1.0}。sentiment_score: 0.0 非常负面，0.5 中性，1.0 非常正面；confidence: 表示模型对该评分的置信度（0-1）。"
            },
            {
                "role": "user",
                "content": batch_content + "\n请严格按照序号顺序，每行一个JSON，不要任何其他内容或说明。"
            }
        ],
        "temperature": 0.0
    }
    
    # 初始化结果列表，默认所有结果都需要人工复查
    results = [(None, True) for _ in filtered_texts]

    retry_count = 0
    # 增加最大重试次数并实现指数退避
    while retry_count <= max_retries:
        try:
            # 确保API服务可用
            if not API_SERVICE_AVAILABLE:
                logger.warning("API服务已被标记为不可用，跳过请求")
                return results
            
            headers = {
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json"
            }
            
            logger.debug(f"发送批量情感分析请求，包含 {len(filtered_texts)} 条文本，第 {retry_count+1}/{max_retries+1} 次尝试")
            
            # 增加超时时间，使用会话对象以重用连接
            session = requests.Session()
            session.keep_alive = True
            
            # 优化超时时间设置
            # 初始超时设置为45秒，每次重试增加15秒，但不超过90秒
            timeout = min(45 + retry_count * 15, 90)
            logger.debug(f"请求超时设置为 {timeout} 秒")
            response = session.post(
                LLM_API_URL, 
                json=payload, 
                headers=headers,
                timeout=timeout  # 动态增加超时时间
            )
            session.close()
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"收到API响应: {json.dumps(result)[:200]}...")
                
                # 处理响应
                try:
                    if 'choices' in result and len(result['choices']) > 0:
                        
                        content_text = result['choices'][0]['message']['content']
                        
                        if content_text:
                            # 按行解析JSON
                            json_lines = content_text.strip().split('\n')
                            for i, line in enumerate(json_lines):
                                if i < len(results):  # 确保索引不越界
                                    try:
                                        line_data = json.loads(line.strip())
                                        score = line_data.get('sentiment_score', None)
                                        confidence = line_data.get('confidence', 0.0)
                                        
                                        # 验证分数格式
                                        if isinstance(score, (int, float)) and 0 <= score <= 1:
                                            # 如果置信度低，则标记为需要人工复查
                                            needs_review = confidence < 0.5
                                            results[i] = (float(score), needs_review)
                                            logger.debug(f"文本 {i+1} 情感分析成功: 分数={score}, 置信度={confidence}")
                                    except json.JSONDecodeError:
                                        logger.error(f"解析文本 {i+1} 的JSON响应失败: {line}")
                        else:
                            logger.error("API返回的content格式不正确或为空")
                            retry_count += 1
                            time.sleep(2 ** retry_count)  # 指数退避
                            continue
                    else:
                        logger.error("API响应不包含choices字段或为空")
                        retry_count += 1
                        time.sleep(2 ** retry_count)  # 指数退避
                        continue
                    
                    # 即使部分解析失败，也返回当前结果
                    return results
                except Exception as e:
                    logger.error(f"批量情感分析发生未预期错误: {str(e)}")
                    logger.debug(traceback.format_exc())
                    retry_count += 1
                    time.sleep(5)
                    continue
            else:
                logger.error(f"API请求失败，状态码: {response.status_code}, 响应: {response.text}")
                retry_count += 1
                time.sleep(2 ** retry_count)  # 指数退避
                
                # 如果达到最大重试次数，标记API服务不可用
                if retry_count > max_retries:
                    logger.error("批量情感分析达到最大重试次数，标记API服务不可用")
                    API_SERVICE_AVAILABLE = False
                continue
            
        except requests.exceptions.RequestException as e:
            logger.error(f"批量情感分析网络请求异常: {str(e)}")
            
            # 检查是否是超时错误 - 超时错误时我们应该继续重试
            if isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ReadTimeout)):
                logger.warning(f"请求超时 (当前超时设置: {timeout}秒)，准备重试...")
            else:
                # 其他网络错误也继续重试，但减少频率
                logger.warning(f"网络错误: {str(e)}，准备重试...")
            
            # 指数退避重试，增加随机因子避免雪崩效应
            wait_time = min(2 ** retry_count + random.uniform(1, 3), 120)  # 最多等待120秒
            logger.info(f"{wait_time:.2f}秒后进行第 {retry_count + 1} 次重试")
            time.sleep(wait_time)
            retry_count += 1
            continue
        except Exception as e:
            logger.error(f"批量情感分析发生未预期错误: {str(e)}")
            logger.debug(traceback.format_exc())
            retry_count += 1
            time.sleep(5)
            continue
    
    # 所有重试都失败，返回失败结果
    logger.error(f"批量情感分析失败，已达到最大重试次数")
    return results  # 返回默认结果


def get_sentiment_label(score: Optional[float]) -> str:
    """
    根据情感分数获取情感标签
    """
    if score is None:
        return "manual_review"
    
    if score < 0.4:
        return "neg"
    elif score > 0.6:
        return "pos"
    else:
        return "neu"



def check_network_connection() -> bool:
    """
    检查网络连接状态（优化版，避免完整API请求消耗token）
    """
    try:
        logger.info("检查网络连接...")
        # 测试域名解析
        import socket
        socket.gethostbyname("ark.cn-beijing.volces.com")
        
        # 使用轻量级的HTTP HEAD请求替代GET请求，减少token消耗
        response = requests.head("https://ark.cn-beijing.volces.com/api/v3/chat/completions", headers={"Authorization": f"Bearer {LLM_API_KEY}"}, timeout=10)
        logger.info(f"网络连接测试成功，状态码: {response.status_code}")
        return True
    except Exception as e:
        logger.error(f"网络连接测试失败: {e}")
        return False


def main():
    try:
        # 定义变量
        df = None
        review_queue = []
        processed_count = 0
        skipped_count = 0
        
        if not check_network_connection():
            logger.error("无法连接到网络，请检查网络设置后重试")
            raise SystemExit("网络连接失败，程序终止")
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 定义文件路径
        base_input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_files', 'merged_all_with_weights.csv')
        output_file = os.path.join(OUTPUT_DIR, "merged_all_with_weights_with_sentiment.csv")
        
        # 首先检查最终输出文件是否存在，如果存在则直接加载它
        if os.path.exists(output_file):
            logger.info(f"正在加载已有处理结果: {output_file}")
            df = pd.read_csv(output_file, encoding='utf-8-sig')
            logger.info(f"成功加载 {len(df)} 条记录")
        else:
            # 如果最终输出文件不存在，则从基础输入文件加载
            if not os.path.exists(base_input_file):
                logger.error(f"基础输入文件不存在: {base_input_file}")
                return
            logger.info(f"正在从基础输入文件加载数据: {base_input_file}")
            df = pd.read_csv(base_input_file, encoding='utf-8-sig')
        
        # 确保comment_id列存在
        if 'comment_id' not in df.columns:
            df['comment_id'] = range(1, len(df) + 1)
        

        

        
        if 'sentiment_score_v1' not in df.columns:
            df['sentiment_score_v1'] = None
        if 'sentiment_label_v1' not in df.columns:
            df['sentiment_label_v1'] = None
        if 'needs_human_review' not in df.columns:
            df['needs_human_review'] = False
        
        logger.info(f"开始处理 {len(df)} 条评论数据")
        
        processed_count = 0
        skipped_count = 0
        review_queue = []
        
        # 检查网络连接和API服务状态
        if not check_network_connection() or not API_SERVICE_AVAILABLE:
            if not API_SERVICE_AVAILABLE:
                logger.error("API服务不可用，无法继续处理")
                raise SystemExit("API服务不可用，程序终止")
            else:
                logger.error("网络连接失败，程序退出")
                return
        
        processed_count = 0
        skipped_count = 0
        review_queue = []
        
        if 'comment_id' not in df.columns:
            df['comment_id'] = range(len(df))
        
        logger.info(f"开始处理数据（{len(df)} 条记录）...")
        
        # 筛选出需要处理的记录
        unprocessed_df = df[df['sentiment_score_v1'].isna()]
        unprocessed_count = len(unprocessed_df)
        logger.info(f"总共 {len(df)} 条记录，其中 {unprocessed_count} 条需要处理，{len(df) - unprocessed_count} 条已处理")
        
        with tqdm(total=len(df), desc="处理进度") as pbar:
            # 先更新已处理的记录的进度条
            for idx in df[~df['sentiment_score_v1'].isna()].index:
                skipped_count += 1
                pbar.update(1)
            
            # 批量处理逻辑
            BATCH_SIZE = 5  # 每批处理的文本数量，保持为5条以配合保存逻辑
            
            # 将未处理数据分批
            unprocessed_indices = list(unprocessed_df.index)
            for i in range(0, len(unprocessed_indices), BATCH_SIZE):
                try:
                    # 检查API服务是否仍然可用
                    if not API_SERVICE_AVAILABLE:
                        logger.error("API服务在处理过程中变得不可用，程序终止")
                        # 保存当前进度
                        save_progress(df, output_file)
                        raise SystemExit("API服务不可用，程序终止")
                    
                    batch_indices = unprocessed_indices[i:i+BATCH_SIZE]
                    batch_data = unprocessed_df.loc[batch_indices]
                    
                    # 收集批量文本
                    batch_texts = []
                    comment_data = []
                    
                    for idx, row in batch_data.iterrows():
                        comment_id = row['comment_id']
                        comment_text = str(row.get('comment_text', '')).strip()
                        
                        if not comment_text:
                            # 空文本直接处理
                            df.at[idx, 'sentiment_score_v1'] = None
                            df.at[idx, 'sentiment_label_v1'] = "empty"
                            pbar.update(1)
                            continue
                        
                        batch_texts.append(comment_text)
                        comment_data.append({
                            'idx': idx,
                            'comment_id': comment_id,
                            'comment_text': comment_text,
                            'platform': row.get('platform', ''),
                            'post_id': row.get('post_id', '')
                        })
                    
                    # 如果批次中有文本需要处理
                    if batch_texts:
                        logger.debug(f"批量处理 {len(batch_texts)} 条评论")
                        
                        # 检查API服务状态
                        if not API_SERVICE_AVAILABLE:
                            logger.error("检测到API服务不可用，停止所有处理并强制退出程序")
                            # 直接保存到最终输出文件
                            temp_output_file = output_file + '.temp'
                            try:
                                df.to_csv(temp_output_file, index=False, encoding='utf-8-sig')
                                if os.path.exists(temp_output_file):
                                    os.replace(temp_output_file, output_file)
                                logger.info("已保存当前进度到最终输出文件")
                            except Exception as e:
                                logger.error(f"保存进度失败: {str(e)}")
                            raise SystemExit("API服务不可用，程序已强制退出")
                        
                        # 使用批量处理函数
                        results = batch_analyze_sentiment_with_llm(batch_texts)
                        
                        # 处理批量结果
                        for data, (sentiment_score, needs_review) in zip(comment_data, results):
                            idx = data['idx']
                            
                            if not needs_review and sentiment_score is not None:
                                df.at[idx, 'sentiment_score_v1'] = sentiment_score
                                df.at[idx, 'sentiment_label_v1'] = get_sentiment_label(sentiment_score)
                            else:
                                df.at[idx, 'sentiment_label_v1'] = "manual_review"
                                review_queue.append({
                                    'comment_id': data['comment_id'],
                                    'comment_text': data['comment_text'],
                                    'platform': data['platform'],
                                    'post_id': data['post_id']
                                })
                            
                            if needs_review:
                                df.at[idx, 'needs_human_review'] = True
                            
                            processed_count += 1
                            pbar.update(1)
                        
                        # 保存进度
                        if processed_count % 10 == 0 or i + BATCH_SIZE >= len(unprocessed_indices):
                            temp_output_file = output_file + '.temp'
                            try:
                                df.to_csv(temp_output_file, index=False, encoding='utf-8-sig')
                                if os.path.exists(temp_output_file):
                                    if os.path.exists(output_file):
                                        # 使用固定名称覆盖之前的备份
                                        backup_output = output_file + '.bak'
                                        shutil.copy2(output_file, backup_output)
                                        logger.info(f"已备份原输出文件到: {backup_output}")
                                    os.replace(temp_output_file, output_file)
                                logger.info(f"已安全保存处理结果，共 {processed_count} 条新处理记录")
                            except Exception as save_error:
                                logger.error(f"保存结果失败: {str(save_error)}")
                                # 使用固定名称覆盖之前的备份
                                backup_file = os.path.join(OUTPUT_DIR, "backup_results.csv")
                                try:
                                    df.to_csv(backup_file, index=False, encoding='utf-8-sig')
                                    logger.warning(f"已保存备用结果到 {backup_file}")
                                except Exception as e2:
                                    logger.error(f"备用保存也失败: {e2}")
                        
                        # 批量处理后只等待一次，而不是每条都等待
                        if not API_SERVICE_AVAILABLE:
                            logger.error("检测到API服务不可用或收到中断信号，跳过休息时间继续处理")
                            # 直接保存到最终输出文件
                            temp_output_file = output_file + '.temp'
                            try:
                                df.to_csv(temp_output_file, index=False, encoding='utf-8-sig')
                                if os.path.exists(temp_output_file):
                                    os.replace(temp_output_file, output_file)
                                logger.info("已保存当前进度到最终输出文件")
                            except Exception as e:
                                logger.error(f"保存进度失败: {str(e)}")
                            continue
                        time.sleep(0.3)  # 每0.3秒调用一次批量API，比之前逐条快约5倍
                except Exception as e:
                    logger.error(f"处理批次 {i//BATCH_SIZE + 1} 时出错: {str(e)}")
                    logger.debug(traceback.format_exc())
                    # 更新进度条，确保不会卡死
                    failed_batch_size = min(BATCH_SIZE, len(unprocessed_indices) - i)
                    pbar.update(failed_batch_size)
            
            final_temp_file = output_file + '.final.tmp'
            df.to_csv(final_temp_file, index=False, encoding='utf-8-sig')
            if os.path.exists(final_temp_file):
                if os.path.exists(output_file):
                    try:
                        # 使用固定名称覆盖之前的备份
                        backup_output = output_file + '.bak'
                        shutil.copy2(output_file, backup_output)
                        logger.info(f"已备份原输出文件到: {backup_output}")
                    except Exception as e:
                        logger.warning(f"备份原文件失败，但仍将继续更新结果: {str(e)}")
                try:
                    os.replace(final_temp_file, output_file)
                except Exception:
                    os.rename(final_temp_file, output_file)
            logger.info(f"最终结果已保存到 {output_file}")
            
            if review_queue:
                review_df = pd.DataFrame(review_queue)
                review_file = os.path.join(OUTPUT_DIR, 'review_queue.csv')
                review_df.to_csv(review_file, index=False, encoding='utf-8-sig')
                logger.info(f"已保存 {len(review_queue)} 条需要人工复查的评论到 {review_file}")
            
            total_processed = processed_count + skipped_count
            completed_count = len(df) - df['sentiment_score_v1'].isna().sum()
            success_count = sum(1 for idx, row in df.iterrows() 
                              if pd.notna(row.get('sentiment_score_v1')) and row.get('sentiment_label_v1') != 'manual_review')
            
            logger.info(f"处理统计:")
            logger.info(f"  总数据量: {len(df)} 条")
            logger.info(f"  本次处理: {processed_count} 条")
            logger.info(f"  跳过/恢复: {skipped_count} 条")
            logger.info(f"  成功处理: {success_count} 条")
            logger.info(f"  需要人工复查: {len(review_queue)} 条")
            logger.info(f"  完成率: {completed_count / len(df) * 100:.2f}%")
            
            if 'needs_human_review' not in df.columns:
                df['needs_human_review'] = False
            
            sentiment_counts = df['sentiment_label_v1'].value_counts().to_dict()
            logger.info(f"情感标签分布: {sentiment_counts}")
            logger.info("已添加needs_human_review列用于标识需要人工查看的记录")

            logger.info("数据处理任务完成！")
    except SystemExit as e:
        logger.info(f"捕获到系统退出信号: {e}")
        if df is not None:
            logger.info("正在保存最后进度...")
            # 直接保存到最终输出文件
            temp_output_file = output_file + '.temp'
            try:
                df.to_csv(temp_output_file, index=False, encoding='utf-8-sig')
                if os.path.exists(temp_output_file):
                    os.replace(temp_output_file, output_file)
                logger.info("进度已保存到最终输出文件，程序即将退出。")
            except Exception as save_error:
                logger.error(f"保存进度失败: {str(save_error)}")
        raise
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        traceback.print_exc()
        if df is not None:
            logger.info("出错时保存最后进度...")
            # 直接保存到最终输出文件
            temp_output_file = output_file + '.temp'
            try:
                df.to_csv(temp_output_file, index=False, encoding='utf-8-sig')
                if os.path.exists(temp_output_file):
                    os.replace(temp_output_file, output_file)
                logger.info("进度已保存到最终输出文件")
            except Exception as save_error:
                logger.error(f"保存进度失败: {str(save_error)}")
        raise

if __name__ == "__main__":
    main()