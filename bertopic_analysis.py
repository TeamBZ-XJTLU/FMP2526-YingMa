import os
import sys
import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import tempfile
from sentence_transformers import SentenceTransformer

# 设置环境变量以解决Windows下的编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义数据路径 - 使用相对路径或更安全的路径处理
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_data_for_analysis.csv")
OUTPUT_FOLDER = "bertopic_results"

# KOL名字列表（完整的KOL名字列表）
KOL_NAMES = ['大哥', '豆豆', '徐老师', '航仔', '老徐', '萌萌', '大哥家', '豆豆家', '徐老师家']

# 中文字符检测函数
def contains_chinese(text):
    return bool(re.search('[\u4e00-\u9fa5]', text))

# 文本预处理函数
def preprocess_text(text):
    """极简预处理 - 只移除URL，保留其他所有内容，包括KOL名字和1字评论
    KOL名字作为评论的重要上下文被保留，如"徐老师推荐的口红"中的"徐老师"
    """
    if not isinstance(text, str) or pd.isna(text):
        return ''
    
    # 只移除URL，保留所有其他内容，包括KOL名字
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 仅过滤空文本，保留所有非空文本（包括1字评论）
    if len(text) < 1:
        return ''
    
    return text

class BERTopicAnalyzer:
    def __init__(self, data_path=DATA_PATH, output_folder=OUTPUT_FOLDER):
        self.data_path = data_path
        self.output_folder = output_folder
        self.data = None
        self.topic_model = None
        self.topics = None
        self.probabilities = None
        
        # 创建输出文件夹
        os.makedirs(self.output_folder, exist_ok=True)
    
    def load_and_preprocess_data(self):
        """
        加载并预处理数据（添加去重功能）
        """
        print(f"正在加载数据: {self.data_path}")
        # 使用编码参数确保正确读取中文
        self.data = pd.read_csv(self.data_path, encoding='utf-8')
        print(f"数据加载成功，形状: {self.data.shape}")
        print(f"数据列: {list(self.data.columns)}")
        
        # 检查必要的列是否存在
        if 'comment_text' not in self.data.columns:
            raise ValueError("数据中缺少 'comment_text' 列")
        if 'platform' not in self.data.columns:
            print("警告: 数据中缺少 'platform' 列，将无法进行平台分析")
        
        # 预处理文本
        print("正在预处理文本...")
        self.data['comment_text_clean'] = self.data['comment_text'].apply(preprocess_text)
        
        # 过滤空文本
        self.data = self.data[self.data['comment_text_clean'] != '']
        print(f"预处理完成，保留数据量: {self.data.shape[0]}")
        
        # 添加去重功能
        original_count = len(self.data)
        self.data = self.data.drop_duplicates(subset=['comment_text_clean'])
        duplicate_count = original_count - len(self.data)
        print(f"去重完成，移除了 {duplicate_count} 条重复评论，当前数据量: {len(self.data)}")
        
        # 检查并打印一些处理后的样本
        sample_size = min(5, len(self.data))
        print(f"\n处理后的样本 ({sample_size}条):")
        for i, (_, row) in enumerate(self.data.sample(sample_size).iterrows()):
            print(f"样本 {i+1}:")
            print(f"  原始: {row['comment_text']}")
            print(f"  处理后: {row['comment_text_clean']}")
    
    def train_bertopic(self, min_topic_size=50, nr_topics='auto', use_local_model=True, local_model_path="C:\model\text2vec-base-chinese"):
        """
        训练BERTopic模型
        
        参数:
        min_topic_size: 最小主题大小（默认50，平衡粒度与噪音）
        use_local_model: 是否使用本地模型（默认True）
        local_model_path: 本地模型路径（默认使用text2vec-base-chinese模型）
        """
        print("\n正在训练BERTopic模型...")
        
        # 设置临时文件夹路径为不包含中文字符的路径
        temp_dir = "C:\\bertopic_temp"  # 修复转义字符问题，使用双反斜杠
        os.makedirs(temp_dir, exist_ok=True)
        os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir
        os.environ['TEMP'] = temp_dir  # 额外设置系统TEMP变量，彻底避免中文路径
        os.environ['TMP'] = temp_dir   # 同时设置TMP变量
        print(f"使用临时目录: {temp_dir}")
        
        # 加载模型
        print("加载模型...")
        try:
            if use_local_model and local_model_path:
                # 从本地路径加载模型
                print(f"从本地加载模型: {local_model_path}")
                model = SentenceTransformer(local_model_path)
                print("本地模型加载成功！")
            else:
                # 尝试在线加载国内友好的中文embedding模型
                print("尝试在线加载国内模型...")
                model = SentenceTransformer("shibing624/text2vec-base-chinese")  # 国内可访问的中文模型
                print("在线模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("\n建议使用本地模型：")
            print("1. 先运行 download_local_model.py 下载模型到本地")
            print("2. 然后使用 --local-model 参数运行此脚本")
            raise
        
        # 初始化优化的UMAP和HDBSCAN参数
        print("初始化UMAP和HDBSCAN参数...")
        umap_model = UMAP(
            n_neighbors=15,        # 增加邻居数，让更多点连接
            min_dist=0.0,         
            n_components=5,        # 减少维度到5
            random_state=42
        )
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=3,         # 微调：从5降到4（平衡严格度和离群点）
            cluster_selection_epsilon=0.15,  # 促进簇合并
            metric='euclidean',
            cluster_selection_method='leaf',
            prediction_data=True  # 添加此参数以生成预测数据
        )
        
        # 添加c-TF-IDF优化（必须）
        from bertopic.vectorizers import ClassTfidfTransformer
        ctfidf_model = ClassTfidfTransformer(
            bm25_weighting=True,
            reduce_frequent_words=True
        )
        
        # 初始化BERTopic时使用加载好的模型和优化参数
        self.topic_model = BERTopic(
            embedding_model=model,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,            # 'auto' 而不是固定数
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            ctfidf_model=ctfidf_model,      
            verbose=True,
            calculate_probabilities=True     # 改为True以便后续优化
        )
        
        # 准备文档列表
        documents = self.data['comment_text_clean'].tolist()
        
        try:
            # 训练模型
            self.topics, self.probabilities = self.topic_model.fit_transform(documents)
            print("BERTopic模型训练完成")
            
            # 主题合并功能
            if nr_topics is not None and nr_topics != 'auto':
                print(f"\n合并相似主题为{nr_topics}个核心主题...")
                self.topic_model.reduce_topics(documents, nr_topics=nr_topics)
            elif nr_topics == 'auto':
                print("\n自动优化主题数量...")
                self.topic_model.reduce_topics(documents, nr_topics='auto')
            else:
                print("\n让模型自行决定最佳主题数量...")
            # 重新获取合并后的主题信息
            topic_info = self.topic_model.get_topic_info()
            print("合并后的主题信息：")
            print(topic_info[['Topic', 'Count', 'Name']])
            
        except Exception as e:
            print(f"模型训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # 强制重新分配离群点
        self.force_reduce_outliers()
        
        # 保存模型
        try:
            model_path = os.path.join(self.output_folder, "bertopic_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.topic_model, f)
            print(f"模型已保存到: {model_path}")
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            # 尝试保存到临时目录
            temp_model_path = os.path.join(temp_dir, "bertopic_model.pkl")
            with open(temp_model_path, 'wb') as f:
                pickle.dump(self.topic_model, f)
            print(f"已保存模型到临时目录: {temp_model_path}")
        
        return self.topic_model
    
    def analyze_topics(self):
        """
        分析主题结果，包括离群点主题统计
        """
        print("\n=== 主题概览 ===")
        topic_info = self.topic_model.get_topic_info()
        print(topic_info)
        
        # 保存主题信息
        topic_info.to_csv(os.path.join(self.output_folder, "topic_info.csv"), index=False, encoding='utf-8-sig')
        
        # 统计离群点主题信息
        outlier_topic = topic_info[topic_info['Topic'] == -1]
        total_documents = topic_info['Count'].sum()
        
        if not outlier_topic.empty:
            outlier_count = outlier_topic['Count'].values[0]
            outlier_percentage = (outlier_count / total_documents) * 100
            
            print(f"\n=== 离群点主题分析 ===")
            print(f"离群点主题（Topic -1）包含 {outlier_count} 条评论")
            print(f"占总文档数的 {outlier_percentage:.2f}%")
            
            if outlier_percentage > 20:
                print("警告：离群点比例较高，可能是预处理过度或模型不匹配导致")
                print("建议：1) 调整预处理逻辑，减少过滤；2) 尝试不同的模型参数")
        else:
            print("\n未发现离群点主题（Topic -1）")
        
        print("\n=== 各主题关键词 ===")
        topic_keywords = []
        
        for topic_num in topic_info['Topic']:
            if topic_num != -1:  # 跳过离群点主题
                keywords = self.topic_model.get_topic(topic_num)
                if keywords:  # 确保有关键词
                    keyword_str = ", ".join([f"{word}({score:.3f})" for word, score in keywords[:5]])
                    count = topic_info[topic_info['Topic'] == topic_num]['Count'].values[0]
                    print(f"主题 {topic_num} (有{count}条评论): {keyword_str}")
                    
                    # 保存关键词信息
                    for rank, (word, score) in enumerate(keywords[:10]):
                        topic_keywords.append({
                            'topic_id': topic_num,
                            'keyword_rank': rank + 1,
                            'keyword': word,
                            'weight': score,
                            'total_documents': count
                        })
        
        # 保存关键词数据
        if topic_keywords:
            keywords_df = pd.DataFrame(topic_keywords)
            keywords_df.to_csv(os.path.join(self.output_folder, "topic_keywords.csv"), index=False, encoding='utf-8-sig')
            print(f"\n关键词数据已保存到: {os.path.join(self.output_folder, 'topic_keywords.csv')}")
    
    def analyze_platform_distribution(self):
        """
        分析不同平台的主题分布
        """
        if 'platform' not in self.data.columns:
            print("\n无法进行平台分析，数据中缺少 'platform' 列")
            return
        
        print("\n=== 平台主题分布分析 ===")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'document': self.data['comment_text_clean'].tolist(),
            'topic': self.topics,
            'platform': self.data['platform'].tolist()
        })
        
        # 过滤掉离群点主题
        results_df = results_df[results_df['topic'] != -1]
        
        # 计算平台主题分布
        platform_topic_counts = results_df.groupby(['platform', 'topic']).size().unstack(fill_value=0)
        platform_topic_prop = platform_topic_counts.div(platform_topic_counts.sum(axis=1), axis=0) * 100
        
        print("\n各平台主题分布(%):")
        print(platform_topic_prop.round(2))
        
        # 保存平台分布数据
        platform_topic_prop.to_csv(os.path.join(self.output_folder, "platform_topic_proportions.csv"), encoding='utf-8-sig')
        
        # 显示每个平台的主要主题
        for platform in platform_topic_prop.index:
            top_topics = platform_topic_prop.loc[platform].nlargest(3)
            print(f"\n{platform}平台主要主题:")
            for topic_num, percentage in top_topics.items():
                # 获取该主题的前3个关键词
                keywords = self.topic_model.get_topic(topic_num)
                if keywords:
                    top_keywords = ", ".join([word for word, _ in keywords[:3]])
                    print(f"  主题 {topic_num}: {percentage:.1f}% - {top_keywords}")
      
    def force_reduce_outliers(self): 
        """强力重新分配离群点到最近的主题""" 
        print("\n=== 强力重新分配离群点 ===") 
        
        documents = self.data['comment_text_clean'].tolist() 
        
        # 统计原始离群点 
        original_outliers = sum(1 for topic in self.topics if topic == -1) 
        original_percent = (original_outliers / len(self.topics)) * 100 
        print(f"重新分配前离群点: {original_outliers}/{len(self.topics)} ({original_percent:.2f}%)") 
        
        try: 
            # 方法1：使用distributions策略（需要probabilities） 
            if self.probabilities is not None: 
                new_topics = self.topic_model.reduce_outliers( 
                    documents, 
                    self.topics, 
                    probabilities=self.probabilities, 
                    strategy="distributions", 
                    threshold=0.05  # 很低的阈值，强制分配 
                ) 
            else: 
                # 方法2：使用c-tf-idf策略 
                new_topics = self.topic_model.reduce_outliers( 
                    documents, 
                    self.topics, 
                    strategy="c-tf-idf" 
                ) 
            
            self.topics = new_topics 
            
            # 统计新离群点 
            new_outliers = sum(1 for topic in self.topics if topic == -1) 
            new_percent = (new_outliers / len(self.topics)) * 100 
            
            print(f"重新分配后离群点: {new_outliers}/{len(self.topics)} ({new_percent:.2f}%)") 
            print(f"离群点减少: {original_outliers - new_outliers} 条评论") 
            
        except Exception as e: 
            print(f"离群点重分配失败: {e}") 
            print("将使用原始主题分配") 
        
        return self.topics
    
    def save_results(self):
        """
        保存完整的结果数据
        """
        print("\n保存完整结果数据...")
        
        # 确保输出文件夹存在
        os.makedirs(self.output_folder, exist_ok=True)
        
        # 添加主题信息到原始数据
        results_df = self.data.copy()
        results_df['topic_id'] = self.topics
        
        # 添加主题标签
        topic_labels = []
        for topic_id in self.topics:
            if topic_id == -1:
                topic_labels.append("离群点")
            else:
                try:
                    keywords = self.topic_model.get_topic(topic_id)
                    if keywords:
                        # 使用前3个关键词作为标签
                        label = ", ".join([word for word, _ in keywords[:3]])
                        topic_labels.append(f"主题{topic_id}: {label}")
                    else:
                        topic_labels.append(f"主题{topic_id}")
                except Exception as e:
                    print(f"获取主题 {topic_id} 标签时出错: {e}")
                    topic_labels.append(f"主题{topic_id}")
        
        results_df['topic_label'] = topic_labels
        
        # 保存结果 - 使用更安全的文件保存方式
        output_path = os.path.join(self.output_folder, "full_topic_results.csv")
        try:
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"完整结果已保存到: {output_path}")
        except Exception as e:
            print(f"保存结果文件时出错: {e}")
            # 尝试保存到临时目录
            temp_output = os.path.join(tempfile.gettempdir(), "full_topic_results.csv")
            results_df.to_csv(temp_output, index=False, encoding='utf-8-sig')
            print(f"已保存到临时目录: {temp_output}")
      
    def run_all(self):
        """
        运行所有分析步骤
        """
        print("="*80)
        print("开始BERTopic主题模型分析")
        print("="*80)
        
        try:
            self.load_and_preprocess_data()
            # 使用更新后的参数配置
            self.train_bertopic(min_topic_size=50, nr_topics='auto')
            self.analyze_topics()
            
            # 简化分析流程，避免可能的额外错误
            try:
                self.analyze_platform_distribution()
            except Exception as e:
                print(f"平台分布分析时出错: {str(e)}")
                

                
            try:
                self.save_results()
            except Exception as e:
                print(f"保存结果时出错: {str(e)}")
                
            # 可视化步骤可能消耗更多资源，可以选择跳过
            print("\n注意：已跳过可视化步骤以减少内存使用")
            
            print("\n" + "="*80)
            print("BERTopic基本分析完成！")
            print(f"主要结果已保存到: {self.output_folder}")
            print("="*80)
            
        except Exception as e:
            print(f"\n分析过程中出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='BERTopic分析工具')
    # 默认使用本地模型，添加--no-local-model参数来禁用本地模型（如果需要）
    parser.add_argument('--no-local-model', action='store_true', help='禁用本地模型（默认使用本地模型）')
    parser.add_argument('--model-path', type=str, 
                        default=os.path.join("C:\model", "text2vec-base-chinese"),
                        help='本地模型路径')
    parser.add_argument('--min-topic-size', type=int, default=50, help='最小主题大小')
    parser.add_argument('--nr-topics', type=str, default='auto', help='主题数量上限 (使用数字或"None"或"auto")')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 计算是否使用本地模型（默认使用，除非显式禁用）
    use_local_model = not args.no_local_model
    
    print("BERTopic分析开始...")
    print(f"使用本地模型: {use_local_model}")
    if use_local_model:
        print(f"本地模型路径: {args.model_path}")
    print(f"最小主题大小: {args.min_topic_size}")
    print(f"主题数量上限: {args.nr_topics}")
    
    # 确保当前工作目录是脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"当前工作目录: {os.getcwd()}")
    
    # 初始化分析器
    analyzer = BERTopicAnalyzer()
    
    # 加载和预处理数据
    analyzer.load_and_preprocess_data()
    
    # 处理nr_topics参数
    nr_topics_param = args.nr_topics
    if nr_topics_param == 'None':
        nr_topics_param = None
    elif nr_topics_param != 'auto':
        try:
            nr_topics_param = int(nr_topics_param)
        except ValueError:
            print("警告: nr_topics参数无效，将使用None")
            nr_topics_param = None
    
    # 训练模型，传入命令行参数
    analyzer.train_bertopic(min_topic_size=args.min_topic_size, 
                            nr_topics=nr_topics_param,
                            use_local_model=use_local_model,
                            local_model_path=args.model_path)
    
    # 继续执行其他分析步骤
    analyzer.analyze_topics()
    

    
    analyzer.save_results()
    print("\nBERTopic分析完成！")
