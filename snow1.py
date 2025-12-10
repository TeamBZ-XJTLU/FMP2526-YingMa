import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from snownlp import SnowNLP
import matplotlib.pyplot as plt
import seaborn as sns
# 优化字体设置，添加中文字体支持以解决乱码问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "DejaVu Sans", "Arial"]  # 增加中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import warnings
warnings.filterwarnings('ignore')

# 创建输出目录 - 修改为在数据采样和离群文件夹下创建output_files子文件夹
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_files')
os.makedirs(output_dir, exist_ok=True)
print(f'输出目录设置为: {output_dir}')

print('Step 1: 读取并合并数据')

# 输入文件路径 - 使用绝对路径
base_dir = os.path.dirname(os.path.abspath(__file__))
file_paths = [
    os.path.join(base_dir, 'viral_analysis_results', 'merged_all_with_flags_doudou.csv'),
    os.path.join(base_dir, 'viral_analysis_results', 'merged_all_with_flags_xls.csv'),
    os.path.join(base_dir, 'viral_analysis_results', 'merged_all_with_flags_zky.csv')
]

dfs = []
for file in file_paths:
    try:
        df = pd.read_csv(file)
        print(f'成功读取 {file}，共 {len(df)} 行数据')
        dfs.append(df)
    except Exception as e:
        print(f'读取文件 {file} 失败: {e}')

# 检查是否有数据可以合并
if not dfs:
    print('警告：没有成功读取任何数据文件，程序将退出')
    exit(1)

df_all = pd.concat(dfs, ignore_index=True)
print(f'合并后总数据行数: {len(df_all)}')


def analyze_sentiment(text):
    """使用SnowNLP分析文本情感，返回0-1之间的情感分数
    
    参数:
        text: str - 从comment_text_clean列获取的评论文本
    """
    if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
        return 0.5  # 中性
    try:
        s = SnowNLP(text)
        return s.sentiments
    except:
        return 0.5  # 出错时返回中性

# 直接使用comment_text_clean列进行情感分析
text_column = 'comment_text_clean'
print(f'使用列 {text_column} 进行情感分析')
# 应用情感分析
df_all['sentiment_score'] = df_all[text_column].apply(analyze_sentiment)
# 添加情感标签
df_all['sentiment_label'] = pd.cut(
    df_all['sentiment_score'],
    bins=[-0.1, 0.4, 0.6, 1.1],
    labels=['neg', 'neu', 'pos']
)
print(f'情感分析完成，生成了 sentiment_score 和 sentiment_label 列')
print(f'sentiment_score统计描述:')
print(df_all['sentiment_score'].describe())

# 保存处理后的数据
output_file = os.path.join(output_dir, 'merged_all_with_weights.csv')
df_all.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f'已保存处理后的数据到: {output_file}')


print('\n=== 基础数据处理完成 ===')