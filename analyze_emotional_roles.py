import pandas as pd
import sys

# 设置中文字体显示
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

try:
    # 读取CSV文件
    df = pd.read_csv('output_results/emotional_analysis/key_node_emotional_analysis.csv')
    
    print("=== 网络中关键用户的情感角色分布 ===")
    # 统计情感角色分布
    role_counts = df['role'].value_counts().sort_values(ascending=False)
    for role, count in role_counts.items():
        print(f"{role}: {count}个用户")
    
    # 计算各情感角色的平均情感得分
    print("\n=== 各情感角色的平均情感得分 ===")
    role_avg_sentiment = df.groupby('role')['node_sentiment'].mean().sort_values(ascending=False)
    for role, avg_sentiment in role_avg_sentiment.items():
        print(f"{role}: {avg_sentiment:.3f}")
    
    print("\n=== 负面关键用户连接情况 ===")
    print("（情感得分≤0.3的用户，按负面邻居比例降序排列）")
    # 筛选负面关键用户
    negative_users = df[df['node_sentiment'] <= 0.3]
    # 确保存在负面邻居比例和正面邻居比例列
    if 'negative_neighbor_ratio' in df.columns and 'positive_neighbor_ratio' in df.columns:
        # 计算负面邻居百分比和正面邻居百分比
        negative_users['负面邻居百分比'] = negative_users['negative_neighbor_ratio'] * 100
        negative_users['正面邻居百分比'] = negative_users['positive_neighbor_ratio'] * 100
        
        # 按负面邻居比例降序排列，显示前10个
        negative_users_sorted = negative_users.sort_values(by='negative_neighbor_ratio', ascending=False)
        
        # 检查是否有负面关键用户
        if not negative_users_sorted.empty:
            # 显示相关信息
            display_columns = ['node_id', 'platform', 'node_sentiment', '负面邻居百分比', '正面邻居百分比']
            print(negative_users_sorted[display_columns].head(10).to_string(index=False))
            
            # 计算负面关键用户的平均负面邻居比例和正面邻居比例
            avg_negative_neighbor_ratio = negative_users['negative_neighbor_ratio'].mean()
            avg_positive_neighbor_ratio = negative_users['positive_neighbor_ratio'].mean()
            
            print(f"\n负面关键用户平均负面邻居比例: {avg_negative_neighbor_ratio:.2%}")
            print(f"负面关键用户平均正面邻居比例: {avg_positive_neighbor_ratio:.2%}")
        else:
            print("没有找到情感得分≤0.3的负面关键用户")
    else:
        print("CSV文件中缺少'negative_neighbor_ratio'或'positive_neighbor_ratio'列")
        print("文件包含的列:", list(df.columns))
        
except Exception as e:
    print(f"分析过程中出现错误: {e}")
    import traceback
    traceback.print_exc()