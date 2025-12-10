import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
import os

from tqdm import tqdm

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
# 改进字体设置，确保包含更多Unicode字符支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Computer Modern', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
# 确保matplotlib使用更好的字体处理方式
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

class NetworkAnalysisForWLS:
    def __init__(self, input_file, output_dir='output_results'):
        """初始化网络分析类
        
        Args:
            input_file (str): 输入CSV文件路径
            output_dir (str): 输出目录路径
        """
        self.input_file = input_file
        self.data = None
        self.user_nodes = None
        self.edges = None
        self.G = None
        self.partition = None
        self.node_metrics = None
        self.output_dir = output_dir
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print(f"输出目录设置为: {os.path.abspath(self.output_dir)}")
    
    def calculate_comment_interaction_score(self, comment_data):
        """
        计算评论级别的互动分数
        使用评论点赞数和子评论数作为主要指标
        """
        # 获取评论级互动数据，直接获取数值，不再使用parse_chinese_number方法
        comment_likes = comment_data.get('comment_like_count', 0)
        # 使用sub_comment_count替代reply_count
        sub_comment_count = comment_data.get('sub_comment_count', 0)
        
        # 确保数据类型为数值
        try:
            comment_likes = float(comment_likes) if comment_likes else 0
            sub_comment_count = float(sub_comment_count) if sub_comment_count else 0
        except (ValueError, TypeError):
            comment_likes = 0
            sub_comment_count = 0
        
        score = 0.75 * comment_likes + 0.25 * sub_comment_count
        
        # 确保分数至少为0.1，避免权重为0的情况
        return max(0.1, round(score, 2))
    
    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        print(f"从文件 {self.input_file} 加载数据...")
        # 加载CSV文件
        self.data = pd.read_csv(self.input_file, encoding='utf-8-sig')
        print(f"数据加载完成，共 {len(self.data)} 条记录")
        
        # 构建节点唯一ID (平台:user_id)
        print("构建节点唯一ID...")
        self.data['post_user_node_id'] = self.data['platform'] + ':' + self.data['post_user_id']
        self.data['comment_user_node_id'] = self.data['platform'] + ':' + self.data['comment_user_id']
        
        # 使用评论级互动分数
        print("计算评论级互动分数...")
        # 使用apply方法对每一行计算评论级互动分数
        self.data['interaction_score'] = self.data.apply(lambda row: self.calculate_comment_interaction_score(row), axis=1)
        
        # 显示互动分数的统计信息
        print(f"评论级互动分数统计 - 最小值: {self.data['interaction_score'].min()}, 最大值: {self.data['interaction_score'].max()}, 平均值: {self.data['interaction_score'].mean():.2f}")
            
        # 验证父子评论映射
        print("验证父子评论映射...")
        # 处理空值
        self.data['parent_comment_id'] = self.data['parent_comment_id'].fillna('')
        
        # 创建用户节点列表
        post_users = self.data['post_user_node_id'].unique()
        comment_users = self.data['comment_user_node_id'].unique()
        all_users = np.union1d(post_users, comment_users)
        
        self.user_nodes = pd.DataFrame({'node_id': all_users})
        print(f"共识别出 {len(self.user_nodes)} 个唯一用户节点")
        
        # 保存节点信息
        self.user_nodes.to_csv(os.path.join(self.output_dir, 'user_nodes.csv'), index=False)
        print("节点信息已保存到 user_nodes.csv")
        
        return self.data
    
    def build_edge_table(self):
        """Step：构建边表（评论级 → 用户级）"""
        print("\n=== Step: 构建边表（评论级 → 用户级） ===")
        
        # 创建评论级边表
        # 1. 帖子作者 -> 评论者 (帖子引发的评论)
        post_to_comment_edges = self.data[[
            'post_id', 'post_user_node_id', 'comment_user_node_id', 
            'comment_time_ts', 'interaction_score', 'sentiment_score_v1'
        ]].copy()
        # 使用原始互动分数
        post_to_comment_edges['source'] = post_to_comment_edges['post_user_node_id']
        post_to_comment_edges['target'] = post_to_comment_edges['comment_user_node_id']
        post_to_comment_edges['edge_type'] = 'post_to_comment'
        
        # 2. 父评论者 -> 子评论者 (回复关系)
        reply_edges = self.data[self.data['parent_comment_id'] != ''][['comment_id', 'parent_comment_id', 'comment_user_node_id',
            'comment_time_ts', 'interaction_score', 'sentiment_score_v1']].copy()
        
        
        if not reply_edges.empty:
            # 找到父评论的作者
            comment_author_map = self.data.set_index('comment_id')['comment_user_node_id'].to_dict()
            reply_edges['parent_author'] = reply_edges['parent_comment_id'].map(comment_author_map)
            reply_edges = reply_edges[reply_edges['parent_author'].notna()]
            reply_edges['source'] = reply_edges['parent_author']
            reply_edges['target'] = reply_edges['comment_user_node_id']
            reply_edges['edge_type'] = 'reply'
        
        # 合并所有边
        if not reply_edges.empty:
            self.edges = pd.concat([
                post_to_comment_edges[['source', 'target', 'edge_type', 'interaction_score', 'sentiment_score_v1']],
                reply_edges[['source', 'target', 'edge_type', 'interaction_score', 'sentiment_score_v1']]
            ])
        else:
            self.edges = post_to_comment_edges[['source', 'target', 'edge_type', 'interaction_score', 'sentiment_score_v1']]
        
        print(f"边表构建完成，共 {len(self.edges)} 条边")
        print(f"边类型统计: {self.edges['edge_type'].value_counts().to_dict()}")
        
        # 保存评论级边表
        self.edges.to_csv(os.path.join(self.output_dir, 'comment_level_edges.csv'), index=False)
        print("评论级边表已保存到 comment_level_edges.csv")
        
        return self.edges
    
    def aggregate_user_edges(self, weight_method='log'):
        """Step：用户级边聚合计算
        
        Args:
            weight_method (str): 权重计算方法：
                'raw': 使用原始互动分数总和
                'log': 使用对数转换后的互动分数 (log1p)（默认）
        """
        print("\n=== Step: 用户级边聚合计算 ===")
        print(f"使用权重计算方法: {weight_method}")
        
        # 聚合用户级边 - 先聚合原始互动分数
        user_edges = self.edges.groupby(['source', 'target']).agg({
            'interaction_score': ['sum', 'mean', 'count'],
            'sentiment_score_v1': 'mean'
        }).reset_index()
        
        # 重命名列
        user_edges.columns = ['source', 'target', 'total_interaction', 
                            'avg_interaction', 'interaction_count', 'avg_sentiment']
        
        # 根据选择的方法计算权重
        if weight_method == 'log':
            # 使用对数转换（log1p避免0值问题）
            user_edges['weight'] = np.log1p(user_edges['total_interaction'])
            # 确保权重最小值不小于0.1
            user_edges.loc[user_edges['weight'] < 0.1, 'weight'] = 0.1
            print("已应用对数转换到权重")
        else:  # 'raw'
            # 使用原始互动分数的总和
            user_edges['weight'] = user_edges['total_interaction']
        
        print(f"用户级边聚合完成，共 {len(user_edges)} 条聚合边")
        print(f"权重统计: 最小值={user_edges['weight'].min():.4f}, 最大值={user_edges['weight'].max():.4f}, 平均值={user_edges['weight'].mean():.4f}")
        
        # 保存用户级边表
        user_edges.to_csv(os.path.join(self.output_dir, 'user_level_aggregated_edges.csv'), index=False)
        print("用户级聚合边表已保存到 user_level_aggregated_edges.csv")
        
        self.user_edges = user_edges
        return user_edges
    
    def build_graph_and_detect_communities(self):
        """Step：图构建与社区检测"""
        print("\n=== Step: 图构建与社区检测 ===")
        
        # 构建加权图
        self.G = nx.Graph()
        
        # 添加节点 - 使用进度条
        print("添加节点到图中...")
        for _, row in tqdm(self.user_nodes.iterrows(), total=len(self.user_nodes), desc="添加节点", unit="节点"):
            self.G.add_node(row['node_id'])
        
        # 添加边 - 使用进度条
        print("添加边到图中...")
        # 使用原始互动分数聚合后的值作为权重，确保权重值范围合理且有意义
        weights = []
        for _, row in tqdm(self.user_edges.iterrows(), total=len(self.user_edges), desc="添加边", unit="边"):
            # 添加边并设置属性
            self.G.add_edge(
                row['source'], 
                row['target'], 
                # 权重使用原始互动分数聚合后的总和
                weight=row['weight'],
                # 保留详细的边属性信息
                total_interaction=row['total_interaction'],  # 原始互动分数总和
                avg_interaction=row['avg_interaction'],      # 平均互动分数
                interaction_count=row['interaction_count'],  # 互动次数
                avg_sentiment=row['avg_sentiment']           # 平均情感分数
            )
            weights.append(row['weight'])
        
        print(f"图构建完成: {len(self.G.nodes)} 个节点, {len(self.G.edges)} 条边")
        # 显示权重分布情况
        weights_array = np.array(weights)
        print(f"边权重分布: 最小值={weights_array.min()}, 最大值={weights_array.max()}, 平均值={weights_array.mean():.2f}, 中位数={np.median(weights_array):.2f}")
        
        # 计算图的基本统计信息
        print(f"图密度: {nx.density(self.G):.6f}")
        print(f"平均度: {2 * len(self.G.edges) / len(self.G.nodes):.2f}")
        
        # 社区检测
        print("执行Louvain社区检测...")
        self.partition = community_louvain.best_partition(self.G, weight='weight')
        
        # 计算modularity值
        modularity_value = community_louvain.modularity(self.partition, self.G, weight='weight')
        print(f"社区检测完成，共发现 {len(set(self.partition.values()))} 个社区")
        print(f"Modularity值: {modularity_value:.4f}")
        
        # 统计社区
        community_sizes = pd.Series(self.partition).value_counts()
        print(f"最大社区大小: {community_sizes.max()}, 最小社区大小: {community_sizes.min()}")
        
        # 保存社区信息
        community_df = pd.DataFrame(list(self.partition.items()), columns=['node_id', 'community_id'])
        community_df.to_csv(os.path.join(self.output_dir, 'community_assignments.csv'), index=False)
        print("社区分配已保存到 community_assignments.csv")
        
        # 保存图结构
        nx.write_gexf(self.G, os.path.join(self.output_dir, 'user_interaction_graph.gexf'))
        print("图结构已保存到 user_interaction_graph.gexf")
        
    def generate_node_metrics(self):
        print("\n=== Step: 生成节点网络指标 ===")
        
        # 获取所有节点并保持一致的顺序
        nodes = list(self.G.nodes())
        n_nodes = len(nodes)
        print(f"总节点数: {n_nodes}")
        
        # degree：表示节点直接连接的边的数量，即一个用户直接连接的其他用户数量
        # weighted_degree：不仅考虑连接数量，还考虑连接的强度（互动频率或强度）
        print("计算基础指标...")
        degree_dict = dict(self.G.degree())
        weighted_degree_dict = dict(self.G.degree(weight='weight'))
        
        # 2. Clustering Coefficient：反映网络的"小集团"现象，聚类系数高表示用户的社交圈较为紧密
        print("  - 计算Clustering Coefficient...")
        if n_nodes > 1000:
            print(f"  - 正在计算{len(nodes)}个节点的聚类系数，这可能需要一些时间...")
        clustering_dict = nx.clustering(self.G, weight='weight')
        print(f"[1/3] 基础指标计算完成：度、加权度、聚类系数")
        
        # 中心性指标 
        print(f"[2/3] 计算中心性指标 (节点数: {n_nodes})...")
        
        # 0. PageRank Centrality：衡量节点在网络中的重要性，值越高表示越重要
        print("  - 计算PageRank Centrality...")
        pagerank_dict = {}        
        try:
            pagerank_dict = nx.pagerank(self.G, weight='weight', alpha=0.85, max_iter=1000)
            print("  ✓ PageRank Centrality计算完成")
        except Exception as e:
            pagerank_dict = {node: 0 for node in nodes}
            print(f"  ✗ PageRank Centrality计算失败: {str(e)}，已设置为0")
        
        # 1. Betweenness Centrality 识别控制信息流动的关键节点，这些节点往往是不同社区间的连接者
        print("  - 计算Betweenness Centrality (可能较慢)...")
        print(f"  当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        betweenness_dict = {node: 0 for node in nodes}
        
        try:
            # 对于任何大于500节点的图，使用近似算法
            if n_nodes > 500:
                print(f"  - 大型图优化处理 (节点数: {n_nodes})，使用近似算法和连通分量分解...")
                print(f"  当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 获取所有连通分量
                connected_components = list(nx.connected_components(self.G))
                print(f"  - 检测到{len(connected_components)}个连通分量")
                
                # 对每个连通分量分别计算
                for i, component in enumerate(connected_components):
                    print(f"    处理连通分量 {i+1}/{len(connected_components)} (大小: {len(component)})...")
                    
                    if len(component) == 1:
                        # 单个节点不需要计算
                        betweenness_dict[list(component)[0]] = 0
                        continue
                    
                    subgraph = self.G.subgraph(component)
                    
                    # 根据连通分量大小选择不同的近似参数
                    if len(component) > 1000:
                        # 非常大的连通分量，使用较少的采样节点
                        print(f"      - 使用近似算法，采样100个节点...")
                        k = 100  # 采样100个节点
                    elif len(component) > 500:
                        # 较大的连通分量
                        print(f"      - 使用近似算法，采样200个节点...")
                        k = 200  # 采样200个节点
                    else:
                        # 较小的连通分量，可以用更多的采样节点
                        print(f"      - 使用近似算法，采样300个节点...")
                        k = 300  # 采样300个节点
                    
                    # 使用近似算法计算betweenness centrality
                    # k参数表示用于近似的源节点数量
                    component_betweenness = nx.betweenness_centrality(subgraph, weight='weight', normalized=True, k=min(k, len(component)-1))
                    
                    # 合并结果
                    for node, value in component_betweenness.items():
                        betweenness_dict[node] = value
                    
                    print(f"      - 完成连通分量 {i+1}/{len(connected_components)}")
                    print(f"      当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                print(f"  ✓ Betweenness Centrality计算完成 (使用近似算法)")
                print(f"  当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                # 对中小型图可以考虑使用精确计算
                if n_nodes > 300:
                    print(f"  - 使用近似算法计算Betweenness Centrality (节点数: {n_nodes})...")
                    betweenness_dict = nx.betweenness_centrality(self.G, weight='weight', normalized=True, k=min(300, n_nodes-1))
                else:
                    print(f"  - 直接计算Betweenness Centrality (节点数: {n_nodes})...")
                    betweenness_dict = nx.betweenness_centrality(self.G, weight='weight', normalized=True)
                print("  ✓ Betweenness Centrality计算完成")
        except KeyboardInterrupt:
            print("\n  ✗ Betweenness Centrality计算被用户中断，将跳过此计算")
            betweenness_dict = {node: 0 for node in nodes}
        except Exception as e:
            betweenness_dict = {node: 0 for node in nodes}
            print(f"  ✗ Betweenness Centrality计算失败: {str(e)}，已设置为0")
        
        # 2. Closeness Centrality 识别能够快速与网络中其他节点通信的中心节点
        print("  - 计算Closeness Centrality...")
        closeness_dict = {}
        try:
            if n_nodes > 1000:
                print(f"  - 正在处理大型图的Closeness Centrality计算 (节点数: {n_nodes})...")
                # 对连通分量进行处理，提供进度反馈
                connected_components = list(nx.connected_components(self.G))
                print(f"  - 检测到{len(connected_components)}个连通分量")
                closeness_dict = {node: 0 for node in nodes}
                
                # 对每个连通分量分别计算
                for i, component in enumerate(connected_components):
                    if len(component) > 1:
                        # 对于较大的连通分量提供进度信息
                        if len(component) > 200:
                            print(f"    - 处理大型连通分量 {i+1}/{len(connected_components)} (节点数: {len(component)})...")
                        subgraph = self.G.subgraph(component)
                        comp_closeness = nx.closeness_centrality(subgraph)
                        closeness_dict.update(comp_closeness)
                print("  ✓ Closeness Centrality计算完成 (分分量处理)")
            else:
                closeness_dict = nx.closeness_centrality(self.G)
                print("  ✓ Closeness Centrality计算完成")
        except Exception as e:
            closeness_dict = {node: 0 for node in nodes}
            print(f"  ✗ Closeness Centrality计算失败: {str(e)}，已设置为0")
        
        # 3. Eigenvector Centrality - 识别与其他重要节点相连的用户，能更好地反映影响力的传播
        print("  - 计算Eigenvector Centrality...")
        eigenvector_dict = {}
        try:
            # 只对连通分量计算特征向量中心性
            largest_cc = max(nx.connected_components(self.G), key=len)
            G_largest = self.G.subgraph(largest_cc)
            print(f"  - 对最大连通分量计算 (节点数: {len(largest_cc)}/{n_nodes})...")
            
            if len(largest_cc) > 500:
                print(f"    - 大型连通分量特征向量计算可能耗时较长...")
            
            # 尝试使用更稳定的numpy实现版本，增加迭代次数和调整容差
            try:
                # 首先尝试使用numpy版本，通常更稳定
                eigenvector_centrality = nx.eigenvector_centrality_numpy(G_largest, weight='weight')
                print("  ✓ 使用numpy实现计算Eigenvector Centrality完成")
            except Exception as e_np:
                print(f"  ⚠ numpy实现失败: {str(e_np)}，尝试使用标准实现...")
                # 如果numpy版本失败，尝试使用标准实现，但增加参数
                eigenvector_centrality = nx.eigenvector_centrality(G_largest, weight='weight', max_iter=2000, tol=1e-6)
                print("  ✓ 使用标准实现计算Eigenvector Centrality完成")
            
            # 使用tqdm进度条扩展结果到所有节点
            eigenvector_dict = {}
            for node in tqdm(nodes, desc="  - 扩展特征向量结果", unit="节点", leave=False):
                eigenvector_dict[node] = eigenvector_centrality.get(node, 0)
                
            # 验证计算结果
            non_zero_count = sum(1 for v in eigenvector_dict.values() if v > 1e-10)
            if non_zero_count == 0:
                print(f"  ⚠ 警告：所有特征向量中心性值都接近0（非零值: {non_zero_count}）")
            else:
                print(f"  ✓ 特征向量中心性计算完成，非零值: {non_zero_count}")
        except Exception as e:
            eigenvector_dict = {node: 0 for node in nodes}
            print(f"  ✗ Eigenvector Centrality计算失败: {str(e)}，已设置为0")
            # 打印更详细的错误信息以便调试
            import traceback
            traceback.print_exc()
        
        print("[2/3] 所有中心性指标计算完成")
        
        # 创建数据字典，确保所有值列表与nodes顺序一致
        print("[3/3] 整合指标数据...")
        metrics = {
            'node_id': nodes,
            'degree': [degree_dict.get(node, 0) for node in nodes],
            'weighted_degree': [weighted_degree_dict.get(node, 0) for node in nodes],
            'weighted_total_degree': [weighted_degree_dict.get(node, 0) for node in nodes],  # 别名，与其他文件保持一致
            'pagerank_wt': [pagerank_dict.get(node, 0) for node in nodes],  # 添加pagerank_wt以满足RQ3
            'clustering_coefficient': [clustering_dict.get(node, 0) for node in nodes],
            'betweenness_centrality': [betweenness_dict.get(node, 0) for node in nodes],
            'closeness_centrality': [closeness_dict.get(node, 0) for node in nodes],
            'eigenvector_centrality': [eigenvector_dict.get(node, 0) for node in nodes]
        }
        
        # 转换为DataFrame
        self.node_metrics = pd.DataFrame(metrics)
        
        # 添加社区信息
        self.node_metrics['community_id'] = self.node_metrics['node_id'].map(self.partition)
        
        # 保存节点指标
        output_file = os.path.join(self.output_dir, 'node_level_metrics.csv')
        self.node_metrics.to_csv(output_file, index=False)
        
        # 验证文件是否存在
        file_status = "✓ 成功" if os.path.exists(output_file) else "✗ 失败"
        file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
        
        print(f"\n[3/3] 节点指标处理完成")
        print(f"  - 处理节点数: {n_nodes}")
        print(f"  - 保存文件: {os.path.basename(output_file)}")
        print(f"  - 状态: {file_status}")
        print(f"  - 文件大小: {file_size:,} 字节")
        
        return self.node_metrics
    
    def join_node_metrics_to_comments(self):
        """Step B2：将节点指标join回评论数据"""
        print("\n=== Step B2: 合并节点指标到评论数据 ===")
        
        # 重命名列以便合并
        comment_metrics = self.node_metrics.copy()
        comment_metrics.columns = [f'commenter_{col}' if col != 'node_id' else 'comment_user_node_id' 
                                 for col in comment_metrics.columns]
        
        post_metrics = self.node_metrics.copy()
        post_metrics.columns = [f'poster_{col}' if col != 'node_id' else 'post_user_node_id' 
                              for col in post_metrics.columns]
        
        # 合并数据
        merged_data = self.data.merge(comment_metrics, on='comment_user_node_id', how='left')
        merged_data = merged_data.merge(post_metrics, on='post_user_node_id', how='left')
        
        print(f"数据合并完成，共 {len(merged_data)} 条记录")
        
        # 保存合并后的数据
        merged_data.to_csv(os.path.join(self.output_dir, 'comments_with_node_metrics.csv'), index=False)
        print("合并后的数据已保存到 comments_with_node_metrics.csv")
        
        self.merged_data = merged_data
        return merged_data
    
    
    def run_complete_analysis(self):
        """运行完整的分析流程"""
        print("开始网络分析流程...")
        
        try:
            print("\n[1/6] Step: 数据准备")
            self.load_and_preprocess_data()
            
            print("\n[2/6] Step: 构建边表")
            self.build_edge_table()
            

            print("\n[3/6] Step: 用户级边聚合")
            self.aggregate_user_edges()
            

            print("\n[4/6] Step: 图构建与社区检测")
            self.build_graph_and_detect_communities()
            

            print("\n[5/6] Step: 生成节点指标")
            self.generate_node_metrics()
            
            print("\n[6/6] Step: 合并节点指标到评论数据")
            self.join_node_metrics_to_comments()
            
            
            print("所有分析结果已保存到:", self.output_dir)
            return True
            
        except Exception as e:
            print(f"\n❌ 分析过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# 主函数
if __name__ == "__main__":
    # 添加命令行参数解析
    import sys
    
    # 默认参数
    input_file = 'cleaned_data_for_analysis.csv'
    output_dir = 'output_results'
    
    # 简单的命令行参数解析
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--input_file' and i + 1 < len(sys.argv):
            input_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--output_dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    print(f"使用参数:")
    print(f"  输入文件: {input_file}")
    print(f"  输出目录: {output_dir}")
    
    # 创建分析实例并执行完整分析流程
    analyzer = NetworkAnalysisForWLS(input_file, output_dir)
    analyzer.run_complete_analysis()
    print("分析完成！")