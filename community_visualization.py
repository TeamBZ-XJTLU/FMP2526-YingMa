import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import random
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CommunityVisualizer:
    def __init__(self, output_dir='output_results', visualization_dir='visualizations'):
        self.output_dir = output_dir
        self.visualization_dir = visualization_dir
        
        # 创建可视化输出目录
        self.vis_output_path = os.path.join(self.output_dir, self.visualization_dir)
        if not os.path.exists(self.vis_output_path):
            os.makedirs(self.vis_output_path)
            print(f"已创建可视化输出目录: {self.vis_output_path}")
    
    def load_data(self):
        """加载社区分配和节点指标数据"""
        # 加载社区分配数据
        community_file = os.path.join(self.output_dir, 'community_assignments.csv')
        self.community_df = pd.read_csv(community_file)
        
        # 加载节点指标数据
        node_metrics_file = os.path.join(self.output_dir, 'node_level_metrics.csv')
        self.node_metrics_df = pd.read_csv(node_metrics_file)
        
        # 合并数据
        self.merged_df = pd.merge(self.community_df, self.node_metrics_df, on=['node_id', 'community_id'], how='inner')
        
        print(f"加载完成: {len(self.community_df)} 个节点，{len(self.community_df['community_id'].unique())} 个社区")
        return self.merged_df
    
    def visualize_community_size_distribution(self):
        """可视化社区大小分布"""
        print("\n1. 生成社区大小分布图...")
        
        # 计算每个社区的大小
        community_sizes = self.community_df['community_id'].value_counts().sort_values(ascending=False)
        
        plt.figure(figsize=(12, 6))
        
        # 绘制社区大小分布直方图
        sns.histplot(community_sizes, bins=20, kde=False)
        plt.title('社区大小分布', fontsize=16)
        plt.xlabel('社区大小', fontsize=12)
        plt.ylabel('社区数量', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        plt.savefig(os.path.join(self.vis_output_path, 'community_size_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 社区大小分布图已保存")
    
    def visualize_top_community_distribution(self, top_n=10):
        """可视化最大的n个社区的节点分布"""
        print(f"\n2. 生成前{top_n}大社区分布饼图...")
        
        # 计算每个社区的大小
        community_sizes = self.community_df['community_id'].value_counts().sort_values(ascending=False)
        
        # 获取前n大社区
        top_communities = community_sizes.head(top_n)
        other_size = community_sizes[top_n:].sum()
        
        # 如果有超过top_n的社区，添加"其他"类别
        if other_size > 0:
            top_communities['其他'] = other_size
        
        plt.figure(figsize=(10, 10))
        
        # 绘制饼图
        wedges, texts, autotexts = plt.pie(
            top_communities.values,
            labels=[f'社区 {i}' for i in top_communities.index],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )
        
        plt.title(f'前{top_n}大社区节点分布', fontsize=16)
        plt.axis('equal')  # 使饼图保持圆形
        
        # 保存图表
        plt.savefig(os.path.join(self.vis_output_path, f'top_{top_n}_community_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 前{top_n}大社区分布图已保存")
    
    def visualize_degree_by_community(self):
        """可视化节点度数按社区分布"""
        print("\n3. 生成节点度数按社区分布图...")
        
        # 获取前20个最大的社区用于可视化
        top_communities = self.community_df['community_id'].value_counts().head(20).index.tolist()
        filtered_df = self.merged_df[self.merged_df['community_id'].isin(top_communities)]
        
        plt.figure(figsize=(14, 8))
        
        # 绘制箱线图
        sns.boxplot(x='community_id', y='degree', data=filtered_df, showfliers=False)
        plt.title('节点度数按社区分布（前20大社区）', fontsize=16)
        plt.xlabel('社区ID', fontsize=12)
        plt.ylabel('节点度数', fontsize=12)
        plt.xticks(rotation=90)
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        plt.savefig(os.path.join(self.vis_output_path, 'degree_by_community.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 节点度数按社区分布图已保存")
    
    def visualize_platform_community_distribution(self):
        """可视化不同平台的节点在社区中的分布"""
        print("\n4. 生成平台节点社区分布对比图...")
        
        # 从node_id提取平台信息（douyin: 抖音, xhs: 小红书）
        self.merged_df['platform'] = self.merged_df['node_id'].str.split(':', expand=True)[0]
        
        # 获取平台统计
        platform_community_stats = self.merged_df.groupby(['platform', 'community_id']).size().reset_index(name='count')
        
        # 获取前10个最大的社区
        top_communities = self.community_df['community_id'].value_counts().head(10).index.tolist()
        filtered_stats = platform_community_stats[platform_community_stats['community_id'].isin(top_communities)]
        
        plt.figure(figsize=(14, 8))
        
        # 绘制分组柱状图
        sns.barplot(x='community_id', y='count', hue='platform', data=filtered_stats)
        plt.title('不同平台的节点在社区中的分布（前10大社区）', fontsize=16)
        plt.xlabel('社区ID', fontsize=12)
        plt.ylabel('节点数量', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title='平台', fontsize=10)
        
        # 保存图表
        plt.savefig(os.path.join(self.vis_output_path, 'platform_community_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 平台节点社区分布图已保存")
    
    def generate_community_summary_statistics(self):
        """生成社区统计摘要"""
        print("\n5. 生成社区统计摘要...")
        
        # 计算社区统计信息
        community_stats = []
        
        for community_id in self.community_df['community_id'].unique():
            # 获取社区节点
            community_nodes = self.community_df[self.community_df['community_id'] == community_id]['node_id'].tolist()
            
            # 计算社区大小
            community_size = len(community_nodes)
            
            # 获取节点指标
            node_metrics = self.node_metrics_df[self.node_metrics_df['node_id'].isin(community_nodes)]
            
            # 计算平均度
            avg_degree = node_metrics['degree'].mean() if not node_metrics.empty else 0
            
            # 计算平均加权度
            avg_weighted_degree = node_metrics['weighted_degree'].mean() if not node_metrics.empty else 0
            
            # 计算平均PageRank
            avg_pagerank = node_metrics['pagerank_wt'].mean() if not node_metrics.empty else 0
            
            # 计算平台分布
            platform_counts = self.merged_df[self.merged_df['community_id'] == community_id]['platform'].value_counts().to_dict()
            
            # 添加到统计列表
            community_stats.append({
                'community_id': community_id,
                'size': community_size,
                'avg_degree': avg_degree,
                'avg_weighted_degree': avg_weighted_degree,
                'avg_pagerank': avg_pagerank,
                'platform_distribution': platform_counts
            })
        
        # 保存统计摘要
        stats_df = pd.DataFrame(community_stats)
        stats_file = os.path.join(self.vis_output_path, 'community_summary_statistics.csv')
        stats_df.to_csv(stats_file, index=False, encoding='utf-8')
        
        print(f"✓ 社区统计摘要已保存到: {stats_file}")
    
    def visualize_community_network_graph(self):
        """绘制社区网络拓扑图，排除最小的社区，展示剩余所有社区"""
        print("\n5. 生成社区网络拓扑图（排除最小社区，展示剩余所有社区）...")
        
        # 加载图数据
        graph_file = os.path.join(self.output_dir, 'user_interaction_graph.gexf')
        if not os.path.exists(graph_file):
            print(f"✗ 未找到图数据文件: {graph_file}")
            return
        
        # 使用networkx读取gexf文件
        G = nx.read_gexf(graph_file)
        print(f"✓ 加载图数据完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
        
        # 获取所有社区并按大小排序
        community_sizes = self.community_df['community_id'].value_counts().sort_values(ascending=False)
        
        # 排除最小的社区，获取剩余所有社区
        selected_communities = community_sizes[1:].index.tolist()
        num_communities = len(selected_communities)
        print(f"✓ 发现{len(community_sizes)}个社区，排除最小社区，展示{num_communities}个社区")
        
        # 创建子图，只包含排除最小社区后的所有社区节点
        selected_community_nodes = self.community_df[self.community_df['community_id'].isin(selected_communities)]['node_id'].tolist()
        G_sub = G.subgraph(selected_community_nodes)
        
        print(f"✓ 创建子图完成，包含 {G_sub.number_of_nodes()} 个节点和 {G_sub.number_of_edges()} 条边")
        
        # 从node_id提取平台信息
        self.community_df['platform'] = self.community_df['node_id'].str.split(':', expand=True)[0]
        
        # 根据图的大小选择合适的布局和参数
        n_nodes = len(G_sub.nodes)
        n_edges = len(G_sub.edges)
        
        # 根据图的大小选择合适的布局和参数
        if n_nodes > 1000:
            # 对于大型图，使用快速布局并降低节点数量
            print(f"✓ 图较大（{n_nodes}节点），使用采样和快速布局")
            # 选择度数最高的前500个节点进行可视化
            degrees = dict(G_sub.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:500]
            G_sample = G_sub.subgraph(top_nodes)
            
            # 优化大型图布局参数，增加社区间距
            pos = nx.spring_layout(G_sample, seed=42, k=0.7, iterations=200)  # 大幅增大k值，显著增加社区间距离
            node_size = 30  # 增大节点大小，提高可见性
            edge_alpha = 0.15  # 降低边透明度，减少视觉干扰
        else:
            # 对于中小型图，使用更详细的布局
            G_sample = G_sub
            # 进一步优化布局参数，使社区之间有更明显的间距
            pos = nx.spring_layout(G_sample, seed=42, k=0.7, iterations=200)  # 大幅增大k值，显著增加社区间距离
            node_size = 30  # 增大节点大小，提高可见性
            edge_alpha = 0.15  # 降低边透明度，减少视觉干扰
        
        # 准备社区颜色
        unique_communities = list(set([self.community_df[self.community_df['node_id'] == node]['community_id'].values[0] 
                                      for node in G_sample.nodes() if not self.community_df[self.community_df['node_id'] == node].empty]))
        community_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_communities)))
        color_map = {community: community_colors[i] for i, community in enumerate(unique_communities)}
        
        # 为不同平台分配不同的节点形状
        platform_shapes = {
            'douyin': 'o',  # 抖音使用圆形
            'xhs': 's'     # 小红书使用方形
        }
        
        # 为采样图创建颜色列表和形状列表
        node_colors = []
        node_shapes = []
        for node in G_sample.nodes():
            node_data = self.community_df[self.community_df['node_id'] == node]
            if not node_data.empty:
                node_community = node_data['community_id'].values[0]
                node_platform = node_data['platform'].values[0]
                node_colors.append(color_map[node_community])
                node_shapes.append(platform_shapes.get(node_platform, 'o'))
            else:
                node_colors.append('lightgray')
                node_shapes.append('o')
        
        # 按形状分组节点
        nodes_by_shape = {shape: [] for shape in set(node_shapes)}
        colors_by_shape = {shape: [] for shape in set(node_shapes)}
        positions_by_shape = {shape: {} for shape in set(node_shapes)}
        
        for node, shape, color in zip(G_sample.nodes(), node_shapes, node_colors):
            nodes_by_shape[shape].append(node)
            colors_by_shape[shape].append(color)
            positions_by_shape[shape][node] = pos[node]
        
        # 创建图表
        plt.figure(figsize=(15, 12))
        
        # 绘制边
        nx.draw_networkx_edges(G_sample, pos, alpha=edge_alpha, width=0.5, edge_color='lightgray')
        
        # 按平台形状绘制节点（按社区着色）
        for shape in nodes_by_shape:
            nx.draw_networkx_nodes(
                G_sample, 
                positions_by_shape[shape], 
                nodelist=nodes_by_shape[shape],
                node_size=node_size, 
                node_color=colors_by_shape[shape], 
                alpha=0.8,
                node_shape=shape
            )
        
        # 添加标题和图例
        plt.title(f'用户互动网络可视化（社区检测结果）\n{len(G_sample.nodes)}节点, {len(G_sample.edges)}边', fontsize=16)
        
        # 创建社区颜色图例和平台形状图例
        legend_elements = []
        
        # 添加社区颜色图例
        for c in sorted(unique_communities[:10]):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[c], 
                                           label=f'社区 {c}', markersize=8))
        
        if len(unique_communities) > 10:
            remaining_communities = len(unique_communities) - 10
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                           label=f'... 还有{remaining_communities}个社区', markersize=8))
        
        # 添加平台形状图例
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                       label='douyin', markersize=8))
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                                       label='xiaohongshu', markersize=8))
        
        plt.legend(handles=legend_elements, loc='best', fontsize=10, title='社区和平台')
        
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(self.vis_output_path, f'community_network_graph_exclude_min_{num_communities}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 分平台绘制前500个节点的网络图
        if n_nodes > 500:
            print("✓ 生成分平台网络图（各平台前500个节点）...")
            platforms = ['douyin', 'xhs']
            platform_names = {'douyin': 'douyin', 'xhs': 'xiaohongshu'}
            
            for platform in platforms:
                # 1. 从完整图中获取该平台的所有节点
                all_platform_nodes = [node for node in G_sub.nodes() 
                                    if not self.community_df[self.community_df['node_id'] == node].empty 
                                    and self.community_df[self.community_df['node_id'] == node]['platform'].values[0] == platform]
                
                if all_platform_nodes:
                    # 2. 计算该平台所有节点的度数
                    platform_degrees = {}  # {node: degree}
                    for node in all_platform_nodes:
                        platform_degrees[node] = G_sub.degree(node)
                    
                    # 3. 选择度数最高的前500个节点
                    top_500_nodes = sorted(platform_degrees, key=platform_degrees.get, reverse=True)[:500]
                    
                    # 4. 创建平台子图（仅包含前500个节点）
                    G_platform = G_sub.subgraph(top_500_nodes)
                    
                    plt.figure(figsize=(18, 15))  # 增大图表尺寸，提供更大的布局空间
                    # 进一步优化布局参数，使社区之间有更明显的间距
                    pos_platform = nx.spring_layout(G_platform, seed=42, k=0.8, iterations=150)  # 大幅增大k值，显著增加社区间距离
                    
                    # 绘制平台网络图
                    nx.draw_networkx_edges(G_platform, pos_platform, alpha=0.4, width=0.7, edge_color='gray')
                    
                    # 创建平台节点颜色列表（按社区着色）
                    node_colors_platform = []
                    for node in G_platform.nodes():
                        node_data = self.community_df[self.community_df['node_id'] == node]
                        if not node_data.empty:
                            node_community = node_data['community_id'].values[0]
                            node_colors_platform.append(color_map[node_community])
                        else:
                            node_colors_platform.append('lightgray')
                    
                    # 绘制节点
                    nx.draw_networkx_nodes(
                        G_platform, 
                        pos_platform, 
                        node_size=100, 
                        node_color=node_colors_platform, 
                        alpha=0.9
                    )
                    
                    # 显示度数最高的5个节点标签
                    top_degrees = dict(G_platform.degree())
                    top_platform_nodes = sorted(top_degrees, key=top_degrees.get, reverse=True)[:5]
                    top_platform_labels = {node: node.split(':')[1][:6] for node in top_platform_nodes}
                    nx.draw_networkx_labels(G_platform, pos_platform, labels=top_platform_labels, font_size=9)
                    
                    plt.title(f'{platform_names[platform]}', fontsize=14)
                    plt.axis('off')
                    plt.tight_layout()
                    
                    # 保存平台网络图
                    platform_file = os.path.join(self.vis_output_path, f'top_500_nodes_{platform}_network_graph_exclude_min_{num_communities}.png')
                    plt.savefig(platform_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✓ {platform_names[platform]}前500节点网络图已保存到 {platform_file}")
        
        print(f"✓ 网络可视化图表已保存到 {output_file}")
        print(f"✓ 图表生成完成！")
    
    def run_all_visualizations(self):
        """运行所有可视化"""
        print("开始社区分布可视化...")
        
        # 加载数据
        self.load_data()
        
        # 只生成用户需要的社区网络图
        self.visualize_community_network_graph()
        
        print(f"\n可视化已完成！输出文件保存在: {self.vis_output_path}")

# 主函数
if __name__ == "__main__":
    visualizer = CommunityVisualizer(output_dir='output_results')
    visualizer.run_all_visualizations()