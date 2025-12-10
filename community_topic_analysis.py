import pandas as pd
import os
import numpy as np
import networkx as nx
import math

class CommunityTopicAnalyzer:
    def __init__(self, data_dir='.'):
        """初始化社区主题分析器
        
        Args:
            data_dir (str): 数据目录路径
        """
        self.data_dir = data_dir
        self.community_assignments = None
        self.topic_data = None
        self.user_community_map = {}
        self.community_topic_dist = {}
        self.user_topic_map = {}  # 用户主题映射
        
    def load_data(self):
        """加载所需数据文件"""
        print("加载数据中...")
        
        # 加载社区分配数据
        comm_file = os.path.join(self.data_dir, 'output_results', 'community_assignments.csv')
        if os.path.exists(comm_file):
            self.community_assignments = pd.read_csv(comm_file)
            print(f"✓ 加载社区分配数据: {len(self.community_assignments)} 条记录")
        else:
            print(f"✗ 未找到社区分配文件: {comm_file}")
            return False
        
        # 加载主题分类数据
        topic_file = os.path.join(self.data_dir, 'topic_category_analysis_results', 'data_with_categories.csv')
        if os.path.exists(topic_file):
            # 只加载需要的列，提高效率
            needed_columns = ['platform', 'post_id', 'comment_user_id', 'category_zh']
            self.topic_data = pd.read_csv(topic_file, usecols=needed_columns)
            print(f"✓ 加载主题分类数据: {len(self.topic_data)} 条记录")
        else:
            print(f"✗ 未找到主题分类文件: {topic_file}")
            return False
        
        return True
    
    def map_users_to_communities(self):
        """将用户ID映射到社区"""
        print("映射用户到社区...")
        
        # 创建用户到社区的映射字典
        for _, row in self.community_assignments.iterrows():
            node_id = row['node_id']
            community_id = row['community_id']
            
            # 解析平台和用户ID
            if ':' in node_id:
                platform, user_id = node_id.split(':', 1)
                self.user_community_map[(platform, user_id)] = community_id
        
        print(f"✓ 已映射 {len(self.user_community_map)} 个用户到社区")
    
    def analyze_community_topics(self):
        """分析每个社区的主题类别分布"""
        if self.community_assignments is None or self.topic_data is None:
            print("请先调用 load_data() 加载数据")
            return
            
        if not self.user_community_map:
            self.map_users_to_communities()
        
        print("分析社区主题分布...")
        
        # 遍历主题数据，将每个评论与社区关联
        total_processed = 0
        matched = 0
        
        for _, row in self.topic_data.iterrows():
            platform = row['platform']
            comment_user_id = str(row['comment_user_id'])  # 转换为字符串
            category_zh = row['category_zh']
            
            total_processed += 1
            
            # 为用户分配主题（使用出现次数最多的主题）
            user_key = (platform, comment_user_id)
            if user_key not in self.user_topic_map and pd.notna(category_zh):
                self.user_topic_map[user_key] = category_zh
            
            # 查找用户所属的社区
            if user_key in self.user_community_map:
                matched += 1
                community_id = self.user_community_map[user_key]
                
                # 初始化社区主题分布
                if community_id not in self.community_topic_dist:
                    self.community_topic_dist[community_id] = {}
                
                # 更新主题计数
                if pd.notna(category_zh):
                    if category_zh not in self.community_topic_dist[community_id]:
                        self.community_topic_dist[community_id][category_zh] = 0
                    self.community_topic_dist[community_id][category_zh] += 1
        
        print(f"✓ 处理了 {total_processed} 条评论数据")
        print(f"✓ 成功匹配到 {matched} 条社区数据")
        print(f"✓ 已记录 {len(self.user_topic_map)} 个用户的主题偏好")
        print("✓ 社区主题分布分析完成")
    
    def generate_report(self):
        """生成社区主题分布报告"""
        if not self.community_topic_dist:
            print("请先调用 analyze_community_topics() 分析社区主题")
            return
        
        print("\n=== 社区主题类别分布报告 ===")
        
        # 按照社区ID排序
        sorted_communities = sorted(self.community_topic_dist.keys())
        
        for community_id in sorted_communities:
            topic_counts = self.community_topic_dist[community_id]
            total_topics = sum(topic_counts.values())
            
            print(f"\n社区ID: {community_id}")
            print(f"总评论数量: {total_topics}")
            print("主题类别分布:")
            print("-" * 80)
            
            # 按照出现次数排序主题
            sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
            
            for topic, count in sorted_topics:
                percentage = (count / total_topics) * 100
                print(f"{topic:<40} | 数量: {count:>5} | 占比: {percentage:>6.2f}%")
            
            print("-" * 80)
    
    def save_report(self):
        """保存社区主题分布报告到CSV文件"""
        if not self.community_topic_dist:
            print("请先调用 analyze_community_topics() 分析社区主题")
            return
        
        print("\n保存社区主题分布报告...")
        
        # 创建结果列表
        results = []
        for community_id, topic_counts in self.community_topic_dist.items():
            total_topics = sum(topic_counts.values())
            
            for topic, count in topic_counts.items():
                percentage = (count / total_topics) * 100
                results.append({
                    'community_id': community_id,
                    'category_zh': topic,
                    'count': count,
                    'total_comments_in_community': total_topics,
                    'percentage': round(percentage, 2)
                })
        
        # 创建数据框
        df = pd.DataFrame(results)
        df = df.sort_values(['community_id', 'count'], ascending=[True, False])
        
        # 保存到CSV
        output_dir = os.path.join(self.data_dir, 'networkdeeper')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, 'community_topic_distribution.csv')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"✓ 报告已保存到: {output_file}")
        print(f"✓ 共记录 {len(df)} 条社区-主题关联")
    
    def identify_bridging_nodes(self, graph, nodes_df, top_n=20):
        """
        识别连接不同主题社区的桥接节点
        
        参数：
        graph: NetworkX图对象
        nodes_df: 包含节点主题信息的DataFrame，必须包含'topic_category'列
        top_n: 返回前N个桥接潜力最大的节点
        
        返回：
        pd.DataFrame: 包含桥接节点信息的DataFrame
        """
        # 检查必要的列是否存在
        if 'topic_category' not in nodes_df.columns:
            raise ValueError("nodes_df必须包含'topic_category'列")
        
        # 1. 计算每个节点的邻居主题分布
        node_topic_entropy = {}
        
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if len(neighbors) < 2:  # 至少需要2个邻居才能成为桥接节点
                continue
            
            # 获取邻居的主题
            neighbor_topics = []
            for neighbor in neighbors:
                neighbor_topic = nodes_df.loc[nodes_df['node_id'] == neighbor, 'topic_category'].values
                if len(neighbor_topic) > 0:
                    neighbor_topics.append(neighbor_topic[0])
            
            if len(neighbor_topics) < 2:  # 至少需要2个有主题的邻居
                continue
            
            # 2. 计算主题分布的香农熵
            topic_counts = {}
            for topic in neighbor_topics:
                if topic in topic_counts:
                    topic_counts[topic] += 1
                else:
                    topic_counts[topic] = 1
            
            # 计算熵
            total = len(neighbor_topics)
            entropy = 0.0
            for count in topic_counts.values():
                p = count / total
                entropy -= p * math.log2(p)
            
            # 3. 计算桥接潜力 = 节点度 × 主题熵
            degree = len(neighbors)
            bridging_potential = degree * entropy
            
            # 4. 检查是否至少连接了2个不同主题
            if len(topic_counts) >= 2:
                node_topic_entropy[node] = {
                    'degree': degree,
                    'topic_count': len(topic_counts),
                    'topic_entropy': entropy,
                    'bridging_potential': bridging_potential,
                    'neighbor_topics': neighbor_topics
                }
        
        # 将结果转换为DataFrame
        bridging_df = pd.DataFrame.from_dict(node_topic_entropy, orient='index').reset_index()
        bridging_df.columns = ['node_id', 'degree', 'topic_count', 'topic_entropy', 'bridging_potential', 'neighbor_topics']
        
        # 排序并返回前N个桥接节点
        bridging_df = bridging_df.sort_values(by='bridging_potential', ascending=False)
        
        return bridging_df.head(top_n)
    
    def analyze_topic_homophily(self):
        """
        分析网络中的主题同质性
        
        计算：
        1. 实际同质性：相同主题用户间的边数 / 总边数
        2. 期望同质性：随机情况下的概率
        3. 同质性指数：(实际-期望)/(1-期望)
        4. 桥接节点识别：识别连接不同主题的关键节点
        
        返回：
            dict: 包含同质性分析结果和桥接节点信息的字典
        """
        print("\n=== 主题同质性分析 ===")
        
        # 加载边数据
        edges_file = os.path.join(self.data_dir, 'output_results', 'user_level_aggregated_edges.csv')
        if not os.path.exists(edges_file):
            print(f"✗ 未找到边数据文件: {edges_file}")
            return
        
        edges_df = pd.read_csv(edges_file)
        print(f"✓ 加载边数据: {len(edges_df)} 条边")
        
        # 分离抖音和小红书数据
        platforms = {}
        
        for platform in ['douyin', 'xhs']:
            # 筛选当前平台的边
            platform_edges = edges_df[
                edges_df['source'].str.startswith(f'{platform}:') & 
                edges_df['target'].str.startswith(f'{platform}:')
            ].copy()
            
            if len(platform_edges) == 0:
                print(f"✗ 未找到{platform}的边数据")
                continue
            
            print(f"\n{platform}边数据: {len(platform_edges)} 条边")
            
            # 解析用户ID并映射主题
            def get_topic(node_id):
                if ':' in node_id:
                    p, user_id = node_id.split(':', 1)
                    return self.user_topic_map.get((p, user_id), None)
                return None
            
            platform_edges['source_topic'] = platform_edges['source'].apply(get_topic)
            platform_edges['target_topic'] = platform_edges['target'].apply(get_topic)
            
            # 过滤掉主题未知的边
            valid_edges = platform_edges[
                platform_edges['source_topic'].notna() & 
                platform_edges['target_topic'].notna()
            ]
            
            print(f"{platform}有效边数(主题已知): {len(valid_edges)} / {len(platform_edges)}")
            
            if len(valid_edges) == 0:
                print(f"✗ 无{platform}的有效主题边数据")
                continue
            
            # 计算实际同质性
            same_topic_edges = valid_edges[valid_edges['source_topic'] == valid_edges['target_topic']]
            actual_homophily = len(same_topic_edges) / len(valid_edges)
            
            # 计算期望同质性 - 正确方式
            # 获取所有唯一用户及其主题
            all_users = pd.concat([valid_edges['source'], valid_edges['target']]).unique()
            
            # 创建用户到主题的映射
            user_topic_dict = {}
            for _, row in valid_edges.iterrows():
                user_topic_dict[row['source']] = row['source_topic']
                user_topic_dict[row['target']] = row['target_topic']
            
            # 统计每个主题的用户数量
            topic_user_counts = {}
            for user in all_users:
                if user in user_topic_dict:
                    topic = user_topic_dict[user]
                    topic_user_counts[topic] = topic_user_counts.get(topic, 0) + 1
            
            total_users = len(all_users)
            expected_homophily = sum((count/total_users)**2 for count in topic_user_counts.values())
            
            # 计算同质性指数
            if expected_homophily < 1:
                homophily_index = (actual_homophily - expected_homophily) / (1 - expected_homophily)
            else:
                homophily_index = 0
            
            # 保存结果
            platforms[platform] = {
                'actual_homophily': actual_homophily,
                'expected_homophily': expected_homophily,
                'homophily_index': homophily_index,
                'total_edges': len(platform_edges),
                'valid_edges': len(valid_edges),
                'same_topic_edges': len(same_topic_edges)
            }
        
        # 打印结果对比
        if len(platforms) >= 2:
            print("\n=== 主题同质性对比 ===")
            
            for platform, result in platforms.items():
                print(f"\n{platform}:")
                print(f"  总边数: {result['total_edges']}")
                print(f"  有效边数: {result['valid_edges']}")
                print(f"  相同主题边数: {result['same_topic_edges']}")
                print(f"  实际同质性: {result['actual_homophily']:.3f}")
                print(f"  期望同质性: {result['expected_homophily']:.3f}")
                print(f"  同质性指数: {result['homophily_index']:.3f}")
            
            # 比较两个平台
            douyin_homophily = platforms['douyin']['homophily_index']
            xhs_homophily = platforms['xhs']['homophily_index']
            
            print(f"\n抖音同质性指数: {douyin_homophily:.3f}")
            print(f"小红书同质性指数: {xhs_homophily:.3f}")
            
            if xhs_homophily > douyin_homophily:
                print("结论：小红书更倾向主题相似的用户互动")
            elif douyin_homophily > xhs_homophily:
                print("结论：抖音更倾向主题相似的用户互动")
            else:
                print("结论：两个平台在主题同质性上表现相似")
        
        # 桥接节点分析
        print("\n=== 桥接节点分析 ===")
        
        # 创建用户节点DataFrame，包含主题信息
        user_nodes = []
        for (platform, user_id), topic in self.user_topic_map.items():
            user_nodes.append({
                'node_id': f'{platform}:{user_id}',
                'platform': platform,
                'user_id': user_id,
                'topic_category': topic
            })
        
        if user_nodes:
            nodes_df = pd.DataFrame(user_nodes)
            
            # 创建图对象
            G = nx.Graph()
            
            # 添加边
            for _, row in edges_df.iterrows():
                G.add_edge(row['source'], row['target'], weight=row['weight'])
            
            print(f"图中共有 {len(G.nodes())} 个节点和 {len(G.edges())} 条边")
            
            # 用于存储平台桥接节点分析结果
            platform_bridge_results = {}
            
            # 分离抖音和小红书的节点
            for platform in ['douyin', 'xhs']:
                platform_nodes = nodes_df[nodes_df['platform'] == platform]
                platform_graph = G.subgraph(platform_nodes['node_id'])
                
                total_nodes = len(platform_graph.nodes())
                total_edges = len(platform_graph.edges())
                print(f"\n{platform}平台节点数: {total_nodes}, 边数: {total_edges}")
                
                if total_nodes < 2:
                    print(f"{platform}平台节点数不足，无法进行桥接节点分析")
                    continue
                
                # 识别桥接节点
                try:
                    bridging_nodes = self.identify_bridging_nodes(platform_graph, platform_nodes)
                    
                    if not bridging_nodes.empty:
                        # 按桥接潜力排序
                        bridging_nodes = bridging_nodes.sort_values('bridging_potential', ascending=False)
                        
                        # 分析全部桥接节点（不止前20个）
                        total_bridges = len(bridging_nodes)
                        bridge_ratio = total_bridges / total_nodes * 100
                        
                        # 计算统计指标
                        avg_degree = bridging_nodes['degree'].mean()
                        avg_topic_count = bridging_nodes['topic_count'].mean()
                        avg_entropy = bridging_nodes['topic_entropy'].mean()
                        avg_potential = bridging_nodes['bridging_potential'].mean()
                        
                        # 计算不同桥接能力的节点分布
                        high_potential_threshold = bridging_nodes['bridging_potential'].quantile(0.9)
                        high_potential_bridges = len(bridging_nodes[bridging_nodes['bridging_potential'] >= high_potential_threshold])
                        
                        print(f"\n{platform}平台桥接节点全面分析:")
                        print(f"- 总桥接节点数: {total_bridges} ({bridge_ratio:.2f}% of total nodes)")
                        print(f"- 平均度: {avg_degree:.2f}")
                        print(f"- 平均连接主题数: {avg_topic_count:.2f}")
                        print(f"- 平均主题熵: {avg_entropy:.3f}")
                        print(f"- 平均桥接潜力: {avg_potential:.2f}")
                        print(f"- 高潜力桥接节点数（前10%）: {high_potential_bridges}")
                        
                        print(f"\n{platform}平台前20个桥接节点:")
                        print(bridging_nodes[['node_id', 'degree', 'topic_count', 'topic_entropy', 'bridging_potential']].head(20))
                        
                        # 保存桥接节点数据
                        output_dir = os.path.join(self.data_dir, 'networkdeeper')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        
                        bridging_nodes.to_csv(os.path.join(output_dir, f'{platform}_bridging_nodes.csv'), 
                                             index=False, encoding='utf-8-sig')
                        
                        # 存储分析结果
                        platform_bridge_results[platform] = {
                            'total_nodes': total_nodes,
                            'total_bridges': total_bridges,
                            'bridge_ratio': bridge_ratio,
                            'avg_degree': avg_degree,
                            'avg_topic_count': avg_topic_count,
                            'avg_entropy': avg_entropy,
                            'avg_potential': avg_potential,
                            'high_potential_bridges': high_potential_bridges
                        }
                    else:
                        print(f"{platform}平台未找到桥接节点")
                    
                except Exception as e:
                    print(f"{platform}平台桥接节点分析失败: {str(e)}")
            
            # 平台间桥接节点比较
            if len(platform_bridge_results) == 2:
                print("\n=== 平台桥接节点比较分析 ===")
                
                douyin = platform_bridge_results.get('douyin', {})
                xhs = platform_bridge_results.get('xhs', {})
                
                # 计算差异百分比
                bridge_count_diff = douyin.get('total_bridges', 0) - xhs.get('total_bridges', 0)
                bridge_ratio_diff = douyin.get('bridge_ratio', 0) - xhs.get('bridge_ratio', 0)
                potential_diff = douyin.get('avg_potential', 0) - xhs.get('avg_potential', 0)
                topic_count_diff = douyin.get('avg_topic_count', 0) - xhs.get('avg_topic_count', 0)
                
                print(f"抖音 vs 小红书桥接节点对比:")
                print(f"- 桥接节点总数差异: {bridge_count_diff} (抖音: {douyin.get('total_bridges')}, 小红书: {xhs.get('total_bridges')})")
                print(f"- 桥接节点比例差异: {bridge_ratio_diff:.2f}% (抖音: {douyin.get('bridge_ratio', 0):.2f}%, 小红书: {xhs.get('bridge_ratio', 0):.2f}%)")
                print(f"- 平均桥接潜力差异: {potential_diff:.2f} (抖音: {douyin.get('avg_potential', 0):.2f}, 小红书: {xhs.get('avg_potential', 0):.2f})")
                print(f"- 平均连接主题数差异: {topic_count_diff:.2f} (抖音: {douyin.get('avg_topic_count', 0):.2f}, 小红书: {xhs.get('avg_topic_count', 0):.2f})")
                
                # 结合同质性指数分析
                print(f"\n=== 桥接节点与同质性指数关联分析 ===")
                
                if 'douyin' in platforms and 'xhs' in platforms:
                    douyin_homophily = platforms['douyin'].get('homophily_index', 0)
                    xhs_homophily = platforms['xhs'].get('homophily_index', 0)
                    
                    print(f"抖音同质性指数: {douyin_homophily:.3f}")
                    print(f"小红书同质性指数: {xhs_homophily:.3f}")
                    
                    if douyin_homophily < xhs_homophily:
                        print("\n结论1: 抖音同质性指数显著低于小红书，表明抖音主题边界更模糊")
                        
                        if bridge_ratio_diff > 0:
                            print("结论2: 抖音桥接节点比例高于小红书，说明抖音存在更多跨主题信息传播通道")
                        elif bridge_ratio_diff < 0:
                            print("结论2: 小红书桥接节点比例高于抖音，但结合低同质性指数，抖音的跨主题传播更高效")
                        else:
                            print("结论2: 两个平台桥接节点比例相似，但抖音的低同质性指数表明其跨主题传播障碍更小")
                    else:
                        print("\n结论: 同质性指数与桥接节点分布的关系需要进一步分析")
        
        return platforms

def main():
    """主函数"""
    # 创建分析器实例
    analyzer = CommunityTopicAnalyzer(data_dir='c:\\Users\\联想\\Desktop\\shuju\\all_analysis')
    
    # 加载数据
    if not analyzer.load_data():
        return
    
    # 分析社区主题分布
    analyzer.analyze_community_topics()
    
    # 生成报告
    analyzer.generate_report()
    
    # 保存报告
    analyzer.save_report()
    
    # 分析主题同质性
    analyzer.analyze_topic_homophily()

if __name__ == "__main__":
    main()