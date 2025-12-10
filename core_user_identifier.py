import pandas as pd
import numpy as np
import os


class CoreUserIdentifier:
    """核心用户识别器
    
    该类提供多种核心用户识别方法，支持跨平台对比分析。
    """
    
    def __init__(self, node_metrics, output_dir=None):
        """初始化核心用户识别器
        
        Args:
            node_metrics (pd.DataFrame): 包含节点指标的DataFrame，必须包含以下列：
                - node_id: 节点ID，格式为 platform:user_id
                - 各种中心性指标列（如pagerank_wt, betweenness_centrality等）
            output_dir (str, optional): 输出目录路径
        """
        self.node_metrics = node_metrics
        self.output_dir = output_dir
        self.core_users = None
        
        # 验证输入数据
        if 'node_id' not in node_metrics.columns:
            raise ValueError("node_metrics必须包含'node_id'列")
        
        # 从node_id中提取平台信息
        self.node_metrics['platform'] = self.node_metrics['node_id'].str.split(':', expand=True)[0]
    
    @classmethod
    def from_file(cls, file_path, output_dir=None):
        """从文件创建CoreUserIdentifier实例
        
        支持读取network_analysis.py生成的node_level_metrics.csv文件
        
        Args:
            file_path (str): 节点指标文件路径，如node_level_metrics.csv
            output_dir (str, optional): 输出目录路径
        
        Returns:
            CoreUserIdentifier: CoreUserIdentifier实例
        """
        # 读取文件
        node_metrics = pd.read_csv(file_path)
        
        # 验证文件内容
        required_columns = ['node_id', 'pagerank_wt', 'betweenness_centrality', 'weighted_degree']
        for col in required_columns:
            if col not in node_metrics.columns:
                raise ValueError(f"文件 {file_path} 缺少必要列 {col}")
        
        print(f"从文件 {file_path} 成功读取节点指标数据")
        print(f"  - 节点数: {len(node_metrics)}")
        print(f"  - 列数: {len(node_metrics.columns)}")
        
        # 创建并返回实例
        return cls(node_metrics, output_dir)
    
    def identify_core_users(self, method='comprehensive', top_percent=5, platform_normalize=True):
        """识别核心用户
        
        Args:
            method (str): 核心用户识别方法：
                'comprehensive': 综合多个中心性指标（默认）
                'pagerank': 基于PageRank中心性
                'betweenness': 基于介数中心性
                'community': 基于社区内部重要性
            top_percent (float): 选择前top_percent%的用户作为核心用户
            platform_normalize (bool): 是否按平台分别进行指标归一化，确保跨平台对比公平性（默认True）
        
        Returns:
            pd.DataFrame: 包含核心用户信息的DataFrame
        """
        print(f"\n=== Step: 识别核心用户 ===")
        print(f"使用方法: {method}, 选择前 {top_percent}% 的用户, 平台归一化: {platform_normalize}")
        
        # 创建核心用户DataFrame
        core_users = self.node_metrics.copy()
        
        # 根据选择的方法计算核心得分
        if method == 'comprehensive':
            # 综合多个中心性指标（归一化后加权平均）
            metrics_to_use = ['pagerank_wt', 'betweenness_centrality', 'weighted_degree']
            
            # 归一化每个指标（0-1范围）
            for metric in metrics_to_use:
                if platform_normalize:
                    # 按平台分别归一化
                    norm_values = []
                    for platform, group in core_users.groupby('platform'):
                        max_val = group[metric].max()
                        min_val = group[metric].min()
                        if max_val != min_val:
                            normalized = (group[metric] - min_val) / (max_val - min_val)
                        else:
                            # 确保返回与group相同长度和索引的Series，而不是整数
                            normalized = pd.Series(0, index=group.index)
                        norm_values.append(normalized)
                    core_users[f'{metric}_norm'] = pd.concat(norm_values)
                else:
                    # 全局归一化
                    max_val = core_users[metric].max()
                    min_val = core_users[metric].min()
                    if max_val != min_val:
                        core_users[f'{metric}_norm'] = (core_users[metric] - min_val) / (max_val - min_val)
                    else:
                        core_users[f'{metric}_norm'] = 0
            
            # 计算综合得分（加权平均）
            weights = {'pagerank_wt_norm': 0.35, 'betweenness_centrality_norm': 0.35, 'weighted_degree_norm': 0.3}
            core_users['core_score'] = 0
            for metric, weight in weights.items():
                core_users['core_score'] += core_users[metric] * weight
                
        elif method == 'pagerank':
            core_users['core_score'] = core_users['pagerank_wt']
            
        elif method == 'betweenness':
            core_users['core_score'] = core_users['betweenness_centrality']
            

            
        elif method == 'community':
            # 基于社区内部重要性：在每个社区内计算综合得分
            if 'community_id' not in core_users.columns:
                raise ValueError("使用'community'方法时，node_metrics必须包含'community_id'列")
                
            metrics_to_use = ['pagerank_wt', 'betweenness_centrality', 'weighted_degree']
            core_users['core_score'] = 0
            
            for community_id, group in core_users.groupby('community_id'):
                for metric in metrics_to_use:
                    if platform_normalize:
                        # 社区内按平台归一化
                        norm_values = []
                        for platform, platform_group in group.groupby('platform'):
                            max_val = platform_group[metric].max()
                            min_val = platform_group[metric].min()
                            if max_val != min_val:
                                normalized = (platform_group[metric] - min_val) / (max_val - min_val)
                            else:
                                # 确保返回与platform_group相同长度和索引的Series，而不是整数
                                normalized = pd.Series(0, index=platform_group.index)
                            norm_values.append(normalized)
                        if norm_values:
                            core_users.loc[group.index, f'{metric}_community_norm'] = pd.concat(norm_values)
                        else:
                            # 确保返回与group相同长度和索引的Series，而不是整数
                            core_users.loc[group.index, f'{metric}_community_norm'] = pd.Series(0, index=group.index)
                    else:
                        # 社区内全局归一化
                        max_val = group[metric].max()
                        min_val = group[metric].min()
                        if max_val != min_val:
                            core_users.loc[group.index, f'{metric}_community_norm'] = (group[metric] - min_val) / (max_val - min_val)
                        else:
                            # 确保返回与group相同长度和索引的Series，而不是整数
                            core_users.loc[group.index, f'{metric}_community_norm'] = pd.Series(0, index=group.index)
            
            # 计算社区内综合得分
            weights = {'pagerank_wt_community_norm': 0.33, 'betweenness_centrality_community_norm': 0.33, 
                      'weighted_degree_community_norm': 0.34}
            for metric, weight in weights.items():
                core_users['core_score'] += core_users[metric] * weight
        
        # 选择前top_percent%的用户作为核心用户
        num_core_users = int(len(core_users) * (top_percent / 100))
        core_users = core_users.nlargest(num_core_users, 'core_score')
        core_users['is_core_user'] = True
        
        # 按平台统计核心用户数量
        print("\n核心用户平台分布:")
        platform_distribution = core_users['platform'].value_counts()
        for platform, count in platform_distribution.items():
            print(f"  - {platform}: {count} 个核心用户")
        
        print(f"识别到 {len(core_users)} 个核心用户")
        
        # 保存核心用户信息
        if self.output_dir is not None:
            output_file = f"{self.output_dir}/core_users.csv"
            core_users.to_csv(output_file, index=False)
            
            # 验证文件是否存在
            file_status = "✓ 成功" if os.path.exists(output_file) else "✗ 失败"
            file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
            
            print(f"\n核心用户识别完成")
            print(f"  - 核心用户数: {len(core_users)}")
            print(f"  - 保存文件: {output_file}")
            print(f"  - 状态: {file_status}")
            print(f"  - 文件大小: {file_size:,} 字节")
        
        self.core_users = core_users
        return core_users
    
    def get_platform_distribution(self):
        """获取核心用户的平台分布
        
        Returns:
            pd.Series: 各平台核心用户数量分布
        """
        if self.core_users is None:
            raise ValueError("请先调用identify_core_users方法")
        return self.core_users['platform'].value_counts()
    
    def get_core_users_by_platform(self, platform):
        """获取指定平台的核心用户
        
        Args:
            platform (str): 平台名称
            
        Returns:
            pd.DataFrame: 指定平台的核心用户DataFrame
        """
        if self.core_users is None:
            raise ValueError("请先调用identify_core_users方法")
        return self.core_users[self.core_users['platform'] == platform]
    
    def get_core_users_by_community(self, community_id):
        """获取指定社区的核心用户
        
        Args:
            community_id (int): 社区ID
            
        Returns:
            pd.DataFrame: 指定社区的核心用户DataFrame
        """
        if self.core_users is None:
            raise ValueError("请先调用identify_core_users方法")
        if 'community_id' not in self.core_users.columns:
            raise ValueError("core_users不包含'community_id'列")
        return self.core_users[self.core_users['community_id'] == community_id]


if __name__ == "__main__":
    """主函数：执行核心用户识别流程"""
    print("=== 核心用户识别程序 ===")
    print("正在加载节点指标数据...")
    
    # 设置文件路径
    input_file = "output_results/node_level_metrics.csv"
    output_dir = "output_results"
    
    try:
        # 创建CoreUserIdentifier实例
        core_identifier = CoreUserIdentifier.from_file(input_file, output_dir)
        
        # 识别核心用户
        core_users = core_identifier.identify_core_users(
            method='comprehensive', 
            top_percent=5, 
            platform_normalize=True
        )
        
        print("\n=== 程序执行完成 ===")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except ValueError as e:
        print(f"错误：{e}")
    except Exception as e:
        print(f"发生未知错误：{e}")
