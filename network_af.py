import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PlatformNetworkAnalysis:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data()
    
    def load_data(self):
        """加载网络分析结果数据"""
        print("加载数据...")
        self.node_metrics = pd.read_csv(f'{self.data_dir}/node_level_metrics.csv')
        self.user_edges = pd.read_csv(f'{self.data_dir}/user_level_aggregated_edges.csv')
        
        # 从节点ID中提取平台信息
        self.node_metrics['platform'] = self.node_metrics['node_id'].apply(lambda x: x.split(':')[0])
        
        
        print(f"加载完成: {len(self.node_metrics)}节点")
    
    def calculate_gini(self, x):
        """计算基尼系数"""
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if cumx[-1] > 0 else 0
    
    def analyze_platform_network_metrics(self):
        """Analyze platform network metrics"""
        print("\n=== Platform Network Metrics Analysis ===")
        
        # Basic network metrics
        platform_stats = self.node_metrics.groupby('platform').agg({
            'node_id': 'count',
            'degree': ['mean', 'std'],
            'weighted_degree': ['mean', 'std'],
            'pagerank_wt': ['mean', 'std'],
            'clustering_coefficient': 'mean',
            'betweenness_centrality': 'mean'
        }).round(4)
        
        # Calculate network concentration (Gini coefficient)
        gini_by_platform = self.node_metrics.groupby('platform')['weighted_degree'].apply(self.calculate_gini)
        
        # Reorganize results
        results = {}
        for platform in self.node_metrics['platform'].unique():
            platform_data = self.node_metrics[self.node_metrics['platform'] == platform]
            results[platform] = {
                'Node Count': len(platform_data),
                'Average Degree': platform_data['degree'].mean(),
                'Average Weighted Degree': platform_data['weighted_degree'].mean(),
                'Average PageRank': platform_data['pagerank_wt'].mean(),
                'Average Clustering Coefficient': platform_data['clustering_coefficient'].mean(),
                'Network Concentration(Gini)': gini_by_platform[platform],
                'Average Betweenness Centrality': platform_data['betweenness_centrality'].mean()
            }
        
        # Convert to DataFrame for display
        platform_comparison = pd.DataFrame(results).T
        print("\nPlatform Network Metrics Comparison:")
        print(platform_comparison)
        
        # Save results to separate directory
        platform_comparison.to_csv(f'{self.output_dir}/platform_network_comparison.csv', encoding='utf-8-sig')
        
        return platform_comparison
    
    
    def create_platform_comparison_visualizations(self):
        """Create platform comparison visualizations with enhanced contrast"""
        print("\n=== Generating Visualizations ===")
        
        # Create visualization directory
        viz_dir = f'{self.output_dir}/visualizations/'
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Platform network metrics comparison charts
        metrics_to_plot = ['Average Degree', 'Average Weighted Degree', 'Average PageRank', 'Average Clustering Coefficient', 'Network Concentration(Gini)']
        platform_data = self.analyze_platform_network_metrics()
        
        # Enhanced color scheme for better contrast
        colors = ['#FF5252', '#2196F3']  # More contrasting colors
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if i < len(axes):
                # Create bar plot with narrower bars for better separation
                platform_data[metric].plot(kind='bar', ax=axes[i], color=colors, width=0.6)
                axes[i].set_title(f'{metric} - Platform Comparison', fontweight='bold')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Set y-axis to start from 0 to show the full difference
                min_val = platform_data[metric].min()
                if min_val > 0:
                    axes[i].set_ylim(bottom=0)
                
                # Add grid lines for better readability
                axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        
        axes[5].axis('off')  # 关闭第6个子图
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/platform_network_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        
        # 3. Network concentration analysis
        self._create_network_concentration_chart(viz_dir)
        
        print(f"Visualizations have been saved to: {viz_dir}")
    
    def _create_network_concentration_chart(self, viz_dir):
        """Create network concentration analysis chart"""
        # 实现网络集中度分析图表
    
       
    def run_complete_analysis(self):
        """Run complete direction analysis"""
        print("Starting Direction Analysis...")
        
        # Analyze platform network metrics
        platform_metrics = self.analyze_platform_network_metrics()
           
        
        # Generate visualizations
        self.create_platform_comparison_visualizations()
        

        
        print("\n✅  Direction Analysis Complete!")
        print(f"All result files have been saved to: {self.output_dir}")
        return platform_metrics

# 运行分析
if __name__ == "__main__":
    # 数据输入目录（从之前的分析结果中读取数据）
    data_dir = 'c:/Users/联想/Desktop/shuju/all_analysis/output_results/'
    # 创建独立的输出目录，避免与其他内容混淆
    output_dir = 'c:/Users/联想/Desktop/shuju/all_analysis/af_analysis_output/'
    
    analyzer = PlatformNetworkAnalysis(data_dir, output_dir)
    analyzer.run_complete_analysis()