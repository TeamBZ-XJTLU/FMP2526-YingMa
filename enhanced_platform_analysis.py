import pandas as pd
import numpy as np
import os
import sys
import traceback
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from datetime import datetime

class TopicCategoryAnalyzer:
    def __init__(self, data_path, output_folder="topic_category_analysis_results"):
        self.data_path = data_path
        self.output_folder = output_folder
        self.data = None
        self.category_mapping = self.define_category_mapping()
        
        # 创建输出文件夹
        os.makedirs(self.output_folder, exist_ok=True)
    
    def define_category_mapping(self):
        """定义主题到八大类别的映射关系"""
        # 主题映射：原始主题ID -> (合并主题英文名称, 合并主题中文名称, 核心描述, 关键价值)
        # 注意：保留中文名称以确保与现有数据处理的向后兼容性
        mapping = {
            # 1. Outliers/Miscellaneous
            -1: ("Outliers / Misc.", "Outliers / Misc.", 
                 "Uncategorized short text, noise, garbled characters, 22% of total", 
                 "Document 'model limitations', not included in main analysis"),
            
            # 2. Emoji & Light Interaction
            0: ("Emoji & Light Interaction", "Emoji & Light Interaction",
                "Light interactions mainly using emojis (blow kiss, laugh with tears, heart)",
                "Reflects 'high participation without substantive content' platform characteristics"),
            4: ("Emoji & Light Interaction", "Emoji & Light Interaction",
                "Light interactions mainly using emojis (blow kiss, laugh with tears, heart)",
                "Reflects 'high participation without substantive content' platform characteristics"),
            
            # 3. 幽默娱乐
            2: ("Humor & Entertainment", "Humor & Entertainment",
                "Memes, doge, nonsensical teasing, highly entertaining",
                "Typical 'Douyin-style' high-engagement topics, distinguishing entertainment-oriented platforms"),
            6: ("Humor & Entertainment", "Humor & Entertainment",
                "Memes, doge, nonsensical teasing, highly entertaining",
                "Typical 'Douyin-style' high-engagement topics, distinguishing entertainment-oriented platforms"),
            23: ("Humor & Entertainment", "Humor & Entertainment",
                "Memes, doge, nonsensical teasing, highly entertaining",
                "Typical 'Douyin-style' high-engagement topics, distinguishing entertainment-oriented platforms"),
            
            # 4. 情感表达
            3: ("Emotional Expression", "Emotional Expression",
                "Covers sadness (tears, sobbing), positive praise (beautiful, love), strong emotions (cracking up)",
                "Matches Xiaohongshu 'emotional resonance' hypothesis, distinguishing emotion-oriented platforms"),
            5: ("Emotional Expression", "Emotional Expression",
                "Covers sadness (tears, sobbing), positive praise (beautiful, love), strong emotions (cracking up)",
                "Matches Xiaohongshu 'emotional resonance' hypothesis, distinguishing emotion-oriented platforms"),
            10: ("Emotional Expression", "Emotional Expression",
                "Covers sadness (tears, sobbing), positive praise (beautiful, love), strong emotions (cracking up)",
                "Matches Xiaohongshu 'emotional resonance' hypothesis, distinguishing emotion-oriented platforms"),
            11: ("Emotional Expression", "Emotional Expression",
                "Covers sadness (tears, sobbing), positive praise (beautiful, love), strong emotions (cracking up)",
                "Matches Xiaohongshu 'emotional resonance' hypothesis, distinguishing emotion-oriented platforms"),
            18: ("Emotional Expression", "Emotional Expression",
                "Covers sadness (tears, sobbing), positive praise (beautiful, love), strong emotions (cracking up)",
                "Matches Xiaohongshu 'emotional resonance' hypothesis, distinguishing emotion-oriented platforms"),
            21: ("Emotional Expression", "Emotional Expression",
                "Covers sadness (tears, sobbing), positive praise (beautiful, love), strong emotions (cracking up)",
                "Matches Xiaohongshu 'emotional resonance' hypothesis, distinguishing emotion-oriented platforms"),
            
            # 5. 实用互动
            1: ("Practical Interaction", "Practical Interaction",
                "Product inquiries, livestream scheduling, product recommendations, user experiences, makeup technique questions",
                "Consumption/advisory oriented, likes = 'useful', distinguishing utility-oriented platforms"),
            14: ("Practical Interaction", "Practical Interaction",
                "Product inquiries, livestream scheduling, product recommendations, user experiences, makeup technique questions",
                "Consumption/advisory oriented, likes = 'useful', distinguishing utility-oriented platforms"),
            19: ("Practical Interaction", "Practical Interaction",
                "Product inquiries, livestream scheduling, product recommendations, user experiences, makeup technique questions",
                "Consumption/advisory oriented, likes = 'useful', distinguishing utility-oriented platforms"),

            20: ("Practical Interaction", "Practical Interaction",
                "Product inquiries, livestream scheduling, product recommendations, user experiences, makeup technique questions",
                "Consumption/advisory oriented, likes = 'useful', distinguishing utility-oriented platforms"),
            22: ("Practical Interaction", "Practical Interaction",
                "Product inquiries, livestream scheduling, product recommendations, user experiences, makeup technique questions",
                "Consumption/advisory oriented, likes = 'useful', distinguishing utility-oriented platforms"),
            
            # 6. 积极仪式互动
            7: ("Positive Ritual Interaction", "Positive Ritual Interaction",
                "Congratulations, sending roses, 'charge' and other ritualized positive expressions",
                "Strong emotional identity, reflecting platform 'positive atmosphere'"),
            8: ("Positive Ritual Interaction", "Positive Ritual Interaction",
                "Congratulations, sending roses, 'charge' and other ritualized positive expressions",
                "Strong emotional identity, reflecting platform 'positive atmosphere'"),
            16: ("Positive Ritual Interaction", "Positive Ritual Interaction",
                "Congratulations, sending roses, 'charge' and other ritualized positive expressions",
                "Strong emotional identity, reflecting platform 'positive atmosphere'"),
            
            # 7. 社区参与
            12: ("Community Engagement", "Community Engagement",
                "Wishing, raising hands to participate, front row seating and other community behaviors",
                "Reflects platform 'community activity', distinguishing community-oriented platforms"),
            13: ("Community Engagement", "Community Engagement",
                "Wishing, raising hands to participate, front row seating and other community behaviors",
                "Reflects platform 'community activity', distinguishing community-oriented platforms"),
            17: ("Community Engagement", "Community Engagement",
                "Wishing, raising hands to participate, front row seating and other community behaviors",
                "Reflects platform 'community activity', distinguishing community-oriented platforms"),
            
            
            # 8. 身份与个人聊天
            9: ("Identity & Personal Chat", "Identity & Personal Chat",
                "Regional origin, zodiac signs, personal affairs sharing",
                "Reflects users' personal expression needs"),
            15: ("Identity & Personal Chat", "Identity & Personal Chat",
                "Regional origin, zodiac signs, personal affairs sharing",
                "Reflects users' personal expression needs"),
        }
        
        # 验证是否覆盖了所有主题
        all_topics = set(range(-1, 24))  # 从-1到23的所有主题
        mapped_topics = set(mapping.keys())
        missing_topics = all_topics - mapped_topics
        
        if missing_topics:
            print(f"警告: 以下主题未在映射中定义: {missing_topics}")
            # 将缺失的主题添加到离群/杂项类别
            for topic in missing_topics:
                mapping[topic] = ("Outliers / Misc.", "离群 / 杂项",
                                 "未归类的短文本、纯噪声、乱码",
                                 "单独说明 '模型限制'，不参与主分析")
        
        return mapping
    
    def load_data(self):
        """加载数据并检查必要的列"""
        print(f"加载数据: {self.data_path}")
        self.data = pd.read_csv(self.data_path, encoding='utf-8', low_memory=False)
        print(f"成功加载{len(self.data)}条记录")
        
        # 检查必要的列
        required_columns = ['platform', 'topic_id', 'like_count']
        for col in required_columns:
            if col not in self.data.columns:
                # 尝试查找类似的列名
                similar_cols = [c for c in self.data.columns if col in c.lower()]
                if similar_cols:
                    print(f"警告: 未找到'{col}'列，使用'{similar_cols[0]}'替代")
                    self.data.rename(columns={similar_cols[0]: col}, inplace=True)
                else:
                    raise ValueError(f"数据中缺少必要的列: {col}")
        
        print(f"数据列: {list(self.data.columns)}")
    
    def map_topics_to_categories(self):
        """将原始主题映射到八大类别"""
        if self.data is None:
            self.load_data()
        
        # 创建映射列
        self.data['category_en'] = self.data['topic_id'].apply(
            lambda x: self.category_mapping.get(x, ("Outliers / Misc.", "", "", ""))[0]
        )
        self.data['category_zh'] = self.data['topic_id'].apply(
            lambda x: self.category_mapping.get(x, ("", "Outliers / Misc.", "", ""))[1]
        )
        self.data['core_description'] = self.data['topic_id'].apply(
            lambda x: self.category_mapping.get(x, ("", "", "Uncategorized content", ""))[2]
        )
        self.data['key_value'] = self.data['topic_id'].apply(
            lambda x: self.category_mapping.get(x, ("", "", "", "Undefined value"))[3]
        )
        
        # 添加用于可视化的topic_category列，直接使用category_en
        self.data['topic_category'] = self.data['category_en']
        
        print("主题映射完成，已添加八大类别列")
        
        # 显示类别分布
        print("\n八大类别主题分布:")
        category_dist = self.data['category_zh'].value_counts()
        for category, count in category_dist.items():
            percentage = (count / len(self.data)) * 100
            print(f"  {category}: {count}条记录 ({percentage:.1f}%)")
        
        return self.data
    
    def analyze_like_statistics(self):
        """分析各平台八大类主题的点赞统计数据"""
        if 'category_en' not in self.data.columns:
            self.map_topics_to_categories()
        
        # 按平台和类别分组计算统计量
        like_stats = self.data.groupby(['platform', 'category_en', 'category_zh']).agg({
            'like_count': ['count', 'mean', 'median', 'std']
        }).round(2)
        
        # 重命名列
        like_stats.columns = ['count', 'mean', 'median', 'std']
        like_stats = like_stats.reset_index()
        
        # 按平台分别显示结果
        platforms = self.data['platform'].unique()
        for platform in platforms:
            print(f"\n=== {platform}平台八大类别点赞统计 ===")
            platform_stats = like_stats[like_stats['platform'] == platform]
            # 按类别排序，保持八大类别的顺序
            category_order = [cat[1] for cat in self.category_mapping.values() if cat[1] != ""][:8]
            platform_stats['category_order'] = platform_stats['category_zh'].apply(
                lambda x: category_order.index(x) if x in category_order else 99
            )
            platform_stats = platform_stats.sort_values('category_order').drop('category_order', axis=1)
            print(platform_stats[['category_zh', 'count', 'mean', 'median', 'std']].to_string(index=False))
        
        # 保存统计结果
        output_path = os.path.join(self.output_folder, 'platform_category_like_statistics.csv')
        like_stats.to_csv(output_path, encoding='utf-8-sig', index=False)
        print(f"\n点赞统计结果已保存至: {output_path}")
        
        # 保存包含八大类别映射的完整数据
        full_data_path = os.path.join(self.output_folder, 'data_with_categories.csv')
        self.data.to_csv(full_data_path, encoding='utf-8-sig', index=False)
        print(f"包含八大类别映射的完整数据已保存至: {full_data_path}")
        
        # 生成八大类别主题说明表
        categories_info = []
        for cat_en, cat_zh, desc, value in set(self.category_mapping.values()):
            categories_info.append({
                '合并主题名称（英文）': cat_en,
                '合并主题名称': cat_zh,
                '核心描述': desc,
                '关键价值': value,
                '包含的原始主题': ', '.join([f"主题 {k}" for k, v in self.category_mapping.items() if v[0] == cat_en])
            })
        
        categories_df = pd.DataFrame(categories_info)
        categories_path = os.path.join(self.output_folder, 'category_definitions.csv')
        categories_df.to_csv(categories_path, encoding='utf-8-sig', index=False)
        print(f"八大类别主题说明表已保存至: {categories_path}")
        
        return like_stats
    
    def add_time_period_column(self):
        """向数据添加时间段分类，确保时间识别成功"""
        if self.data is None:
            self.load_data()
        
        # 优先处理特定时间列
        if 'comment_dt' in self.data.columns:
            # 使用comment_dt列（带时区的ISO格式）
            date_column = 'comment_dt'
            print(f"使用日期列: {date_column}")
            try:
                # 解析带时区的ISO格式
                self.data['post_date'] = pd.to_datetime(self.data[date_column], utc=True)
                # 如需转换到本地时区
                # self.data['post_date'] = self.data['post_date'].dt.tz_convert('Asia/Shanghai')
            except Exception as e:
                raise ValueError(f"解析comment_dt列时出错: {e}。请确保时间格式正确。")
        elif 'comment_time_ts' in self.data.columns:
            # 使用comment_time_ts列（Unix时间戳）
            date_column = 'comment_time_ts'
            print(f"使用日期列: {date_column}")
            try:
                # 将时间戳转换为datetime
                self.data['post_date'] = pd.to_datetime(self.data[date_column], unit='ms')
            except Exception as e:
                raise ValueError(f"解析comment_time_ts列时出错: {e}。请确保时间戳格式正确。")
        else:
            # 回退到标准列检测
            self._fallback_time_parsing()
        
        # 基于小时创建时间段 - 使用请求的四个时间段
        self.data['hour'] = self.data['post_date'].dt.hour
        self.data['time_period'] = pd.cut(
            self.data['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Late Night (0-6AM)', 'Morning (6-12PM)', 'Afternoon (12-6PM)', 'Evening (6-12AM)'],
            include_lowest=True
        )
        
        # 验证时间段创建是否成功
        if 'time_period' not in self.data.columns or self.data['time_period'].isna().any():
            raise ValueError("时间段分类失败。请检查数据中的时间格式是否正确。")
        
        print("时间段分类完成")
        print("时间段分布:")
        print(self.data['time_period'].value_counts().sort_index())
        
    def _fallback_time_parsing(self):
        """当优先列不可用时的时间解析回退方法，确保解析成功"""
        # 检查与时间相关的列
        date_columns = [col for col in self.data.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'post'])]
        
        if not date_columns:
            raise ValueError("错误：未找到合适的时间列。请确保数据中包含有效的时间信息列。")
        
        # 使用找到的第一个日期列
        date_column = date_columns[0]
        print(f"使用日期列: {date_column}")
        
        # 尝试不同的日期格式转换
        parsing_error = None
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S']:
            try:
                self.data['post_date'] = pd.to_datetime(self.data[date_column], format=fmt)
                parsing_error = None
                break
            except Exception as e:
                parsing_error = e
        else:
            # 如果解析失败，使用默认解析
            try:
                self.data['post_date'] = pd.to_datetime(self.data[date_column])
            except Exception as e:
                raise ValueError(f"解析{date_column}列时出错: {parsing_error or e}。请确保时间格式正确。")
        
        # 基于小时创建时间段 - 使用请求的四个时间段
        self.data['hour'] = self.data['post_date'].dt.hour
        self.data['time_period'] = pd.cut(
            self.data['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Late Night (0-6AM)', 'Morning (6-12PM)', 'Afternoon (12-6PM)', 'Evening (6-12AM)'],
            include_lowest=True
        )
        
        print("时间段分类完成")
        print("时间段分布:")
        print(self.data['time_period'].value_counts().sort_index())
    
    def analyze_time_based_likes(self):
        """分析每个主题在不同时间段的平均点赞数"""
        if 'category_zh' not in self.data.columns:
            self.map_topics_to_categories()
        
        if 'time_period' not in self.data.columns:
            self.add_time_period_column()
        
        print("\n" + "="*80)
        print("分析每个主题在不同时间段的平均点赞数")
        print("="*80)
        
        # 分析每个主题在不同时间段的平均点赞数
        like_time_analysis = self.data.groupby(['platform', 'category_zh', 'time_period'], observed=True).agg({
            'like_count': 'mean'
        }).reset_index()
        
        # 保存分析结果
        time_analysis_path = os.path.join(self.output_folder, 'time_period_like_mean_analysis.csv')
        like_time_analysis.to_csv(time_analysis_path, encoding='utf-8-sig', index=False)
        print(f"时间段平均点赞数分析结果已保存至: {time_analysis_path}")
        
        return like_time_analysis
    
    def create_time_period_heatmap(self, like_time_analysis):
        """创建指定主题在不同时间段的平均点赞热图"""
        # 定义要显示的三个主题
        specified_topics = ['Practical Interaction', 'Humor & Entertainment', 'Emotional Expression']
        
        # 过滤数据以仅包含指定主题
        filtered_analysis = like_time_analysis[like_time_analysis['category_zh'].isin(specified_topics)]
        
        # 为热图数据创建透视表
        pivot_likes = filtered_analysis.pivot_table(
            index=['platform', 'category_zh'],
            columns='time_period',
            values='like_count',
            fill_value=0,
            observed=True
        )
        
        # 为每个平台单独创建热图
        platforms = filtered_analysis['platform'].unique()
        
        for platform in platforms:
            # 过滤当前平台数据
            platform_data = pivot_likes.loc[platform]
            
            # 设置图形大小
            plt.figure(figsize=(14, 8))
            
            # Create heatmap
            sns.heatmap(platform_data, annot=True, fmt='.1f', cmap='YlOrRd',
                       cbar_kws={'label': 'Mean Likes'})
            
            # Set title and labels
            plt.title(f'{platform} Platform Mean Likes by Topic and Time Period', fontsize=16)
            plt.xlabel('Time Period', fontsize=12)
            plt.ylabel('Topic Category', fontsize=12)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存热图
            heatmap_path = os.path.join(self.output_folder, f'{platform}_time_period_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ {platform}平台热图已保存至: {heatmap_path}")
        
        # 创建所有平台的组合热图
        plt.figure(figsize=(18, 10))
        
        # 创建组合热图
        sns.heatmap(pivot_likes, annot=True, fmt='.1f', cmap='YlOrRd',
                   cbar_kws={'label': 'Mean Likes'})
        
        # Set title and labels
        plt.title('Mean Likes by Platform, Topic and Time Period', fontsize=18)
        plt.xlabel('Time Period', fontsize=14)
        plt.ylabel('Platform - Topic Category', fontsize=14)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存组合热图
        combined_heatmap_path = os.path.join(self.output_folder, 'combined_platform_time_period_heatmap.png')
        plt.savefig(combined_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 组合热图已保存至: {combined_heatmap_path}")
    
    def create_topic_comparison_chart(self, analysis_df): 
        """创建三个主题类别的平台平均点赞比较图（无对数转换，优化美学效果）"""
        print("\n" + "="*80)
        print("为情感表达、幽默娱乐和实用互动主题创建平台比较图")
        print("="*80)
        
        # 设置字体以获得更好的显示效果
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        # 确保非交互式后端以获得更好的稳定性
        mpl.use('Agg')
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 定义三个主题类别
        topic_mapping = {
            "Emotional Expression": [
                "Emotional Expression"
            ],
            "Humor & Entertainment": [
                "Humor & Entertainment"
            ],
            "Practical Interaction": [
                "Practical Interaction"
            ]
        }
        
        # 转换数据格式以便于处理
        topic_data = []
        for main_topic, sub_topics in topic_mapping.items():
            # 过滤此类别下的所有子主题数据
            sub_data = analysis_df[analysis_df['topic_category'].isin(sub_topics)].copy()
            # 计算各平台的平均点赞数
            platform_avg = sub_data.groupby('platform')['like_count'].mean().reset_index()
            # 添加主主题类别
            platform_avg['main_topic'] = main_topic
            topic_data.append(platform_avg)
        
        # 合并为单个数据框
        comparison_df = pd.concat(topic_data, ignore_index=True)
        
        # 设置图形大小
        plt.figure(figsize=(14, 8))
        
        # 定义美观的配色方案
        colors = {'douyin': '#FF2442', 'weibo': '#E6162D', 'xhs': '#FE2C55', 'other_platform': '#6B7A8F'}
        # 如果平台数量超过预设，使用替代颜色映射
        all_platforms = comparison_df['platform'].unique()
        if len(all_platforms) > len(colors):
            colors = dict(zip(all_platforms, sns.color_palette('husl', len(all_platforms))))
        
        # 替换平台名称为英文
        platform_mapping = {
            'douyin': 'Douyin',
            'weibo': 'Weibo',
            'xhs': 'Xiaohongshu'
        }
        comparison_df['platform'] = comparison_df['platform'].map(platform_mapping)
        
        # 使用更专业的颜色方案
        professional_colors = {'Douyin': '#3498db', 'Weibo': '#e74c3c', 'Xiaohongshu': '#9b59b6'}
        
        # 创建分组条形图，使用更专业的样式
        ax = sns.barplot(
            data=comparison_df,
            x='like_count',
            y='main_topic',
            hue='platform',
            palette=professional_colors,
            edgecolor='white',  # 使用白色边框增加现代感
            linewidth=1.5,
            saturation=0.85  # 稍微降低饱和度使颜色更柔和
        )
        
        # 添加数据标签，使用更优雅的格式
        for i, container in enumerate(ax.containers):
            ax.bar_label(
                container,
                fmt='%.1f',
                label_type='edge',
                fontsize=11,
                fontweight='bold',
                padding=8,
                color='black'
            )
        
        # 设置更专业的标题和标签
        plt.title('Average Likes Comparison by Topic and Platform',
                  fontsize=18, fontweight='bold', pad=20, fontname='Arial')
        plt.xlabel('Average Likes', fontsize=14, labelpad=12, fontname='Arial')
        plt.ylabel('Topic Category', fontsize=14, labelpad=12, fontname='Arial')
        
        # 优化刻度样式
        plt.xticks(fontsize=12, fontname='Arial')
        plt.yticks(fontsize=12, fontname='Arial')
        
        # 美化图例 - 使用英文标题
        legend = plt.legend(title='Platform', title_fontsize=13, fontsize=11,
                   loc='upper right', frameon=True, edgecolor='lightgray',
                   facecolor='white', shadow=True)
        
        # 为图例文本设置字体
        for text in legend.get_texts():
            text.set_fontname('Arial')
        legend.get_title().set_fontname('Arial')
        
        # 调整网格线，使其更精细
        plt.grid(axis='x', linestyle='-', alpha=0.2, linewidth=0.8)
        
        # 去除边框，使用更现代的无边框设计
        sns.despine(top=True, right=True, left=True, bottom=False)
        
        # 添加轻微的背景色渐变效果
        ax.set_facecolor('#f8f9fa')
        plt.gcf().set_facecolor('white')
        
        # 调整布局
        plt.tight_layout(pad=2.0)
        
        # 保存图表
        chart_path = os.path.join(self.output_folder, 'three_topics_platform_comparison.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 三主题平台比较图已保存至: {chart_path}")
    
    def run_analysis(self):
        """运行完整的分析流程"""
        try:
            print("开始八大类别主题点赞统计分析...")
            
            # 1. 加载数据
            self.load_data()
            
            # 2. 将主题映射到八大类别
            self.map_topics_to_categories()
            
            # 3. 分析点赞统计数据
            self.analyze_like_statistics()
            
            # 4. 创建可视化图表
            self.create_topic_comparison_chart(self.data)
            
            # 5. 分析基于时间的点赞并创建热图
            like_time_analysis = self.analyze_time_based_likes()
            self.create_time_period_heatmap(like_time_analysis)
            
            print("\n✅ 八大类别主题点赞统计分析完成!")
            print(f"所有结果已保存至: {self.output_folder}")
            
        except Exception as e:
            print(f"分析过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # 默认参数
    data_path = "bertopic_results/full_topic_results.csv"  # 包含topic_id和like_count的数据文件
    output_folder = "topic_category_analysis_results"
    
    # 如果提供了命令行参数，则使用它们
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    
    print(f"使用数据文件: {data_path}")
    print(f"输出文件夹: {output_folder}")
    
    # 创建分析器实例并运行分析
    analyzer = TopicCategoryAnalyzer(data_path, output_folder)
    analyzer.run_analysis()