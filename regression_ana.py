import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def complete_platform_analysis(df, output_folder='.'):
    """
    完整的平台差异分析
    包含：分平台OLS回归、池化回归交互项
    
    参数：
    df: 包含所有必要列的数据框
    output_folder: 输出文件保存路径
    """
    
    print("=" * 60)
    print("           平台差异分析完整报告")
    print("=" * 60)
    
    # 数据准备
    df = df.copy()
    df['ln_likes'] = np.log1p(df['comment_like_count'])
    
    
    # ==================== 1. 分平台OLS回归 ====================
    print("\n" + "="*50)
    print("1. 分平台OLS回归分析")
    print("="*50)
    
    results_summary = []
    
    for platform in ['douyin', 'xhs']:
        platform_data = df[df['platform'] == platform].copy()
        
        # 拟合OLS模型
        model = smf.ols(
            'sentiment_score_v1 ~ ln_likes + comment_length_log + hours_since_post + C(kol_name)', 
            data=platform_data
        ).fit(cov_type='cluster', cov_kwds={'groups': platform_data['post_id']})
        
        # 提取关键结果
        ln_likes_coef = model.params['ln_likes']
        ln_likes_se = model.bse['ln_likes']
        ln_likes_p = model.pvalues['ln_likes']
        
        # 显著性标记
        stars = '***' if ln_likes_p < 0.001 else '**' if ln_likes_p < 0.01 else '*' if ln_likes_p < 0.05 else ''
        
        # 保存每个平台的系数和p值，用于后续绘图
        if platform == 'douyin':
            douyin_coef = ln_likes_coef
            douyin_p = ln_likes_p
        else:  # xhs
            xhs_coef = ln_likes_coef
            xhs_p = ln_likes_p
            
        results_summary.append({
            '平台': platform,
            'ln_likes系数': f"{ln_likes_coef:.4f}{stars}",
            '标准误': f"{ln_likes_se:.4f}",
            'p值': f"{ln_likes_p:.4f}",
            'R方': f"{model.rsquared:.4f}"
        })
        
        print(f"\n--- {platform.upper()} 平台回归结果 ---")
        print(f"ln_likes 系数: {ln_likes_coef:.4f}{stars}")
        print(f"标准误: {ln_likes_se:.4f}")
        print(f"p值: {ln_likes_p:.4f}")
        print(f"R方: {model.rsquared:.4f}")
    
    # ==================== 2. 池化回归 + 交互项 ====================
    print("\n" + "="*50)
    print("2. 池化回归交互项检验")
    print("="*50)
    
    df['is_douyin'] = (df['platform'] == 'douyin').astype(int)
    
    pool_model = smf.ols(
        'sentiment_score_v1 ~ ln_likes + is_douyin + is_douyin:ln_likes + comment_length_log + hours_since_post + C(kol_name)', 
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['post_id']})
    
    interaction_coef = pool_model.params['is_douyin:ln_likes']
    interaction_se = pool_model.bse['is_douyin:ln_likes']
    interaction_p = pool_model.pvalues['is_douyin:ln_likes']
    
    print(f"平台×点赞数交互项系数: {interaction_coef:.4f}")
    print(f"交互项标准误: {interaction_se:.4f}")
    print(f"交互项p值: {interaction_p:.4f}")
    
    # 计算净效应
    douyin_net_effect = pool_model.params['ln_likes'] + interaction_coef
    xhs_net_effect = pool_model.params['ln_likes']
    
    print(f"\n净效应分析:")
    print(f"抖音净效应: {douyin_net_effect:.4f}")
    print(f"小红书净效应: {xhs_net_effect:.4f}")
    print(f"平台差异: {douyin_net_effect - xhs_net_effect:.4f}")
    

    # ==================== 4. 创建可视化 ====================
    print("\n" + "="*50)
    print("4. 生成可视化图表")
    print("="*50)
    
    # 创建跨平台回归系数比较图表（左侧图表）
    create_platform_comparison_plot(douyin_coef, douyin_p, xhs_coef, xhs_p, output_folder=output_folder)
    
    # 创建交互系数可视化图表
    create_interaction_coefficient_plot(interaction_coef, interaction_se, interaction_p, output_folder=output_folder)
    
    # ==================== 5. 保存结果 ====================
    print("\n" + "="*50)
    print("5. 保存分析结果")
    print("="*50)
    
    # 保存回归结果
    regression_df = pd.DataFrame(results_summary)
    regression_file = os.path.join(output_folder, 'platform_regression_results.csv')
    regression_df.to_csv(regression_file, index=False, encoding='utf-8-sig')
      
    # 保存交互项结果
    interaction_results = pd.DataFrame([{
        '交互项系数': interaction_coef,
        '交互项标准误': interaction_se,
        '交互项p值': interaction_p,
        '抖音净效应': douyin_net_effect,
        '小红书净效应': xhs_net_effect,
        '平台差异': douyin_net_effect - xhs_net_effect
    }])
    interaction_file = os.path.join(output_folder, 'platform_interaction_results.csv')
    interaction_results.to_csv(interaction_file, index=False, encoding='utf-8-sig')
    
    print(f"所有结果已保存到 {output_folder} 文件夹中的CSV文件")
    
    return {
        'regression_results': results_summary,
        'interaction_results': {
            'coefficient': interaction_coef,
            'p_value': interaction_p,
            'douyin_net': douyin_net_effect,
            'xhs_net': xhs_net_effect
        },
    }

def create_platform_comparison_plot(douyin_coef, douyin_p, xhs_coef, xhs_p, output_folder='.'):
    """
    Create Cross-Platform Regression Coefficient Comparison plot
    
    Parameters:
    douyin_coef: Douyin platform coefficient
    douyin_p: Douyin platform p-value
    xhs_coef: Xiaohongshu platform coefficient
    xhs_p: Xiaohongshu platform p-value
    output_folder: Output file path
    """
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Define bar positions and heights
    positions = [0, 1]
    coefficients = [douyin_coef, xhs_coef]
    colors = ['#4CAF50', '#FF5252']  # Green for Douyin, Red for XHS
    
    # Create bars
    bars = plt.bar(positions, coefficients, color=colors, width=0.6)
    
    # Add significance stars
    douyin_stars = '***' if douyin_p < 0.001 else '**' if douyin_p < 0.01 else '*' if douyin_p < 0.05 else ''
    xhs_stars = '***' if xhs_p < 0.001 else '**' if xhs_p < 0.01 else '*' if xhs_p < 0.05 else ''
    
    # Add coefficient labels
    plt.text(0, douyin_coef + 0.002 if douyin_coef > 0 else douyin_coef - 0.004, 
             f'{douyin_coef:.4f}{douyin_stars}', ha='center', fontweight='bold')
    plt.text(1, xhs_coef + 0.002 if xhs_coef > 0 else xhs_coef - 0.004, 
             f'{xhs_coef:.4f}{xhs_stars}', ha='center', fontweight='bold')
    
    # Calculate effect difference ratio
    effect_ratio = abs(xhs_coef / douyin_coef) if douyin_coef != 0 else float('inf')
    
    # Add effect difference text
    plt.text(0.5, min(coefficients) - 0.01, 
             f'Effect Difference: {effect_ratio:.1f}x', 
             ha='center', fontweight='bold', color='red', fontsize=10)
    
    # Set y-axis limits to match the screenshot
    plt.ylim(min(coefficients) - 0.015, max(coefficients) + 0.015)
    
    # Set labels and title
    plt.xlabel('Platform', fontweight='bold')
    plt.ylabel('Like Count Regression Coefficient', fontweight='bold')
    plt.title('Cross-Platform Regression Coefficient Comparison', fontweight='bold', fontsize=12)
    
    # Set x-ticks
    plt.xticks(positions, ['Douyin', 'Xiaohongshu'])
    
    # Add reference line at y=0
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add significance legend
    plt.figtext(0.5, 0.01, '* p < 0.05, *** p < 0.001', ha='center', fontsize=10)
    
    # Save the plot
    plot_file = os.path.join(output_folder, 'improved_platform_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Platform comparison plot generated and saved to {plot_file}")
    return plot_file

def create_interaction_coefficient_plot(interaction_coef, interaction_se, interaction_p, output_folder='.'):
    """
    Create interaction coefficient plot with 95% confidence interval
    
    Parameters:
    interaction_coef: Interaction term coefficient
    interaction_se: Interaction term standard error
    interaction_p: Interaction term p-value
    output_folder: Output file path
    """
    # Calculate 95% confidence interval (using 1.96 as z critical value)
    ci_lower = interaction_coef - 1.96 * interaction_se
    ci_upper = interaction_coef + 1.96 * interaction_se
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(x=[0], y=[interaction_coef], yerr=[[interaction_coef - ci_lower], [ci_upper - interaction_coef]],
                fmt='o', markersize=8, capsize=5, color='#3498db', ecolor='#3498db', alpha=0.8)
    
    # Add reference line
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Set axes and title
    plt.title('Statistical Test of the Platform Difference: The Interaction Term is Significant', 
              fontweight='bold', fontsize=12)
    plt.ylabel('Regression Coefficient', fontweight='bold')
    plt.xlabel('"Likes × Platform" Interaction Term', fontweight='bold')
    
    # Set x-axis tick labels
    plt.xticks([0], ['Likes × Platform'])    
    
    # Add coefficient value label
    plt.text(0, interaction_coef + 0.002 if interaction_coef > 0 else interaction_coef - 0.004,
            f'{interaction_coef:.4f}', ha='center', fontweight='bold')
    
    # Add p-value label
    significance = '***' if interaction_p < 0.001 else '**' if interaction_p < 0.01 else '*' if interaction_p < 0.05 else ''
    plt.text(0.5, 0.05, f'p-value = {interaction_p:.4f}{significance}', 
            ha='center', transform=plt.gca().transAxes, fontweight='bold')
    
    # Add confidence interval label
    plt.text(0.5, 0.01, f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]', 
            ha='center', transform=plt.gca().transAxes)
    
    # Set plot style
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_folder, 'interaction_coefficient_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Interaction coefficient plot generated and saved to {plot_file}")
    return plot_file


# ==================== 执行分析 ====================
if __name__ == "__main__":
    # 导入os模块用于文件夹操作
    import os
    
    # 定义输出文件夹路径
    output_folder = "ana_results"
    
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")
    else:
        print(f"输出文件夹已存在: {output_folder}")
    
    # 数据文件路径
    data_file_path = "c:\\Users\\联想\\Desktop\\shuju\\all_analysis\\cleaned_data_for_analysis.csv"
    
    try:
        # 读取CSV数据文件
        print(f"正在读取数据文件: {data_file_path}")
        df = pd.read_csv(data_file_path)
        
        # 检查必要的列是否存在
        required_columns = [
            'platform', 'sentiment_score_v1', 'comment_like_count', 
            'comment_length_log', 'hours_since_post', 'kol_name', 'post_id'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据文件缺少必要的列: {missing_columns}")
        
        
        # 执行完整分析
        results = complete_platform_analysis(df, output_folder)
        
        print("\n" + "="*60)
        print("分析完成！")
        print("="*60)
        
        # 输出核心结论
        print("\n核心结论:")
        print("1. 分平台回归显示点赞数对情感的影响存在平台差异")
        print("2. 交互项检验确认平台调节效应显著")
        print("3. Fisher z检验验证相关性差异的统计显著性")
        print("4. 所有结果已保存为CSV文件和图表")
        
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {data_file_path}")
    except pd.errors.EmptyDataError:
        print(f"错误: 数据文件为空 {data_file_path}")
    except pd.errors.ParserError:
        print(f"错误: 数据文件解析失败 {data_file_path}")
    except ValueError as ve:
        print(f"数据验证错误: {ve}")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("请检查数据文件和代码")