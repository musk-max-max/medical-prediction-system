#!/usr/bin/env python3
"""
医疗预测系统 - 数据分析和预处理
Framingham Heart Study 数据分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data(file_path):
    """加载和探索数据"""
    print("🔍 加载数据...")
    df = pd.read_csv(file_path)
    
    print(f"📊 数据形状: {df.shape}")
    print(f"📋 列数: {df.shape[1]}")
    print(f"📝 行数: {df.shape[0]}")
    
    print("\n📑 数据信息:")
    print(df.info())
    
    print("\n📈 数值型特征描述统计:")
    print(df.describe())
    
    return df

def analyze_missing_values(df):
    """分析缺失值"""
    print("\n🔍 缺失值分析:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        '缺失数量': missing,
        '缺失比例(%)': missing_percent
    })
    missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
    
    if not missing_df.empty:
        print(missing_df)
        
        # 可视化缺失值
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        missing_df['缺失数量'].plot(kind='bar')
        plt.title('缺失值数量')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        missing_df['缺失比例(%)'].plot(kind='bar')
        plt.title('缺失值比例 (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 缺失值分析图已保存: missing_values_analysis.png")
    else:
        print("✅ 没有缺失值")
    
    return missing_df

def analyze_target_variable(df):
    """分析目标变量"""
    # 候选目标变量
    potential_targets = ['CVD', 'MI_FCHD', 'ANYCHD', 'STROKE', 'DEATH', 'ANGINA', 'HOSPMI']
    
    print("\n🎯 目标变量分析:")
    for target in potential_targets:
        if target in df.columns:
            value_counts = df[target].value_counts()
            print(f"\n{target}:")
            print(value_counts)
            print(f"患病率: {(value_counts.get(1, 0) / len(df)) * 100:.2f}%")

def correlation_analysis(df):
    """相关性分析"""
    print("\n🔗 相关性分析:")
    
    # 选择数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("📊 相关性热力图已保存: correlation_heatmap.png")
        
        # 找出高相关性特征对
        high_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > 0.7:  # 高相关性阈值
                    high_corr.append({
                        'Feature1': correlation_matrix.columns[i],
                        'Feature2': correlation_matrix.columns[j],
                        'Correlation': correlation_matrix.iloc[i, j]
                    })
        
        if high_corr:
            print("\n⚠️ 高相关性特征对 (|r| > 0.7):")
            for pair in high_corr:
                print(f"{pair['Feature1']} - {pair['Feature2']}: {pair['Correlation']:.3f}")

def feature_distributions(df):
    """特征分布分析"""
    print("\n📊 特征分布分析:")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 移除ID列
    numeric_cols = [col for col in numeric_cols if 'ID' not in col.upper()]
    
    if len(numeric_cols) > 0:
        n_cols = 4
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(20, 5 * n_rows))
        
        for i, col in enumerate(numeric_cols[:16]):  # 限制显示前16个特征
            plt.subplot(n_rows, n_cols, i + 1)
            df[col].hist(bins=30, alpha=0.7)
            plt.title(f'{col} 分布')
            plt.xlabel(col)
            plt.ylabel('频数')
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        print("📊 特征分布图已保存: feature_distributions.png")

def main():
    """主函数"""
    print("🏥 医疗预测系统 - 数据分析")
    print("=" * 50)
    
    # 加载数据
    df = load_and_explore_data('../frmgham_data.csv')
    
    # 缺失值分析
    missing_info = analyze_missing_values(df)
    
    # 目标变量分析
    analyze_target_variable(df)
    
    # 相关性分析
    correlation_analysis(df)
    
    # 特征分布分析
    feature_distributions(df)
    
    # 保存清理后的数据信息
    print("\n💾 保存数据摘要...")
    with open('data_summary.txt', 'w', encoding='utf-8') as f:
        f.write("医疗预测系统数据摘要\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"数据形状: {df.shape}\n")
        f.write(f"特征数量: {df.shape[1]}\n")
        f.write(f"样本数量: {df.shape[0]}\n\n")
        
        f.write("列名:\n")
        for col in df.columns:
            f.write(f"- {col}\n")
    
    print("✅ 数据分析完成!")
    print("📁 生成的文件:")
    print("  - missing_values_analysis.png")
    print("  - correlation_heatmap.png") 
    print("  - feature_distributions.png")
    print("  - data_summary.txt")

if __name__ == "__main__":
    main() 