import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('frmgham_data.csv')
print('数据集形状:', df.shape)
print('\n所有列名:')
print(df.columns.tolist())

# 找出TIME前缀的列
time_cols = [col for col in df.columns if col.startswith('TIME')]
print('\nTIME前缀的列:')
print(time_cols)

# 找出所有疾病相关的列
disease_cols = []
for col in df.columns:
    if any(keyword in col.upper() for keyword in ['CVD', 'CHD', 'STROKE', 'HYPERT', 'DIABETES', 'ANGINA', 'HOSPMI', 'MI', 'AP', 'IC', 'DEATH']):
        disease_cols.append(col)

print('\n疾病相关的列:')
print(disease_cols)

# 分析每个TIME列
print('\n各TIME列的详细分析:')
for col in time_cols:
    print(f'\n{col}:')
    print(f'  唯一值数: {df[col].nunique()}')
    print(f'  最小值: {df[col].min()}')
    print(f'  最大值: {df[col].max()}')
    print(f'  平均值: {df[col].mean():.2f}')
    print(f'  中位数: {df[col].median():.2f}')
    print(f'  缺失值: {df[col].isnull().sum()}')
    
    # 显示前10个最常见的值
    print(f'  最常见的值:')
    print(df[col].value_counts().head())

# 检查对应的疾病结果列
print('\n\n疾病结果列分析:')
for col in disease_cols:
    if col in df.columns:
        print(f'\n{col}:')
        print(f'  唯一值: {df[col].unique()}')
        print(f'  值分布:')
        print(df[col].value_counts())
        
        # 如果有对应的TIME列，显示关系
        time_col = f'TIME{col}'
        if time_col in df.columns:
            print(f'  对应时间列: {time_col}')
            # 分析患病和未患病的时间分布
            sick = df[df[col] == 1]
            healthy = df[df[col] == 0]
            if len(sick) > 0:
                print(f'  患病者平均时间: {sick[time_col].mean():.2f}')
            if len(healthy) > 0:
                print(f'  未患病者平均时间: {healthy[time_col].mean():.2f}')

# 创建疾病-时间对应表
print('\n\n疾病与时间字段对应关系:')
disease_time_pairs = []
for time_col in time_cols:
    # 去掉TIME前缀找对应的疾病列
    disease_name = time_col.replace('TIME', '')
    if disease_name in df.columns:
        disease_time_pairs.append((disease_name, time_col))
        print(f'{disease_name} <-> {time_col}')

print(f'\n找到 {len(disease_time_pairs)} 个疾病-时间对应关系') 