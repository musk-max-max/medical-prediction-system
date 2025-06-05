#!/usr/bin/env python3
"""
探索弗雷明汉纵向研究数据结构
分析PERIOD、CVD、PREVCHD、TIMECVD等关键字段的含义
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv('../frmgham_data.csv')

print('🔍 弗雷明汉纵向研究数据结构分析')
print('=' * 60)

print('\n📊 数据基本信息:')
print(f'总行数: {len(df)}')
print(f'总列数: {len(df.columns)}')

print('\n🗓️ PERIOD分布:')
period_counts = df['PERIOD'].value_counts().sort_index()
print(period_counts)

print('\n👥 独特病人数:')
unique_patients = df['RANDID'].nunique()
print(f'独特RANDID数量: {unique_patients}')

print('\n📈 每个病人的体检次数分布:')
visit_counts = df.groupby('RANDID').size()
visit_distribution = visit_counts.value_counts().sort_index()
print(visit_distribution)

print('\n❤️ CVD相关字段分析:')
print(f'CVD=1的比例: {(df["CVD"]==1).mean():.3f}')
print(f'PREVCHD=1的比例: {(df["PREVCHD"]==1).mean():.3f}')

print('\n⏱️ TIMECVD字段统计:')
timecvd_stats = df['TIMECVD'].describe()
print(timecvd_stats)

print('\n📝 示例: 查看前5个病人的纵向数据')
sample_patients = df['RANDID'].unique()[:5]
for randid in sample_patients:
    patient_data = df[df['RANDID'] == randid][
        ['RANDID', 'PERIOD', 'AGE', 'CVD', 'PREVCHD', 'TIMECVD', 'TIME']
    ].sort_values('PERIOD')
    print(f'\n病人 {randid}:')
    print(patient_data.to_string(index=False))

# 分析CVD发病时间模式
print('\n\n🎯 CVD发病时间模式分析:')

# 找到CVD=1的患者
cvd_patients = df[df['CVD'] == 1]['RANDID'].unique()
print(f'有CVD的患者数: {len(cvd_patients)}')

# 分析这些患者的TIMECVD分布
cvd_data = df[df['CVD'] == 1].groupby('RANDID')['TIMECVD'].first()
print(f'TIMECVD平均值: {cvd_data.mean():.0f} 小时 ({cvd_data.mean()/24:.0f} 天)')
print(f'TIMECVD中位数: {cvd_data.median():.0f} 小时 ({cvd_data.median()/24:.0f} 天)')

# 分析PREVCHD的含义
print('\n\n📋 PREVCHD字段含义分析:')
# 对于每个患者，查看PREVCHD在不同PERIOD中的变化
sample_analysis = []
for randid in sample_patients:
    patient_data = df[df['RANDID'] == randid][
        ['RANDID', 'PERIOD', 'PREVCHD', 'CVD', 'TIMECVD']
    ].sort_values('PERIOD')
    if len(patient_data) > 1:
        sample_analysis.append(patient_data)

print('\n几个患者的PREVCHD变化模式:')
for i, patient_data in enumerate(sample_analysis[:3]):
    print(f'\n患者样本 {i+1}:')
    print(patient_data.to_string(index=False))

# 创建数据理解总结
print('\n\n📚 数据结构理解总结:')
print('1. PERIOD: 表示第几次体检 (1=第一次, 2=第二次, 3=第三次)')
print('2. CVD: 在整个研究期间是否曾经患有心血管疾病 (0=否, 1=是)')
print('3. PREVCHD: 在当前体检时是否已经患有冠心病 (0=否, 1=是)')
print('4. TIMECVD: 从第一次体检到首次诊断CVD的时间间隔(小时)')
print('5. 每个患者最多有3次体检记录')
print('6. 数据可用于生存分析和时间序列预测')

# 保存探索结果
with open('data_structure_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('弗雷明汉纵向研究数据结构分析\n')
    f.write('=' * 40 + '\n\n')
    f.write(f'总病人数: {unique_patients}\n')
    f.write(f'总记录数: {len(df)}\n')
    f.write(f'CVD患病率: {(df["CVD"]==1).mean():.3f}\n')
    f.write('\n体检次数分布:\n')
    for visits, count in visit_distribution.items():
        f.write(f'{visits}次体检: {count}人\n')

print('\n✅ 数据结构分析完成，结果已保存到 data_structure_analysis.txt') 