#!/usr/bin/env python3
import pandas as pd
import numpy as np

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

print('\n⏱️ TIMECVD字段统计 (小时):')
timecvd_stats = df['TIMECVD'].describe()
print(timecvd_stats)
print(f'转换为天数: 平均 {timecvd_stats["mean"]/24:.0f} 天, 中位数 {timecvd_stats["50%"]/24:.0f} 天')

print('\n📝 示例: 查看前3个病人的纵向数据')
sample_patients = df['RANDID'].unique()[:3]
for randid in sample_patients:
    patient_data = df[df['RANDID'] == randid][
        ['RANDID', 'PERIOD', 'AGE', 'CVD', 'PREVCHD', 'TIMECVD', 'TIME']
    ].sort_values('PERIOD')
    print(f'\n病人 {randid}:')
    print(patient_data.to_string(index=False))

print('\n\n📚 数据结构理解总结:')
print('1. PERIOD: 表示第几次体检 (1=第一次, 2=第二次, 3=第三次)')
print('2. CVD: 在整个研究期间是否曾经患有心血管疾病 (0=否, 1=是)')
print('3. PREVCHD: 在当前体检时是否已经患有冠心病 (0=否, 1=是)')
print('4. TIMECVD: 从第一次体检到首次诊断CVD的时间间隔(小时)')
print('5. 每个患者最多有3次体检记录')
print('6. 数据可用于生存分析和时间序列预测') 