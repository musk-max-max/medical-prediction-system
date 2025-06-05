#!/usr/bin/env python3
"""
弗雷明汉心血管疾病24年随访预测模型
基于正确理解的TIMECVD字段含义：
- TIMECVD = 8766天: 患者在24年随访期内未患CVD (删失数据)  
- TIMECVD < 8766天: 患者患CVD的具体时间点 (事件数据)
- 单位：天
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib
import json

print('🏥 弗雷明汉心血管疾病24年随访预测模型')
print('基于正确理解的TIMECVD字段含义（单位：天）')
print('=' * 60)

# 加载数据
df = pd.read_csv('../frmgham_data.csv')
print(f'总记录数: {len(df)}')
print(f'独特患者数: {df["RANDID"].nunique()}')

# 分析TIMECVD和CVD字段的关系（单位：天）
print(f'\n📊 24年随访生存数据分析:')
print(f'研究总时长: {8766}天 = {8766/365.25:.1f}年')
print(f'未患病患者 (TIMECVD=8766天, CVD=0): {((df["TIMECVD"]==8766) & (df["CVD"]==0)).sum()}')
print(f'患病患者 (TIMECVD<8766天, CVD=1): {((df["TIMECVD"]<8766) & (df["CVD"]==1)).sum()}')
print(f'24年随访CVD发病率: {(df["CVD"]==1).mean():.3f}')

# 转换时间单位为年
df['time_to_event_years'] = df['TIMECVD'] / 365.25
print(f'\n⏱️ 详细时间分析:')
print(f'最长随访时间: {df["time_to_event_years"].max():.1f} 年')

# 患病患者的时间分布
cvd_patients = df[df['CVD'] == 1]
avg_time_to_cvd = cvd_patients['time_to_event_years'].mean()
median_time_to_cvd = cvd_patients['time_to_event_years'].median()
print(f'CVD平均发病时间: {avg_time_to_cvd:.1f} 年')
print(f'CVD中位发病时间: {median_time_to_cvd:.1f} 年')

# 分析不同时间段的发病情况
print(f'\n📈 CVD发病时间分布:')
time_bins = [0, 5, 10, 15, 20, 25]
cvd_time_dist = pd.cut(cvd_patients['time_to_event_years'], bins=time_bins, include_lowest=True)
print(cvd_time_dist.value_counts().sort_index())

# 使用第一次体检数据进行建模
print(f'\n📋 准备建模数据 (使用第一次体检数据):')
baseline_data = df[df['PERIOD'] == 1].copy()

# 创建生存分析的目标变量
baseline_data['event'] = baseline_data['CVD']  # 是否发生CVD事件
baseline_data['time_years'] = baseline_data['TIMECVD'] / 365.25  # 时间（年）

print(f'基线数据样本数: {len(baseline_data)}')
print(f'CVD事件发生数: {baseline_data["event"].sum()}')
print(f'删失数据数: {(baseline_data["event"]==0).sum()}')
print(f'CVD发病率: {baseline_data["event"].mean():.3f}')

# 选择预测特征（排除既往CVD相关特征，因为这是预测模型）
risk_factors = [
    'SEX', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 
    'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 
    'BPMEDS', 'HEARTRTE', 'GLUCOSE'
]

available_features = [col for col in risk_factors if col in baseline_data.columns]
X = baseline_data[available_features].copy()
y = baseline_data['event']  # 预测24年内是否会发生CVD事件

# 特征工程
print(f'\n🔧 特征工程:')
if 'SYSBP' in X.columns and 'DIABP' in X.columns:
    X['PULSE_PRESSURE'] = X['SYSBP'] - X['DIABP']
    print('+ 脉压差 (收缩压-舒张压)')

if 'TOTCHOL' in X.columns and 'AGE' in X.columns:
    X['CHOL_AGE_RATIO'] = X['TOTCHOL'] / (X['AGE'] + 1)
    print('+ 胆固醇年龄比')

# 年龄分组
if 'AGE' in X.columns:
    X['AGE_GROUP'] = pd.cut(X['AGE'], bins=[0, 45, 55, 65, 100], labels=[0, 1, 2, 3]).astype(float)
    print('+ 年龄分组')

# 高血压分类
if 'SYSBP' in X.columns and 'DIABP' in X.columns:
    # 按照标准血压分类
    X['HYPERTENSION_STAGE'] = 0  # 正常
    X.loc[(X['SYSBP'] >= 130) | (X['DIABP'] >= 80), 'HYPERTENSION_STAGE'] = 1  # 1期高血压
    X.loc[(X['SYSBP'] >= 140) | (X['DIABP'] >= 90), 'HYPERTENSION_STAGE'] = 2  # 2期高血压
    print('+ 高血压分期')

# 吸烟风险评分
if 'CURSMOKE' in X.columns and 'CIGPDAY' in X.columns:
    X['SMOKING_RISK'] = X['CURSMOKE'] * (1 + X['CIGPDAY'].fillna(0) / 20)
    print('+ 吸烟风险评分')

print(f'最终特征数量: {X.shape[1]}')
print(f'特征列表: {list(X.columns)}')

# 处理缺失值
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f'\n🤖 训练24年CVD风险预测模型:')
print(f'训练集大小: {len(X_train)}')
print(f'测试集大小: {len(X_test)}')
print(f'训练集CVD率: {y_train.mean():.3f}')

# 训练多个模型
models = {
    'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
}

results = {}
for name, model in models.items():
    print(f'\n📈 训练 {name}...')
    
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc': auc,
        'y_pred_proba': y_pred_proba
    }
    
    print(f'准确率: {accuracy:.4f}')
    print(f'AUC: {auc:.4f}')
    
    # 详细分类报告
    print('分类报告:')
    print(classification_report(y_test, y_pred, target_names=['No CVD', 'CVD'], digits=3))

# 选择最佳模型
best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
best_model_result = results[best_model_name]

print(f'\n🏆 最佳模型: {best_model_name}')
print(f'🎯 AUC分数: {best_model_result["auc"]:.4f}')
print(f'📈 准确率: {best_model_result["accuracy"]:.4f}')

# 特征重要性分析
if best_model_name == 'Random_Forest':
    feature_importance = best_model_result['model'].feature_importances_
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f'\n📊 特征重要性排序:')
    for i, row in importance_df.head(10).iterrows():
        print(f'{row["feature"]}: {row["importance"]:.4f}')
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importance_df.head(10))), importance_df.head(10)['importance'])
    plt.yticks(range(len(importance_df.head(10))), importance_df.head(10)['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importance for 24-Year CVD Risk Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('cvd_24year_feature_importance.png', dpi=300, bbox_inches='tight')
    print('\n特征重要性图已保存: cvd_24year_feature_importance.png')

# 保存最佳模型
joblib.dump(best_model_result['model'], 'framingham_24year_cvd_model.pkl')
joblib.dump(scaler, 'framingham_24year_scaler.pkl')
joblib.dump(imputer, 'framingham_24year_imputer.pkl')

# 保存特征名称
with open('framingham_24year_features.json', 'w') as f:
    json.dump(list(X.columns), f, indent=2)

# 保存模型信息
model_info = {
    'model_name': best_model_name,
    'model_type': '24_year_cvd_risk_prediction',
    'auc_score': float(best_model_result['auc']),
    'accuracy': float(best_model_result['accuracy']),
    'features': list(X.columns),
    'description': '基于弗雷明汉24年随访数据的CVD风险预测模型',
    'study_details': {
        'total_patients': int(baseline_data['RANDID'].nunique()),
        'cvd_events': int(baseline_data['event'].sum()),
        'censored': int((baseline_data['event']==0).sum()),
        'follow_up_years': 24.0,
        'avg_time_to_cvd_years': float(avg_time_to_cvd),
        'median_time_to_cvd_years': float(median_time_to_cvd)
    }
}

with open('framingham_24year_model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

print(f'\n💾 模型文件已保存:')
print('- framingham_24year_cvd_model.pkl (24年CVD风险预测模型)')
print('- framingham_24year_scaler.pkl (标准化器)')
print('- framingham_24year_imputer.pkl (缺失值处理器)')
print('- framingham_24year_features.json (特征名称)')
print('- framingham_24year_model_info.json (模型信息)')

print(f'\n💡 弗雷明汉24年随访研究总结:')
print('1. 这是一个24年长期随访的前瞻性队列研究')
print('2. TIMECVD单位为天，8766天 = 24年')
print('3. 平均CVD发病时间约12年，中位发病时间约{:.1f}年'.format(median_time_to_cvd))
print('4. 模型可预测患者24年内CVD发病风险')
print('5. 这是真正的长期心血管疾病风险评估工具')
print('6. 适用于临床长期风险分层和预防决策')

print(f'\n✅ 24年CVD风险预测模型构建完成!') 