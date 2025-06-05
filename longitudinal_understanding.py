#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import json

print('🏥 弗雷明汉纵向心血管疾病风险预测模型')
print('基于正确理解的纵向数据结构')
print('=' * 60)

# 加载数据
df = pd.read_csv('../frmgham_data.csv')
print(f'总记录数: {len(df)}')
print(f'独特患者数: {df["RANDID"].nunique()}')

# 理解数据结构
print(f'\nPERIOD分布:')
print(df['PERIOD'].value_counts().sort_index())

# 分析CVD字段的含义
print(f'\nCVD字段分析:')
print(f'CVD=1的患者数: {(df["CVD"]==1).sum()}')
print(f'CVD=1的比例: {(df["CVD"]==1).mean():.3f}')

# 查看几个患者的纵向数据
print(f'\n查看前3个患者的纵向数据示例:')
for randid in df['RANDID'].unique()[:3]:
    patient_data = df[df['RANDID'] == randid][['RANDID', 'PERIOD', 'AGE', 'CVD', 'PREVCHD', 'TIMECVD']].sort_values('PERIOD')
    print(f'患者 {randid}:')
    print(patient_data.to_string(index=False))
    print()

# 基线预测模型：使用第一次体检数据预测CVD风险
print('📊 创建基线预测模型（使用第一次体检数据）')
baseline_data = df[df['PERIOD'] == 1].copy()

# 选择重要的心血管风险因素
risk_factors = [
    'SEX', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 
    'CURSMOKE', 'BMI', 'DIABETES', 'PREVCHD', 
    'PREVMI', 'PREVSTRK'
]

available_features = [col for col in risk_factors if col in baseline_data.columns]
X = baseline_data[available_features].copy()
y = baseline_data['CVD']  # CVD=1表示在研究期间曾患心血管疾病

# 简单特征工程
if 'SYSBP' in X.columns and 'DIABP' in X.columns:
    X['PULSE_PRESSURE'] = X['SYSBP'] - X['DIABP']

print(f'特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}')
print(f'CVD患病率: {y.mean():.3f}')

# 处理缺失值和标准化
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 训练模型
models = {
    'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, y_pred_proba)
    results[name] = {'accuracy': accuracy, 'auc': auc, 'model': model}
    print(f'{name}: 准确率={accuracy:.4f}, AUC={auc:.4f}')

# 保存最佳模型
best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
best_model = results[best_model_name]

# 保存模型和相关文件
joblib.dump(best_model['model'], 'longitudinal_baseline_model.pkl')
joblib.dump(scaler, 'longitudinal_baseline_scaler.pkl') 
joblib.dump(imputer, 'longitudinal_baseline_imputer.pkl')

# 保存特征名称
feature_names = available_features + ['PULSE_PRESSURE']
with open('longitudinal_baseline_features.json', 'w') as f:
    json.dump(feature_names, f, indent=2)

# 保存模型信息
model_info = {
    'model_name': best_model_name,
    'model_type': 'baseline_longitudinal',
    'auc_score': best_model['auc'],
    'accuracy': best_model['accuracy'],
    'features': feature_names,
    'description': '使用第一次体检数据预测患者在研究期间是否会患CVD'
}

with open('longitudinal_baseline_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

print(f'\n🏆 最佳基线模型: {best_model_name}')
print(f'🎯 AUC分数: {best_model["auc"]:.4f}')
print(f'📈 准确率: {best_model["accuracy"]:.4f}')
print('✅ 纵向基线模型已保存')

print('\n💡 数据理解总结:')
print('1. 数据包含4434名患者，最多3次体检记录（PERIOD 1-3）')
print('2. CVD=1表示患者在整个研究期间曾经患心血管疾病')
print('3. PREVCHD=1表示在当前体检时已经患有冠心病')
print('4. TIMECVD表示从第一次体检到首次CVD诊断的时间间隔（小时）')
print('5. 基线模型使用第一次体检数据预测患者未来CVD风险')
print('6. 这是一个前瞻性队列研究的正确建模方法')

print('\n📁 生成的文件:')
print('- longitudinal_baseline_model.pkl (基线模型)')
print('- longitudinal_baseline_scaler.pkl (标准化器)')  
print('- longitudinal_baseline_imputer.pkl (缺失值处理器)')
print('- longitudinal_baseline_features.json (特征名称)')
print('- longitudinal_baseline_info.json (模型信息)') 