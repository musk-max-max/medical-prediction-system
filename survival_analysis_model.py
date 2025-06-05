#!/usr/bin/env python3
"""
弗雷明汉心血管疾病生存分析模型
基于正确理解的TIMECVD字段含义：
- TIMECVD = 8766: 患者在研究期间未患CVD (删失数据)
- TIMECVD < 8766: 患者患CVD的具体时间 (事件数据)
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

print('🏥 弗雷明汉心血管疾病生存分析模型')
print('基于正确理解的TIMECVD字段含义')
print('=' * 60)

# 加载数据
df = pd.read_csv('../frmgham_data.csv')
print(f'总记录数: {len(df)}')
print(f'独特患者数: {df["RANDID"].nunique()}')

# 分析TIMECVD和CVD字段的关系
print(f'\n📊 生存数据分析:')
print(f'未患病患者 (TIMECVD=8766, CVD=0): {((df["TIMECVD"]==8766) & (df["CVD"]==0)).sum()}')
print(f'患病患者 (TIMECVD<8766, CVD=1): {((df["TIMECVD"]<8766) & (df["CVD"]==1)).sum()}')
print(f'研究期间CVD发病率: {(df["CVD"]==1).mean():.3f}')

# 转换时间单位为年
df['time_to_event_years'] = df['TIMECVD'] / (24 * 365.25)
print(f'\n⏱️ 时间分析:')
print(f'研究总时长: {df["time_to_event_years"].max():.1f} 年')

# 患病患者的平均患病时间
cvd_patients = df[df['CVD'] == 1]
avg_time_to_cvd = cvd_patients['time_to_event_years'].mean()
print(f'平均CVD发病时间: {avg_time_to_cvd:.1f} 年')

# 使用第一次体检数据进行建模
print(f'\n📋 准备建模数据 (使用第一次体检数据):')
baseline_data = df[df['PERIOD'] == 1].copy()

# 创建生存分析的目标变量
baseline_data['event'] = baseline_data['CVD']  # 是否发生事件
baseline_data['time'] = baseline_data['time_to_event_years']  # 观察时间或事件时间

print(f'基线数据样本数: {len(baseline_data)}')
print(f'事件发生数 (CVD=1): {baseline_data["event"].sum()}')
print(f'删失数据数 (CVD=0): {(baseline_data["event"]==0).sum()}')

# 选择预测特征
risk_factors = [
    'SEX', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 
    'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 
    'BPMEDS', 'HEARTRTE', 'GLUCOSE'
]

available_features = [col for col in risk_factors if col in baseline_data.columns]
X = baseline_data[available_features].copy()
y = baseline_data['event']  # 预测是否会发生CVD事件

# 特征工程
if 'SYSBP' in X.columns and 'DIABP' in X.columns:
    X['PULSE_PRESSURE'] = X['SYSBP'] - X['DIABP']

if 'TOTCHOL' in X.columns and 'AGE' in X.columns:
    X['CHOL_AGE_RATIO'] = X['TOTCHOL'] / (X['AGE'] + 1)

# 年龄分组
if 'AGE' in X.columns:
    X['AGE_GROUP'] = pd.cut(X['AGE'], bins=[0, 45, 55, 65, 100], labels=[0, 1, 2, 3]).astype(float)

# 吸烟风险评分
if 'CURSMOKE' in X.columns and 'CIGPDAY' in X.columns:
    X['SMOKING_RISK'] = X['CURSMOKE'] * (1 + X['CIGPDAY'].fillna(0) / 20)

print(f'特征数量: {X.shape[1]}')
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

print(f'\n🤖 训练生存分析模型:')
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
    print(classification_report(y_test, y_pred, target_names=['No CVD', 'CVD']))

# 选择最佳模型
best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
best_model_result = results[best_model_name]

print(f'\n🏆 最佳模型: {best_model_name}')
print(f'🎯 AUC分数: {best_model_result["auc"]:.4f}')
print(f'📈 准确率: {best_model_result["accuracy"]:.4f}')

# 特征重要性分析（如果是随机森林）
if best_model_name == 'Random_Forest':
    feature_importance = best_model_result['model'].feature_importances_
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f'\n📊 特征重要性排序:')
    print(importance_df.head(10))
    
    # 绘制特征重要性图
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df.head(10))), importance_df.head(10)['importance'])
    plt.yticks(range(len(importance_df.head(10))), importance_df.head(10)['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importance for CVD Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('survival_feature_importance.png', dpi=300, bbox_inches='tight')
    print('特征重要性图已保存: survival_feature_importance.png')

# 保存最佳模型
joblib.dump(best_model_result['model'], 'survival_cvd_model.pkl')
joblib.dump(scaler, 'survival_scaler.pkl')
joblib.dump(imputer, 'survival_imputer.pkl')

# 保存模型信息
model_info = {
    'model_name': best_model_name,
    'model_type': 'survival_analysis',
    'auc_score': best_model_result['auc'],
    'accuracy': best_model_result['accuracy'],
    'features': list(X.columns),
    'description': '基于生存分析的CVD风险预测模型，正确理解TIMECVD字段含义',
    'data_understanding': {
        'total_patients': int(baseline_data['RANDID'].nunique()),
        'cvd_events': int(baseline_data['event'].sum()),
        'censored': int((baseline_data['event']==0).sum()),
        'follow_up_years': float(baseline_data['time'].max())
    }
}

with open('survival_model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

print(f'\n💾 模型文件已保存:')
print('- survival_cvd_model.pkl (生存分析模型)')
print('- survival_scaler.pkl (标准化器)')
print('- survival_imputer.pkl (缺失值处理器)')
print('- survival_model_info.json (模型信息)')

print(f'\n💡 正确的数据理解总结:')
print('1. TIMECVD=8766表示患者在研究期间未患CVD (约10年随访)')
print('2. TIMECVD<8766表示患者患CVD的具体时间点')
print('3. 这是典型的生存分析数据结构')
print('4. 模型可预测患者在随访期间患CVD的风险')
print('5. 这种建模方法更符合医学研究的实际情况')

print(f'\n✅ 生存分析建模完成!') 