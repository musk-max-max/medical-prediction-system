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
print('=' * 60)

# 加载数据
df = pd.read_csv('../frmgham_data.csv')
print(f'总记录数: {len(df)}')
print(f'独特患者数: {df["RANDID"].nunique()}')
print(f'CVD患病率: {(df["CVD"]==1).mean():.3f}')

# 基线模型：使用第一次体检数据
baseline_data = df[df['PERIOD'] == 1].copy()
risk_factors = ['SEX', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 'BMI', 'DIABETES', 'PREVCHD', 'PREVMI', 'PREVSTRK']
available_features = [col for col in risk_factors if col in baseline_data.columns]
X = baseline_data[available_features].copy()
y = baseline_data['CVD']

# 特征工程
if 'SYSBP' in X.columns and 'DIABP' in X.columns:
    X['PULSE_PRESSURE'] = X['SYSBP'] - X['DIABP']

print(f'\n基线模型特征数: {X.shape[1]}, 样本数: {X.shape[0]}')
print(f'CVD患病率: {y.mean():.3f}')

# 处理缺失值和标准化
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

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
joblib.dump(best_model['model'], 'final_cvd_model.pkl')
joblib.dump(scaler, 'final_scaler.pkl')
joblib.dump(imputer, 'final_imputer.pkl')

model_info = {
    'model_name': best_model_name,
    'auc_score': best_model['auc'],
    'accuracy': best_model['accuracy'],
    'features': available_features + ['PULSE_PRESSURE']
}
with open('final_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f'\n🏆 最佳模型: {best_model_name}')
print(f'🎯 AUC分数: {best_model["auc"]:.4f}')
print('✅ 模型已保存')

print('\n💡 数据理解总结:')
print('1. 数据包含4434名患者，最多3次体检记录')
print('2. CVD=1表示在整个研究期间曾患心血管疾病')
print('3. PREVCHD=1表示在当前体检时已患冠心病')
print('4. TIMECVD表示从第一次体检到CVD诊断的时间(小时)')
print('5. 本模型使用第一次体检数据预测未来CVD风险') 