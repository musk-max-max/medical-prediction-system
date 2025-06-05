#!/usr/bin/env python3
"""
弗雷明汉多疾病预测模型训练
为每种疾病训练单独的预测模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print('🏥 弗雷明汉心血管疾病风险预测模型训练')
print('=' * 60)

# 加载数据
df = pd.read_csv('../frmgham_data.csv')
print('🔍 加载弗雷明汉心脏研究数据...')
print(f'原始数据形状: {df.shape}')

# 选择特征
features = [
    'SEX', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY',
    'BMI', 'DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE'
]

# 选择目标变量
targets = {
    'CVD': 'CVD',
    'CHD': 'ANYCHD',
    'STROKE': 'STROKE',
    'ANGINA': 'ANGINA',
    'MI': 'MI_FCHD',
    'HYPERTENSION': 'HYPERTEN',
    'DEATH': 'DEATH'
}

# 数据预处理
print('🔧 进行特征工程...')
df = df[features + list(targets.values())]
print(f'过滤后数据形状: {df.shape}')
print(f'使用的特征: {features}')

# 特征工程
def engineer_features(df):
    # 脉压差
    df['PULSE_PRESSURE'] = df['SYSBP'] - df['DIABP']
    
    # 胆固醇年龄比
    df['CHOL_AGE_RATIO'] = df['TOTCHOL'] / (df['AGE'] + 1)
    
    # 年龄分组
    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 45, 55, 65, 100], labels=[0, 1, 2, 3]).astype(float)
    
    # 高血压分期
    df['HYPERTENSION_STAGE'] = 0  # 正常
    df.loc[(df['SYSBP'] >= 130) | (df['DIABP'] >= 80), 'HYPERTENSION_STAGE'] = 1  # 1期高血压
    df.loc[(df['SYSBP'] >= 140) | (df['DIABP'] >= 90), 'HYPERTENSION_STAGE'] = 2  # 2期高血压
    
    # 吸烟风险评分
    df['SMOKING_RISK'] = df['CURSMOKE'] * (1 + df['CIGPDAY'].fillna(0) / 20)
    
    # BMI分类
    df['BMI_CATEGORY'] = 0  # 正常
    df.loc[df['BMI'] < 18.5, 'BMI_CATEGORY'] = 1  # 偏瘦
    df.loc[df['BMI'] >= 25, 'BMI_CATEGORY'] = 2  # 超重
    df.loc[df['BMI'] >= 30, 'BMI_CATEGORY'] = 3  # 肥胖
    
    return df

df = engineer_features(df)
feature_names = df.columns.tolist()
feature_names.remove('CVD')
feature_names.remove('ANYCHD')
feature_names.remove('STROKE')
feature_names.remove('ANGINA')
feature_names.remove('MI_FCHD')
feature_names.remove('HYPERTEN')
feature_names.remove('DEATH')

print('📊 准备特征数据...')
print(f'特征数量: {len(feature_names)}')
print(f'样本数量: {len(df)}')

# 保存特征名称
with open('framingham_multi_disease_features.json', 'w', encoding='utf-8') as f:
    json.dump(feature_names, f, ensure_ascii=False, indent=2)

# 为每种疾病训练模型
for disease, target in targets.items():
    print(f'\n🤖 训练 {disease} 预测模型...')
    print(f'患病率: {df[target].mean()*100:.2f}%')
    
    # 准备数据
    X = df[feature_names]
    y = df[target]
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 预处理
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # 训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f'准确率: {accuracy:.4f}')
    print(f'AUC: {auc:.4f}')
    
    # 保存模型和预处理器
    joblib.dump(model, f'framingham_{disease.lower()}_model.pkl')
    joblib.dump(scaler, f'framingham_{disease.lower()}_scaler.pkl')
    joblib.dump(imputer, f'framingham_{disease.lower()}_imputer.pkl')
    
    print(f'✅ {disease} 模型已保存')

print('\n🎉 所有模型训练完成!')
print('📁 生成的文件:')
for disease in targets.keys():
    print(f'  - framingham_{disease.lower()}_model.pkl')
    print(f'  - framingham_{disease.lower()}_scaler.pkl')
    print(f'  - framingham_{disease.lower()}_imputer.pkl')
print('  - framingham_multi_disease_features.json') 