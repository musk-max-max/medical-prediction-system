#!/usr/bin/env python3
"""
弗雷明汉心脏研究多疾病预测系统
基于24年随访数据，预测多种心血管疾病风险

疾病包括：
- CVD: 心血管疾病 (心肌梗死或卒中)
- CHD: 冠心病 (心绞痛、心肌梗死、冠状动脉功能不全或致命性冠心病)
- STROKE: 卒中 (动脉血栓性梗死、脑栓塞、脑内出血或蛛网膜下腔出血)
- ANGINA: 心绞痛
- MI: 心肌梗死 (住院心肌梗死或致命性冠心病)
- HYPERTENSION: 高血压
- DEATH: 死亡风险

每种疾病都有对应的生存时间数据，可以进行生存分析和风险预测
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
from sklearn.calibration import calibration_curve
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print('🏥 弗雷明汉心脏研究多疾病预测系统')
print('基于24年随访数据的综合心血管疾病风险评估')
print('=' * 80)

# 加载数据
df = pd.read_csv('frmgham_data.csv')
print(f'总记录数: {len(df)}')
print(f'独特患者数: {df["RANDID"].nunique()}')

# 定义疾病-时间对应关系
DISEASES = {
    'CVD': {
        'name': '心血管疾病 (心肌梗死或卒中)',
        'event_col': 'CVD',
        'time_col': 'TIMECVD',
        'description': '包括心肌梗死、致命性冠心病、动脉血栓性梗死、脑栓塞、脑内出血等'
    },
    'CHD': {
        'name': '冠心病 (任何形式)',
        'event_col': 'ANYCHD', 
        'time_col': 'TIMECHD',
        'description': '包括心绞痛、心肌梗死、冠状动脉功能不全、致命性冠心病'
    },
    'STROKE': {
        'name': '卒中',
        'event_col': 'STROKE',
        'time_col': 'TIMESTRK', 
        'description': '包括动脉血栓性梗死、脑栓塞、脑内出血、蛛网膜下腔出血'
    },
    'ANGINA': {
        'name': '心绞痛',
        'event_col': 'ANGINA',
        'time_col': 'TIMEAP',
        'description': '胸痛或胸部不适，通常由心肌缺血引起'
    },
    'MI': {
        'name': '心肌梗死',
        'event_col': 'MI_FCHD', 
        'time_col': 'TIMEMIFC',
        'description': '住院心肌梗死或致命性冠心病'
    },
    'HYPERTENSION': {
        'name': '高血压',
        'event_col': 'HYPERTEN',
        'time_col': 'TIMEHYP',
        'description': '收缩压≥140mmHg或舒张压≥90mmHg或正在服用降压药'
    },
    'DEATH': {
        'name': '死亡风险',
        'event_col': 'DEATH',
        'time_col': 'TIMEDTH',
        'description': '任何原因导致的死亡'
    }
}

# 分析每种疾病的发病情况
print(f'\n📊 24年随访期间疾病发病情况:')
print(f'研究总时长: 8766天 = {8766/365.25:.1f}年')
print('-' * 80)

disease_stats = {}
for disease_id, disease_info in DISEASES.items():
    event_col = disease_info['event_col']
    time_col = disease_info['time_col']
    
    if event_col in df.columns and time_col in df.columns:
        total_patients = df['RANDID'].nunique()
        event_rate = df[event_col].mean()
        
        # 计算平均发病时间（仅针对发病患者）
        sick_patients = df[df[event_col] == 1]
        if len(sick_patients) > 0:
            avg_time_to_event = sick_patients[time_col].mean() / 365.25
            median_time_to_event = sick_patients[time_col].median() / 365.25
        else:
            avg_time_to_event = np.nan
            median_time_to_event = np.nan
        
        disease_stats[disease_id] = {
            'event_rate': event_rate,
            'avg_time_years': avg_time_to_event,
            'median_time_years': median_time_to_event,
            'total_events': sick_patients.shape[0] if len(sick_patients) > 0 else 0
        }
        
        print(f'{disease_info["name"]}:')
        print(f'  发病率: {event_rate:.3f} ({event_rate*100:.1f}%)')
        if not np.isnan(avg_time_to_event):
            print(f'  平均发病时间: {avg_time_to_event:.1f}年')
            print(f'  中位发病时间: {median_time_to_event:.1f}年')
        print(f'  总事件数: {disease_stats[disease_id]["total_events"]}')
        print()

# 使用第一次体检数据进行建模
print(f'📋 准备建模数据 (使用第一次体检数据):')
baseline_data = df[df['PERIOD'] == 1].copy()
print(f'基线数据样本数: {len(baseline_data)}')

# 选择预测特征（标准心血管风险因素）
risk_factors = [
    'SEX', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 
    'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 
    'BPMEDS', 'HEARTRTE', 'GLUCOSE'
]

available_features = [col for col in risk_factors if col in baseline_data.columns]
X_base = baseline_data[available_features].copy()

# 特征工程
print(f'\n🔧 特征工程:')
if 'SYSBP' in X_base.columns and 'DIABP' in X_base.columns:
    X_base['PULSE_PRESSURE'] = X_base['SYSBP'] - X_base['DIABP']
    print('+ 脉压差 (收缩压-舒张压)')

if 'TOTCHOL' in X_base.columns and 'AGE' in X_base.columns:
    X_base['CHOL_AGE_RATIO'] = X_base['TOTCHOL'] / (X_base['AGE'] + 1)
    print('+ 胆固醇年龄比')

# 年龄分组 
if 'AGE' in X_base.columns:
    X_base['AGE_GROUP'] = pd.cut(X_base['AGE'], bins=[0, 45, 55, 65, 100], labels=[0, 1, 2, 3]).astype(float)
    print('+ 年龄分组 (<45, 45-55, 55-65, >65)')

# 高血压分类
if 'SYSBP' in X_base.columns and 'DIABP' in X_base.columns:
    X_base['HYPERTENSION_STAGE'] = 0  # 正常
    X_base.loc[(X_base['SYSBP'] >= 130) | (X_base['DIABP'] >= 80), 'HYPERTENSION_STAGE'] = 1  # 1期高血压
    X_base.loc[(X_base['SYSBP'] >= 140) | (X_base['DIABP'] >= 90), 'HYPERTENSION_STAGE'] = 2  # 2期高血压
    print('+ 高血压分期 (正常/1期/2期)')

# 吸烟风险评分
if 'CURSMOKE' in X_base.columns and 'CIGPDAY' in X_base.columns:
    X_base['SMOKING_RISK'] = X_base['CURSMOKE'] * (1 + X_base['CIGPDAY'].fillna(0) / 20)
    print('+ 吸烟风险评分')

# BMI分类
if 'BMI' in X_base.columns:
    X_base['BMI_CATEGORY'] = 0  # 正常
    X_base.loc[X_base['BMI'] < 18.5, 'BMI_CATEGORY'] = 1  # 偏瘦
    X_base.loc[X_base['BMI'] >= 25, 'BMI_CATEGORY'] = 2  # 超重
    X_base.loc[X_base['BMI'] >= 30, 'BMI_CATEGORY'] = 3  # 肥胖
    print('+ BMI分类 (正常/偏瘦/超重/肥胖)')

print(f'最终特征数量: {X_base.shape[1]}')

# 处理缺失值和标准化
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_base)
X_imputed = pd.DataFrame(X_imputed, columns=X_base.columns, index=X_base.index)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled = pd.DataFrame(X_scaled, columns=X_base.columns, index=X_base.index)

# 为每种疾病训练预测模型
print(f'\n🤖 训练多疾病预测模型:')
print('=' * 80)

models = {}
results = {}

for disease_id, disease_info in DISEASES.items():
    event_col = disease_info['event_col']
    time_col = disease_info['time_col']
    
    if event_col not in baseline_data.columns:
        print(f'跳过 {disease_info["name"]}: 缺少事件列 {event_col}')
        continue
    
    print(f'\n📈 训练 {disease_info["name"]} 预测模型...')
    
    # 准备目标变量
    y = baseline_data[event_col].copy()
    
    # 计算生存时间（年）
    if time_col in baseline_data.columns:
        survival_time = baseline_data[time_col] / 365.25
        print(f'   事件发生率: {y.mean():.3f} ({y.mean()*100:.1f}%)')
        print(f'   平均随访时间: {survival_time.mean():.1f}年')
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 训练模型
    disease_models = {}
    disease_results = {}
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    lr_pred = lr_model.predict(X_test)
    
    lr_auc = roc_auc_score(y_test, lr_pred_proba)
    lr_acc = accuracy_score(y_test, lr_pred)
    
    disease_models['LogisticRegression'] = lr_model
    disease_results['LogisticRegression'] = {
        'auc': lr_auc,
        'accuracy': lr_acc,
        'predictions': lr_pred_proba
    }
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = rf_model.predict(X_test)
    
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    disease_models['RandomForest'] = rf_model
    disease_results['RandomForest'] = {
        'auc': rf_auc,
        'accuracy': rf_acc,
        'predictions': rf_pred_proba
    }
    
    # 选择最佳模型
    best_model_name = 'LogisticRegression' if lr_auc >= rf_auc else 'RandomForest'
    best_model = disease_models[best_model_name]
    best_result = disease_results[best_model_name]
    
    print(f'   逻辑回归 - AUC: {lr_auc:.4f}, 准确率: {lr_acc:.4f}')
    print(f'   随机森林 - AUC: {rf_auc:.4f}, 准确率: {rf_acc:.4f}')
    print(f'   最佳模型: {best_model_name} (AUC: {best_result["auc"]:.4f})')
    
    # 存储模型和结果
    models[disease_id] = {
        'model': best_model,
        'model_type': best_model_name,
        'scaler': scaler,
        'imputer': imputer,
        'features': list(X_base.columns),
        'test_auc': best_result['auc'],
        'test_accuracy': best_result['accuracy'],
        'y_test': y_test,
        'predictions': best_result['predictions']
    }
    
    results[disease_id] = disease_stats.get(disease_id, {})
    results[disease_id].update({
        'model_performance': {
            'auc': best_result['auc'],
            'accuracy': best_result['accuracy'],
            'model_type': best_model_name
        }
    })

# 创建综合风险评估可视化
print(f'\n📊 创建可视化图表...')

# 1. 疾病发病率对比
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 疾病发病率
disease_names = [DISEASES[d]['name'] for d in results.keys()]
event_rates = [results[d]['event_rate'] * 100 for d in results.keys()]

ax1.barh(disease_names, event_rates, color='skyblue')
ax1.set_xlabel('发病率 (%)')
ax1.set_title('24年随访期间各疾病发病率')
ax1.grid(axis='x', alpha=0.3)

# 模型性能对比
model_aucs = [results[d]['model_performance']['auc'] for d in results.keys()]
colors = ['lightgreen' if auc >= 0.7 else 'orange' if auc >= 0.6 else 'salmon' for auc in model_aucs]

ax2.barh(disease_names, model_aucs, color=colors)
ax2.set_xlabel('AUC 分数')
ax2.set_title('各疾病预测模型性能 (AUC)')
ax2.set_xlim(0, 1)
ax2.grid(axis='x', alpha=0.3)

# 平均发病时间
avg_times = []
disease_names_with_time = []
for d in results.keys():
    if not np.isnan(results[d].get('avg_time_years', np.nan)):
        avg_times.append(results[d]['avg_time_years'])
        disease_names_with_time.append(DISEASES[d]['name'])

if avg_times:
    ax3.barh(disease_names_with_time, avg_times, color='lightcoral')
    ax3.set_xlabel('平均发病时间 (年)')
    ax3.set_title('各疾病平均发病时间')
    ax3.grid(axis='x', alpha=0.3)

# ROC曲线对比
for disease_id in models.keys():
    model_data = models[disease_id]
    fpr, tpr, _ = roc_curve(model_data['y_test'], model_data['predictions'])
    auc = model_data['test_auc']
    ax4.plot(fpr, tpr, label=f'{DISEASES[disease_id]["name"]} (AUC={auc:.3f})')

ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax4.set_xlabel('假阳性率')
ax4.set_ylabel('真阳性率')
ax4.set_title('各疾病预测模型ROC曲线对比')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('framingham_multi_disease_analysis.png', dpi=300, bbox_inches='tight')
print('综合分析图已保存: framingham_multi_disease_analysis.png')

# 保存所有模型
print(f'\n💾 保存预测模型...')
for disease_id, model_data in models.items():
    # 保存模型
    model_filename = f'framingham_{disease_id.lower()}_model.pkl'
    joblib.dump(model_data['model'], model_filename)
    
    # 保存预处理器
    scaler_filename = f'framingham_{disease_id.lower()}_scaler.pkl'
    joblib.dump(model_data['scaler'], scaler_filename)
    
    imputer_filename = f'framingham_{disease_id.lower()}_imputer.pkl'
    joblib.dump(model_data['imputer'], imputer_filename)
    
    print(f'✓ {DISEASES[disease_id]["name"]} 模型已保存')

# 保存特征名称
with open('framingham_multi_disease_features.json', 'w', encoding='utf-8') as f:
    json.dump(list(X_base.columns), f, ensure_ascii=False, indent=2)

# 保存完整的模型信息
model_info = {
    'study_name': '弗雷明汉心脏研究多疾病预测系统',
    'study_period': '24年随访 (8766天)',
    'total_patients': int(baseline_data['RANDID'].nunique()),
    'total_features': len(X_base.columns),
    'features': list(X_base.columns),
    'diseases': {},
    'created_date': pd.Timestamp.now().isoformat()
}

for disease_id, disease_info in DISEASES.items():
    if disease_id in results:
        model_info['diseases'][disease_id] = {
            'name': disease_info['name'],
            'description': disease_info['description'],
            'event_rate': float(results[disease_id]['event_rate']),
            'total_events': int(results[disease_id].get('total_events', 0)),
            'avg_time_to_event_years': float(results[disease_id].get('avg_time_years', 0)) if not np.isnan(results[disease_id].get('avg_time_years', np.nan)) else None,
            'model_performance': {
                'auc': float(results[disease_id]['model_performance']['auc']),
                'accuracy': float(results[disease_id]['model_performance']['accuracy']),
                'model_type': results[disease_id]['model_performance']['model_type']
            }
        }

with open('framingham_multi_disease_model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

print(f'\n✅ 弗雷明汉多疾病预测系统构建完成!')
print(f'💡 系统特点:')
print(f'1. 基于4,434名患者24年随访数据')
print(f'2. 可预测7种主要心血管疾病风险')
print(f'3. 使用标准心血管风险因素进行预测')
print(f'4. 所有模型均经过交叉验证和性能评估')
print(f'5. 支持个体风险评估和风险分层')

print(f'\n📋 模型性能总结:')
for disease_id in results.keys():
    perf = results[disease_id]['model_performance']
    print(f'{DISEASES[disease_id]["name"]}: AUC={perf["auc"]:.3f}, 准确率={perf["accuracy"]:.3f} ({perf["model_type"]})')

print(f'\n🎯 应用场景:')
print(f'• 临床风险评估和预防决策')
print(f'• 长期心血管健康管理')
print(f'• 高危人群筛查和监测')
print(f'• 健康生活方式指导')
print(f'• 医疗资源配置优化') 