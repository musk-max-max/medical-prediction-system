#!/usr/bin/env python3
"""
弗雷明汉纵向心血管疾病风险预测模型
正确处理每个患者的多次体检数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def load_and_analyze_longitudinal_data(file_path):
    """加载和分析纵向数据"""
    print("🔍 加载弗雷明汉纵向心脏研究数据...")
    df = pd.read_csv(file_path)
    
    print(f"数据基本信息:")
    print(f"  总记录数: {len(df)}")
    print(f"  独特患者数: {df['RANDID'].nunique()}")
    print(f"  CVD患病率: {(df['CVD']==1).mean():.3f}")
    
    # 分析体检次数分布
    visit_counts = df.groupby('RANDID').size()
    print(f"  体检次数分布:")
    for visits, count in visit_counts.value_counts().sort_index().items():
        print(f"    {visits}次体检: {count}人")
    
    return df

def create_baseline_model(df):
    """创建基线模型：使用第一次体检数据预测CVD风险"""
    print("\n📊 方法1: 基线预测模型 (使用第一次体检数据)")
    
    # 获取第一次体检数据
    baseline_data = df[df['PERIOD'] == 1].copy()
    
    # 选择重要特征
    risk_factors = [
        'SEX', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 
        'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 
        'BPMEDS', 'HEARTRTE', 'GLUCOSE', 'PREVCHD',
        'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP'
    ]
    
    available_features = [col for col in risk_factors if col in baseline_data.columns]
    X = baseline_data[available_features].copy()
    y = baseline_data['CVD']
    
    # 特征工程
    if 'SYSBP' in X.columns and 'DIABP' in X.columns:
        X['PULSE_PRESSURE'] = X['SYSBP'] - X['DIABP']
    
    if 'TOTCHOL' in X.columns and 'AGE' in X.columns:
        X['CHOL_AGE_RATIO'] = X['TOTCHOL'] / (X['AGE'] + 1)
    
    print(f"特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
    print(f"CVD患病率: {y.mean():.3f}")
    
    return X, y, '基线预测模型'

def create_longitudinal_model(df):
    """创建纵向模型：利用多次体检的变化趋势"""
    print("\n📊 方法2: 纵向变化模型 (利用体检变化趋势)")
    
    longitudinal_features = []
    
    # 获取有多次体检的患者
    multi_visit_patients = df.groupby('RANDID').size()
    multi_visit_patients = multi_visit_patients[multi_visit_patients >= 2].index
    
    for patient_id in multi_visit_patients:
        patient_data = df[df['RANDID'] == patient_id].sort_values('PERIOD')
        
        if len(patient_data) < 2:
            continue
            
        # 基线数据（第一次体检）
        baseline = patient_data.iloc[0]
        features = {}
        
        # 基线特征
        baseline_vars = ['SEX', 'AGE', 'DIABETES', 'PREVCHD', 'PREVMI', 'PREVSTRK']
        for var in baseline_vars:
            if var in baseline and pd.notna(baseline[var]):
                features[f'baseline_{var}'] = baseline[var]
        
        # 计算变化趋势
        change_vars = ['TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'HEARTRTE', 'GLUCOSE']
        
        for var in change_vars:
            if var in patient_data.columns:
                values = patient_data[var].dropna()
                if len(values) >= 2:
                    # 线性趋势
                    slope = np.polyfit(range(len(values)), values, 1)[0]
                    features[f'{var}_slope'] = slope
                    
                    # 变化率
                    first_val = values.iloc[0]
                    last_val = values.iloc[-1]
                    if first_val != 0:
                        features[f'{var}_change_rate'] = (last_val - first_val) / first_val
        
        # 吸烟状态变化
        if 'CURSMOKE' in patient_data.columns:
            smoke_values = patient_data['CURSMOKE'].dropna()
            if len(smoke_values) >= 2:
                features['smoking_change'] = smoke_values.iloc[-1] - smoke_values.iloc[0]
        
        # 目标变量
        features['CVD'] = baseline['CVD']
        longitudinal_features.append(features)
    
    # 转换为DataFrame
    long_df = pd.DataFrame(longitudinal_features)
    
    if len(long_df) == 0:
        return None, None, None
    
    # 分离特征和目标
    feature_cols = [col for col in long_df.columns if col != 'CVD']
    X = long_df[feature_cols]
    y = long_df['CVD']
    
    print(f"特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
    print(f"CVD患病率: {y.mean():.3f}")
    
    return X, y, '纵向变化模型'

def train_and_evaluate_models(X, y, model_name):
    """训练和评估模型"""
    if X is None or y is None:
        print(f"❌ {model_name}: 数据准备失败")
        return {}
    
    print(f"\n🤖 训练 {model_name}...")
    
    # 处理缺失值
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 定义模型
    models = {
        'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[f"{model_name}_{name}"] = {
            'model': model,
            'scaler': scaler,
            'imputer': imputer,
            'accuracy': accuracy,
            'auc': auc,
            'model_type': model_name
        }
        
        print(f"  {name}: 准确率={accuracy:.4f}, AUC={auc:.4f}")
    
    return results

def compare_and_save_best_model(all_results):
    """比较模型性能并保存最佳模型"""
    if not all_results:
        print("❌ 没有可比较的模型")
        return
    
    print("\n📊 模型性能比较:")
    print("=" * 60)
    
    # 创建比较表
    comparison_data = []
    for name, result in all_results.items():
        comparison_data.append({
            'Model': name,
            'Model_Type': result['model_type'],
            'Accuracy': result['accuracy'],
            'AUC': result['auc']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('AUC', ascending=False)
    print(comparison_df.round(4).to_string(index=False))
    
    # 找到最佳模型
    best_model_name = comparison_df.iloc[0]['Model']
    best_result = all_results[best_model_name]
    
    print(f"\n🏆 最佳模型: {best_model_name}")
    print(f"🎯 AUC分数: {best_result['auc']:.4f}")
    print(f"📈 准确率: {best_result['accuracy']:.4f}")
    
    # 保存最佳模型
    joblib.dump(best_result['model'], 'best_longitudinal_model.pkl')
    joblib.dump(best_result['scaler'], 'longitudinal_scaler.pkl')
    joblib.dump(best_result['imputer'], 'longitudinal_imputer.pkl')
    
    # 保存模型信息
    model_info = {
        'model_name': best_model_name,
        'model_type': best_result['model_type'],
        'auc_score': best_result['auc'],
        'accuracy': best_result['accuracy']
    }
    
    import json
    with open('longitudinal_model_info.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    # 绘制性能比较图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.barh(range(len(comparison_df)), comparison_df['AUC'])
    plt.yticks(range(len(comparison_df)), comparison_df['Model'])
    plt.xlabel('AUC Score')
    plt.title('模型AUC性能比较')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    model_types = comparison_df.groupby('Model_Type')['AUC'].max()
    plt.bar(range(len(model_types)), model_types.values)
    plt.xticks(range(len(model_types)), model_types.index, rotation=45)
    plt.ylabel('Best AUC Score')
    plt.title('不同建模方法的最佳AUC')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('longitudinal_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 性能比较图已保存: longitudinal_comparison.png")
    
    return best_model_name, best_result

def main():
    """主函数"""
    print("🏥 弗雷明汉纵向心血管疾病风险预测模型")
    print("=" * 60)
    
    # 加载数据
    df = load_and_analyze_longitudinal_data('../frmgham_data.csv')
    
    # 创建不同建模方法
    all_results = {}
    
    # 方法1: 基线模型
    X1, y1, name1 = create_baseline_model(df)
    results1 = train_and_evaluate_models(X1, y1, name1)
    all_results.update(results1)
    
    # 方法2: 纵向模型
    X2, y2, name2 = create_longitudinal_model(df)
    results2 = train_and_evaluate_models(X2, y2, name2)
    all_results.update(results2)
    
    # 比较和保存最佳模型
    best_name, best_model = compare_and_save_best_model(all_results)
    
    print("\n🎉 纵向建模完成!")
    print("📁 生成的文件:")
    print("  - best_longitudinal_model.pkl (最佳模型)")
    print("  - longitudinal_scaler.pkl (标准化器)")
    print("  - longitudinal_imputer.pkl (缺失值处理器)")
    print("  - longitudinal_model_info.json (模型信息)")
    print("  - longitudinal_comparison.png (性能比较图)")
    
    print("\n💡 数据理解总结:")
    print("1. 数据包含4434名患者，最多3次体检记录")
    print("2. CVD=1表示在整个研究期间曾患心血管疾病")
    print("3. PREVCHD=1表示在当前体检时已患冠心病")
    print("4. TIMECVD表示从第一次体检到CVD诊断的时间(小时)")
    print("5. 基线模型使用第一次体检预测未来CVD风险")
    print("6. 纵向模型利用多次体检的变化趋势")

if __name__ == "__main__":
    main() 