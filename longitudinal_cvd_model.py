#!/usr/bin/env python3
"""
弗雷明汉纵向心血管疾病风险预测模型
专门处理每个患者的多次体检数据，进行时间序列分析和生存分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和随机种子
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

class LongitudinalFraminghamPredictor:
    """弗雷明汉纵向心血管疾病风险预测器"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        
    def load_and_analyze_data(self, file_path):
        """加载和分析纵向数据"""
        print("🔍 加载弗雷明汉纵向心脏研究数据...")
        df = pd.read_csv(file_path)
        
        print(f"数据基本信息:")
        print(f"  总记录数: {len(df)}")
        print(f"  独特患者数: {df['RANDID'].nunique()}")
        print(f"  体检次数分布:")
        visit_counts = df.groupby('RANDID').size()
        for visits, count in visit_counts.value_counts().sort_index().items():
            print(f"    {visits}次体检: {count}人")
        
        print(f"  CVD总患病率: {(df['CVD']==1).mean():.3f}")
        
        return df
    
    def create_modeling_approaches(self, df):
        """创建不同的建模方法"""
        approaches = {}
        
        # 方法1: 使用第一次体检数据预测未来CVD风险
        print("\n📊 方法1: 基线预测模型 (使用第一次体检数据)")
        baseline_data = df[df['PERIOD'] == 1].copy()
        baseline_data['future_cvd'] = baseline_data['CVD']  # CVD表示在研究期间是否发生
        approaches['baseline'] = self.prepare_baseline_features(baseline_data)
        
        # 方法2: 生存分析方法 - 预测CVD发生时间
        print("\n📊 方法2: 生存分析模型 (预测CVD发生时间)")
        approaches['survival'] = self.prepare_survival_features(df)
        
        # 方法3: 纵向变化模型 - 利用多次体检的变化趋势
        print("\n📊 方法3: 纵向变化模型 (利用体检变化趋势)")
        approaches['longitudinal'] = self.prepare_longitudinal_features(df)
        
        return approaches
    
    def prepare_baseline_features(self, baseline_data):
        """准备基线特征(第一次体检)"""
        # 选择重要的心血管风险因素
        risk_factors = [
            'SEX', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 
            'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 
            'BPMEDS', 'HEARTRTE', 'GLUCOSE', 'PREVCHD',
            'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP'
        ]
        
        # 过滤存在的列
        available_features = [col for col in risk_factors if col in baseline_data.columns]
        
        # 特征工程
        df_features = baseline_data[available_features].copy()
        
        # 创建衍生特征
        if 'SYSBP' in df_features.columns and 'DIABP' in df_features.columns:
            df_features['PULSE_PRESSURE'] = df_features['SYSBP'] - df_features['DIABP']
        
        if 'TOTCHOL' in df_features.columns and 'AGE' in df_features.columns:
            df_features['CHOL_AGE_RATIO'] = df_features['TOTCHOL'] / (df_features['AGE'] + 1)
        
        if 'BMI' in df_features.columns:
            df_features['BMI_CATEGORY'] = pd.cut(
                df_features['BMI'], 
                bins=[0, 18.5, 25, 30, float('inf')], 
                labels=[0, 1, 2, 3]
            ).astype(float)
        
        if 'CURSMOKE' in df_features.columns and 'CIGPDAY' in df_features.columns:
            df_features['SMOKING_RISK'] = (
                df_features['CURSMOKE'] * (1 + df_features['CIGPDAY'].fillna(0) / 20)
            )
        
        # 年龄相关风险评分
        if 'AGE' in df_features.columns and 'SEX' in df_features.columns:
            # 男性和女性的年龄风险不同
            df_features['AGE_SEX_RISK'] = df_features['AGE'] * (1.2 if df_features['SEX'].iloc[0] == 1 else 1.0)
        
        # 目标变量
        y = baseline_data['future_cvd']
        
        return {
            'X': df_features,
            'y': y,
            'description': '基线预测模型 - 使用第一次体检数据预测未来CVD风险'
        }
    
    def prepare_survival_features(self, df):
        """准备生存分析特征"""
        # 获取每个患者的第一次体检数据
        first_visits = df[df['PERIOD'] == 1].copy()
        
        # 创建生存分析目标变量
        # event: 是否发生CVD (0=未发生, 1=发生)
        # time_to_event: 观察时间或事件发生时间
        first_visits['event'] = first_visits['CVD']
        
        # 对于发生CVD的患者，使用TIMECVD作为事件时间
        # 对于未发生CVD的患者，使用最后一次体检时间作为删失时间
        first_visits['time_to_event'] = first_visits['TIMECVD']
        
        # 对于未发生CVD的患者，计算最后一次观察时间
        max_time_per_patient = df.groupby('RANDID')['TIME'].max()
        no_cvd_mask = first_visits['CVD'] == 0
        first_visits.loc[no_cvd_mask, 'time_to_event'] = first_visits.loc[no_cvd_mask, 'RANDID'].map(max_time_per_patient)
        
        # 转换为年为单位
        first_visits['time_to_event_years'] = first_visits['time_to_event'] / (24 * 365.25)
        
        # 特征选择（类似基线模型）
        risk_factors = [
            'SEX', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 
            'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 
            'BPMEDS', 'HEARTRTE', 'GLUCOSE'
        ]
        
        available_features = [col for col in risk_factors if col in first_visits.columns]
        X = first_visits[available_features].copy()
        
        # 添加特征工程
        if 'SYSBP' in X.columns and 'DIABP' in X.columns:
            X['PULSE_PRESSURE'] = X['SYSBP'] - X['DIABP']
        
        return {
            'X': X,
            'y': first_visits['event'],
            'time': first_visits['time_to_event_years'],
            'description': '生存分析模型 - 预测CVD发生时间和概率'
        }
    
    def prepare_longitudinal_features(self, df):
        """准备纵向变化特征"""
        # 计算每个患者的变化趋势
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
            
            # 计算关键指标的变化
            features = {'RANDID': patient_id}
            
            # 基线特征
            baseline_vars = ['SEX', 'AGE', 'DIABETES', 'PREVCHD', 'PREVMI', 'PREVSTRK']
            for var in baseline_vars:
                if var in baseline:
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
                        
                        # 变异性（标准差）
                        features[f'{var}_variability'] = values.std()
            
            # 吸烟状态变化
            if 'CURSMOKE' in patient_data.columns:
                smoke_values = patient_data['CURSMOKE'].dropna()
                if len(smoke_values) >= 2:
                    features['smoking_change'] = smoke_values.iloc[-1] - smoke_values.iloc[0]
            
            # 用药状态变化
            if 'BPMEDS' in patient_data.columns:
                med_values = patient_data['BPMEDS'].dropna()
                if len(med_values) >= 2:
                    features['medication_change'] = med_values.iloc[-1] - med_values.iloc[0]
            
            # 目标变量
            features['CVD'] = baseline['CVD']
            
            longitudinal_features.append(features)
        
        # 转换为DataFrame
        long_df = pd.DataFrame(longitudinal_features)
        
        if len(long_df) == 0:
            return None
        
        # 分离特征和目标
        feature_cols = [col for col in long_df.columns if col not in ['RANDID', 'CVD']]
        X = long_df[feature_cols]
        y = long_df['CVD']
        
        return {
            'X': X,
            'y': y,
            'description': '纵向变化模型 - 利用多次体检的变化趋势预测CVD风险'
        }
    
    def train_and_evaluate_approach(self, approach_name, approach_data):
        """训练和评估特定建模方法"""
        if approach_data is None:
            print(f"❌ {approach_name}: 数据准备失败")
            return None
            
        print(f"\n🤖 训练 {approach_name}: {approach_data['description']}")
        
        X = approach_data['X']
        y = approach_data['y']
        
        print(f"特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        print(f"CVD患病率: {y.mean():.3f}")
        
        # 处理缺失值
        X_imputed = self.imputer.fit_transform(X)
        X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X_imputed)
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
            print(f"  训练 {name}...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 评估指标
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[f"{approach_name}_{name}"] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'approach': approach_name,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"    准确率: {accuracy:.4f}, AUC: {auc:.4f}")
        
        return results
    
    def compare_approaches(self, all_results):
        """比较不同建模方法的性能"""
        print("\n📊 建模方法性能比较:")
        print("=" * 60)
        
        comparison_data = []
        for name, result in all_results.items():
            comparison_data.append({
                'Model': name,
                'Approach': result['approach'],
                'Accuracy': result['accuracy'],
                'AUC': result['auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC', ascending=False)
        
        print(comparison_df.round(4))
        
        # 绘制比较图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC比较
        ax1.barh(range(len(comparison_df)), comparison_df['AUC'])
        ax1.set_yticks(range(len(comparison_df)))
        ax1.set_yticklabels(comparison_df['Model'])
        ax1.set_xlabel('AUC Score')
        ax1.set_title('模型AUC性能比较')
        ax1.grid(True, alpha=0.3)
        
        # 按方法分组的AUC比较
        approach_auc = comparison_df.groupby('Approach')['AUC'].max()
        ax2.bar(range(len(approach_auc)), approach_auc.values)
        ax2.set_xticks(range(len(approach_auc)))
        ax2.set_xticklabels(approach_auc.index, rotation=45)
        ax2.set_ylabel('Best AUC Score')
        ax2.set_title('不同建模方法的最佳AUC')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('longitudinal_model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n📊 性能比较图已保存: longitudinal_model_comparison.png")
        
        return comparison_df
    
    def save_best_longitudinal_model(self, all_results):
        """保存最佳纵向模型"""
        best_result = max(all_results.items(), key=lambda x: x[1]['auc'])
        best_name, best_model_data = best_result
        
        print(f"\n💾 保存最佳纵向模型: {best_name}")
        print(f"AUC: {best_model_data['auc']:.4f}")
        
        # 保存模型
        joblib.dump(best_model_data['model'], 'best_longitudinal_cvd_model.pkl')
        joblib.dump(self.scaler, 'longitudinal_scaler.pkl')
        joblib.dump(self.imputer, 'longitudinal_imputer.pkl')
        
        # 保存模型信息
        model_info = {
            'model_name': best_name,
            'approach': best_model_data['approach'],
            'auc_score': best_model_data['auc'],
            'accuracy': best_model_data['accuracy']
        }
        
        import json
        with open('longitudinal_model_info.json', 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        return best_name, best_model_data

def main():
    """主函数"""
    print("🏥 弗雷明汉纵向心血管疾病风险预测模型")
    print("=" * 60)
    
    # 创建预测器
    predictor = LongitudinalFraminghamPredictor()
    
    # 加载和分析数据
    df = predictor.load_and_analyze_data('../frmgham_data.csv')
    
    # 创建不同的建模方法
    approaches = predictor.create_modeling_approaches(df)
    
    # 训练和评估所有方法
    all_results = {}
    
    for approach_name, approach_data in approaches.items():
        results = predictor.train_and_evaluate_approach(approach_name, approach_data)
        if results:
            all_results.update(results)
    
    # 比较不同方法
    if all_results:
        comparison_df = predictor.compare_approaches(all_results)
        
        # 保存最佳模型
        best_name, best_model = predictor.save_best_longitudinal_model(all_results)
        
        print("\n🎉 纵向建模完成!")
        print("📁 生成的文件:")
        print("  - best_longitudinal_cvd_model.pkl (最佳纵向模型)")
        print("  - longitudinal_scaler.pkl (标准化器)")
        print("  - longitudinal_imputer.pkl (缺失值处理器)")
        print("  - longitudinal_model_info.json (模型信息)")
        print("  - longitudinal_model_comparison.png (性能比较图)")
        
        print(f"\n🏆 最佳模型: {best_name}")
        print(f"🎯 AUC分数: {best_model['auc']:.4f}")
    else:
        print("❌ 所有建模方法都失败了")

if __name__ == "__main__":
    main() 