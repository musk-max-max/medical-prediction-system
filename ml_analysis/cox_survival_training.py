#!/usr/bin/env python3
"""
Cox时间变化生存分析模型训练
使用CoxTimeVaryingFitter处理Framingham纵向数据
"""

import pandas as pd
import numpy as np
import pickle
from lifelines import CoxTimeVaryingFitter
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path='../frmgham_data.csv'):
    """加载并准备纵向数据"""
    print("📊 加载Framingham纵向数据...")
    df = pd.read_csv(file_path)
    
    print(f"原始数据形状: {df.shape}")
    print(f"独特患者数: {df['RANDID'].nunique()}")
    print(f"PERIOD分布:\n{df['PERIOD'].value_counts().sort_index()}")
    
    return df

def prepare_cox_tv_data(df):
    """准备Cox时间变化模型数据"""
    print("🔄 准备Cox时间变化数据格式...")
    
    # 核心特征选择
    features = [
        'SEX', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 
        'CIGPDAY', 'BMI', 'DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE',
        'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP'
    ]
    
    # 创建时间变化数据格式
    tv_data = []
    
    for patient_id in df['RANDID'].unique():
        patient_data = df[df['RANDID'] == patient_id].sort_values('PERIOD')
        
        if len(patient_data) == 0:
            continue
            
        # 获取患者的最终状态
        max_time = patient_data['TIME'].max()
        final_cvd = patient_data['CVD'].iloc[-1]
        
        # 为每个PERIOD创建记录
        for i, (_, row) in enumerate(patient_data.iterrows()):
            # 计算时间区间
            start_time = 0 if i == 0 else patient_data.iloc[i-1]['TIME']
            stop_time = row['TIME']
            
            # 修复时间间隔为0的问题
            if start_time == stop_time:
                stop_time = start_time + 0.5  # 添加最小时间间隔
            
            # 事件只在最后一个记录中发生
            event = final_cvd if i == len(patient_data) - 1 else 0
            
            # 构建记录
            record = {
                'id': patient_id,
                'start': start_time,
                'stop': stop_time,
                'event': event
            }
            
            # 添加特征
            for feature in features:
                if feature in row and pd.notna(row[feature]):
                    record[feature] = row[feature]
                else:
                    # 使用默认值
                    defaults = {
                        'SEX': 1, 'AGE': 50, 'TOTCHOL': 200, 'SYSBP': 120, 'DIABP': 80,
                        'CURSMOKE': 0, 'CIGPDAY': 0, 'BMI': 25, 'DIABETES': 0,
                        'BPMEDS': 0, 'HEARTRTE': 70, 'GLUCOSE': 90,
                        'PREVCHD': 0, 'PREVAP': 0, 'PREVMI': 0, 'PREVSTRK': 0, 'PREVHYP': 0
                    }
                    record[feature] = defaults.get(feature, 0)
            
            tv_data.append(record)
    
    tv_df = pd.DataFrame(tv_data)
    print(f"时间变化数据形状: {tv_df.shape}")
    print(f"事件数量: {tv_df['event'].sum()}")
    
    return tv_df, features

def train_cox_tv_model(tv_df, features):
    """训练Cox时间变化模型"""
    print("🧠 训练Cox时间变化模型...")
    
    # 数据预处理
    print("📋 数据预处理...")
    
    # 填充缺失值
    imputer = SimpleImputer(strategy='median')
    tv_df[features] = imputer.fit_transform(tv_df[features])
    
    # 标准化特征
    scaler = StandardScaler()
    tv_df[features] = scaler.fit_transform(tv_df[features])
    
    # 训练Cox时间变化模型
    print("🔬 训练模型...")
    ctv = CoxTimeVaryingFitter()
    
    try:
        ctv.fit(tv_df, 
                id_col='id', 
                start_col='start', 
                stop_col='stop', 
                event_col='event',
                show_progress=True)
        
        print("✅ Cox时间变化模型训练成功！")
        
        # 显示模型摘要
        print("\n📊 模型摘要:")
        print(ctv.summary.head(10))
        
        # 显示一致性指数 (CoxTimeVaryingFitter用不同的属性名)
        try:
            c_index = ctv.concordance_index_
            print(f"\n🎯 一致性指数 (C-index): {c_index:.4f}")
        except AttributeError:
            print("\n🎯 模型训练成功，C-index将在评估阶段计算")
        
        return ctv, scaler, imputer, tv_df
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return None, None, None, None

def save_models(ctv, scaler, imputer, features):
    """保存训练好的模型"""
    print("💾 保存模型...")
    
    try:
        # 保存Cox模型
        with open('cox_timevarying_model.pkl', 'wb') as f:
            pickle.dump(ctv, f)
        
        # 保存预处理器
        joblib.dump(scaler, 'cox_tv_scaler.pkl')
        joblib.dump(imputer, 'cox_tv_imputer.pkl')
        
        # 保存特征名称
        with open('cox_tv_features.pkl', 'wb') as f:
            pickle.dump(features, f)
        
        print("✅ 模型保存成功！")
        print("📁 保存的文件:")
        print("  - cox_timevarying_model.pkl")
        print("  - cox_tv_scaler.pkl") 
        print("  - cox_tv_imputer.pkl")
        print("  - cox_tv_features.pkl")
        
    except Exception as e:
        print(f"❌ 模型保存失败: {e}")

def evaluate_model(ctv, tv_df):
    """评估模型性能"""
    print("\n📈 模型评估:")
    
    try:
        # 一致性指数 (对于CoxTimeVaryingFitter需要手动计算)
        try:
            c_index = ctv.concordance_index_
        except AttributeError:
            # 如果没有concordance_index_属性，使用None或跳过
            c_index = None
            print("C-index: 将在实际预测中评估")
        
        if c_index is not None:
            print(f"C-index: {c_index:.4f}")
        
        # 风险比分析
        hazard_ratios = ctv.hazard_ratios_
        print("\n🔍 主要风险因子 (前10个):")
        top_hr = hazard_ratios.abs().sort_values(ascending=False).head(10)
        for feature, hr in top_hr.items():
            direction = "↑" if hazard_ratios[feature] > 1 else "↓"
            print(f"  {feature}: HR={hazard_ratios[feature]:.3f} {direction}")
        
        # 保存评估结果
        evaluation = {
            'c_index': float(c_index) if c_index is not None else None,
            'n_events': int(tv_df['event'].sum()),
            'n_observations': len(tv_df),
            'n_patients': tv_df['id'].nunique(),
            'hazard_ratios': hazard_ratios.to_dict()
        }
        
        with open('cox_tv_evaluation.pkl', 'wb') as f:
            pickle.dump(evaluation, f)
        
        print("📊 评估结果已保存: cox_tv_evaluation.pkl")
        
    except Exception as e:
        print(f"❌ 模型评估失败: {e}")

def main():
    """主函数"""
    print("🏥 Cox时间变化生存分析模型训练")
    print("=" * 50)
    
    # 1. 加载数据
    df = load_and_prepare_data()
    
    # 2. 准备时间变化数据
    tv_df, features = prepare_cox_tv_data(df)
    
    if tv_df is None or len(tv_df) == 0:
        print("❌ 数据准备失败")
        return
    
    # 3. 训练模型
    ctv, scaler, imputer, processed_tv_df = train_cox_tv_model(tv_df, features)
    
    if ctv is None:
        print("❌ 模型训练失败")
        return
    
    # 4. 评估模型
    evaluate_model(ctv, processed_tv_df)
    
    # 5. 保存模型
    save_models(ctv, scaler, imputer, features)
    
    print("\n🎉 Cox时间变化模型训练完成！")
    print("🚀 现在可以部署新的生存分析系统了！")

if __name__ == "__main__":
    main() 