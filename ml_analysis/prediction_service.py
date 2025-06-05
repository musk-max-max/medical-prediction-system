#!/usr/bin/env python3
"""
弗雷明汉多疾病预测服务
加载训练好的模型，对输入的健康数据进行疾病风险预测
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Dict, List, Any
import sys

class FraminghamPredictor:
    """弗雷明汉多疾病预测器"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        self.feature_names = []
        self.diseases = [
            'CVD', 'CHD', 'STROKE', 'ANGINA', 
            'MI', 'HYPERTENSION', 'DEATH'
        ]
        
    def load_models(self):
        """加载所有疾病的模型"""
        # 只在测试模式下打印加载信息
        if len(sys.argv) == 1:
            print("🔍 加载预测模型...")
        
        # 加载特征名称
        with open('framingham_multi_disease_features.json', 'r', encoding='utf-8') as f:
            self.feature_names = json.load(f)
        
        # 加载每个疾病的模型
        for disease in self.diseases:
            model_path = f'framingham_{disease.lower()}_model.pkl'
            scaler_path = f'framingham_{disease.lower()}_scaler.pkl'
            imputer_path = f'framingham_{disease.lower()}_imputer.pkl'
            
            if os.path.exists(model_path):
                self.models[disease] = joblib.load(model_path)
                self.scalers[disease] = joblib.load(scaler_path)
                self.imputers[disease] = joblib.load(imputer_path)
                if len(sys.argv) == 1:
                    print(f"✅ 已加载 {disease} 模型")
            else:
                if len(sys.argv) == 1:
                    print(f"⚠️ 未找到 {disease} 模型")
    
    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """预处理输入数据"""
        # 将字段名转换为大写
        uppercase_data = {}
        for key, value in data.items():
            uppercase_data[key.upper()] = value
        
        # 创建特征DataFrame
        df = pd.DataFrame([uppercase_data])
        
        # 特征工程
        df['PULSE_PRESSURE'] = df['SYSBP'] - df['DIABP']
        df['CHOL_AGE_RATIO'] = df['TOTCHOL'] / (df['AGE'] + 1)
        
        # 年龄分组
        df['AGE_GROUP'] = pd.cut(
            df['AGE'], 
            bins=[0, 45, 55, 65, 100], 
            labels=[0, 1, 2, 3]
        ).astype(float)
        
        # 高血压分期
        df['HYPERTENSION_STAGE'] = 0
        df.loc[(df['SYSBP'] >= 130) | (df['DIABP'] >= 80), 'HYPERTENSION_STAGE'] = 1
        df.loc[(df['SYSBP'] >= 140) | (df['DIABP'] >= 90), 'HYPERTENSION_STAGE'] = 2
        
        # 吸烟风险评分
        df['SMOKING_RISK'] = df['CURSMOKE'] * (1 + df['CIGPDAY'].fillna(0) / 20)
        
        # BMI分类
        df['BMI_CATEGORY'] = 0
        df.loc[df['BMI'] < 18.5, 'BMI_CATEGORY'] = 1
        df.loc[df['BMI'] >= 25, 'BMI_CATEGORY'] = 2
        df.loc[df['BMI'] >= 30, 'BMI_CATEGORY'] = 3
        
        return df
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, float]:
        """预测所有疾病的风险"""
        if not self.models:
            self.load_models()
        
        # 预处理输入数据
        df = self.preprocess_input(data)
        
        # 确保所有特征都存在
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # 按特征名称排序
        df = df[self.feature_names]
        
        # 预测每种疾病的风险
        predictions = {}
        for disease in self.diseases:
            if disease in self.models:
                # 处理缺失值
                X_imputed = self.imputers[disease].transform(df)
                
                # 标准化特征
                X_scaled = self.scalers[disease].transform(X_imputed)
                
                # 预测概率
                risk = self.models[disease].predict_proba(X_scaled)[0, 1]
                predictions[disease] = float(risk)
        
        return predictions

def main():
    """主函数"""
    predictor = FraminghamPredictor()
    predictor.load_models()

    if len(sys.argv) == 2:
        # 从json文件读取输入
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        predictions = predictor.predict(input_data)
        print(json.dumps(predictions, ensure_ascii=False))
    else:
        # 用测试数据
        test_data = {
            'SEX': 1,  # 1=男性, 0=女性
            'AGE': 45,
            'TOTCHOL': 200,
            'SYSBP': 120,
            'DIABP': 80,
            'CURSMOKE': 0,
            'CIGPDAY': 0,
            'BMI': 25,
            'DIABETES': 0,
            'BPMEDS': 0,
            'HEARTRTE': 75,
            'GLUCOSE': 100
        }
        predictions = predictor.predict(test_data)
        print(json.dumps(predictions, ensure_ascii=False))

if __name__ == "__main__":
    main() 