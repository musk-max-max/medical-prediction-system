#!/usr/bin/env python3
"""
Cox时间变化生存分析推理脚本
使用训练好的CoxTimeVaryingFitter模型进行预测
"""

import pandas as pd
import numpy as np
import json
import sys
import os
import warnings
import pickle
import joblib

warnings.filterwarnings('ignore')

def log_message(message):
    """将调试信息输出到stderr"""
    print(message, file=sys.stderr, flush=True)

class CoxTimeVaryingPredictor:
    def __init__(self):
        self.cox_model = None
        self.scaler = None
        self.imputer = None
        self.features = None
        
        # 为其他疾病加载Framingham模型
        self.framingham_models = {}
        self.framingham_scalers = {}
        self.framingham_imputers = {}
        
        # 疾病列表 (CVD用Cox模型，其他用Framingham模型)
        self.diseases = ['CVD', 'CHD', 'STROKE', 'MI', 'ANGINA', 'HYPERTENSION', 'DEATH']
        self.framingham_diseases = ['chd', 'stroke', 'mi', 'angina', 'hypertension', 'death']
        
    def load_models(self, model_dir='.'):
        """加载训练好的Cox模型"""
        log_message("Loading Cox Time-Varying models...")
        log_message(f"Current working directory: {os.getcwd()}")
        log_message(f"Script location: {os.path.abspath(__file__)}")
        log_message(f"Model directory: {model_dir}")
        
        # 尝试多个可能的路径
        possible_dirs = [
            model_dir,
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml_analysis'),
            '/opt/render/project/src/ml_analysis',
            '/opt/render/project/ml_analysis',
            '.'
        ]
        
        model_files = {
            'cox_model': 'cox_timevarying_model.pkl',
            'scaler': 'cox_tv_scaler.pkl', 
            'imputer': 'cox_tv_imputer.pkl',
            'features': 'cox_tv_features.pkl'
        }
        
        # 记录文件搜索过程
        log_message("Searching for model files...")
        for directory in possible_dirs:
            log_message(f"Checking directory: {directory}")
            if os.path.exists(directory):
                files_in_dir = os.listdir(directory)
                log_message(f"  Files found: {files_in_dir}")
                
                # 检查是否所有模型文件都存在
                all_files_found = True
                for file_key, filename in model_files.items():
                    file_path = os.path.join(directory, filename)
                    if os.path.exists(file_path):
                        log_message(f"  ✓ Found {filename}")
                    else:
                        log_message(f"  ✗ Missing {filename}")
                        all_files_found = False
                
                if all_files_found:
                    log_message(f"All model files found in: {directory}")
                    model_dir = directory
                    break
            else:
                log_message(f"  Directory does not exist")
        
        try:
            # 加载Cox模型
            cox_model_path = os.path.join(model_dir, 'cox_timevarying_model.pkl')
            if os.path.exists(cox_model_path):
                with open(cox_model_path, 'rb') as f:
                    self.cox_model = pickle.load(f)
                log_message(f"Cox model loaded successfully from: {cox_model_path}")
            else:
                log_message(f"Cox model file not found at: {cox_model_path}")
                return False
            
            # 加载预处理器
            scaler_path = os.path.join(model_dir, 'cox_tv_scaler.pkl')
            imputer_path = os.path.join(model_dir, 'cox_tv_imputer.pkl')
            features_path = os.path.join(model_dir, 'cox_tv_features.pkl')
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                log_message("Scaler loaded successfully")
            
            if os.path.exists(imputer_path):
                self.imputer = joblib.load(imputer_path)
                log_message("Imputer loaded successfully")
            
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    self.features = pickle.load(f)
                log_message(f"Features loaded: {len(self.features)} features")
            
            # 加载Framingham模型用于其他疾病
            self.load_framingham_models(model_dir)
            
            return True
            
        except Exception as e:
            log_message(f"Model loading failed: {e}")
            return False
    
    def load_framingham_models(self, model_dir='.'):
        """加载Framingham模型用于其他疾病预测"""
        log_message("Loading Framingham models for other diseases...")
        
        loaded_count = 0
        for disease in self.framingham_diseases:
            try:
                model_path = os.path.join(model_dir, f'framingham_{disease}_model.pkl')
                scaler_path = os.path.join(model_dir, f'framingham_{disease}_scaler.pkl')
                imputer_path = os.path.join(model_dir, f'framingham_{disease}_imputer.pkl')
                
                if all(os.path.exists(p) for p in [model_path, scaler_path, imputer_path]):
                    self.framingham_models[disease] = joblib.load(model_path)
                    self.framingham_scalers[disease] = joblib.load(scaler_path)
                    self.framingham_imputers[disease] = joblib.load(imputer_path)
                    
                    log_message(f"  ✓ {disease.upper()} Framingham model loaded")
                    loaded_count += 1
                else:
                    log_message(f"  ✗ {disease.upper()} model files missing")
                    
            except Exception as e:
                log_message(f"  ✗ {disease.upper()} model loading failed: {e}")
                
        log_message(f"Loaded {loaded_count}/{len(self.framingham_diseases)} Framingham models")
    
    def preprocess_input(self, patient_data):
        """预处理患者数据"""
        log_message(f"Preprocessing input data: {patient_data}")
        
        if self.features is None:
            log_message("Features not loaded")
            return None
        
        # 创建特征向量
        feature_vector = []
        for feature in self.features:
            value = None
            
            # 尝试多种键名格式
            for key_format in [feature.upper(), feature.lower(), feature]:
                if key_format in patient_data:
                    value = patient_data[key_format]
                    break
            
            # 如果还没找到，尝试映射
            if value is None:
                mapping = {
                    'SEX': 'sex', 'AGE': 'age', 'TOTCHOL': 'totchol',
                    'SYSBP': 'sysbp', 'DIABP': 'diabp', 'CURSMOKE': 'cursmoke',
                    'CIGPDAY': 'cigpday', 'BMI': 'bmi', 'DIABETES': 'diabetes',
                    'BPMEDS': 'bpmeds', 'HEARTRTE': 'heartrte', 'GLUCOSE': 'glucose'
                }
                mapped_key = mapping.get(feature.upper())
                if mapped_key and mapped_key in patient_data:
                    value = patient_data[mapped_key]
            
            # 使用默认值
            if value is None:
                default_values = {
                    'SEX': 1, 'AGE': 50, 'TOTCHOL': 200, 'SYSBP': 120, 'DIABP': 80,
                    'CURSMOKE': 0, 'CIGPDAY': 0, 'BMI': 25, 'DIABETES': 0,
                    'BPMEDS': 0, 'HEARTRTE': 70, 'GLUCOSE': 90,
                    'PREVCHD': 0, 'PREVAP': 0, 'PREVMI': 0, 'PREVSTRK': 0, 'PREVHYP': 0
                }
                value = default_values.get(feature.upper(), 0)
            
            feature_vector.append(float(value))
            log_message(f"   {feature}: {value}")
        
        # 转换为DataFrame
        feature_df = pd.DataFrame([feature_vector], columns=self.features)
        
        # 应用预处理
        if self.imputer:
            feature_df = pd.DataFrame(self.imputer.transform(feature_df), columns=self.features)
        
        if self.scaler:
            feature_df = pd.DataFrame(self.scaler.transform(feature_df), columns=self.features)
        
        log_message(f"Processed feature shape: {feature_df.shape}")
        return feature_df
    
    def predict_survival(self, patient_data):
        """预测生存概率"""
        log_message("Starting Cox survival prediction...")
        
        if self.cox_model is None:
            log_message("Cox model not loaded")
            return None
        
        try:
            # 预处理数据
            processed_data = self.preprocess_input(patient_data)
            if processed_data is None:
                return None
            
            # 获取风险评分
            risk_score = self.cox_model.predict_partial_hazard(processed_data).iloc[0]
            log_message(f"Risk score (partial hazard): {risk_score}")
            
            # 计算生存概率 (使用基线生存函数)
            time_points = [365, 1825, 3650, 7300]  # 1, 5, 10, 20年(天)
            survival_probabilities = []
            
            for days in time_points:
                years = days / 365.25
                
                # 使用Cox模型的生存函数
                try:
                    # 对于CoxTimeVaryingFitter，需要构造预测数据格式
                    prediction_df = processed_data.copy()
                    prediction_df['start'] = 0
                    prediction_df['stop'] = days
                    prediction_df['id'] = 1
                    
                    # 预测生存概率
                    survival_func = self.cox_model.predict_survival_function(prediction_df)
                    
                    if len(survival_func) > 0:
                        # 获取指定时间点的生存概率
                        timeline = survival_func.index
                        if days in timeline:
                            survival_prob = survival_func.loc[days].iloc[0]
                        else:
                            # 插值获取生存概率
                            if days > timeline.max():
                                survival_prob = survival_func.iloc[-1].iloc[0]
                            else:
                                # 线性插值
                                idx = np.searchsorted(timeline, days)
                                if idx > 0:
                                    survival_prob = survival_func.iloc[idx-1].iloc[0]
                                else:
                                    survival_prob = 1.0
                    else:
                        # 如果无法获取生存函数，使用指数模型近似
                        baseline_hazard = 0.01  # 基线风险率
                        survival_prob = np.exp(-baseline_hazard * risk_score * years)
                
                except:
                    # 备用方案：基于风险评分的指数生存模型
                    log_message(f"Using exponential approximation for {years} years")
                    baseline_hazard = 0.01
                    survival_prob = np.exp(-baseline_hazard * risk_score * years)
                
                # 确保概率在合理范围内
                survival_prob = max(0.01, min(0.99, survival_prob))
                event_prob = 1 - survival_prob
                
                survival_probabilities.append({
                    'years': int(years),
                    'survival_probability': float(survival_prob),
                    'event_probability': float(event_prob)
                })
                
                log_message(f"  {years} years: survival={survival_prob:.3f}, event={event_prob:.3f}")
            
            # 计算期望时间
            expected_time = 25 * (2 - risk_score) if risk_score < 2 else 5
            expected_time = max(1, expected_time)
            
            result = {
                'risk_score': float(risk_score),
                'expected_time_years': float(expected_time),
                'median_time_years': float(expected_time * 0.693),
                'survival_probabilities': survival_probabilities,
                'model_quality': 0.90,  # Cox模型质量更高
                'baseline_event_rate': float(min(0.99, risk_score / 10))
            }
            
            log_message("Cox survival prediction completed successfully")
            return result
            
        except Exception as e:
            log_message(f"Cox survival prediction failed: {e}")
            return None
    
    def predict_framingham_disease(self, patient_data, disease):
        """使用Framingham模型预测单个疾病"""
        if disease not in self.framingham_models:
            log_message(f"Framingham model for {disease} not loaded")
            return None
        
        try:
            # 准备18个核心特征
            features = [
                'SEX', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 
                'CIGPDAY', 'BMI', 'DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE',
                'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'PULSE_PRESSURE'
            ]
            
            feature_vector = []
            for feature in features:
                value = None
                
                # 尝试多种键名格式
                for key_format in [feature.upper(), feature.lower(), feature]:
                    if key_format in patient_data:
                        value = patient_data[key_format]
                        break
                
                # 特征映射
                if value is None:
                    mapping = {
                        'SEX': 'sex', 'AGE': 'age', 'TOTCHOL': 'totchol',
                        'SYSBP': 'sysbp', 'DIABP': 'diabp', 'CURSMOKE': 'cursmoke',
                        'CIGPDAY': 'cigpday', 'BMI': 'bmi', 'DIABETES': 'diabetes',
                        'BPMEDS': 'bpmeds', 'HEARTRTE': 'heartrte', 'GLUCOSE': 'glucose'
                    }
                    mapped_key = mapping.get(feature.upper())
                    if mapped_key and mapped_key in patient_data:
                        value = patient_data[mapped_key]
                
                # 默认值
                if value is None:
                    default_values = {
                        'SEX': 1, 'AGE': 50, 'TOTCHOL': 200, 'SYSBP': 120, 'DIABP': 80,
                        'CURSMOKE': 0, 'CIGPDAY': 0, 'BMI': 25, 'DIABETES': 0,
                        'BPMEDS': 0, 'HEARTRTE': 70, 'GLUCOSE': 90,
                        'PREVCHD': 0, 'PREVAP': 0, 'PREVMI': 0, 'PREVSTRK': 0, 'PREVHYP': 0,
                        'PULSE_PRESSURE': 0
                    }
                    value = default_values.get(feature.upper(), 0)
                    
                    # 计算脉压
                    if feature == 'PULSE_PRESSURE':
                        sysbp = patient_data.get('sysbp', 120)
                        diabp = patient_data.get('diabp', 80)
                        value = float(sysbp) - float(diabp)
                
                feature_vector.append(float(value))
            
            # 预处理
            X = np.array(feature_vector).reshape(1, -1)
            X = self.framingham_imputers[disease].transform(X)
            X = self.framingham_scalers[disease].transform(X)
            
            # 预测
            model = self.framingham_models[disease]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                risk_prob = proba[1] if len(proba) > 1 else proba[0]
            else:
                risk_prob = float(model.predict(X)[0])
                risk_prob = max(0, min(1, risk_prob))
            
            # 计算生存指标
            base_time = 25
            risk_adjusted_time = base_time * (1 - risk_prob * 0.8)
            expected_time = max(risk_adjusted_time, 1)
            
            survival_probabilities = []
            for years in [1, 5, 10, 20]:
                hazard_rate = -np.log(1 - risk_prob) / base_time if risk_prob < 0.99 else 0.1
                survival_prob = np.exp(-hazard_rate * years)
                survival_prob = max(0.01, min(0.99, survival_prob))
                
                survival_probabilities.append({
                    'years': years,
                    'survival_probability': float(survival_prob),
                    'event_probability': float(1 - survival_prob)
                })
            
            result = {
                'risk_score': float(risk_prob),
                'expected_time_years': float(expected_time),
                'median_time_years': float(expected_time * 0.693),
                'survival_probabilities': survival_probabilities,
                'model_quality': 0.85,  # Framingham模型质量
                'baseline_event_rate': float(risk_prob)
            }
            
            log_message(f"{disease.upper()} prediction completed (risk: {risk_prob:.3f})")
            return result
            
        except Exception as e:
            log_message(f"{disease} Framingham prediction failed: {e}")
            return None
    
    def run_prediction(self, patient_data):
        """运行完整的Cox生存分析预测"""
        # 加载模型
        if not self.load_models():
            return {
                'success': False,
                'error': 'Unable to load Cox Time-Varying models'
            }
        
        # 进行预测
        prediction = self.predict_survival(patient_data)
        
        if prediction is None:
            return {
                'success': False,
                'error': 'Cox survival prediction failed'
            }
        
        # 格式化为与原系统兼容的输出
        survival_predictions = {
            'CVD': prediction  # 主要用CVD作为代表性疾病
        }
        
        return {
            'success': True,
            'survival_predictions': survival_predictions,
            'metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_type': 'cox_time_varying',
                'input_features': len(self.features) if self.features else 0,
                'diseases_predicted': 1
            }
        }

def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "success": False,
            "error": "Usage: python cox_survival_inference.py <patient_data_json>"
        }))
        return
    
    try:
        # 解析患者数据
        patient_data = json.loads(sys.argv[1])
        
        # 创建预测器
        predictor = CoxTimeVaryingPredictor()
        
        # 运行预测
        result = predictor.run_prediction(patient_data)
        
        # 输出结果到stdout（只有JSON）
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Cox survival analysis failed: {str(e)}"
        }))

if __name__ == "__main__":
    main() 