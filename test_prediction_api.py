#!/usr/bin/env python3
"""
测试医疗预测API
"""

import requests
import json
import random
import time

API_BASE = "http://127.0.0.1:5000/api"

def test_health_check():
    """测试健康检查接口"""
    print("🔍 测试健康检查接口...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print("✅ 健康检查通过")
            print(response.json())
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")

def test_register_and_predict():
    """测试注册和预测流程"""
    print("\n🔍 测试注册和预测流程...")
    
    # 1. 注册用户 - 使用随机用户名
    print("1. 注册用户...")
    random_id = int(time.time())
    register_data = {
        "username": f"testuser{random_id}",
        "email": f"test{random_id}@example.com",
        "password": "password123"
    }
    
    try:
        register_response = requests.post(f"{API_BASE}/auth/register", json=register_data)
        if register_response.status_code == 201:
            print("✅ 用户注册成功")
            auth_data = register_response.json()
            token = auth_data["token"]
        else:
            print(f"❌ 用户注册失败: {register_response.text}")
            return
    except Exception as e:
        print(f"❌ 注册请求失败: {e}")
        return
    
    # 2. 预测疾病风险
    print("2. 预测疾病风险...")
    health_data = {
        "sex": 1,
        "age": 45,
        "totchol": 200,
        "sysbp": 120,
        "diabp": 80,
        "cursmoke": 0,
        "cigpday": 0,
        "bmi": 25,
        "diabetes": 0,
        "bpmeds": 0,
        "heartrte": 75,
        "glucose": 100
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        predict_response = requests.post(f"{API_BASE}/predict", json=health_data, headers=headers, timeout=60)
        if predict_response.status_code == 200:
            print("✅ 预测成功")
            result = predict_response.json()
            print("📊 预测结果:")
            if "predictions" in result:
                for disease, info in result["predictions"].items():
                    print(f"  {disease}: {info['risk_probability']*100:.1f}% ({info['risk_level']})")
            
            if "overall_risk" in result:
                print(f"\n总体风险: {result['overall_risk']['total_risk_score']*100:.1f}% ({result['overall_risk']['risk_category']})")
        else:
            print(f"❌ 预测失败: {predict_response.text}")
    except Exception as e:
        print(f"❌ 预测请求失败: {e}")

if __name__ == "__main__":
    print("🏥 医疗预测API测试")
    print("=" * 50)
    
    test_health_check()
    test_register_and_predict()
    
    print("\n✨ 测试完成") 