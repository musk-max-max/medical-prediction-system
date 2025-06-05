#!/usr/bin/env python3
import requests
import json
import time

def test_with_different_users():
    """测试不同用户的预测"""
    for i in range(3):
        user_id = int(time.time()) + i
        
        print(f"\n=== 测试用户 {user_id} ===")
        
        # 注册
        register_data = {
            "username": f"testuser{user_id}",
            "email": f"test{user_id}@example.com", 
            "password": "password123"
        }
        
        try:
            register_response = requests.post("http://127.0.0.1:5000/api/auth/register", 
                                            json=register_data, timeout=10)
            
            if register_response.status_code != 201:
                print(f"❌ 注册失败: {register_response.text}")
                continue
                
            token = register_response.json()["token"]
            print("✅ 注册成功")
            
            # 预测 - 分别测试不同的超时时间
            health_data = {
                "sex": 1, "age": 45, "totchol": 200, "sysbp": 120, "diabp": 80,
                "cursmoke": 0, "cigpday": 0, "bmi": 25, "diabetes": 0, 
                "bpmeds": 0, "heartrte": 75, "glucose": 100
            }
            
            headers = {"Authorization": f"Bearer {token}"}
            
            for timeout in [30, 60, 120]:
                print(f"⏱️ 尝试预测 (超时: {timeout}秒)...")
                try:
                    predict_response = requests.post("http://127.0.0.1:5000/api/predict", 
                                                   json=health_data, headers=headers, 
                                                   timeout=timeout)
                    
                    if predict_response.status_code == 200:
                        print("✅ 预测成功!")
                        result = predict_response.json()
                        print(f"📊 获得 {len(result.get('predictions', {}))} 种疾病的预测结果")
                        return True
                    else:
                        print(f"❌ 预测失败 {predict_response.status_code}: {predict_response.text}")
                        
                except requests.exceptions.Timeout:
                    print(f"⏰ 请求超时 ({timeout}秒)")
                except Exception as e:
                    print(f"❌ 请求异常: {e}")
                    
                time.sleep(1)  # 等待1秒再试
                
        except Exception as e:
            print(f"❌ 注册异常: {e}")
    
    return False

if __name__ == "__main__":
    print("🔍 简单API测试")
    
    # 先检查健康状态
    try:
        health_response = requests.get("http://127.0.0.1:5000/api/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ 后端服务正常")
        else:
            print(f"❌ 后端服务异常: {health_response.status_code}")
            exit(1)
    except Exception as e:
        print(f"❌ 无法连接后端: {e}")
        exit(1)
    
    # 测试预测功能
    success = test_with_different_users()
    
    if success:
        print("\n🎉 测试成功!")
    else:
        print("\n💔 所有测试都失败了") 