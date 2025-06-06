#!/usr/bin/env python3
"""
测试Cox模型文件是否存在
"""
import os
import json
import sys

def test_cox_files():
    """检查Cox模型文件是否存在"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    cox_files = [
        'cox_timevarying_model.pkl',
        'cox_tv_scaler.pkl', 
        'cox_tv_imputer.pkl',
        'cox_tv_features.pkl'
    ]
    
    result = {
        'success': True,
        'current_directory': current_dir,
        'files_status': {},
        'errors': []
    }
    
    for filename in cox_files:
        filepath = os.path.join(current_dir, filename)
        exists = os.path.exists(filepath)
        
        result['files_status'][filename] = {
            'exists': exists,
            'path': filepath,
            'size': os.path.getsize(filepath) if exists else 0
        }
        
        if not exists:
            result['success'] = False
            result['errors'].append(f"Missing file: {filename}")
    
    # 列出当前目录所有文件
    try:
        all_files = os.listdir(current_dir)
        result['all_files'] = sorted(all_files)
    except Exception as e:
        result['all_files'] = []
        result['errors'].append(f"Cannot list directory: {e}")
    
    return result

if __name__ == "__main__":
    result = test_cox_files()
    print(json.dumps(result, indent=2, ensure_ascii=False)) 