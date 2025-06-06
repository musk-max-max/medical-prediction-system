#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

def diagnose_environment():
    """诊断Render环境中的文件系统结构"""
    
    result = {
        "success": True,
        "environment": {
            "python_version": sys.version,
            "current_directory": os.getcwd(),
            "script_location": os.path.abspath(__file__),
            "python_executable": sys.executable
        },
        "file_structure": {},
        "cox_files": {},
        "errors": []
    }
    
    try:
        # 检查当前目录结构
        current_dir = Path.cwd()
        result["file_structure"]["current_dir_contents"] = []
        
        for item in current_dir.iterdir():
            item_info = {
                "name": item.name,
                "type": "file" if item.is_file() else "directory",
                "path": str(item)
            }
            result["file_structure"]["current_dir_contents"].append(item_info)
        
        # 查找ml_analysis目录
        ml_analysis_paths = []
        
        # 检查相对路径
        possible_paths = [
            Path("."),
            Path(".."),
            Path("../.."),
            Path("../../.."),
            Path("./ml_analysis"),
            Path("../ml_analysis"),
            Path("../../ml_analysis"),
            Path("../../../ml_analysis"),
            Path("/opt/render/project/src/ml_analysis"),
            Path("/opt/render/project/ml_analysis")
        ]
        
        for path in possible_paths:
            if path.exists():
                ml_analysis_path = path / "ml_analysis" if not path.name == "ml_analysis" else path
                if ml_analysis_path.exists():
                    ml_analysis_paths.append(str(ml_analysis_path.absolute()))
        
        result["file_structure"]["ml_analysis_paths"] = ml_analysis_paths
        
        # 检查Cox文件
        cox_files = [
            "cox_timevarying_model.pkl",
            "cox_tv_scaler.pkl", 
            "cox_tv_imputer.pkl",
            "cox_tv_features.pkl",
            "cox_tv_evaluation.pkl"
        ]
        
        for ml_path in ml_analysis_paths:
            ml_dir = Path(ml_path)
            result["cox_files"][str(ml_dir)] = {}
            
            for cox_file in cox_files:
                file_path = ml_dir / cox_file
                if file_path.exists():
                    try:
                        file_stat = file_path.stat()
                        result["cox_files"][str(ml_dir)][cox_file] = {
                            "exists": True,
                            "size": file_stat.st_size,
                            "readable": os.access(file_path, os.R_OK)
                        }
                    except Exception as e:
                        result["cox_files"][str(ml_dir)][cox_file] = {
                            "exists": True,
                            "error": str(e)
                        }
                else:
                    result["cox_files"][str(ml_dir)][cox_file] = {"exists": False}
        
        # 检查环境变量
        result["environment"]["env_vars"] = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "PWD": os.environ.get("PWD", ""),
            "HOME": os.environ.get("HOME", "")
        }
        
    except Exception as e:
        result["success"] = False
        result["errors"].append(str(e))
    
    return result

if __name__ == "__main__":
    try:
        diagnosis = diagnose_environment()
        print(json.dumps(diagnosis, indent=2, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)) 