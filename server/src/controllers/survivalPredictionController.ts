import { Request, Response } from 'express';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';
import { v4 as uuidv4 } from 'uuid';

interface SurvivalPrediction {
  risk_score: number;
  expected_time_years: number;
  median_time_years: number;
  survival_probabilities: Array<{
    years: number;
    survival_probability: number;
    event_probability: number;
  }>;
  model_quality: number;
  baseline_event_rate: number;
}

interface SurvivalPredictionResponse {
  success: boolean;
  survival_predictions?: Record<string, SurvivalPrediction>;
  message?: string;
  error?: string;
}

export const predictSurvivalTimes = async (req: Request, res: Response) => {
  try {
    console.log('🔮 开始生存分析预测...');

    // 验证输入数据
    const healthData = req.body;
    const requiredFields = ['sex', 'age'];
    
    for (const field of requiredFields) {
      if (healthData[field] === undefined || healthData[field] === null) {
        return res.status(400).json({
          success: false,
          message: `缺少必填字段: ${field}`
        });
      }
    }

    // 设置默认值
    const processedData = {
      sex: Number(healthData.sex),
      age: Number(healthData.age),
      totchol: Number(healthData.totchol) || 200,
      sysbp: Number(healthData.sysbp) || 120,
      diabp: Number(healthData.diabp) || 80,
      cursmoke: Number(healthData.cursmoke) || 0,
      cigpday: Number(healthData.cigpday) || 0,
      bmi: Number(healthData.bmi) || 25,
      diabetes: Number(healthData.diabetes) || 0,
      bpmeds: Number(healthData.bpmeds) || 0,
      heartrte: Number(healthData.heartrte) || 70,
      glucose: Number(healthData.glucose) || 90
    };

    console.log('📋 处理后的健康数据:', processedData);

    // 调用Python生存分析脚本
    const pythonScript = path.resolve(__dirname, '../../../ml_analysis/survival_inference.py');
    const inputData = JSON.stringify(processedData);
    
    console.log('🐍 调用生存分析脚本:', pythonScript);
    console.log('📝 输入数据:', inputData);

    const pythonProcess = spawn('python3', [pythonScript, inputData], {
      cwd: path.resolve(__dirname, '../../../ml_analysis'),
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    // 设置超时
    const timeout = setTimeout(() => {
      pythonProcess.kill('SIGTERM');
      console.error('❌ Python脚本执行超时 (3分钟)');
    }, 180000); // 3分钟超时

    pythonProcess.on('close', async (code) => {
      clearTimeout(timeout);

      if (code !== 0) {
        console.error('❌ Python脚本执行失败:');
        console.error('stdout:', stdout);
        console.error('stderr:', stderr);
        
        return res.status(500).json({
          success: false,
          message: '生存分析预测失败',
          error: stderr || '未知错误'
        });
      }

      try {
        console.log('📊 Python脚本原始输出:', stdout);
        console.log('🔍 stderr内容:', stderr);
        
        // 清理stdout，移除可能的空白字符
        const cleanedOutput = stdout.trim();
        
        if (!cleanedOutput) {
          return res.status(500).json({
            success: false,
            message: '生存分析预测失败 - 空输出',
            error: 'Python脚本没有返回任何数据'
          });
        }

        // 解析Python脚本输出
        const result: SurvivalPredictionResponse = JSON.parse(cleanedOutput);
        
        if (!result.success) {
          return res.status(500).json({
            success: false,
            message: result.message || '生存分析预测失败',
            error: result.error
          });
        }

        console.log('✅ 生存分析预测成功');
        
        // 返回预测结果
        res.json({
          success: true,
          message: '生存分析预测完成',
          survival_predictions: result.survival_predictions,
          metadata: {
            timestamp: new Date().toISOString(),
            model_type: 'survival_analysis',
            input_features: Object.keys(processedData).length
          }
        });

      } catch (parseError) {
        console.error('❌ 解析预测结果失败:', parseError);
        console.error('Python stdout:', stdout);
        console.error('Python stderr:', stderr);
        
        res.status(500).json({
          success: false,
          message: '解析生存分析结果失败 - JSON格式错误',
          error: `解析错误: ${String(parseError)}. 输出: ${stdout.slice(0, 200)}...`
        });
      }
    });

    pythonProcess.on('error', (error) => {
      clearTimeout(timeout);
      console.error('❌ Python进程启动失败:', error);
      
      res.status(500).json({
        success: false,
        message: '启动生存分析模型失败',
        error: String(error)
      });
    });

  } catch (error) {
    console.error('❌ 生存分析预测控制器错误:', error);
    res.status(500).json({
      success: false,
      message: '生存分析预测失败',
      error: String(error)
    });
  }
};

// 获取生存分析模型信息
export const getSurvivalModelInfo = async (req: Request, res: Response) => {
  try {
    const modelInfo = {
      diseases: [
        'CVD', 'CHD', 'STROKE', 'ANGINA', 'MI', 'HYPERTENSION', 'DEATH'
      ],
      features: [
        'sex', 'age', 'totchol', 'sysbp', 'diabp', 'cursmoke',
        'cigpday', 'bmi', 'diabetes', 'bpmeds', 'heartrte', 'glucose'
      ],
      model_types: ['cox_proportional_hazard', 'weibull_aft'],
      time_units: 'years',
      prediction_horizons: [1, 5, 10, 20],
      description: '基于弗雷明汉心脏研究数据的生存分析模型，可预测心血管疾病发生时间'
    };

    res.json({
      success: true,
      model_info: modelInfo
    });

  } catch (error) {
    console.error('❌ 获取生存模型信息失败:', error);
    res.status(500).json({
      success: false,
      message: '获取模型信息失败',
      error: String(error)
    });
  }
}; 