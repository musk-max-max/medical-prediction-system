import express from 'express';
import { register, login, registerValidation, loginValidation } from '../controllers/authController';
import { predict, getPredictionHistory, deletePredictionHistory } from '../controllers/predictionController';
import { predictSurvivalTimes, getSurvivalModelInfo } from '../controllers/survivalPredictionController';
import { authenticateToken } from '../utils/auth';

const router = express.Router();

// 认证路由
router.post('/auth/register', registerValidation, register);
router.post('/auth/login', loginValidation, login);

// 预测路由
router.post('/predict', authenticateToken, predict);
router.get('/predict/history', authenticateToken, getPredictionHistory);
router.delete('/predict/history', authenticateToken, deletePredictionHistory);

// 生存分析路由（需要认证）
router.post('/survival/predict', authenticateToken, predictSurvivalTimes);
router.get('/survival/model-info', authenticateToken, getSurvivalModelInfo);

// 临时测试端点 - 无需认证的Cox模型测试
router.post('/test-survival-predict', predictSurvivalTimes);
router.get('/test-survival-info', getSurvivalModelInfo);

// 健康检查
router.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    message: '医疗预测系统运行正常',
    timestamp: new Date().toISOString(),
    features: {
      risk_prediction: true,
      survival_analysis: true
    }
  });
});

// 临时测试端点 - 检查Cox文件
router.get('/test-cox-files', (req, res) => {
  const { spawn } = require('child_process');
  const path = require('path');
  
  const pythonScript = path.resolve(__dirname, '../../ml_analysis/test_cox_files.py');
  const pythonProcess = spawn('python', [pythonScript], {
    cwd: path.resolve(__dirname, '../../ml_analysis')
  });

  let stdout = '';
  let stderr = '';

  pythonProcess.stdout.on('data', (data) => {
    stdout += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    stderr += data.toString();
  });

  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({
        success: false,
        error: stderr,
        code: code
      });
    }

    try {
      const result = JSON.parse(stdout);
      res.json(result);
    } catch (e) {
      res.status(500).json({
        success: false,
        error: 'Failed to parse output',
        stdout: stdout,
        stderr: stderr
      });
    }
  });
});

// 环境诊断端点
router.get('/diagnose-env', (req, res) => {
  const { spawn } = require('child_process');
  const path = require('path');
  
  const pythonScript = path.resolve(__dirname, '../../ml_analysis/diagnose_render_env.py');
  const pythonProcess = spawn('python', [pythonScript], {
    cwd: path.resolve(__dirname, '../../ml_analysis')
  });

  let stdout = '';
  let stderr = '';

  pythonProcess.stdout.on('data', (data) => {
    stdout += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    stderr += data.toString();
  });

  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({
        success: false,
        error: stderr,
        code: code,
        stdout: stdout
      });
    }

    try {
      const result = JSON.parse(stdout);
      res.json(result);
    } catch (e) {
      res.status(500).json({
        success: false,
        error: 'Failed to parse output',
        stdout: stdout,
        stderr: stderr
      });
    }
  });
});

export default router; 