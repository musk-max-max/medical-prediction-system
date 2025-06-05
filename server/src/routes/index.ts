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

export default router; 