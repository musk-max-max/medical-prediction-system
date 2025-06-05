import express from 'express';
import { authenticateToken } from '../middleware/auth';
import { 
  predict, 
  getPredictionHistory, 
  deletePredictionHistory 
} from '../controllers/predictionController';

const router = express.Router();

// 预测路由
router.post('/predict', authenticateToken, predict);

// 获取历史记录
router.get('/history', authenticateToken, getPredictionHistory);

// 删除历史记录
router.delete('/history', authenticateToken, deletePredictionHistory);

export default router; 