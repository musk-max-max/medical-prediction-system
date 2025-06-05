import express from 'express';
import { auth, adminAuth } from '../middleware/auth';
import { getHistory } from '../controllers/prediction';

const router = express.Router();

// 获取历史记录 - 需要认证
router.get('/history', auth, getHistory);

export default router; 