import express from 'express';
import { login, register } from '../controllers/auth';

const router = express.Router();

// 登录
router.post('/login', login);

// 注册
router.post('/register', register);

export default router; 