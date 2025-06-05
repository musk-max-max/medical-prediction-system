import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { AuthRequest } from '../types';

export const auth = async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');

    if (!token) {
      throw new Error();
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key') as any;

    req.user = {
      id: decoded.id,
      username: decoded.username,
      email: decoded.email || '',
      is_admin: decoded.is_admin || false,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };
    next();
  } catch (error) {
    res.status(401).json({ error: '请先登录' });
  }
};

export const adminAuth = async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    await auth(req, res, () => {
      if (!req.user?.is_admin) {
        return res.status(403).json({ error: '需要管理员权限' });
      }
      next();
    });
  } catch (error) {
    res.status(401).json({ error: '请先登录' });
  }
};

// 别名导出，兼容其他文件的导入
export const authenticateToken = auth; 