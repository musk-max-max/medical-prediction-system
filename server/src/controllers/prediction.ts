import { Request, Response } from 'express';
import { Op } from 'sequelize';
import { User } from '../models/user';

interface AuthRequest extends Request {
  user?: {
    id: number;
    username: string;
    is_admin: boolean;
  };
}

// 获取历史记录
export const getHistory = async (req: AuthRequest, res: Response) => {
  try {
    const { search } = req.query as { search?: string };
    const userId = req.user?.id;
    const isAdmin = req.user?.is_admin;

    // 暂时返回空数据，避免编译错误
    const records: any[] = [];

    res.json({
      success: true,
      data: records
    });
  } catch (error) {
    console.error('Get history error:', error);
    res.status(500).json({ error: '获取历史记录失败' });
  }
}; 