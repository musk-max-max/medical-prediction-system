import { Request, Response } from 'express';
import { Op } from 'sequelize';
import { Prediction } from '../models/prediction';
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

    // 构建查询条件
    const where: any = {};
    
    // 如果不是管理员，只能查看自己的记录
    if (!isAdmin) {
      where.user_id = userId;
    }
    
    // 如果是管理员且提供了搜索条件
    if (isAdmin && search) {
      where['$User.username$'] = {
        [Op.like]: `%${search}%`
      };
    }

    const records = await Prediction.findAll({
      where,
      include: [{
        model: User,
        attributes: ['username'],
        required: true
      }],
      order: [['created_at', 'DESC']]
    });

    // 格式化返回数据
    const formattedRecords = records.map(record => ({
      ...record.toJSON(),
      username: record.User?.username
    }));

    res.json({
      success: true,
      data: formattedRecords
    });
  } catch (error) {
    console.error('Get history error:', error);
    res.status(500).json({ error: '获取历史记录失败' });
  }
}; 