import { Request, Response } from 'express';
import { body, validationResult } from 'express-validator';
import { database } from '../config/database';
import { hashPassword, comparePassword, generateToken } from '../utils/auth';
import { User, RegisterRequest, LoginRequest, AuthResponse } from '../types';

export const registerValidation = [
  body('username').isLength({ min: 3, max: 30 }).withMessage('用户名长度必须在3-30个字符之间'),
  body('email').isEmail().withMessage('邮箱格式不正确'),
  body('password').isLength({ min: 6 }).withMessage('密码长度至少6个字符'),
];

export const loginValidation = [
  body('username').notEmpty().withMessage('用户名不能为空'),
  body('password').notEmpty().withMessage('密码不能为空'),
];

export const register = async (req: Request, res: Response): Promise<void> => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      res.status(400).json({ errors: errors.array() });
      return;
    }

    const { username, email, password }: RegisterRequest = req.body;
    const db = database.getDatabase();

    // 检查用户是否已存在
    const existingUser = await new Promise<User | undefined>((resolve, reject) => {
      db.get(
        'SELECT * FROM users WHERE username = ? OR email = ?',
        [username, email],
        (err: any, row: User) => {
          if (err) reject(err);
          else resolve(row);
        }
      );
    });

    if (existingUser) {
      res.status(400).json({ error: '用户名或邮箱已存在' });
      return;
    }

    // 创建新用户
    const hashedPassword = await hashPassword(password);
    const userId = await new Promise<number>((resolve, reject) => {
      db.run(
        'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
        [username, email, hashedPassword],
        function(err: any) {
          if (err) reject(err);
          else resolve(this.lastID);
        }
      );
    });

    // 生成token
    const token = generateToken(userId, username, false);
    const response: AuthResponse = {
      token,
      user: {
        id: userId,
        username,
        email,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }
    };

    res.status(201).json(response);
  } catch (error) {
    console.error('注册错误:', error);
    res.status(500).json({ error: '服务器内部错误' });
  }
};

export const login = async (req: Request, res: Response): Promise<void> => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      console.log('登录验证错误:', errors.array());
      res.status(400).json({ error: '输入数据验证失败' });
      return;
    }

    const { username, password }: LoginRequest = req.body;
    console.log('尝试登录用户:', username);
    
    const db = database.getDatabase();

    // 查找用户
    const user = await new Promise<User | undefined>((resolve, reject) => {
      db.get(
        'SELECT * FROM users WHERE username = ?',
        [username],
        (err: any, row: User) => {
          if (err) {
            console.error('数据库查询错误:', err);
            reject(err);
          } else {
            console.log('查询到的用户:', row ? '找到用户' : '未找到用户');
            resolve(row);
          }
        }
      );
    });

    if (!user) {
      console.log('用户不存在:', username);
      res.status(400).json({ error: '用户名或密码错误' });
      return;
    }

    // 验证密码
    const isValidPassword = await comparePassword(password, user.password!);
    console.log('密码验证结果:', isValidPassword ? '密码正确' : '密码错误');
    
    if (!isValidPassword) {
      res.status(400).json({ error: '用户名或密码错误' });
      return;
    }

    // 生成 JWT token
    const token = generateToken(user.id, user.username, Boolean(user.is_admin));
    console.log('生成token成功');

    res.json({
      token,
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        is_admin: Boolean(user.is_admin)
      }
    });
  } catch (error) {
    console.error('登录错误:', error);
    res.status(500).json({ error: '服务器内部错误' });
  }
}; 