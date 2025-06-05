import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import routes from './routes';
import { database } from './config/database';
import bcrypt from 'bcrypt';
import path from 'path';

const app = express();
const PORT = process.env.PORT || 5000;

// 创建管理员账户
const createAdminUser = () => {
  const db = database.getDatabase();
  
  // 首先检查是否已存在管理员账户
  db.get("SELECT * FROM users WHERE username = 'admin'", (err, row) => {
    if (err) {
      console.error('检查管理员账户失败:', err);
      return;
    }
    
    if (!row) {
      // 创建管理员账户
      bcrypt.hash('admin123', 10, (hashErr, hashedPassword) => {
        if (hashErr) {
          console.error('密码加密失败:', hashErr);
          return;
        }
        
        db.run(
          "INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)",
          ['admin', 'admin@system.com', hashedPassword, 1],
          function(insertErr) {
            if (insertErr) {
              console.error('创建管理员账户失败:', insertErr);
            } else {
              console.log('管理员账户创建成功 (ID:', this.lastID, ')');
            }
          }
        );
      });
    } else {
      console.log('管理员账户已存在');
    }
  });
};

// 开发环境下宽松的安全配置
if (process.env.NODE_ENV !== 'production') {
  app.use(helmet({
    crossOriginResourcePolicy: false,
    contentSecurityPolicy: false
  }));
} else {
  app.use(helmet());
}

// CORS配置 - 支持生产环境
const corsOptions = {
  origin: process.env.NODE_ENV === 'production' 
    ? [
        'https://your-domain.com',
        'https://your-vercel-app.vercel.app',
        /\.vercel\.app$/
      ]
    : ['http://localhost:3000', 'http://127.0.0.1:3000'],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
};

app.use(cors(corsOptions));

// 请求限制
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15分钟
  max: 100, // 限制每个IP 15分钟内最多100个请求
  message: '请求过于频繁，请稍后再试'
});
app.use(limiter);

// 特殊的预测接口限制
const predictionLimiter = rateLimit({
  windowMs: 60 * 1000, // 1分钟
  max: 10, // 限制每个IP 1分钟内最多10次预测
  message: '预测请求过于频繁，请稍后再试'
});

// JSON解析
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// 预测路由特殊限制
app.use('/api/predict', predictionLimiter);

// 路由
app.use('/api', routes);

// 错误处理中间件
app.use((err: any, req: any, res: any, next: any) => {
  console.error('服务器错误:', err);
  res.status(500).json({
    error: '服务器内部错误',
    message: process.env.NODE_ENV === 'development' ? err.message : '请稍后重试'
  });
});

// 404处理
app.use('*', (req: any, res: any) => {
  res.status(404).json({
    error: '接口不存在',
    message: '请检查请求路径'
  });
});

// 优雅关闭
process.on('SIGTERM', () => {
  console.log('收到SIGTERM信号，正在关闭服务器...');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('收到SIGINT信号，正在关闭服务器...');
  process.exit(0);
});

app.listen(PORT, () => {
  console.log('🚀 医疗预测系统后端服务启动成功');
  console.log(`📡 服务器运行在: http://localhost:${PORT}`);
  console.log(`🏥 API地址: http://localhost:${PORT}/api`);
  console.log(`💊 健康检查: http://localhost:${PORT}/api/health`);
  console.log(`🔒 环境: ${process.env.NODE_ENV || 'development'}`);
  
  // 创建管理员账户
  setTimeout(createAdminUser, 1000); // 延迟1秒执行，确保数据库初始化完成
}); 