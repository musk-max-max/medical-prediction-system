import express from 'express';
import cors from 'cors';
import { sequelize } from './config/database';
import authRoutes from './routes/auth';
import predictionRoutes from './routes/prediction';
import { User } from './models/user';

const app = express();

// 中间件
app.use(cors());
app.use(express.json());

// 路由
app.use('/api/auth', authRoutes);
app.use('/api/predict', predictionRoutes);

// 健康检查
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

// 数据库同步和初始化
const initDatabase = async () => {
  try {
    // 同步数据库模型
    await sequelize.sync({ force: true }); // 注意：force: true 会删除现有表
    console.log('数据库同步完成');

    // 创建管理员账户
    await User.createAdminUser();

    console.log('数据库初始化完成');
  } catch (error) {
    console.error('数据库初始化失败:', error);
  }
};

// 启动服务器
const PORT = process.env.PORT || 5000;
app.listen(PORT, async () => {
  console.log(`服务器运行在端口 ${PORT}`);
  await initDatabase();
});

export default app; 