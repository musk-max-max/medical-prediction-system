# 医疗预测系统部署指南

## 🚀 快速部署（推荐）

### 选项1: Vercel + Railway
**优点**: 免费额度充足，部署简单，性能好
1. 前端部署到Vercel（免费）
2. 后端部署到Railway（免费500小时/月）

### 选项2: Netlify + Heroku
**优点**: 老牌稳定，文档丰富
1. 前端部署到Netlify（免费）
2. 后端部署到Heroku（免费额度有限）

### 选项3: Render（全栈）
**优点**: 一站式解决方案，配置简单
1. 前后端一起部署到Render

## 📦 部署前检查清单

- [ ] 代码推送到GitHub
- [ ] 环境变量配置完成
- [ ] 数据库迁移脚本准备
- [ ] CORS配置正确
- [ ] Python依赖文件存在

## 🔧 环境变量设置

### 前端环境变量
```
REACT_APP_API_URL=https://your-backend-url.com
```

### 后端环境变量
```
NODE_ENV=production
JWT_SECRET=your-super-secret-key
PORT=5000
PYTHON_PATH=/app/.venv/bin/python
```

## 🌐 域名配置

1. 在域名服务商设置DNS
2. 添加CNAME记录指向部署平台
3. 配置SSL证书（平台自动提供）

## 📊 监控和维护

- 设置错误监控
- 配置日志收集
- 定期备份数据库
- 监控API响应时间 