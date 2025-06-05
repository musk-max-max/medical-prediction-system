# 使用Node.js官方镜像
FROM node:18-slim

# 安装Python和必要的系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制package.json文件
COPY server/package*.json ./server/
COPY package*.json ./

# 安装Node.js依赖
RUN cd server && npm install

# 创建Python虚拟环境并安装依赖
RUN python3 -m venv /app/.venv
COPY requirements.txt ./
RUN /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY server/ ./server/
COPY ml_analysis/ ./ml_analysis/
COPY models/ ./models/

# 编译TypeScript
RUN cd server && npm run build

# 暴露端口
EXPOSE 5000

# 设置环境变量
ENV NODE_ENV=production
ENV PYTHON_PATH=/app/.venv/bin/python

# 启动应用
CMD ["node", "server/dist/index.js"] 