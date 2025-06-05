// API配置
const config = {
  // 开发环境
  development: {
    API_BASE_URL: 'http://localhost:5000/api'
  },
  // 生产环境 - 请替换为您的Render部署URL
  production: {
    API_BASE_URL: 'https://YOUR-ACTUAL-RENDER-URL.onrender.com/api'
  }
};

// 根据环境选择配置
const environment = process.env.NODE_ENV || 'development';
export const API_CONFIG = config[environment as keyof typeof config];

export default API_CONFIG; 