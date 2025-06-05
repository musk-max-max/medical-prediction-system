import { API_CONFIG } from '../config';

class ApiService {
  private baseURL: string;

  constructor() {
    this.baseURL = API_CONFIG.API_BASE_URL;
  }

  // 通用请求方法
  private async request(endpoint: string, options: RequestInit = {}): Promise<any> {
    const url = `${this.baseURL}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
      },
    };

    const config = { ...defaultOptions, ...options };
    
    // 添加认证token
    const token = localStorage.getItem('token');
    if (token) {
      config.headers = {
        ...config.headers,
        'Authorization': `Bearer ${token}`,
      };
    }

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // 健康检查
  async healthCheck() {
    return this.request('/health');
  }

  // 用户注册
  async register(userData: { username: string; email: string; password: string }) {
    return this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  // 用户登录
  async login(credentials: { username: string; password: string }) {
    return this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    });
  }

  // 医疗预测
  async predict(healthData: any) {
    return this.request('/prediction/predict', {
      method: 'POST',
      body: JSON.stringify(healthData),
    });
  }

  // 获取预测历史
  async getPredictionHistory() {
    return this.request('/prediction/history');
  }
}

export const apiService = new ApiService();
export default apiService; 