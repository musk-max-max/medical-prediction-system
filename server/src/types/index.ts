export interface User {
  id: number;
  username: string;
  email: string;
  password?: string;
  is_admin?: boolean; // 0: 普通用户, 1: 管理员
  created_at: string;
  updated_at: string;
}

export interface HealthData {
  id?: number;
  user_id: number;
  sex: number; // 0: 女性, 1: 男性
  age: number;
  totchol?: number; // 总胆固醇
  sysbp?: number; // 收缩压
  diabp?: number; // 舒张压
  cursmoke?: number; // 是否吸烟 0: 否, 1: 是
  cigpday?: number; // 每天吸烟支数
  bmi?: number; // 身体质量指数
  diabetes?: number; // 是否糖尿病 0: 否, 1: 是
  bpmeds?: number; // 是否服用降压药 0: 否, 1: 是
  heartrte?: number; // 心率
  glucose?: number; // 血糖
  created_at?: string;
}

export interface PredictionResult {
  id?: number;
  user_id: number;
  health_data_id: number;
  prediction_result: number; // 预测概率 0-1
  risk_level: 'Low' | 'Medium' | 'High'; // 风险等级
  model_version: string;
  created_at?: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface AuthResponse {
  token: string;
  user: Omit<User, 'password'>;
}

export interface PredictionRequest extends HealthData {
  // 继承HealthData的所有字段
}

// Express扩展接口
import { Request } from 'express';

export interface AuthRequest extends Request {
  user?: User;
} 