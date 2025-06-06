import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import { translations } from './locales';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// 接口定义
interface User {
  id: number;
  username: string;
  email: string;
  is_admin: boolean;
}

interface AuthData {
  token: string;
  user: User;
}

interface HealthData {
  sex: number;
  age: number;
  totchol?: number;
  sysbp?: number;
  diabp?: number;
  cursmoke?: number;
  cigpday?: number;
  bmi?: number;
  diabetes?: number;
  bpmeds?: number;
  heartrte?: number;
  glucose?: number;
}

interface PredictionResult {
  [disease: string]: {
    name: string;
    risk_probability: number;
    risk_level: string;
    description: string;
    recommendations: string[];
  };
}

interface PredictionResponse {
  success: boolean;
  message?: string;
  predictions?: PredictionResult;
  overall_risk?: {
    high_risk_diseases: string[];
    total_risk_score: number;
    risk_category: string;
  };
  ai_advice?: {
    enabled: boolean;
    content: string;
    generated_by: 'ai' | 'fallback';
  };
}

// 生存分析接口
interface SurvivalPrediction {
  risk_score: number;
  expected_time_years: number;
  median_time_years: number;
  survival_probabilities: Array<{
    years: number;
    survival_probability: number;
    event_probability: number;
  }>;
  model_quality: number;
  baseline_event_rate: number;
}

interface SurvivalPredictionResponse {
  success: boolean;
  message?: string;
  survival_predictions?: Record<string, SurvivalPrediction>;
  metadata?: {
    timestamp: string;
    model_type: string;
    input_features: number;
  };
}

// API配置
const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://medical-prediction-api.onrender.com/api';
axios.defaults.baseURL = API_BASE_URL;
axios.defaults.timeout = 60000; // 增加到60秒超时，用于AI预测
axios.defaults.headers.common['Content-Type'] = 'application/json';
axios.defaults.withCredentials = true; // 启用 credentials 以支持跨域请求

// 添加请求拦截器
axios.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url, config.data);
    return config;
  },
  (error) => {
    console.error('请求错误:', error);
    return Promise.reject(error);
  }
);

// 添加响应拦截器
axios.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.data);
    return response;
  },
  (error) => {
    console.error('响应错误:', error);
    if (error.code === 'ECONNABORTED') {
      console.error('请求超时');
    } else if (error.code === 'ERR_NETWORK') {
      console.error('网络连接错误');
    }
    return Promise.reject(error);
  }
);

interface HistoryRecord {
  id: number;
  created_at: string;
  age: number;
  sex: number;
  totchol: number | null;
  sysbp: number | null;
  diabp: number | null;
  cursmoke: number;
  cigpday: number;
  bmi: number | null;
  diabetes: number;
  bpmeds: number;
  heartrte: number | null;
  glucose: number | null;
  username?: string;
}

const App: React.FC = () => {
  const [user, setUser] = useState<User | null>(null);
  const [currentView, setCurrentView] = useState<'login' | 'register' | 'prediction' | 'history'>('login');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [language, setLanguage] = useState<'en' | 'zh'>(() => {
    const savedLanguage = localStorage.getItem('language');
    return (savedLanguage === 'zh' || savedLanguage === 'en') ? savedLanguage : 'en';
  });

  // 获取当前语言的翻译
  const t = translations[language];

  // 认证状态
  const [authForm, setAuthForm] = useState({
    username: '',
    email: '',
    password: ''
  });

  // 健康数据表单
  const [healthForm, setHealthForm] = useState<HealthData>({
    sex: 1,
    age: 50,
    totchol: undefined,
    sysbp: undefined,
    diabp: undefined,
    cursmoke: 0,
    cigpday: 0,
    bmi: undefined,
    diabetes: 0,
    bpmeds: 0,
    heartrte: undefined,
    glucose: undefined
  });

  // 预测结果
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [survivalResult, setSurvivalResult] = useState<SurvivalPredictionResponse | null>(null);
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [selectedRecords, setSelectedRecords] = useState<number[]>([]);
  const [selectAll, setSelectAll] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isAdmin, setIsAdmin] = useState(false);
  const [userAgreement, setUserAgreement] = useState(false); // 用户协议同意状态
  const [useAIAdvice, setUseAIAdvice] = useState(true); // AI建议选项
  const [showTrendChart, setShowTrendChart] = useState(false); // 趋势图显示状态

  // 初始化
  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    const savedUser = localStorage.getItem('user');
    
    if (savedToken && savedUser) {
      const userData = JSON.parse(savedUser);
      setUser(userData);
      setIsAdmin(userData.is_admin || false);
      setCurrentView('prediction');
      axios.defaults.headers.common['Authorization'] = `Bearer ${savedToken}`;
    }

    // 测试API连接
    const testAPIConnection = async () => {
      try {
        console.log('测试API连接...');
        const response = await axios.get('/health');
        console.log('API连接成功:', response.data);
      } catch (error: any) {
        console.error('API连接失败:', error);
        if (error.code === 'ERR_NETWORK') {
          setError('无法连接到服务器，请确保后端服务正在运行');
        }
      }
    };

    testAPIConnection();
  }, []);

  // 清除消息
  const clearMessages = () => {
    setError(null);
    setSuccess(null);
  };

  // 手动测试API连接
  const testConnection = async () => {
    setLoading(true);
    clearMessages();
    
    try {
      console.log('手动测试API连接...');
      const response = await axios.get('/health');
      console.log('API连接成功:', response.data);
      setSuccess('API连接测试成功！');
    } catch (error: any) {
      console.error('API连接失败:', error);
      const errorMessage = error.code === 'ERR_NETWORK' 
        ? '网络连接错误：无法连接到后端服务器' 
        : error.message || '连接测试失败';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // 自动清除成功消息
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => {
        setSuccess(null);
      }, 5000); // 5秒后自动清除成功消息
      return () => clearTimeout(timer);
    }
  }, [success]);

  // 登录
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    clearMessages();

    try {
      const response = await axios.post('/auth/login', { username: authForm.username, password: authForm.password });
      const authData: AuthData = response.data;
      
      console.log('Login response:', authData);
      console.log('User is admin:', authData.user.is_admin);
      
      // 存储登录信息
      localStorage.setItem('token', authData.token);
      localStorage.setItem('user', JSON.stringify(authData.user));
      
      // 设置用户状态
      setUser(authData.user);
      setIsAdmin(authData.user.is_admin);
      axios.defaults.headers.common['Authorization'] = `Bearer ${authData.token}`;
      setCurrentView('prediction');
      setSuccess(language === 'en' ? 'Login successful!' : '登录成功！');
    } catch (error: any) {
      setError(error.response?.data?.error || '登录失败');
    } finally {
      setLoading(false);
    }
  };

  // 注册
  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    clearMessages();

    try {
      const response = await axios.post('/auth/register', {
        username: authForm.username,
        email: authForm.email,
        password: authForm.password
      });

      const authData: AuthData = response.data;
      setUser(authData.user);
      localStorage.setItem('token', authData.token);
      localStorage.setItem('user', JSON.stringify(authData.user));
      axios.defaults.headers.common['Authorization'] = `Bearer ${authData.token}`;
      setCurrentView('prediction');
      setSuccess(language === 'en' ? 'Registration successful!' : '注册成功！');
    } catch (error: any) {
      setError(error.response?.data?.error || '注册失败');
    } finally {
      setLoading(false);
    }
  };

  // 登出
  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    delete axios.defaults.headers.common['Authorization'];
    setCurrentView('login');
    setPredictionResult(null);
    setSurvivalResult(null);
    setSuccess(language === 'en' ? 'Logged out successfully' : '已安全登出');
  };

  // 检查是否有空值
  const checkEmptyValues = () => {
    const emptyFields = [];
    if (healthForm.totchol === undefined) {
      emptyFields.push(language === 'en' ? 'Total Cholesterol' : '总胆固醇');
    }
    if (healthForm.sysbp === undefined) {
      emptyFields.push(language === 'en' ? 'Systolic Blood Pressure' : '收缩压');
    }
    if (healthForm.diabp === undefined) {
      emptyFields.push(language === 'en' ? 'Diastolic Blood Pressure' : '舒张压');
    }
    if (healthForm.bmi === undefined) {
      emptyFields.push('BMI');
    }
    if (healthForm.heartrte === undefined) {
      emptyFields.push(language === 'en' ? 'Heart Rate' : '心率');
    }
    if (healthForm.glucose === undefined) {
      emptyFields.push(language === 'en' ? 'Fasting Glucose' : '空腹血糖');
    }
    return emptyFields;
  };

  // 综合预测 - 同时进行风险评估和生存分析
  const handleComprehensivePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    clearMessages();

    // 检查空值
    const emptyFields = checkEmptyValues();
    if (emptyFields.length > 0) {
      const confirmMessage = language === 'en' 
        ? `The following fields are empty:\n${emptyFields.map(field => `• ${field}`).join('\n')}\n\nThis may affect prediction accuracy. Continue?`
        : `以下字段为空：\n${emptyFields.map(field => `• ${field}`).join('\n')}\n\n这可能会影响预测准确性。是否继续？`;
      
      if (!window.confirm(confirmMessage)) {
        setLoading(false);
        return;
      }
    }

    console.log('开始综合预测，健康数据:', healthForm);
    console.log('当前用户:', user);
    console.log('Authorization token:', axios.defaults.headers.common['Authorization']);

    try {
      // 分别发起两个请求，避免其中一个失败影响另一个
      const requestData = {
        ...healthForm,
        useAIAdvice: useAIAdvice,
        language: language
      };

      console.log('发起风险评估请求...');
      const riskPromise = axios.post('/predict', requestData, {
        timeout: 120000 // 增加到120秒超时，给AI分析更多时间
      });

      console.log('发起生存分析请求...');
      const survivalPromise = axios.post('/survival/predict', healthForm, {
        timeout: 120000 // 增加到120秒超时
      });

      // 等待两个请求都完成
      const [riskResponse, survivalResponse] = await Promise.all([
        riskPromise,
        survivalPromise
      ]);

      console.log('风险评估响应:', riskResponse.data);
      console.log('生存分析响应:', survivalResponse.data);

      setPredictionResult(riskResponse.data);
      setSurvivalResult(survivalResponse.data);
      setSuccess('🎉 综合分析完成！');
    } catch (error: any) {
      console.error('预测错误详情:', error);
      console.error('错误响应数据:', error.response?.data);
      console.error('错误状态码:', error.response?.status);
      console.error('错误消息:', error.message);
      
      let errorMessage = '预测失败，请稍后重试';
      
      if (error.code === 'ECONNABORTED') {
        errorMessage = '🕐 请求超时：AI分析时间过长，请稍后重试';
      } else if (error.code === 'ERR_NETWORK') {
        errorMessage = '🔌 网络连接错误：无法连接到服务器';
      } else if (error.response?.status === 429) {
        errorMessage = '⏳ 请求过于频繁，请稍后再试';
      } else if (error.response?.status === 401) {
        errorMessage = '🔐 认证失效，请重新登录';
      } else if (error.response?.data?.error) {
        errorMessage = error.response.data.error;
      } else if (error.response?.data?.message) {
        errorMessage = error.response.data.message;
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // 获取历史记录
  const fetchHistory = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/predict/history', {
        params: {
          search: searchQuery,
          is_admin: isAdmin
        }
      });
      console.log('History response:', response.data);
      console.log('Is admin:', isAdmin);
      console.log('First record:', response.data.data?.[0]);
      setHistory(response.data.data || []);
    } catch (error: any) {
      setError(language === 'en' ? 
        'Failed to fetch history records' : 
        '获取历史记录失败');
    } finally {
      setLoading(false);
    }
  };

  // 添加搜索处理函数
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    fetchHistory();
  };

  // 格式化日期
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return language === 'en' 
      ? date.toLocaleString('en-US', { 
          year: 'numeric', 
          month: 'short', 
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        })
      : date.toLocaleString('zh-CN', { 
          year: 'numeric', 
          month: 'long', 
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        });
  };

  // 格式化数值
  const formatValue = (value: number | undefined | null) => {
    if (value === undefined || value === null) return '-';
    return value.toFixed(1);
  };

  // 风险等级颜色
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return '#4CAF50';
      case 'medium': return '#FF9800';
      case 'high': return '#F44336';
      default: return '#9E9E9E';
    }
  };

  // 风险等级文本
  const getRiskText = (level: string) => {
    switch (level) {
      case 'low': return '低风险';
      case 'medium': return '中等风险';
      case 'high': return '高风险';
      default: return '未知';
    }
  };

  // 切换语言
  const toggleLanguage = () => {
    setLanguage(prev => prev === 'en' ? 'zh' : 'en');
  };

  // 切换选择所有记录
  const toggleSelectAll = () => {
    if (selectAll) {
      setSelectedRecords([]);
    } else {
      setSelectedRecords(history.map(record => record.id));
    }
    setSelectAll(!selectAll);
  };

  // 切换单个记录的选择
  const toggleRecordSelection = (recordId: number) => {
    setSelectedRecords(prev => {
      if (prev.includes(recordId)) {
        return prev.filter(id => id !== recordId);
      } else {
        return [...prev, recordId];
      }
    });
  };

  // 删除选中的记录
  const deleteSelectedRecords = async () => {
    if (!window.confirm(language === 'en' ? 
      'Are you sure you want to delete the selected records?' : 
      '确定要删除选中的记录吗？')) {
      return;
    }

    setLoading(true);
    try {
      await axios.delete('/predict/history', {
        data: { ids: selectedRecords }
      });
      setSuccess(language === 'en' ? 
        'Selected records deleted successfully' : 
        '已成功删除选中的记录');
      setSelectedRecords([]);
      setSelectAll(false);
      // 立即重新加载历史记录
      await fetchHistory();
    } catch (error: any) {
      setError(language === 'en' ? 
        'Failed to delete records' : 
        '删除记录失败');
    } finally {
      setLoading(false);
    }
  };

  // 导出选中的记录
  const exportSelectedRecords = () => {
    const selectedData = history.filter(record => 
      selectedRecords.includes(record.id)
    );

    // 转换为CSV格式
    const headers = [
      'ID', 'Created At', 'Age', 'Sex', 'Total Cholesterol', 
      'Systolic BP', 'Diastolic BP', 'Current Smoker', 
      'Cigarettes per Day', 'BMI', 'Diabetes', 
      'BP Medication', 'Heart Rate', 'Glucose'
    ];

    const csvContent = [
      headers.join(','),
      ...selectedData.map(record => [
        record.id,
        record.created_at,
        record.age,
        record.sex,
        record.totchol || '',
        record.sysbp || '',
        record.diabp || '',
        record.cursmoke,
        record.cigpday,
        record.bmi || '',
        record.diabetes,
        record.bpmeds,
        record.heartrte || '',
        record.glucose || ''
      ].join(','))
    ].join('\n');

    // 创建并下载文件
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `medical_records_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // 显示趋势图
  const showTrendAnalysis = () => {
    if (selectedRecords.length < 2) {
      setError(language === 'en' ? 
        'Please select at least 2 records to view trend analysis' : 
        '请至少选择2条记录以查看趋势分析');
      return;
    }
    setShowTrendChart(true);
  };

  // 关闭趋势图
  const closeTrendChart = () => {
    setShowTrendChart(false);
  };

  // 认证表单
  const renderAuthForm = () => (
    <div className="auth-container">
      <div className="auth-card">
        <div className="language-switch">
          <button 
            className={`lang-btn ${language === 'en' ? 'active' : ''}`}
            onClick={() => setLanguage('en')}
          >
            EN
          </button>
          <button 
            className={`lang-btn ${language === 'zh' ? 'active' : ''}`}
            onClick={() => setLanguage('zh')}
          >
            中
          </button>
        </div>
        
        <h2>{currentView === 'login' ? t.login.title : t.register.title}</h2>
        
        {/* 错误和成功消息显示 */}
        {error && (
          <div className="alert alert-error">
            {error}
          </div>
        )}
        
        {success && (
          <div className="alert alert-success">
            {success}
          </div>
        )}
        
        <form onSubmit={currentView === 'login' ? handleLogin : handleRegister}>
          <div className="form-group">
            <label>{t.login.username}</label>
            <input
              type="text"
              value={authForm.username}
              onChange={(e) => setAuthForm({...authForm, username: e.target.value})}
              required
            />
          </div>
          
          {currentView === 'register' && (
            <div className="form-group">
              <label>{t.register.email}</label>
              <input
                type="email"
                value={authForm.email}
                onChange={(e) => setAuthForm({...authForm, email: e.target.value})}
                required
              />
            </div>
          )}
          
          <div className="form-group">
            <label>{t.login.password}</label>
            <input
              type="password"
              value={authForm.password}
              onChange={(e) => setAuthForm({...authForm, password: e.target.value})}
              required
            />
          </div>
          
          <button 
            type="submit" 
            disabled={loading}
            style={{
              fontSize: '1.2rem',
              padding: '12px 24px',
              width: '100%',
              marginTop: '1rem'
            }}
          >
            {loading ? t.login.loading : (currentView === 'login' ? t.login.submit : t.register.submit)}
          </button>
          
          <p>
            {currentView === 'login' ? t.login.noAccount : t.register.haveAccount}
            <button 
              type="button" 
              className="link-button"
              onClick={() => setCurrentView(currentView === 'login' ? 'register' : 'login')}
            >
              {currentView === 'login' ? t.login.register : t.register.login}
            </button>
          </p>
        </form>
      </div>
    </div>
  );

  // 预测表单
  const renderPredictionForm = () => (
    <div className="prediction-container">
      <h2>{t.prediction.title}</h2>
      <p className="subtitle">{t.prediction.subtitle}</p>
      
      {/* 错误和成功消息显示 */}
      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}
      
      {success && (
        <div className="alert alert-success">
          {success}
        </div>
      )}
      
      <form onSubmit={handleComprehensivePredict} className="health-form">
        {/* 基本信息 */}
        <div className="form-section">
          <h3 className="section-title">
            {t.prediction.basicInfo.title}
          </h3>
          <div className="form-row">
            <div className="form-group">
              <label>{t.prediction.basicInfo.gender} <span className="required">*</span></label>
              <select
                value={healthForm.sex}
                onChange={(e) => setHealthForm({...healthForm, sex: Number(e.target.value)})}
                required
              >
                <option value={1}>{t.prediction.basicInfo.male}</option>
                <option value={0}>{t.prediction.basicInfo.female}</option>
              </select>
            </div>
            
            <div className="form-group">
              <label>{t.prediction.basicInfo.age} <span className="required">*</span></label>
              <input
                type="number"
                min="30"
                max="62"
                value={healthForm.age}
                onChange={(e) => {
                  const age = Number(e.target.value);
                  if (age < 30 || age > 62) {
                    setError(language === 'en' ? 
                      'Age must be between 30 and 62 years' : 
                      '年龄必须在30-62岁之间');
                  } else {
                    setError(null);
                  }
                  setHealthForm({...healthForm, age: age});
                }}
                required
              />
              <small>{t.prediction.basicInfo.ageRange}</small>
            </div>
            
            <div className="form-group">
              <label>{t.prediction.basicInfo.bmi}</label>
              <input
                type="number"
                step="0.1"
                min="15"
                max="40"
                value={healthForm.bmi || ''}
                onChange={(e) => {
                  const value = e.target.value === '' ? undefined : Number(e.target.value);
                  setHealthForm({...healthForm, bmi: value});
                }}
              />
              <small>{t.prediction.basicInfo.bmiRange}</small>
            </div>
          </div>
        </div>

        {/* 心血管指标 */}
        <div className="form-section">
          <h3 className="section-title">
            {t.prediction.cardiovascular.title}
          </h3>
          <div className="form-row">
            <div className="form-group">
              <label>{t.prediction.cardiovascular.systolicBP}</label>
              <input
                type="number"
                min="60"
                max="250"
                value={healthForm.sysbp || ''}
                onChange={(e) => {
                  const value = e.target.value === '' ? undefined : Number(e.target.value);
                  if (value !== undefined && (value < 90 || value > 200)) {
                    setError(language === 'en' ? 
                      '⚠️ Warning: Systolic blood pressure is outside normal range (90-200 mmHg). Please confirm this value.' : 
                      '⚠️ 警告：收缩压超出正常范围（90-200 mmHg）。请确认此数值。');
                  } else {
                    setError(null);
                  }
                  setHealthForm({...healthForm, sysbp: value});
                }}
              />
              <small>{t.prediction.cardiovascular.normalBP}</small>
            </div>
            
            <div className="form-group">
              <label>{t.prediction.cardiovascular.diastolicBP}</label>
              <input
                type="number"
                min="40"
                max="150"
                value={healthForm.diabp || ''}
                onChange={(e) => {
                  const value = e.target.value === '' ? undefined : Number(e.target.value);
                  if (value !== undefined && (value < 60 || value > 120)) {
                    setError(language === 'en' ? 
                      '⚠️ Warning: Diastolic blood pressure is outside normal range (60-120 mmHg). Please confirm this value.' : 
                      '⚠️ 警告：舒张压超出正常范围（60-120 mmHg）。请确认此数值。');
                  } else {
                    setError(null);
                  }
                  setHealthForm({...healthForm, diabp: value});
                }}
              />
              <small>{t.prediction.cardiovascular.normalBP}</small>
            </div>
            
            <div className="form-group">
              <label>{t.prediction.cardiovascular.heartRate}</label>
              <input
                type="number"
                min="30"
                max="200"
                value={healthForm.heartrte || ''}
                onChange={(e) => {
                  const value = e.target.value === '' ? undefined : Number(e.target.value);
                  if (value !== undefined && (value < 50 || value > 120)) {
                    setError(language === 'en' ? 
                      '⚠️ Warning: Heart rate is outside normal range (50-120 bpm). Please confirm this value.' : 
                      '⚠️ 警告：心率超出正常范围（50-120次/分钟）。请确认此数值。');
                  } else {
                    setError(null);
                  }
                  setHealthForm({...healthForm, heartrte: value});
                }}
              />
              <small>{t.prediction.cardiovascular.normalHR}</small>
            </div>
          </div>
        </div>

        {/* 生化指标 */}
        <div className="form-section">
          <h3 className="section-title">
            {t.prediction.biochemical.title}
          </h3>
          <div className="form-row">
            <div className="form-group">
              <label>{t.prediction.biochemical.cholesterol}</label>
              <input
                type="number"
                min="80"
                max="600"
                value={healthForm.totchol || ''}
                onChange={(e) => {
                  const value = e.target.value === '' ? undefined : Number(e.target.value);
                  if (value !== undefined && (value < 120 || value > 400)) {
                    setError(language === 'en' ? 
                      '⚠️ Warning: Total cholesterol is outside normal range (120-400 mg/dL). Please confirm this value.' : 
                      '⚠️ 警告：总胆固醇超出正常范围（120-400 mg/dL）。请确认此数值。');
                  } else {
                    setError(null);
                  }
                  setHealthForm({...healthForm, totchol: value});
                }}
              />
              <small>{t.prediction.biochemical.idealCholesterol}</small>
            </div>
            
            <div className="form-group">
              <label>{t.prediction.biochemical.glucose}</label>
              <input
                type="number"
                min="40"
                max="400"
                value={healthForm.glucose || ''}
                onChange={(e) => {
                  const value = e.target.value === '' ? undefined : Number(e.target.value);
                  if (value !== undefined && (value < 70 || value > 200)) {
                    setError(language === 'en' ? 
                      '⚠️ Warning: Fasting glucose is outside normal range (70-200 mg/dL). Please confirm this value.' : 
                      '⚠️ 警告：空腹血糖超出正常范围（70-200 mg/dL）。请确认此数值。');
                  } else {
                    setError(null);
                  }
                  setHealthForm({...healthForm, glucose: value});
                }}
              />
              <small>{t.prediction.biochemical.normalGlucose}</small>
            </div>
          </div>
        </div>

        {/* 生活习惯 */}
        <div className="form-section">
          <h3 className="section-title">
            {t.prediction.lifestyle.title}
          </h3>
          <div className="form-row">
            <div className="form-group">
              <label>{t.prediction.lifestyle.smokingStatus} <span className="required">*</span></label>
              <select
                value={healthForm.cursmoke}
                onChange={(e) => {
                  const smoking = Number(e.target.value);
                  setHealthForm({
                    ...healthForm, 
                    cursmoke: smoking,
                    cigpday: smoking === 0 ? 0 : healthForm.cigpday
                  });
                }}
                required
              >
                <option value={0}>{t.prediction.lifestyle.noSmoking}</option>
                <option value={1}>{t.prediction.lifestyle.smoking}</option>
              </select>
            </div>
            
            <div className="form-group">
              <label>
                {t.prediction.lifestyle.cigarettesPerDay}
                {healthForm.cursmoke === 1 && <span className="required">*</span>}
              </label>
              <input
                type="number"
                min="0"
                max="60"
                value={healthForm.cigpday}
                onChange={(e) => {
                  const value = Number(e.target.value);
                  if (value < 0 || value > 60) {
                    setError(language === 'en' ? 
                      '⚠️ Warning: Cigarettes per day is outside normal range (0-60). Please confirm this value.' : 
                      '⚠️ 警告：每日吸烟量超出正常范围（0-60支）。请确认此数值。');
                  } else {
                    setError(null);
                  }
                  setHealthForm({...healthForm, cigpday: value});
                }}
                disabled={healthForm.cursmoke === 0}
                required={healthForm.cursmoke === 1}
                style={{ 
                  backgroundColor: healthForm.cursmoke === 0 ? '#f5f5f5' : 'white',
                  opacity: healthForm.cursmoke === 0 ? 0.6 : 1
                }}
              />
              <small>{healthForm.cursmoke === 0 ? t.prediction.lifestyle.autoZero : ''}</small>
            </div>
          </div>
        </div>

        {/* 疾病史与用药 */}
        <div className="form-section">
          <h3 className="section-title">
            {t.prediction.medicalHistory.title}
          </h3>
          <div className="form-row">
            <div className="form-group">
              <label>{t.prediction.medicalHistory.diabetes} <span className="required">*</span></label>
              <select
                value={healthForm.diabetes}
                onChange={(e) => setHealthForm({...healthForm, diabetes: Number(e.target.value)})}
                required
              >
                <option value={0}>{t.prediction.medicalHistory.no}</option>
                <option value={1}>{t.prediction.medicalHistory.yes}</option>
              </select>
            </div>
            
            <div className="form-group">
              <label>{t.prediction.medicalHistory.bpMeds} <span className="required">*</span></label>
              <select
                value={healthForm.bpmeds}
                onChange={(e) => setHealthForm({...healthForm, bpmeds: Number(e.target.value)})}
                required
              >
                <option value={0}>{t.prediction.medicalHistory.notUsing}</option>
                <option value={1}>{t.prediction.medicalHistory.using}</option>
              </select>
            </div>
          </div>
        </div>

        {/* 用户协议 */}
        <div className="user-agreement-section">
          <h3 className="section-title">
            📋 {language === 'en' ? 'Medical Research Data Usage Agreement' : '将数据应用于医学研究的用户协议'}
          </h3>
          <div className="agreement-content">
            <div className="agreement-text">
              {language === 'en' ? (
                <>
                  <p><strong>Data Usage for Medical Research:</strong></p>
                  <ul>
                    <li>Your health data will be anonymized and may be used for medical research purposes</li>
                    <li>No personal identifying information will be shared with third parties</li>
                    <li>Data will be used to improve cardiovascular disease prediction models</li>
                    <li>Your participation helps advance medical science and potentially benefit future patients</li>
                    <li>You can withdraw your consent at any time by contacting our support team</li>
                  </ul>
                  <p><em>By checking the box below, you consent to the use of your anonymized health data for medical research purposes.</em></p>
                </>
              ) : (
                <>
                  <p><strong>数据用于医学研究说明：</strong></p>
                  <ul>
                    <li>您的健康数据将被匿名化处理，可能用于医学研究目的</li>
                    <li>不会与第三方共享任何个人身份信息</li>
                    <li>数据将用于改进心血管疾病预测模型</li>
                    <li>您的参与有助于推进医学科学发展，并可能造福未来患者</li>
                    <li>您可随时通过联系我们的支持团队撤回同意</li>
                  </ul>
                  <p><em>勾选下方复选框即表示您同意将匿名化的健康数据用于医学研究目的。</em></p>
                </>
              )}
            </div>
            {/* AI建议选项 */}
            <div className="ai-advice-section">
              <h4 style={{ margin: '1rem 0 0.5rem 0', color: '#2c3e50' }}>
                🤖 {language === 'en' ? 'AI Health Advice' : 'AI健康建议'}
              </h4>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={useAIAdvice}
                  onChange={(e) => setUseAIAdvice(e.target.checked)}
                />
                <span className="checkmark"></span>
                {language === 'en' ? 
                  'Generate personalized health advice using AI (OpenAI GPT)' : 
                  '使用AI生成个性化健康建议 (OpenAI GPT)'}
              </label>
              <p style={{ 
                fontSize: '0.85rem', 
                color: '#666', 
                margin: '0.3rem 0 1rem 1.5rem',
                lineHeight: '1.4'
              }}>
                {language === 'en' ? 
                  '💡 AI will analyze your health data and risk factors to provide tailored advice. If disabled, standard recommendations will be used.' : 
                  '💡 AI将分析您的健康数据和风险因素，提供个性化建议。如禁用，将使用标准建议。'}
              </p>
            </div>

            <div className="agreement-checkbox">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={userAgreement}
                  onChange={(e) => setUserAgreement(e.target.checked)}
                />
                <span className="checkmark"></span>
                {language === 'en' ? 
                  'I agree to allow my anonymized health data to be used for medical research' : 
                  '我同意将我的匿名化健康数据用于医学研究'}
              </label>
            </div>
          </div>
        </div>

        <div className="form-actions">
          <button type="submit" disabled={loading} className="predict-button">
            {loading ? (
              <>
                <span className="loading-spinner"></span>
                {t.prediction.analyzing}
              </>
            ) : (
              <>
                {t.prediction.submit}
              </>
            )}
          </button>
          
          {loading && (
            <p className="form-note">
              {t.prediction.note}
            </p>
          )}
        </div>
      </form>

      {/* 预测结果展示 */}
      {predictionResult && (
        <div className="prediction-results">
          <h2>{t.results.title}</h2>
          
          {/* 风险评分 */}
          <div className="risk-scores">
            <h3>
              <span className="section-icon">📊</span>
              {t.results.riskScores.title}
            </h3>
            <p className="analysis-note">
              {t.results.riskScores.note}<br/>
              <span className="risk-level low">{t.results.riskScores.lowRisk}</span><br/>
              <span className="risk-level medium">{t.results.riskScores.mediumRisk}</span><br/>
              <span className="risk-level high">{t.results.riskScores.highRisk}</span>
            </p>
            <div className="risk-grid">
              {Object.entries(predictionResult.predictions || {}).map(([disease, data]) => {
                const riskLevel = data.risk_probability > 0.3 ? 'high' : 
                                data.risk_probability > 0.15 ? 'medium' : 'low';
                const riskColor = riskLevel === 'high' ? '#e74c3c' : 
                                riskLevel === 'medium' ? '#f39c12' : '#2ecc71';
                return (
                  <div key={disease} className="risk-item">
                    <div className="disease-name">
                      {(() => {
                        const icon = {
                          'HYPERTENSION': '🫀',
                          'CHD': '❤️',
                          'STROKE': '🧠',
                          'DEATH': '⚡',
                          'CVD': '💓',
                          'ANGINA': '💔',
                          'MI': '🫁'
                        }[disease] || '';

                        const name = {
                          'CVD': language === 'en' ? 'Cardiovascular Disease' : '心血管疾病',
                          'CHD': language === 'en' ? 'Coronary Heart Disease' : '冠心病',
                          'STROKE': language === 'en' ? 'Stroke' : '卒中',
                          'ANGINA': language === 'en' ? 'Angina' : '心绞痛',
                          'MI': language === 'en' ? 'Myocardial Infarction' : '心肌梗死',
                          'HYPERTENSION': language === 'en' ? 'Hypertension' : '高血压',
                          'DEATH': language === 'en' ? 'Death Risk' : '死亡风险'
                        }[disease] || '';

                        return `${icon} ${name}`;
                      })()}
                    </div>
                    <div className="risk-bar-container">
                      <div 
                        className="risk-bar" 
                        style={{
                          width: `${data.risk_probability * 100}%`,
                          backgroundColor: riskColor
                        }}
                      />
                    </div>
                    <div className="risk-details">
                      <div className="risk-value" style={{ color: riskColor }}>
                        {(data.risk_probability * 100).toFixed(1)}%
                      </div>
                      <div className={`risk-level-badge ${riskLevel}`}>
                        {riskLevel === 'high' ? t.results.riskScores.highRisk.split(' - ')[0] : 
                         riskLevel === 'medium' ? t.results.riskScores.mediumRisk.split(' - ')[0] : 
                         t.results.riskScores.lowRisk.split(' - ')[0]}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 生存分析结果 */}
          {survivalResult?.survival_predictions && (
            <div className="survival-analysis">
              <h3>{t.results.survival.title}</h3>
              <p className="analysis-note">
                {t.results.survival.note}
              </p>
              <div className="survival-grid">
                {Object.entries(survivalResult.survival_predictions).map(([disease, data]) => (
                  <div key={disease} className="survival-item">
                    <h4>
                      {(() => {
                        const name = {
                          'CVD': language === 'en' ? 'Cardiovascular Disease' : '心血管疾病',
                          'CHD': language === 'en' ? 'Coronary Heart Disease' : '冠心病',
                          'STROKE': language === 'en' ? 'Stroke' : '卒中',
                          'ANGINA': language === 'en' ? 'Angina' : '心绞痛',
                          'MI': language === 'en' ? 'Myocardial Infarction' : '心肌梗死',
                          'HYPERTENSION': language === 'en' ? 'Hypertension' : '高血压',
                          'DEATH': language === 'en' ? 'Death Risk' : '死亡风险'
                        }[disease] || '';
                        return name;
                      })()}
                    </h4>
                    <div className="survival-details">
                      <div className="survival-probs">
                        {data.survival_probabilities.map(prob => (
                          <div key={prob.years} className="prob-item">
                            <span className="prob-label">{prob.years}{language === 'en' ? ' years survival rate:' : '年生存率:'}</span>
                            <span className="prob-value" style={{
                              color: prob.survival_probability > 0.9 ? '#4CAF50' :
                                     prob.survival_probability > 0.7 ? '#FF9800' : '#F44336'
                            }}>
                              {(prob.survival_probability * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* AI健康建议 */}
          {predictionResult?.ai_advice && (
            <div className="health-advice">
              <h3>
                {predictionResult.ai_advice.generated_by === 'ai' ? '🤖' : '💡'} 
                {t.results.advice.title}
                {predictionResult.ai_advice.generated_by === 'ai' && (
                  <span style={{ 
                    fontSize: '0.8rem', 
                    color: '#667eea', 
                    marginLeft: '0.5rem',
                    fontWeight: 'normal'
                  }}>
                    {language === 'en' ? '(AI Generated)' : '(AI生成)'}
                  </span>
                )}
              </h3>
              <div className="advice-content">
                <div className="advice-text" style={{
                  whiteSpace: 'pre-line',
                  backgroundColor: predictionResult.ai_advice.generated_by === 'ai' ? '#f8f9ff' : '#f8fafc',
                  border: predictionResult.ai_advice.generated_by === 'ai' ? '1px solid #e0e6ff' : '1px solid #e2e8f0',
                  borderLeft: `4px solid ${predictionResult.ai_advice.generated_by === 'ai' ? '#667eea' : '#4a5568'}`
                }}>
                  {predictionResult.ai_advice.content}
                </div>
                {predictionResult.ai_advice.generated_by === 'ai' && (
                  <p style={{ 
                    fontSize: '0.75rem', 
                    color: '#666', 
                    margin: '0.5rem 0 0 0',
                    fontStyle: 'italic'
                  }}>
                    {language === 'en' ? 
                      '⚠️ This advice is generated by AI and should not replace professional medical consultation.' : 
                      '⚠️ 此建议由AI生成，不应替代专业医疗咨询。'}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* 传统建议（当没有AI建议时） */}
          {predictionResult?.predictions && !predictionResult?.ai_advice && (
            <div className="health-advice">
              <h3>{t.results.advice.title}</h3>
              <p className="advice-text">
                {(() => {
                  const advice = [];
                  const predictions = predictionResult.predictions;
                  
                  if (predictions.HYPERTENSION?.risk_probability > 0.3) {
                    advice.push(t.results.advice.hypertension);
                  }
                  if (predictions.CHD?.risk_probability > 0.2) {
                    advice.push(t.results.advice.heartDisease);
                  }
                  if (predictions.STROKE?.risk_probability > 0.15) {
                    advice.push(t.results.advice.stroke);
                  }
                  if (predictions.DEATH?.risk_probability > 0.2) {
                    advice.push(t.results.advice.death);
                  }
                  return advice.length > 0 ? advice.join('，') + '。' : t.results.advice.default;
                })()}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );

  // 主界面
  if (!user) {
    return (
      <div className="app">
        <header className="app-header">
          <h1>{t.header.title}</h1>
          <p>{t.header.subtitle}</p>
        </header>
        {renderAuthForm()}
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>{t.header.title}</h1>
        <div className="user-info">
          <span>{t.header.welcome} {user.username}</span>
          <button className="btn btn-secondary" onClick={handleLogout}>{t.header.logout}</button>
        </div>
      </header>
      
      <nav className="app-nav">
        <button 
          className={`nav-btn ${currentView === 'prediction' ? 'active' : ''}`}
          onClick={() => setCurrentView('prediction')}
        >
          🎯 {language === 'en' ? 'Risk Assessment' : '风险评估'}
        </button>
        <button 
          className={`nav-btn ${currentView === 'history' ? 'active' : ''}`}
          onClick={() => {
            setCurrentView('history');
            fetchHistory();
          }}
        >
          📋 {language === 'en' ? 'History' : '历史记录'}
        </button>
      </nav>
      
      <main className="app-main">
        {currentView === 'prediction' && (
          <>
            {renderPredictionForm()}
          </>
        )}
        
        {currentView === 'history' && (
          <div className="history-container">
            <h2>
              <span className="section-icon">📋</span>
              {language === 'en' ? 'Assessment History' : '历史评估记录'}
            </h2>

            <div className="admin-controls">
              <div className="admin-controls-row">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder={translations[language].history.searchPlaceholder}
                  className="search-input"
                />
                <button 
                  className="btn-search" 
                  onClick={handleSearch}
                  disabled={loading}
                >
                  {translations[language].history.search}
                </button>
                <button 
                  className="btn-clear-search" 
                  onClick={() => setSearchQuery('')}
                  disabled={loading}
                >
                  {translations[language].history.clearSearch}
                </button>
              </div>

              <div className="admin-controls-row">
                <button 
                  className="btn-trend" 
                  onClick={showTrendAnalysis} 
                  disabled={loading || selectedRecords.length < 2}
                  style={{
                    backgroundColor: selectedRecords.length < 2 ? '#ccc' : '#4CAF50',
                    cursor: selectedRecords.length < 2 ? 'not-allowed' : 'pointer'
                  }}
                >
                  {translations[language].history.trendAnalysis} ({selectedRecords.length} {translations[language].history.recordsCount})
                </button>
              </div>
            </div>

            {showTrendChart && (
              <div className="trend-chart-modal">
                <div className="trend-chart-content">
                  <div className="trend-chart-header">
                    <h3>{translations[language].history.trendChart}</h3>
                    <button className="btn-close-trend" onClick={closeTrendChart}>
                      ×
                    </button>
                  </div>
                  <div className="trend-chart-container">
                    <Line 
                      data={{
                        labels: history
                          .filter(record => selectedRecords.includes(record.id))
                          .map(record => new Date(record.created_at).toLocaleDateString()),
                                              datasets: [
                        {
                          label: language === 'en' ? 'Systolic BP (mmHg)' : '收缩压 (mmHg)',
                          data: history
                            .filter(record => selectedRecords.includes(record.id))
                            .map(record => record.sysbp || 0),
                          borderColor: 'rgb(255, 99, 132)',
                          backgroundColor: 'rgba(255, 99, 132, 0.2)',
                          yAxisID: 'y',
                        },
                        {
                          label: language === 'en' ? 'Diastolic BP (mmHg)' : '舒张压 (mmHg)',
                          data: history
                            .filter(record => selectedRecords.includes(record.id))
                            .map(record => record.diabp || 0),
                          borderColor: 'rgb(255, 159, 64)',
                          backgroundColor: 'rgba(255, 159, 64, 0.2)',
                          yAxisID: 'y',
                        },
                        {
                          label: language === 'en' ? 'Total Cholesterol (mg/dL)' : '总胆固醇 (mg/dL)',
                          data: history
                            .filter(record => selectedRecords.includes(record.id))
                            .map(record => record.totchol || 0),
                          borderColor: 'rgb(54, 162, 235)',
                          backgroundColor: 'rgba(54, 162, 235, 0.2)',
                          yAxisID: 'y1',
                        },
                        {
                          label: language === 'en' ? 'Fasting Glucose (mg/dL)' : '空腹血糖 (mg/dL)',
                          data: history
                            .filter(record => selectedRecords.includes(record.id))
                            .map(record => record.glucose || 0),
                          borderColor: 'rgb(153, 102, 255)',
                          backgroundColor: 'rgba(153, 102, 255, 0.2)',
                          yAxisID: 'y1',
                        },
                        {
                          label: 'BMI',
                          data: history
                            .filter(record => selectedRecords.includes(record.id))
                            .map(record => record.bmi || 0),
                          borderColor: 'rgb(255, 205, 86)',
                          backgroundColor: 'rgba(255, 205, 86, 0.2)',
                          yAxisID: 'y2',
                        },
                        {
                          label: language === 'en' ? 'Heart Rate (bpm)' : '心率 (次/分钟)',
                          data: history
                            .filter(record => selectedRecords.includes(record.id))
                            .map(record => record.heartrte || 0),
                          borderColor: 'rgb(75, 192, 192)',
                          backgroundColor: 'rgba(75, 192, 192, 0.2)',
                          yAxisID: 'y2',
                        },
                        {
                          label: language === 'en' ? 'Age (years)' : '年龄 (岁)',
                          data: history
                            .filter(record => selectedRecords.includes(record.id))
                            .map(record => record.age || 0),
                          borderColor: 'rgb(199, 199, 199)',
                          backgroundColor: 'rgba(199, 199, 199, 0.2)',
                          yAxisID: 'y2',
                        },
                      ],
                      }}
                                          options={{
                      responsive: true,
                      interaction: {
                        mode: 'index' as const,
                        intersect: false,
                      },
                      plugins: {
                        legend: {
                          position: 'top' as const,
                        },
                        title: {
                          display: true,
                          text: language === 'en' ? 'Health Indicators Trend Chart' : '健康指标趋势图',
                          font: {
                            size: 16
                          }
                        },
                        tooltip: {
                          callbacks: {
                            title: function(context: any) {
                              return language === 'en' ? `Date: ${context[0].label}` : `日期: ${context[0].label}`;
                            }
                          }
                        }
                      },
                      scales: {
                        x: {
                          display: true,
                          title: {
                            display: true,
                            text: language === 'en' ? 'Date' : '日期'
                          }
                        },
                        y: {
                          type: 'linear' as const,
                          display: true,
                          position: 'left' as const,
                          title: {
                            display: true,
                            text: language === 'en' ? 'Blood Pressure (mmHg)' : '血压 (mmHg)'
                          },
                        },
                        y1: {
                          type: 'linear' as const,
                          display: true,
                          position: 'right' as const,
                          title: {
                            display: true,
                            text: language === 'en' ? 'Cholesterol/Glucose (mg/dL)' : '胆固醇/血糖 (mg/dL)'
                          },
                          grid: {
                            drawOnChartArea: false,
                          },
                        },
                        y2: {
                          type: 'linear' as const,
                          display: true,
                          position: 'right' as const,
                          title: {
                            display: true,
                            text: language === 'en' ? 'BMI/Heart Rate/Age' : 'BMI/心率/年龄'
                          },
                          grid: {
                            drawOnChartArea: false,
                          },
                          offset: true,
                        },
                      },
                    }}
                    />
                  </div>
                </div>
              </div>
            )}

            {selectedRecords.length > 0 && (
              <div className="bulk-actions">
                <span className="selected-count">
                  {language === 'en' ? 
                    `${selectedRecords.length} records selected` : 
                    `已选择 ${selectedRecords.length} 条记录`}
                </span>
                <button 
                  className="btn btn-danger" 
                  onClick={deleteSelectedRecords}
                  disabled={loading}
                >
                  {language === 'en' ? 'Delete Selected' : '删除选中'}
                </button>
                <button 
                  className="btn btn-primary" 
                  onClick={exportSelectedRecords}
                  disabled={loading}
                >
                  {language === 'en' ? 'Export Selected' : '导出选中'}
                </button>
              </div>
            )}
            
            {loading ? (
              <div className="loading-container">
                <div className="loading-spinner"></div>
                <p>{language === 'en' ? 'Loading...' : '加载中...'}</p>
              </div>
            ) : history.length === 0 ? (
              <div className="empty-state">
                <p>{language === 'en' ? 'No assessment records found' : '未找到评估记录'}</p>
              </div>
            ) : (
              <div className="history-list">
                <div className="history-header bulk-select">
                  <label className="checkbox-container">
                    <input
                      type="checkbox"
                      checked={selectAll}
                      onChange={toggleSelectAll}
                    />
                    <span className="checkmark"></span>
                  </label>
                  <span className="select-all-text">
                    {language === 'en' ? 'Select All' : '全选'}
                  </span>
                </div>
                {history.map((record) => (
                  <div key={record.id} className="history-item">
                    <div className="history-header">
                      <div className="history-header-left">
                        <label className="checkbox-container">
                          <input
                            type="checkbox"
                            checked={selectedRecords.includes(record.id)}
                            onChange={() => toggleRecordSelection(record.id)}
                          />
                          <span className="checkmark"></span>
                        </label>
                        <div className="history-date">
                          {formatDate(record.created_at)}
                        </div>
                        {isAdmin && record.username && (
                          <div className="history-username">
                            {language === 'en' ? 'User: ' : '用户: '}{record.username}
                          </div>
                        )}
                      </div>
                      <div className="history-id">
                        #{record.id}
                      </div>
                    </div>
                    
                    <div className="history-content">
                      <div className="history-section">
                        <h4>{language === 'en' ? 'Basic Information' : '基本信息'}</h4>
                        <div className="history-grid">
                          <div className="history-item">
                            <span className="label">{language === 'en' ? 'Age' : '年龄'}:</span>
                            <span className="value">{record.age}</span>
                          </div>
                          <div className="history-item">
                            <span className="label">{language === 'en' ? 'Gender' : '性别'}:</span>
                            <span className="value">{record.sex === 1 ? (language === 'en' ? 'Male' : '男') : (language === 'en' ? 'Female' : '女')}</span>
                          </div>
                          <div className="history-item">
                            <span className="label">BMI:</span>
                            <span className="value">{formatValue(record.bmi)}</span>
                          </div>
                        </div>
                      </div>

                      <div className="history-section">
                        <h4>{language === 'en' ? 'Cardiovascular Indicators' : '心血管指标'}</h4>
                        <div className="history-grid">
                          <div className="history-item">
                            <span className="label">{language === 'en' ? 'Blood Pressure' : '血压'}:</span>
                            <span className="value">{formatValue(record.sysbp)}/{formatValue(record.diabp)} mmHg</span>
                          </div>
                          <div className="history-item">
                            <span className="label">{language === 'en' ? 'Heart Rate' : '心率'}:</span>
                            <span className="value">{formatValue(record.heartrte)} bpm</span>
                          </div>
                        </div>
                      </div>

                      <div className="history-section">
                        <h4>{language === 'en' ? 'Biochemical Indicators' : '生化指标'}</h4>
                        <div className="history-grid">
                          <div className="history-item">
                            <span className="label">{language === 'en' ? 'Total Cholesterol' : '总胆固醇'}:</span>
                            <span className="value">{formatValue(record.totchol)} mg/dL</span>
                          </div>
                          <div className="history-item">
                            <span className="label">{language === 'en' ? 'Fasting Glucose' : '空腹血糖'}:</span>
                            <span className="value">{formatValue(record.glucose)} mg/dL</span>
                          </div>
                        </div>
                      </div>

                      <div className="history-section">
                        <h4>{language === 'en' ? 'Lifestyle & Medical History' : '生活习惯与病史'}</h4>
                        <div className="history-grid">
                          <div className="history-item">
                            <span className="label">{language === 'en' ? 'Smoking Status' : '吸烟状态'}:</span>
                            <span className="value">
                              {record.cursmoke === 1 
                                ? `${language === 'en' ? 'Smoking' : '吸烟'} (${record.cigpday} ${language === 'en' ? 'cigarettes/day' : '支/天'})`
                                : language === 'en' ? 'Non-smoker' : '不吸烟'}
                            </span>
                          </div>
                          <div className="history-item">
                            <span className="label">{language === 'en' ? 'Diabetes' : '糖尿病'}:</span>
                            <span className="value">{record.diabetes === 1 ? (language === 'en' ? 'Yes' : '是') : (language === 'en' ? 'No' : '否')}</span>
                          </div>
                          <div className="history-item">
                            <span className="label">{language === 'en' ? 'BP Medication' : '降压药'}:</span>
                            <span className="value">{record.bpmeds === 1 ? (language === 'en' ? 'Using' : '使用中') : (language === 'en' ? 'Not using' : '未使用')}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
