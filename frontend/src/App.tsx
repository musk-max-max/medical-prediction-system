import React, { useState, useEffect } from 'react';
import './App.css';
import { apiService } from './services/api';

function App() {
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // 测试后端连接
    const checkHealth = async () => {
      try {
        setLoading(true);
        console.log('开始健康检查，API URL:', apiService);
        const response = await apiService.healthCheck();
        console.log('健康检查响应:', response);
        setHealthStatus(response);
        setError(null);
      } catch (err: any) {
        const errorMessage = `无法连接到后端服务器: ${err.message}`;
        setError(errorMessage);
        console.error('健康检查失败:', err);
        console.error('错误详情:', {
          message: err.message,
          stack: err.stack,
          status: err.status
        });
      } finally {
        setLoading(false);
      }
    };

    checkHealth();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>🏥 医疗预测系统</h1>
        
        <div style={{ margin: '20px 0' }}>
          <h2>后端连接状态</h2>
          {loading && <p>检查中...</p>}
          {error && (
            <div style={{ color: '#ff6b6b', background: '#ffe0e0', padding: '10px', borderRadius: '5px' }}>
              ❌ {error}
            </div>
          )}
          {healthStatus && (
            <div style={{ color: '#51cf66', background: '#e0ffe0', padding: '10px', borderRadius: '5px' }}>
              ✅ 后端连接成功！
              <br />
              状态: {healthStatus.status}
              <br />
              消息: {healthStatus.message}
              <br />
              时间: {new Date(healthStatus.timestamp).toLocaleString()}
            </div>
          )}
        </div>

        <div style={{ marginTop: '30px' }}>
          <h3>🚀 部署状态</h3>
          <ul style={{ textAlign: 'left', maxWidth: '500px' }}>
            <li>✅ React 前端已启动</li>
            <li>{healthStatus ? '✅' : '⏳'} 后端 API 连接</li>
            <li>⏳ 准备部署到 Vercel</li>
          </ul>
        </div>

        <div style={{ marginTop: '30px', fontSize: '14px', opacity: 0.8 }}>
          <p>医疗预测系统包含以下功能：</p>
          <ul style={{ textAlign: 'left', maxWidth: '400px' }}>
            <li>用户注册和登录</li>
            <li>健康数据输入</li>
            <li>7种疾病风险预测</li>
            <li>预测历史管理</li>
            <li>管理员功能</li>
          </ul>
        </div>
      </header>
    </div>
  );
}

export default App;
