const fetch = require('node-fetch');

async function testAPI() {
  try {
    console.log('测试API连接到: http://localhost:5000/api/health');
    const response = await fetch('http://localhost:5000/api/health');
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('✅ API连接成功!');
    console.log('响应数据:', JSON.stringify(data, null, 2));
  } catch (error) {
    console.error('❌ API连接失败:', error.message);
  }
}

testAPI(); 