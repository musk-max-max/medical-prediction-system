const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// 创建测试输入
const testData = {
  sex: 1,
  age: 45,
  totchol: 200,
  sysbp: 120,
  diabp: 80,
  cursmoke: 0,
  cigpday: 0,
  bmi: 25,
  diabetes: 0,
  bpmeds: 0,
  heartrte: 75,
  glucose: 100
};

const tempFile = path.resolve(__dirname, 'test_node_input.json');
fs.writeFileSync(tempFile, JSON.stringify(testData));

console.log('🔍 测试Node.js调用Python脚本...');

const pythonScript = path.resolve(__dirname, 'ml_analysis/prediction_service.py');
console.log(`Python脚本路径: ${pythonScript}`);
console.log(`临时文件路径: ${tempFile}`);
console.log(`工作目录: ${path.resolve(__dirname, 'ml_analysis')}`);

const pythonProcess = spawn('python3', [pythonScript, tempFile], {
  cwd: path.resolve(__dirname, 'ml_analysis')
});

let output = '';
let errorOutput = '';

pythonProcess.stdout.on('data', (data) => {
  output += data.toString();
  console.log('stdout:', data.toString());
});

pythonProcess.stderr.on('data', (data) => {
  errorOutput += data.toString();
  console.error('stderr:', data.toString());
});

pythonProcess.on('close', (code) => {
  console.log(`进程退出代码: ${code}`);
  console.log(`完整输出: ${output}`);
  console.log(`错误输出: ${errorOutput}`);
  
  // 清理文件
  fs.unlinkSync(tempFile);
  
  if (code === 0) {
    try {
      const result = JSON.parse(output.trim());
      console.log('✅ 解析成功:', result);
    } catch (e) {
      console.error('❌ JSON解析失败:', e.message);
    }
  } else {
    console.error('❌ Python脚本执行失败');
  }
});

pythonProcess.on('error', (error) => {
  console.error('❌ 进程启动失败:', error);
}); 