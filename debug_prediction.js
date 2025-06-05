const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
app.use(express.json());

app.post('/test-predict', async (req, res) => {
  console.log('📥 收到预测请求:', req.body);
  
  try {
    const inputData = {
      sex: req.body.sex || 1,
      age: req.body.age || 45,
      totchol: req.body.totchol || 200,
      sysbp: req.body.sysbp || 120,
      diabp: req.body.diabp || 80,
      cursmoke: req.body.cursmoke || 0,
      cigpday: req.body.cigpday || 0,
      bmi: req.body.bmi || 25,
      diabetes: req.body.diabetes || 0,
      bpmeds: req.body.bpmeds || 0,
      heartrte: req.body.heartrte || 70,
      glucose: req.body.glucose || 90
    };

    const tempInputFile = path.resolve(__dirname, 'debug_temp_input.json');
    fs.writeFileSync(tempInputFile, JSON.stringify(inputData));
    console.log('📝 临时文件已创建:', tempInputFile);

    const pythonScript = path.resolve(__dirname, 'ml_analysis/prediction_service.py');
    console.log('🐍 Python脚本路径:', pythonScript);
    
    const pythonProcess = spawn('python3', [pythonScript, tempInputFile], {
      cwd: path.resolve(__dirname, 'ml_analysis')
    });

    let output = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
      console.log('📤 Python输出:', data.toString());
    });

    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
      console.error('❌ Python错误:', data.toString());
    });

    pythonProcess.on('close', (code) => {
      console.log('🏁 Python进程结束，代码:', code);
      console.log('📊 完整输出:', output);
      console.log('⚠️ 错误输出:', errorOutput);
      
      // 清理临时文件
      try {
        fs.unlinkSync(tempInputFile);
        console.log('🗑️ 临时文件已清理');
      } catch (e) {
        console.warn('⚠️ 清理临时文件失败:', e);
      }

      if (code === 0) {
        try {
          const result = JSON.parse(output.trim());
          console.log('✅ JSON解析成功');
          res.json({
            success: true,
            predictions: result,
            message: '预测成功'
          });
        } catch (e) {
          console.error('❌ JSON解析失败:', e.message);
          res.status(500).json({
            success: false,
            message: `JSON解析失败: ${output}`
          });
        }
      } else {
        console.error('❌ Python脚本执行失败');
        res.status(500).json({
          success: false,
          message: `Python脚本执行失败 (代码 ${code}): ${errorOutput}`
        });
      }
    });

    pythonProcess.on('error', (error) => {
      console.error('💥 Python进程启动失败:', error);
      res.status(500).json({
        success: false,
        message: `启动Python进程失败: ${error.message}`
      });
    });

  } catch (error) {
    console.error('💥 预测过程出错:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

const PORT = 3001;
app.listen(PORT, () => {
  console.log(`🚀 调试服务器启动在 http://localhost:${PORT}`);
  console.log('📝 测试命令: curl -X POST -H "Content-Type: application/json" -d \'{"sex":1,"age":45}\' http://localhost:3001/test-predict');
}); 