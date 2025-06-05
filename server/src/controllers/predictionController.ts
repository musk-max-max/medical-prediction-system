import { Response } from 'express';
import { body, validationResult } from 'express-validator';
import { spawn } from 'child_process';
import { database } from '../config/database';
import { AuthRequest } from '../types';
import { HealthData, PredictionResult } from '../types';
import path from 'path';
import fs from 'fs';

export const predictionValidation = [
  body('sex').isInt({ min: 0, max: 1 }).withMessage('性别必须是0(女)或1(男)'),
  body('age').isInt({ min: 18, max: 120 }).withMessage('年龄必须在18-120之间'),
  body('totchol').optional().isFloat({ min: 50, max: 1000 }).withMessage('总胆固醇范围50-1000'),
  body('sysbp').optional().isFloat({ min: 70, max: 300 }).withMessage('收缩压范围70-300'),
  body('diabp').optional().isFloat({ min: 40, max: 200 }).withMessage('舒张压范围40-200'),
  body('cursmoke').optional().isInt({ min: 0, max: 1 }).withMessage('吸烟状态必须是0或1'),
  body('cigpday').optional().isFloat({ min: 0, max: 100 }).withMessage('每日吸烟量范围0-100'),
  body('bmi').optional().isFloat({ min: 10, max: 60 }).withMessage('BMI范围10-60'),
  body('diabetes').optional().isInt({ min: 0, max: 1 }).withMessage('糖尿病状态必须是0或1'),
  body('bpmeds').optional().isInt({ min: 0, max: 1 }).withMessage('降压药使用必须是0或1'),
  body('heartrte').optional().isFloat({ min: 30, max: 250 }).withMessage('心率范围30-250'),
  body('glucose').optional().isFloat({ min: 30, max: 500 }).withMessage('血糖范围30-500')
];

interface PredictionResponse {
  success: boolean;
  message?: string;
  predictions?: {
    [disease: string]: {
      name: string;
      risk_probability: number;
      risk_level: string;
      description: string;
      recommendations: string[];
    };
  };
  overall_risk?: {
    high_risk_diseases: string[];
    total_risk_score: number;
    risk_category: string;
  };
  health_data_id?: number;
}

// 疾病信息配置
const DISEASE_INFO = {
  'CVD': {
    name: '心血管疾病',
    description: '包括心肌梗死、致命性冠心病、卒中等',
    high_risk_threshold: 0.3,
    recommendations: {
      low: ['保持健康生活方式', '定期体检', '控制体重'],
      medium: ['加强心血管监测', '控制血压血脂', '戒烟限酒', '增加运动'],
      high: ['立即就医咨询', '严格控制危险因素', '遵医嘱用药', '密切监测']
    }
  },
  'CHD': {
    name: '冠心病',
    description: '包括心绞痛、心肌梗死、冠状动脉功能不全',
    high_risk_threshold: 0.25,
    recommendations: {
      low: ['均衡饮食', '适量运动', '避免过度劳累'],
      medium: ['控制胆固醇', '监测血压', '减少饱和脂肪摄入'],
      high: ['心脏专科就诊', '考虑冠脉造影', '抗血小板治疗']
    }
  },
  'STROKE': {
    name: '卒中',
    description: '包括脑梗死、脑出血等脑血管疾病',
    high_risk_threshold: 0.2,
    recommendations: {
      low: ['控制血压', '健康饮食', '适量运动'],
      medium: ['定期神经科检查', '控制血脂血糖', '戒烟'],
      high: ['神经科专科就诊', '头颅影像检查', '抗凝治疗评估']
    }
  },
  'ANGINA': {
    name: '心绞痛',
    description: '胸痛或胸部不适，通常由心肌缺血引起',
    high_risk_threshold: 0.3,
    recommendations: {
      low: ['避免剧烈运动', '保持心情愉快', '充足睡眠'],
      medium: ['心电图检查', '运动负荷试验', '硝酸甘油备用'],
      high: ['心内科就诊', '冠脉CT或造影', '药物治疗']
    }
  },
  'MI': {
    name: '心肌梗死',
    description: '心肌细胞坏死，严重危及生命',
    high_risk_threshold: 0.2,
    recommendations: {
      low: ['健康生活方式', '定期心电图', '控制危险因素'],
      medium: ['心脏彩超检查', '运动耐量评估', '药物预防'],
      high: ['立即心内科就诊', '急诊绿色通道', '介入治疗准备']
    }
  },
  'HYPERTENSION': {
    name: '高血压',
    description: '血压持续升高的慢性疾病',
    high_risk_threshold: 0.4,
    recommendations: {
      low: ['低盐饮食', '适量运动', '控制体重'],
      medium: ['家庭血压监测', '限制钠盐', '增加钾摄入'],
      high: ['降压药物治疗', '靶器官保护', '血压达标管理']
    }
  },
  'DEATH': {
    name: '死亡风险',
    description: '24年内死亡的综合风险评估',
    high_risk_threshold: 0.5,
    recommendations: {
      low: ['健康生活方式', '定期健康体检', '疾病预防'],
      medium: ['加强健康管理', '慢病筛查', '生活方式干预'],
      high: ['全面健康评估', '多学科会诊', '积极治疗']
    }
  }
};

// 获取风险等级
function getRiskLevel(probability: number, disease: string): string {
  const threshold = DISEASE_INFO[disease as keyof typeof DISEASE_INFO]?.high_risk_threshold || 0.3;
  
  if (probability >= threshold) return 'High';
  if (probability >= threshold * 0.6) return 'Medium';
  return 'Low';
}

// 获取推荐建议
function getRecommendations(riskLevel: string, disease: string): string[] {
  const diseaseInfo = DISEASE_INFO[disease as keyof typeof DISEASE_INFO];
  if (!diseaseInfo) return ['请咨询医生获取专业建议'];
  
  const level = riskLevel.toLowerCase() as 'low' | 'medium' | 'high';
  return diseaseInfo.recommendations[level] || ['请咨询医生获取专业建议'];
}

// 调用Python预测脚本
function callPythonPredictor(healthData: HealthData): Promise<any> {
  return new Promise((resolve, reject) => {
    let isResolved = false;
    
    // 创建临时输入文件
    const inputData = {
      sex: healthData.sex,
      age: healthData.age,
      totchol: healthData.totchol || 200,
      sysbp: healthData.sysbp || 120,
      diabp: healthData.diabp || 80,
      cursmoke: healthData.cursmoke || 0,
      cigpday: healthData.cigpday || 0,
      bmi: healthData.bmi || 25,
      diabetes: healthData.diabetes || 0,
      bpmeds: healthData.bpmeds || 0,
      heartrte: healthData.heartrte || 70,
      glucose: healthData.glucose || 90
    };

    const tempInputFile = path.resolve(__dirname, '../../temp_input.json');
    
    try {
      fs.writeFileSync(tempInputFile, JSON.stringify(inputData));
    } catch (error: any) {
      reject(new Error(`写入临时文件失败: ${error.message}`));
      return;
    }

    // 调用Python脚本 - 修复路径
    const pythonScript = path.resolve(__dirname, '../../../ml_analysis/prediction_service.py');
    console.log(`调用Python脚本: ${pythonScript}`);
    console.log(`输入文件: ${tempInputFile}`);
    
    // 强制超时机制
    const timeoutHandle = setTimeout(() => {
      if (!isResolved) {
        isResolved = true;
        try {
          pythonProcess.kill('SIGKILL');
        } catch (e) {
          // 忽略kill错误
        }
        
        // 清理临时文件
        try {
          fs.unlinkSync(tempInputFile);
        } catch (e) {
          // 忽略清理错误
        }
        
        reject(new Error('Python脚本执行超时 (15秒)'));
      }
    }, 15000); // 15秒超时

    // 使用虚拟环境中的Python
    const pythonExecutable = path.resolve(__dirname, '../../../.venv/bin/python');
    const pythonProcess = spawn(pythonExecutable, [pythonScript, tempInputFile], {
      cwd: path.resolve(__dirname, '../../../ml_analysis'),
      stdio: ['pipe', 'pipe', 'pipe'],
      detached: false
    });

    let output = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
      console.error('Python stderr:', data.toString());
    });

    pythonProcess.on('close', (code) => {
      if (isResolved) return;
      isResolved = true;
      
      clearTimeout(timeoutHandle);
      
      console.log(`Python进程退出，代码: ${code}`);
      console.log(`输出: ${output}`);
      console.log(`错误输出: ${errorOutput}`);
      
      // 清理临时文件
      try {
        fs.unlinkSync(tempInputFile);
      } catch (e) {
        console.warn('清理临时文件失败:', e);
      }

      if (code === 0) {
        try {
          const result = JSON.parse(output.trim());
          resolve(result);
        } catch (e) {
          reject(new Error(`解析Python输出失败: ${output}`));
        }
      } else {
        reject(new Error(`Python脚本执行失败 (退出代码 ${code}): ${errorOutput}`));
      }
    });

    pythonProcess.on('error', (error) => {
      if (isResolved) return;
      isResolved = true;
      
      clearTimeout(timeoutHandle);
      console.error('Python进程错误:', error);
      
      // 清理临时文件
      try {
        fs.unlinkSync(tempInputFile);
      } catch (e) {
        // 忽略清理错误
      }
      
      reject(new Error(`启动Python进程失败: ${error.message}`));
    });
  });
}

// 保存健康数据到数据库
function saveHealthData(userId: number, healthData: HealthData): Promise<number> {
  return new Promise((resolve, reject) => {
    const db = database.getDatabase();
    
    db.run(
      `INSERT INTO user_health_data 
       (user_id, sex, age, totchol, sysbp, diabp, cursmoke, cigpday, bmi, diabetes, bpmeds, heartrte, glucose) 
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        userId,
        healthData.sex,
        healthData.age,
        healthData.totchol,
        healthData.sysbp,
        healthData.diabp,
        healthData.cursmoke,
        healthData.cigpday,
        healthData.bmi,
        healthData.diabetes,
        healthData.bpmeds,
        healthData.heartrte,
        healthData.glucose
      ],
      function(err: any) {
        if (err) reject(err);
        else resolve(this.lastID);
      }
    );
  });
}

// 保存预测结果到数据库
function savePredictionResults(userId: number, healthDataId: number, predictions: any): Promise<void> {
  return new Promise((resolve, reject) => {
    const db = database.getDatabase();
    
    const promises = Object.entries(predictions).map(([disease, result]: [string, any]) => {
      return new Promise<void>((resolve, reject) => {
        db.run(
          `INSERT INTO predictions 
           (user_id, health_data_id, prediction_result, risk_level, model_version) 
           VALUES (?, ?, ?, ?, ?)`,
          [
            userId,
            healthDataId,
            result.risk_probability,
            result.risk_level,
            `${disease}_v1.0`
          ],
          function(err: any) {
            if (err) reject(err);
            else resolve();
          }
        );
      });
    });

    Promise.all(promises)
      .then(() => resolve())
      .catch(reject);
  });
}

export const predict = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      res.status(400).json({ 
        success: false,
        message: '输入数据验证失败',
        errors: errors.array() 
      });
      return;
    }

      const userId = req.user?.id;
  if (!userId) {
      res.status(401).json({ 
        success: false,
        message: '用户未认证' 
      });
      return;
    }

    const healthData: HealthData = {
      user_id: userId,
      ...req.body
    };

    // 1. 保存健康数据
    const healthDataId = await saveHealthData(userId, healthData);

    // 2. 调用Python预测模型
    const pythonPredictions = await callPythonPredictor(healthData);

    // 3. 处理预测结果
    const predictions: any = {};
    let highRiskDiseases: string[] = [];
    let totalRiskScore = 0;

    for (const [disease, probability] of Object.entries(pythonPredictions)) {
      const prob = probability as number;
      const riskLevel = getRiskLevel(prob, disease);
      const diseaseInfo = DISEASE_INFO[disease as keyof typeof DISEASE_INFO];

      if (diseaseInfo) {
        predictions[disease] = {
          name: diseaseInfo.name,
          risk_probability: prob,
          risk_level: riskLevel,
          description: diseaseInfo.description,
          recommendations: getRecommendations(riskLevel, disease)
        };

        if (riskLevel === 'High') {
          highRiskDiseases.push(diseaseInfo.name);
        }

        totalRiskScore += prob;
      }
    }

    // 4. 计算总体风险
    const avgRiskScore = totalRiskScore / Object.keys(predictions).length;
    let riskCategory = 'Low';
    if (avgRiskScore >= 0.4) riskCategory = 'High';
    else if (avgRiskScore >= 0.25) riskCategory = 'Medium';

    const overallRisk = {
      high_risk_diseases: highRiskDiseases,
      total_risk_score: Math.round(avgRiskScore * 100) / 100,
      risk_category: riskCategory
    };

    // 5. 保存预测结果到数据库
    await savePredictionResults(userId, healthDataId, predictions);

    // 6. 返回结果
    const response: PredictionResponse = {
      success: true,
      message: '预测完成',
      predictions,
      overall_risk: overallRisk,
      health_data_id: healthDataId
    };

    res.status(200).json(response);

  } catch (error: any) {
    console.error('预测错误:', error);
    res.status(500).json({ 
      success: false,
      message: error.message || '预测服务暂时不可用，请稍后重试'
    });
  }
};

// 获取用户历史预测记录
export const getPredictionHistory = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
      const userId = req.user?.id;
  const isAdmin = req.user?.is_admin;
  const { search } = req.query as { search?: string };
    
    if (!userId) {
      res.status(401).json({ error: '用户未认证' });
      return;
    }

    const db = database.getDatabase();

    // 管理员可以查看所有用户记录，普通用户只能查看自己的记录
    let whereClause = '';
    let queryParams: any[] = [];

    if (isAdmin) {
      if (search && search.trim() !== '') {
        // 管理员精准搜索：完全匹配用户名
        whereClause = 'WHERE u.username = ?';
        queryParams = [search.trim()];
      } else {
        // 管理员查看所有记录
        whereClause = '';
        queryParams = [];
      }
    } else {
      // 普通用户只能查看自己的记录
      whereClause = 'WHERE hd.user_id = ?';
      queryParams = [userId];
    }

    // 获取用户的健康数据记录（去重）
    const records = await new Promise<any[]>((resolve, reject) => {
      db.all(`
        SELECT DISTINCT
          hd.id,
          hd.user_id,
          hd.sex,
          hd.age,
          hd.totchol,
          hd.sysbp,
          hd.diabp,
          hd.cursmoke,
          hd.cigpday,
          hd.bmi,
          hd.diabetes,
          hd.bpmeds,
          hd.heartrte,
          hd.glucose,
          hd.created_at,
          u.username,
          u.email,
          (SELECT COUNT(*) FROM predictions WHERE health_data_id = hd.id) as prediction_count
        FROM user_health_data hd
        LEFT JOIN users u ON hd.user_id = u.id
        ${whereClause}
        ORDER BY hd.created_at DESC
        LIMIT ${isAdmin ? 50 : 10}
      `, queryParams, (err: any, rows: any[]) => {
        if (err) reject(err);
        else resolve(rows || []);
      });
    });

    const searchInfo = search && search.trim() !== '' 
      ? ` (精准搜索: "${search.trim()}")`
      : '';

    res.status(200).json({
      success: true,
      data: records,
      isAdmin: isAdmin,
      message: isAdmin 
        ? `已显示所有用户的预测记录${searchInfo}` 
        : '已显示您的预测记录'
    });

  } catch (error) {
    console.error('获取历史记录错误:', error);
    res.status(500).json({ error: '获取历史记录失败' });
  }
};

// 删除预测记录
export const deletePredictionHistory = async (req: AuthRequest, res: Response) => {
  try {
    const { ids } = req.body;
      const userId = req.user?.id;
  const isAdmin = req.user?.is_admin;

    if (!Array.isArray(ids) || ids.length === 0) {
      return res.status(400).json({ error: 'Invalid record IDs' });
    }

    const db = database.getDatabase();

    // 开始事务
    await new Promise<void>((resolve, reject) => {
      db.run('BEGIN TRANSACTION', (err) => {
        if (err) reject(err);
        else resolve();
      });
    });

    try {
      // 如果是管理员，可以删除任何记录
      // 如果是普通用户，只能删除自己的记录
      const whereClause = isAdmin ? 
        `id IN (${ids.join(',')})` : 
        `id IN (${ids.join(',')}) AND user_id = ?`;

      // 删除预测结果
      await new Promise<void>((resolve, reject) => {
        db.run(
          `DELETE FROM predictions WHERE health_data_id IN (${ids.join(',')})`,
          [],
          (err) => {
            if (err) reject(err);
            else resolve();
          }
        );
      });

      // 删除健康数据
      await new Promise<void>((resolve, reject) => {
        db.run(
          `DELETE FROM user_health_data WHERE ${whereClause}`,
          isAdmin ? [] : [userId],
          (err) => {
            if (err) reject(err);
            else resolve();
          }
        );
      });

      // 提交事务
      await new Promise<void>((resolve, reject) => {
        db.run('COMMIT', (err) => {
          if (err) reject(err);
          else resolve();
        });
      });

      res.json({ 
        success: true,
        message: 'Records deleted successfully', 
        count: ids.length 
      });
    } catch (error) {
      // 如果出错，回滚事务
      await new Promise<void>((resolve) => {
        db.run('ROLLBACK', () => resolve());
      });
      throw error;
    }
  } catch (error) {
    console.error('Error deleting prediction history:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to delete records' 
    });
  }
}; 