import { Response } from 'express';
import { body, validationResult } from 'express-validator';
import { database } from '../config/database';
import { AuthRequest } from '../utils/auth';
import { HealthData } from '../types';

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

// 简化的风险计算函数
function calculateRisks(healthData: HealthData): { [disease: string]: number } {
  const age = healthData.age;
  const sex = healthData.sex; // 1=男性, 0=女性
  const sysbp = healthData.sysbp || 120;
  const totchol = healthData.totchol || 200;
  const smoking = healthData.cursmoke || 0;
  const diabetes = healthData.diabetes || 0;
  const bmi = healthData.bmi || 25;

  // 基础风险评分
  let baseRisk = 0;
  
  // 年龄因子
  if (age >= 65) baseRisk += 0.15;
  else if (age >= 55) baseRisk += 0.10;
  else if (age >= 45) baseRisk += 0.05;
  
  // 性别因子
  if (sex === 1) baseRisk += 0.05; // 男性风险更高
  
  // 血压因子
  if (sysbp >= 140) baseRisk += 0.10;
  else if (sysbp >= 130) baseRisk += 0.05;
  
  // 胆固醇因子
  if (totchol >= 240) baseRisk += 0.08;
  else if (totchol >= 200) baseRisk += 0.04;
  
  // 吸烟因子
  if (smoking === 1) baseRisk += 0.12;
  
  // 糖尿病因子
  if (diabetes === 1) baseRisk += 0.15;
  
  // BMI因子
  if (bmi >= 30) baseRisk += 0.08;
  else if (bmi >= 25) baseRisk += 0.04;

  // 为不同疾病计算特定风险
  const risks = {
    CVD: Math.min(baseRisk * 1.0 + Math.random() * 0.1, 0.8),
    CHD: Math.min(baseRisk * 0.9 + Math.random() * 0.1, 0.7),
    STROKE: Math.min(baseRisk * 0.6 + Math.random() * 0.05, 0.5),
    ANGINA: Math.min(baseRisk * 0.8 + Math.random() * 0.1, 0.6),
    MI: Math.min(baseRisk * 0.7 + Math.random() * 0.08, 0.6),
    HYPERTENSION: Math.min((sysbp >= 130 ? 0.6 : 0.2) + baseRisk * 0.5 + Math.random() * 0.1, 0.9),
    DEATH: Math.min(baseRisk * 0.8 + Math.random() * 0.1, 0.7)
  };

  return risks;
}

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

    const userId = req.user?.userId;
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

    console.log('📊 开始预测，用户ID:', userId);

    // 1. 保存健康数据
    const healthDataId = await saveHealthData(userId, healthData);
    console.log('💾 健康数据已保存，ID:', healthDataId);

    // 2. 计算风险
    const riskProbabilities = calculateRisks(healthData);
    console.log('🎯 风险计算完成:', riskProbabilities);

    // 3. 处理预测结果
    const predictions: any = {};
    let highRiskDiseases: string[] = [];
    let totalRiskScore = 0;

    for (const [disease, probability] of Object.entries(riskProbabilities)) {
      const riskLevel = getRiskLevel(probability, disease);
      const diseaseInfo = DISEASE_INFO[disease as keyof typeof DISEASE_INFO];

      if (diseaseInfo) {
        predictions[disease] = {
          name: diseaseInfo.name,
          risk_probability: probability,
          risk_level: riskLevel,
          description: diseaseInfo.description,
          recommendations: getRecommendations(riskLevel, disease)
        };

        if (riskLevel === 'High') {
          highRiskDiseases.push(diseaseInfo.name);
        }

        totalRiskScore += probability;
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

    console.log('✅ 预测完成');

    // 5. 返回结果
    res.status(200).json({
      success: true,
      message: '预测完成',
      predictions,
      overall_risk: overallRisk,
      health_data_id: healthDataId
    });

  } catch (error: any) {
    console.error('❌ 预测错误:', error);
    res.status(500).json({ 
      success: false,
      message: error.message || '预测服务暂时不可用，请稍后重试'
    });
  }
};

// 获取用户历史预测记录
export const getPredictionHistory = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const userId = req.user?.userId;
    const isAdmin = req.user?.isAdmin; // 使用 isAdmin 字段
    const { search } = req.query as { search?: string };

    if (!userId) {
      res.status(401).json({ error: '用户未认证' });
      return;
    }

    const db = database.getDatabase();

    let query = '';
    let params: any[] = [];

    if (isAdmin) {
      // 管理员可以看到所有记录
      if (search) {
        // 如果有搜索条件，按用户名搜索
        query = `
          SELECT uhd.*, u.username, u.email 
          FROM user_health_data uhd
          JOIN users u ON uhd.user_id = u.id
          WHERE u.username LIKE ?
          ORDER BY uhd.created_at DESC
        `;
        params = [`%${search}%`];
      } else {
        // 管理员默认显示所有记录
        query = `
          SELECT uhd.*, u.username, u.email 
          FROM user_health_data uhd
          JOIN users u ON uhd.user_id = u.id
          ORDER BY uhd.created_at DESC
        `;
        params = [];
      }
    } else {
      // 普通用户只能看自己的记录
      query = `
        SELECT uhd.*, u.username, u.email 
        FROM user_health_data uhd
        JOIN users u ON uhd.user_id = u.id
        WHERE uhd.user_id = ?
        ORDER BY uhd.created_at DESC
        LIMIT 10
      `;
      params = [userId];
    }

    const records = await new Promise<any[]>((resolve, reject) => {
      db.all(query, params, (err: any, rows: any[]) => {
        if (err) reject(err);
        else resolve(rows || []);
      });
    });

    res.status(200).json({
      success: true,
      data: records
    });

  } catch (error) {
    console.error('获取历史记录错误:', error);
    res.status(500).json({ error: '获取历史记录失败' });
  }
}; 