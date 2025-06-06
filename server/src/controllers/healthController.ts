import { Request, Response } from 'express';
import { aiAdviceService } from '../services/aiAdviceService';

export const getHealthStatus = (req: Request, res: Response) => {
  res.json({
    status: 'ok',
    message: '医疗预测系统运行正常',
    timestamp: new Date().toISOString(),
    features: {
      risk_prediction: true,
      survival_analysis: true
    }
  });
};

export const getAIStatus = (req: Request, res: Response) => {
  res.json({
    status: 'ok',
    ai_enabled: aiAdviceService.getIsEnabled(),
    message: aiAdviceService.getIsEnabled() ? 
      'OpenAI API已配置，AI建议功能可用' : 
      'OpenAI API未配置，使用默认建议',
    timestamp: new Date().toISOString()
  });
}; 