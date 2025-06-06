import OpenAI from 'openai';
import { HealthData } from '../types';

interface PredictionResult {
  [disease: string]: {
    name: string;
    risk_probability: number;
    risk_level: string;
    description: string;
    recommendations: string[];
  };
}

interface AIAdviceOptions {
  language: 'en' | 'zh';
  includePersonalization: boolean;
  focusAreas?: string[];
}

class AIAdviceService {
  private openai: OpenAI | null = null;
  private isEnabled: boolean = false;

  // 疾病名称映射到英文
  private getDiseaseNameInEnglish(disease: string): string {
    const diseaseMap: {[key: string]: string} = {
      'CVD': 'Cardiovascular Disease',
      'CHD': 'Coronary Heart Disease', 
      'STROKE': 'Stroke',
      'ANGINA': 'Angina',
      'MI': 'Myocardial Infarction',
      'HYPERTENSION': 'Hypertension',
      'DEATH': 'Death Risk'
    };
    return diseaseMap[disease] || disease;
  }

  constructor() {
    const apiKey = process.env.OPENAI_API_KEY;
    if (apiKey && apiKey.trim() !== '') {
      this.openai = new OpenAI({
        apiKey: apiKey,
      });
      this.isEnabled = true;
      console.log('✅ OpenAI API initialized');
    } else {
      console.log('⚠️ OpenAI API key not found, using default recommendations');
    }
  }

  public getIsEnabled(): boolean {
    return this.isEnabled;
  }

  // 生成AI健康建议
  public async generateHealthAdvice(
    healthData: HealthData,
    predictions: PredictionResult,
    options: AIAdviceOptions = { language: 'zh', includePersonalization: true }
  ): Promise<string> {
    if (!this.isEnabled || !this.openai) {
      return this.getFallbackAdvice(predictions, options.language);
    }

    try {
      const prompt = this.buildPrompt(healthData, predictions, options);
      
      const completion = await Promise.race([
        this.openai.chat.completions.create({
          model: "gpt-3.5-turbo",
          messages: [
                      {
            role: "system",
            content: "You are a professional medical health advisor. Based on user health data and disease risk predictions, provide personalized, professional, and understandable health advice. Use a gentle, professional tone, avoid excessive panic, and focus on prevention and lifestyle improvements. IMPORTANT: Always respond in English only, regardless of the input language."
          },
            {
              role: "user",
              content: prompt
            }
          ],
          max_tokens: 800,
          temperature: 0.7,
          top_p: 0.9,
        }),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('OpenAI API timeout after 30 seconds')), 30000)
        )
      ]) as any;

      const aiAdvice = completion.choices[0]?.message?.content;
      if (aiAdvice && aiAdvice.trim() !== '') {
        console.log('✅ AI advice generated successfully');
        return aiAdvice.trim();
      } else {
        console.log('⚠️ AI response was empty, using fallback');
        return this.getFallbackAdvice(predictions, options.language);
      }
    } catch (error) {
      console.error('❌ OpenAI API error:', error);
      return this.getFallbackAdvice(predictions, options.language);
    }
  }

  // 构建AI提示词
  private buildPrompt(healthData: HealthData, predictions: PredictionResult, options: AIAdviceOptions): string {
    const { language, includePersonalization } = options;
    
    // 构建患者基本信息
    const patientInfo = {
      age: healthData.age,
      gender: healthData.sex === 1 ? (language === 'zh' ? '男性' : 'male') : (language === 'zh' ? '女性' : 'female'),
      bmi: healthData.bmi || 'unknown',
      smoking: healthData.cursmoke === 1 ? (language === 'zh' ? '吸烟' : 'smoker') : (language === 'zh' ? '不吸烟' : 'non-smoker'),
      diabetes: healthData.diabetes === 1 ? (language === 'zh' ? '有糖尿病' : 'diabetic') : (language === 'zh' ? '无糖尿病' : 'non-diabetic'),
      bpMeds: healthData.bpmeds === 1 ? (language === 'zh' ? '服用降压药' : 'on BP medication') : (language === 'zh' ? '未服用降压药' : 'not on BP medication')
    };

    // 构建风险信息
    const highRiskDiseases = Object.entries(predictions)
      .filter(([_, pred]) => pred.risk_level === 'High')
      .map(([_, pred]) => pred.name);
    
    const mediumRiskDiseases = Object.entries(predictions)
      .filter(([_, pred]) => pred.risk_level === 'Medium')
      .map(([_, pred]) => pred.name);

    // Always use English prompts to ensure English responses
    const patientInfoEn = {
      age: healthData.age,
      gender: healthData.sex === 1 ? 'male' : 'female',
      bmi: healthData.bmi || 'unknown',
      smoking: healthData.cursmoke === 1 ? 'smoker' : 'non-smoker',
      diabetes: healthData.diabetes === 1 ? 'diabetic' : 'non-diabetic',
      bpMeds: healthData.bpmeds === 1 ? 'on BP medication' : 'not on BP medication'
    };

    // Convert disease names to English
    const highRiskDiseasesEn = Object.entries(predictions)
      .filter(([_, pred]) => pred.risk_level === 'High')
      .map(([disease, _]) => this.getDiseaseNameInEnglish(disease));
    
    const mediumRiskDiseasesEn = Object.entries(predictions)
      .filter(([_, pred]) => pred.risk_level === 'Medium')
      .map(([disease, _]) => this.getDiseaseNameInEnglish(disease));

    return `
Please provide personalized health advice based on the following patient information:

Patient Basic Information:
- Age: ${patientInfoEn.age} years old
- Gender: ${patientInfoEn.gender}
- BMI: ${patientInfoEn.bmi}
- Smoking status: ${patientInfoEn.smoking}
- Diabetes: ${patientInfoEn.diabetes}
- BP medication: ${patientInfoEn.bpMeds}
${healthData.sysbp ? `- Systolic BP: ${healthData.sysbp} mmHg` : ''}
${healthData.totchol ? `- Total cholesterol: ${healthData.totchol} mg/dL` : ''}

Disease Risk Assessment Results:
${highRiskDiseasesEn.length > 0 ? `High-risk diseases: ${highRiskDiseasesEn.join(', ')}` : 'No high-risk diseases identified'}
${mediumRiskDiseasesEn.length > 0 ? `Medium-risk diseases: ${mediumRiskDiseasesEn.join(', ')}` : 'No medium-risk diseases identified'}

Please provide:
1. Specific prevention recommendations for high-risk diseases
2. Lifestyle improvement suggestions (diet, exercise, sleep)
3. Medical checkup and follow-up recommendations
4. Mental health and stress management advice

Please use a gentle, encouraging tone, avoid panic, and focus on positive lifestyle changes. Respond in English only.
    `;
  }

  // 备用建议（当AI不可用时）
  private getFallbackAdvice(predictions: PredictionResult, language: 'en' | 'zh'): string {
    const highRiskDiseases = Object.entries(predictions)
      .filter(([_, pred]) => pred.risk_level === 'High')
      .map(([disease, _]) => {
        // 总是使用英文疾病名称
        return this.getDiseaseNameInEnglish(disease);
      });

    if (language === 'zh') {
      if (highRiskDiseases.length === 0) {
        return "您的健康状况整体良好！建议继续保持健康的生活方式，包括均衡饮食、适量运动、充足睡眠和定期体检。";
      } else {
        return `您在 ${highRiskDiseases.join('、')} 方面存在较高风险。建议：1）及时咨询专科医生；2）控制相关危险因素；3）改善生活方式；4）定期监测相关指标；5）遵医嘱进行治疗。请保持积极心态，通过科学的管理可以有效降低疾病风险。`;
      }
    } else {
      if (highRiskDiseases.length === 0) {
        return "Your overall health condition is good! We recommend continuing to maintain a healthy lifestyle, including balanced diet, moderate exercise, adequate sleep, and regular check-ups.";
      } else {
        return `You have higher risks for ${highRiskDiseases.join(', ')}. Recommendations: 1) Consult specialists promptly; 2) Control related risk factors; 3) Improve lifestyle; 4) Monitor relevant indicators regularly; 5) Follow medical advice for treatment. Please maintain a positive attitude - effective disease risk reduction is achievable through scientific management.`;
      }
    }
  }

  // 生成针对特定疾病的建议
  public async generateDiseaseSpecificAdvice(
    disease: string,
    riskLevel: string,
    healthData: HealthData,
    language: 'en' | 'zh' = 'zh'
  ): Promise<string[]> {
    if (!this.isEnabled || !this.openai) {
      return this.getFallbackDiseaseAdvice(disease, riskLevel, language);
    }

    try {
      // Always use English prompts with English disease names
      const diseaseNameEn = this.getDiseaseNameInEnglish(disease);
      const prompt = `Please provide ${riskLevel} risk management advice for ${diseaseNameEn} for a ${healthData.age}-year-old ${healthData.sex === 1 ? 'male' : 'female'} patient. Provide 3-5 specific, actionable recommendations, each within 30 words. Respond in English only.`;

      const completion = await this.openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "system",
            content: "You are a professional medical advisor providing concise and practical health advice. Always respond in English only."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        max_tokens: 300,
        temperature: 0.6,
      });

      const response = completion.choices[0]?.message?.content;
      if (response) {
        return response.split('\n').filter(line => line.trim() !== '').slice(0, 5);
      }
    } catch (error) {
      console.error('❌ OpenAI API error for disease advice:', error);
    }

    return this.getFallbackDiseaseAdvice(disease, riskLevel, language);
  }

  private getFallbackDiseaseAdvice(disease: string, riskLevel: string, language: 'en' | 'zh'): string[] {
    // 这里可以根据疾病类型返回预设建议
    if (language === 'zh') {
      return ['定期医疗检查', '改善生活方式', '控制危险因素', '遵医嘱用药', '保持心理健康'];
    } else {
      return ['Regular medical check-ups', 'Improve lifestyle', 'Control risk factors', 'Follow medical advice', 'Maintain mental health'];
    }
  }
}

export const aiAdviceService = new AIAdviceService(); 