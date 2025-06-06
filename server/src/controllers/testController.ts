import { Request, Response } from 'express';
import OpenAI from 'openai';

export const testOpenAI = async (req: Request, res: Response) => {
  try {
    const apiKey = process.env.OPENAI_API_KEY;
    console.log('🔍 Testing OpenAI API...');
    console.log('🔍 API Key found:', apiKey ? `${apiKey.substring(0, 10)}...` : 'NOT FOUND');
    
    if (!apiKey || apiKey.trim() === '') {
      return res.json({
        success: false,
        error: 'OpenAI API key not found in environment variables',
        env_vars: Object.keys(process.env).filter(key => key.includes('OPENAI'))
      });
    }

    const openai = new OpenAI({
      apiKey: apiKey,
    });

    console.log('🔍 Making test API call...');
    
    const completion = await Promise.race([
      openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "system",
            content: "You are a helpful assistant. Respond with exactly: 'OpenAI API test successful!'"
          },
          {
            role: "user",
            content: "Please confirm the API is working"
          }
        ],
        max_tokens: 50,
        temperature: 0,
      }),
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('OpenAI API timeout after 30 seconds')), 30000)
      )
    ]) as any;

    const response = completion.choices[0]?.message?.content;
    console.log('✅ OpenAI response received:', response);

    res.json({
      success: true,
      message: 'OpenAI API test completed',
      response: response,
      completion_details: {
        model: completion.model,
        usage: completion.usage,
        finish_reason: completion.choices[0]?.finish_reason
      }
    });

  } catch (error: any) {
    console.error('❌ OpenAI API test failed:', error);
    console.error('🔍 Error details:', {
      name: error.name,
      message: error.message,
      code: error.code,
      type: error.type,
      param: error.param,
      status: error.status
    });

    res.json({
      success: false,
      error: 'OpenAI API test failed',
      error_details: {
        name: error.name,
        message: error.message,
        code: error.code || 'unknown',
        type: error.type || 'unknown',
        status: error.status || 'unknown'
      },
      api_key_status: process.env.OPENAI_API_KEY ? 'present' : 'missing'
    });
  }
}; 