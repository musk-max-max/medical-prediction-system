export const translations = {
  en: {
    login: {
      title: 'Login',
      username: 'Username',
      password: 'Password',
      submit: 'Login',
      register: 'Register',
      noAccount: "Don't have an account?",
      haveAccount: 'Already have an account?',
      testConnection: '🔗 Test API Connection',
      loading: 'Loading...',
      testing: 'Testing...'
    },
    register: {
      title: 'Register',
      username: 'Username',
      email: 'Email',
      password: 'Password',
      submit: 'Register',
      login: 'Login',
      noAccount: "Don't have an account?",
      haveAccount: 'Already have an account?',
      testConnection: '🔗 Test API Connection',
      loading: 'Loading...',
      testing: 'Testing...'
    },
    header: {
      title: '🏥 Intelligent Medical Prediction System',
      subtitle: 'AI-based Cardiovascular Disease Risk Assessment Platform',
      welcome: 'Welcome, ',
      logout: 'Logout'
    },
    nav: {
      prediction: '🎯 Risk Assessment',
      history: '📋 History'
    },
    prediction: {
      title: '🏥 Medical Health Risk Assessment',
      subtitle: 'AI-based Cardiovascular Disease Risk Assessment and Survival Analysis System',
      basicInfo: {
        title: '👤 Basic Information',
        gender: 'Gender',
        male: 'Male',
        female: 'Female',
        age: 'Age',
        ageRange: 'Range: 30-62 years',
        ageWarning: 'Age must be between 30 and 62 years',
        bmi: 'BMI',
        bmiRange: 'Range: 15-40'
      },
      cardiovascular: {
        title: '❤️ Cardiovascular Indicators',
        systolicBP: 'Systolic BP (mmHg)',
        diastolicBP: 'Diastolic BP (mmHg)',
        heartRate: 'Heart Rate (bpm)',
        normalBP: 'Normal: <120',
        normalHR: 'Normal: 60-100'
      },
      biochemical: {
        title: '🩸 Biochemical Indicators',
        cholesterol: 'Total Cholesterol (mg/dL)',
        glucose: 'Fasting Glucose (mg/dL)',
        idealCholesterol: 'Ideal: <200',
        normalGlucose: 'Normal: 70-100'
      },
      lifestyle: {
        title: '🚬 Lifestyle',
        smokingStatus: 'Smoking Status',
        noSmoking: 'Non-smoker',
        smoking: 'Smoker',
        cigarettesPerDay: 'Cigarettes per Day',
        autoZero: 'Auto 0 for non-smokers'
      },
      medicalHistory: {
        title: '💊 Medical History & Medication',
        diabetes: 'Diabetes History',
        no: 'No',
        yes: 'Yes',
        bpMeds: 'Blood Pressure Medication',
        notUsing: 'Not Using',
        using: 'Currently Using'
      },
      submit: '🔮 Start Comprehensive Analysis',
      analyzing: '🧠 AI is analyzing, please wait...',
      note: '💡 Risk assessment and survival analysis in progress, this may take 30-60 seconds...',
      completed: '🎉 Comprehensive analysis completed!'
    },
    results: {
      title: '🎯 Prediction Results',
      riskScores: {
        title: '📊 Risk Scores',
        note: '💡 Risk Score Description:',
        lowRisk: 'Low Risk (0-15%) - Low risk, maintain healthy lifestyle',
        mediumRisk: 'Medium Risk (15-30%) - Need attention, regular check-ups recommended',
        highRisk: 'High Risk (>30%) - High risk, medical attention recommended'
      },
      survival: {
        title: 'Survival Analysis',
        note: '💡 Survival rate prediction based on three medical examinations (follow-up period up to 20 years)'
      },
      advice: {
        title: '💡 Health Advice',
        default: 'Your health condition is good, maintain your current lifestyle.',
        hypertension: 'Blood pressure control recommended',
        heartDisease: 'Pay attention to heart health',
        stroke: 'Prevent stroke risk',
        death: 'Comprehensive health management needed'
      }
    },
    history: {
      title: '📋 Assessment History',
      loading: 'Loading...',
      noRecords: 'No assessment records',
      date: 'Date',
      age: 'Age',
      gender: 'Gender',
      male: 'Male',
      female: 'Female',
      bmi: 'BMI',
      bloodPressure: 'Blood Pressure',
      search: 'Search',
      clearSearch: 'Clear Search',
      searchPlaceholder: 'Exact match by username...',
      trendAnalysis: 'Trend Analysis',
      trendChart: 'Health Data Trend Analysis',
      recordsCount: 'records'
    },
    errors: {
      serverConnection: 'Unable to connect to server, please ensure backend service is running',
      networkError: '🔌 Network connection error: Unable to connect to server',
      predictionFailed: 'Prediction failed, please try again later',
      requestTimeout: '🕐 Request timeout: AI analysis taking too long, please try again later',
      tooManyRequests: '⏳ Too many requests, please try again later',
      authenticationFailed: '🔐 Authentication failed, please login again',
      loginFailed: 'Login failed',
      registrationFailed: 'Registration failed',
      fetchHistoryFailed: 'Failed to fetch history records',
      deleteRecordsFailed: 'Failed to delete records',
      connectionTestFailed: 'Connection test failed'
    },
    success: {
      apiConnectionSuccess: 'API connection test successful!',
      loginSuccess: 'Login successful!',
      registrationSuccess: 'Registration successful!',
      logoutSuccess: 'Logged out successfully',
      recordsDeletedSuccess: 'Selected records deleted successfully'
    },
    confirmations: {
      deleteRecords: 'Are you sure you want to delete the selected records?',
      emptyFieldsWarning: 'The following fields are empty:\n{fields}\n\nThis may affect prediction accuracy. Continue?'
    },
    fields: {
      totalCholesterol: 'Total Cholesterol',
      systolicBP: 'Systolic Blood Pressure', 
      diastolicBP: 'Diastolic Blood Pressure',
      heartRate: 'Heart Rate',
      fastingGlucose: 'Fasting Glucose'
    }
  },
  zh: {
    login: {
      title: '登录',
      username: '用户名',
      password: '密码',
      submit: '登录',
      register: '注册',
      noAccount: '没有账户？',
      haveAccount: '已有账户？',
      testConnection: '🔗 测试API连接',
      loading: '处理中...',
      testing: '测试中...'
    },
    register: {
      title: '注册',
      username: '用户名',
      email: '邮箱',
      password: '密码',
      submit: '注册',
      login: '登录',
      noAccount: '没有账户？',
      haveAccount: '已有账户？',
      testConnection: '🔗 测试API连接',
      loading: '处理中...',
      testing: '测试中...'
    },
    header: {
      title: '🏥 智能医疗预测系统',
      subtitle: '基于AI的心血管疾病风险评估平台',
      welcome: '欢迎，',
      logout: '登出'
    },
    nav: {
      prediction: '🎯 风险评估',
      history: '📋 历史记录'
    },
    prediction: {
      title: '🏥 医疗健康风险评估',
      subtitle: '基于AI的心血管疾病风险评估与生存分析系统',
      basicInfo: {
        title: '👤 基本信息',
        gender: '性别',
        male: '男',
        female: '女',
        age: '年龄',
        ageRange: '范围：30-62岁',
        ageWarning: '年龄必须在30-62岁之间',
        bmi: 'BMI',
        bmiRange: '范围：15-40'
      },
      cardiovascular: {
        title: '❤️ 心血管指标',
        systolicBP: '收缩压 (mmHg)',
        diastolicBP: '舒张压 (mmHg)',
        heartRate: '心率 (次/分钟)',
        normalBP: '正常: <120',
        normalHR: '正常: 60-100'
      },
      biochemical: {
        title: '🩸 生化指标',
        cholesterol: '总胆固醇 (mg/dL)',
        glucose: '空腹血糖 (mg/dL)',
        idealCholesterol: '理想: <200',
        normalGlucose: '正常: 70-100'
      },
      lifestyle: {
        title: '🚬 生活习惯',
        smokingStatus: '吸烟状态',
        noSmoking: '不吸烟',
        smoking: '吸烟',
        cigarettesPerDay: '每日吸烟量 (支)',
        autoZero: '不吸烟时自动为0'
      },
      medicalHistory: {
        title: '💊 疾病史与用药',
        diabetes: '糖尿病史',
        no: '无',
        yes: '有',
        bpMeds: '血压药物使用',
        notUsing: '未使用',
        using: '正在使用'
      },
      submit: '🔮 开始综合分析',
      analyzing: '🧠 AI正在分析中，请稍候...',
      note: '💡 正在进行风险评估和生存分析，这可能需要30-60秒时间...',
      completed: '🎉 综合分析完成！'
    },
    results: {
      title: '🎯 预测结果',
      riskScores: {
        title: '📊 风险评分',
        note: '💡 风险评分说明：',
        lowRisk: '低风险 (0-15%) - 风险较低，建议保持健康生活方式',
        mediumRisk: '中风险 (15-30%) - 需要关注，建议定期检查',
        highRisk: '高风险 (>30%) - 风险较高，建议及时就医'
      },
      survival: {
        title: '生存分析',
        note: '💡 基于三次体检随访数据（最长随访时间约20年）的生存率预测'
      },
      advice: {
        title: '💡 健康建议',
        default: '您的健康状况良好，建议保持当前的生活方式。',
        hypertension: '建议控制血压',
        heartDisease: '注意心脏健康',
        stroke: '预防中风风险',
        death: '需要全面健康管理'
      }
    },
    history: {
      title: '📋 历史评估记录',
      loading: '加载中...',
      noRecords: '暂无评估记录',
      date: '日期',
      age: '年龄',
      gender: '性别',
      male: '男',
      female: '女',
      bmi: 'BMI',
      bloodPressure: '血压',
      search: '搜索',
      clearSearch: '清除搜索',
      searchPlaceholder: '按用户名精确匹配...',
      trendAnalysis: '趋势分析',
      trendChart: '健康数据趋势分析',
      recordsCount: '记录'
    },
    errors: {
      serverConnection: '无法连接到服务器，请确保后端服务正在运行',
      networkError: '🔌 网络连接错误：无法连接到服务器',
      predictionFailed: '预测失败，请稍后重试',
      requestTimeout: '🕐 请求超时：AI分析时间过长，请稍后重试',
      tooManyRequests: '⏳ 请求过于频繁，请稍后再试',
      authenticationFailed: '🔐 认证失效，请重新登录',
      loginFailed: '登录失败',
      registrationFailed: '注册失败',
      fetchHistoryFailed: '获取历史记录失败',
      deleteRecordsFailed: '删除记录失败',
      connectionTestFailed: '连接测试失败'
    },
    success: {
      apiConnectionSuccess: 'API连接测试成功！',
      loginSuccess: '登录成功！',
      registrationSuccess: '注册成功！',
      logoutSuccess: '已安全登出',
      recordsDeletedSuccess: '已成功删除选中的记录'
    },
    confirmations: {
      deleteRecords: '确定要删除选中的记录吗？',
      emptyFieldsWarning: '以下字段为空：\n{fields}\n\n这可能会影响预测准确性。是否继续？'
    },
    fields: {
      totalCholesterol: '总胆固醇',
      systolicBP: '收缩压', 
      diastolicBP: '舒张压',
      heartRate: '心率',
      fastingGlucose: '空腹血糖'
    }
  }
}; 