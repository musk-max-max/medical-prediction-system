module.exports = {
  apps: [
    {
      name: 'medical-prediction-server',
      script: './server/dist/index.js',
      cwd: '/var/www/medical-prediction',
      env_file: './server/.env.production',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      error_file: '/var/log/pm2/medical-prediction-error.log',
      out_file: '/var/log/pm2/medical-prediction-out.log',
      log_file: '/var/log/pm2/medical-prediction.log',
      time: true,
      env: {
        NODE_ENV: 'production',
        PORT: 5000
      }
    }
  ]
}; 