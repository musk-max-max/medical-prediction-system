import sqlite3 from 'sqlite3';
import { Sequelize } from 'sequelize';
import path from 'path';

const dbPath = path.resolve(__dirname, '../../data/medical_prediction.db');

// SQLite 数据库连接
const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error('数据库连接失败:', err);
  } else {
    console.log('数据库连接成功');
    // 初始化数据库表
    initializeTables();
  }
});

// 初始化数据库表
function initializeTables() {
  // 用户表
  db.run(`CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    is_admin INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )`);

  // 健康数据表
  db.run(`CREATE TABLE IF NOT EXISTS user_health_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    sex INTEGER NOT NULL,
    age INTEGER NOT NULL,
    totchol REAL,
    sysbp REAL,
    diabp REAL,
    cursmoke INTEGER,
    cigpday REAL,
    bmi REAL,
    diabetes INTEGER,
    bpmeds INTEGER,
    heartrte REAL,
    glucose REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
  )`);

  // 预测结果表
  db.run(`CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    health_data_id INTEGER NOT NULL,
    prediction_result REAL NOT NULL,
    risk_level TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (health_data_id) REFERENCES user_health_data (id)
  )`);

  console.log('数据库表初始化完成');
}

// Sequelize 实例
export const sequelize = new Sequelize({
  dialect: 'sqlite',
  storage: dbPath,
  logging: false
});

// 测试数据库连接
sequelize.authenticate()
  .then(() => {
    console.log('Sequelize 数据库连接成功');
  })
  .catch(err => {
    console.error('Sequelize 数据库连接失败:', err);
  });

export const database = {
  getDatabase: () => db
}; 