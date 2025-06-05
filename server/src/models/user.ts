import { Model, DataTypes } from 'sequelize';
import { sequelize } from '../config/database';
import bcrypt from 'bcrypt';

export interface UserAttributes {
  id: number;
  username: string;
  email: string;
  password: string;
  is_admin: boolean;
}

export class User extends Model<UserAttributes> implements UserAttributes {
  public id!: number;
  public username!: string;
  public email!: string;
  public password!: string;
  public is_admin!: boolean;

  // 创建管理员账户的静态方法
  public static async createAdminUser() {
    try {
      const adminExists = await this.findOne({ where: { username: 'admin' } });
      if (!adminExists) {
        const hashedPassword = await bcrypt.hash('admin123', 10);
        await this.create({
          username: 'admin',
          email: 'admin@system.com',
          password: hashedPassword,
          is_admin: true
        });
        console.log('管理员账户创建成功');
      }
    } catch (error) {
      console.error('创建管理员账户失败:', error);
    }
  }
}

User.init(
  {
    id: {
      type: DataTypes.INTEGER,
      autoIncrement: true,
      primaryKey: true,
    },
    username: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: true,
    },
    email: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: true,
    },
    password: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    is_admin: {
      type: DataTypes.BOOLEAN,
      allowNull: false,
      defaultValue: false,
    },
  },
  {
    sequelize,
    modelName: 'User',
    tableName: 'users',
  }
); 