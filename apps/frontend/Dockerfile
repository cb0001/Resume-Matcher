FROM node:20-alpine

# 设置工作目录
WORKDIR /app

# 复制package.json和package-lock.json
COPY package.json package-lock.json ./

# 安装依赖
RUN npm install

# 复制环境变量文件
COPY .env ./

# 复制源代码
COPY . .

# 构建应用
RUN npm run build

# 暴露端口
EXPOSE 80

# 启动应用
CMD ["npm", "start"]