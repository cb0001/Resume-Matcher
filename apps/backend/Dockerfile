# 使用官方Python 3.12.0镜像作为基础镜像
FROM python:3.12.0

# 设置环境变量
ENV ENV="production"
ENV PYTHONPATH=/app
RUN echo "Building ENV: $ENV"

# 设置工作目录
WORKDIR /app

# 将app目录的内容复制到容器的/app目录
COPY app /app/app
COPY launch_uvicorn.py /app/launch_uvicorn.py

# 复制环境变量文件
COPY .env /app/.env

# 将requirements.txt文件复制到容器中
COPY requirements.txt /app/requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir -r /app/requirements.txt

# 设置启动命令和暴露端口
CMD ["python3", "launch_uvicorn.py"]
EXPOSE 80