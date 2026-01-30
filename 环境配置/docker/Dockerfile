# 1) 基础镜像
FROM python:3.10-slim
FROM pytorch/pytorch:latest
FROM tensorflow/tensorflow:latest

# 2) 更换apt源
RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
         /etc/apt/sources.list && \
    sed -i 's|http://security.debian.org|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
         /etc/apt/sources.list

# 3) 安装系统组件
RUN apt-get update && \
    apt-get install -y \
        libgl1 \
        libglib2.0-0 \
        libsm6 libxext6 libxrender1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 4) 创建工作目录
WORKDIR /workspace

# 5) 复制代码
COPY . .

# 6) 设置环境变量
ENV PYTHONPATH = /workspace

# 7) 复制依赖文件
COPY requirements.txt .

# 8) 安装依赖
RUN pip3 install --break-system-packages \
    --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 9) 声明入口：容器启动后自动执行 main.py
ENTRYPOINT ["python3", "/workspace/main.py"]
CMD ["bash"]