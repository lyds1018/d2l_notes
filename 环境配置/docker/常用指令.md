# Docker 常用指令速查表

## 1. 镜像操作
```bash
# 查看本地镜像
docker images

# 拉取镜像
docker pull ubuntu:20.04

# 删除镜像
docker rmi ubuntu:20.04

# 构建镜像（当前目录 Dockerfile）
docker build -t myimage:1.0 .

# 构建镜像（指定 Dockerfile）
docker build -f ./Dockerfile.prod -t myimage:prod .
```

## 2. 容器操作
```bash
# 运行容器（交互模式）
docker run -it pytorch/pytorch:latest bash

# 运行容器（后台模式 + 端口映射）
docker run -d -p 80:80 nginx

# 运行容器（挂载目录）
docker run -it \
  -v /path/to/data:/data \
  -v /path/to/project:/workspace \
  myimage:v1

# 启动已停止的容器
docker start container_id

# 停止容器
docker stop container_id

# 删除容器
docker rm container_id

# 自动删除容器（退出时）
docker run --rm myimage
```

## 3. 查看容器信息
```bash
# 查看正在运行的容器
docker ps

# 查看所有容器（包括已停止）
docker ps -a

# 查看容器日志
docker logs container_id

# 查看容器详细信息
docker inspect container_id

# 进入正在运行的容器
docker exec -it container_id /bin/bash
```

## 4. 清理资源
```bash
# 删除所有已停止的容器
docker container prune

# 删除所有无用镜像
docker image prune

# 一次性清理无用数据（容器+镜像+网络+缓存）
docker system prune -a
```

## 5. 镜像导入导出
```bash
# 保存镜像到文件
docker save myimage:1.0 -o myimage.tar

# 从文件加载镜像
docker load -i myimage.tar

# 导出容器文件系统
docker export container_id -o container.tar

# 从文件导入容器文件系统
docker import container.tar myimage:imported
```
