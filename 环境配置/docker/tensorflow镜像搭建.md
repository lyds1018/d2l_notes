# Windows 11 + Docker + TensorFlow GPU 容器使用指南

## 1. 系统要求

- **操作系统**: Windows 11 (版本 21H2 或更高)
- **GPU**: NVIDIA RTX 系列或支持 CUDA 的显卡
- **NVIDIA 驱动**: 最新 Game/Studio 驱动，支持 WSL2
- **Docker Desktop**: >= 4.44.1，启用 WSL2 后端

## 2. Docker Desktop 配置

1. 打开 Docker Desktop
2. 选择 **Settings → General → Use the WSL 2 based engine**
3. 确认并重启 Docker Desktop

## 3. 拉取 NVIDIA TensorFlow GPU 镜像

```bash
docker pull nvcr.io/nvidia/tensorflow:24.08-tf2-py3
```

## 4. 运行容器并测试 GPU

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it nvcr.io/nvidia/tensorflow:24.08-tf2-py3

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

输出示例:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

说明 GPU 已被 TensorFlow 识别。

## 5. 注意事项

- 不需要在 WSL2 或容器内部单独安装 CUDA
- docker官方 `tensorflow/tensorflow:latest-gpu` 镜像可能和 RTX 40 系列驱动不兼容，使用 NVIDIA 官方tensorflow镜像
- 大模型训练建议加 `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864` 参数，防止共享内存不足
