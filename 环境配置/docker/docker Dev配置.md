## 📄 `.devcontainer/devcontainer.json`

```
{
  "name": "TensorFlow DevContainer",
  // 使用已有镜像（比如你已经构建好的 tfgpu:v1）
  "image": "tfgpu:v1",

  // 容器启动参数
  "runArgs": [
    "--gpus", "all",
    "-p", "8888:8888",  // Jupyter Notebook
    "-p", "6006:6006",  // TensorBoard
    "-p", "6064:6064"   // 其他服务
  ],

  // 容器里的工作目录
  "workspaceFolder": "/workspace",

  // 挂载本地目录到容器
  "mounts": [
    "source=D:/tflearn,target=/workspace,type=bind"
  ],

  // VS Code 自动安装的扩展
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-toolsai.jupyter",
        "ms-azuretools.vscode-docker"
      ]
    }
  },

  // 使用 root 用户（避免权限问题）
  "remoteUser": "root"
}
```

------

## 使用方式

1. 在项目目录（比如 `D:/tflearn`）下创建上述配置文件
2. 打开 VS Code → `Ctrl+Shift+P` → 输入 **Dev Containers: Reopen in Container**