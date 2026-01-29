## 一、核心思想与数学原理

[# 残差网络（ResNet）](https://zh.d2l.ai/chapter_convolutional-modern/resnet.html)

![[Pasted image 20260127212636.png]]

---

## 二、模型结构与使用

### 1. 残差块

![[Pasted image 20260127212536.png]]

让我们聚焦于神经网络局部部：如图 7.6.2 所示，假设我们的原始输入为$x$，而希望学出的理想映射为$f(\mathbf{x})$（作为 图7.6.2上方激活函数的输入）。左图虚线框中的部分需要直接拟合出该映射$f(\mathbf{x})$，而右图虚线框中的部分则需要拟合出残差映射$f(\mathbf{x}) - \mathbf{x}$。残差映射在现实中往往更容易优化。

![[Pasted image 20260127213648.png]]

==如图，残差块也可以通过 1x1 卷积调整通道和分辨率。==
==若不使用 1x1 卷积，则要求2个卷积层的输出与输入形状一致。==

### 2. ResNet-18 架构

![[Pasted image 20260127214106.png]]

- 虚线框内为残差块

### 3. 调用 ResNet 模型

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(pretrained=True)   # True = 加载 ImageNet 预训练权重

# 修改类别数
num_classes = 10   # 例如 CIFAR10
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失 + 优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train...
```