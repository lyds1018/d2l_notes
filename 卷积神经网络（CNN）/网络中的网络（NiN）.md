## 一、动机与思想

### 1. 传统 CNN 问题

LeNet、AlexNet和VGG都有一个共同的设计模式：通过一系列的卷积层与汇聚层来提取空间结构特征；然后通过全连接层对特征的表征进行处理。然而，如果使用了全连接层，可能会完全放弃表征的空间结构。

### 2. 核心思想

网络中的网络（NiN）提供了一个非常简单的解决方案：在每个像素的通道上分别使用多层感知机。

NiN 块的作用：

1. 使用 1x1 卷积对每个像素进行多通道加权求和，等价于在每个像素应用了 MLP

2. 堆叠1x1 卷积 + ReLU，使得在不改变特征图空间尺寸的情况下，大幅增加网络的**深度**

---

## 二、模型结构与使用

### 1. NiN 块

```
Conv(k×k) → ReLU → 1×1 Conv → ReLU → 1×1 Conv → ReLU
```

- 普通卷积（负责空间感受野）
- 两层 1×1 卷积（构成 MLP）

### 2. NiN 网络

![[Pasted image 20260127053015.png]]

- 全局平均汇聚层的输出通道数(等于输入通道数)，即为标签类别数

### 3. 模板

```python
import torch
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, strides=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

net = nn.Sequential(
    nin_block(3, 96, 11, s=4),
    nn.MaxPool2d(3, stride=2),
    
    nin_block(96, 256, 5, p=2),
    nn.MaxPool2d(3, stride=2),
    
    nin_block(256, 384, 3, p=1),
    nn.MaxPool2d(3, stride=2),
    
    nin_block(384, 10, 3, p=1),
    
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)
```

