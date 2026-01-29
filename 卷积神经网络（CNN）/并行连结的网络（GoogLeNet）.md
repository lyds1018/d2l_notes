## 一、动机与思想

传统 CNN每一层通常只有**一种尺度卷积核**，由此引发“什么样大小的卷积核最合适”的问题。

GoogLeNet 模型认为使用不同大小的卷积核组合是有利的，从而设计了 Inception 块。

---

## 二、模型结构与使用

### 1. Inception 块

![[Pasted image 20260127055401.png]]

同一层并行使用多种尺度提取特征，再并行输出：

1. 前三条路径使用 1x1、3x3 和 5x5 的卷积层，从不同空间大小中提取信息。 
2. 中间两条路径在输入上执行 1x1 卷积，以减少通道数，降低模型的复杂性。 
3. 第四条路径使用 3x3 最大汇聚层，然后使用 1x1 卷积层来改变通道数。
4. 最后在通道维度上并行连结，构成Inception块的输出。

### 2. GoogLeNet 网络

![[Pasted image 20260127060212.png]]

### 3. 模板

```python
import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，1×1
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        
        # 线路2，1×1 -> 3×3
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        
        # 线路3，1×1 -> 5×5
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        
        # 线路4，Maxpool -> 1×1
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)

net = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(inplace=True),
    ...
    
    Inception(64, 32, 48, 64, 8, 16, 16),
    Inception(128, 64, 64, 96, 16, 32, 32),
    Inception(224, 64, 96, 128, 16, 32, 32),
    ...
    
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(1024, 10)
)
```



