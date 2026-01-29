## 一、动机与思想

### 1. 深层网络训练的问题

内部协变量偏移（Internal Covariate Shift）。即前一层参数更新 → 后一层输入分布不断变化

导致：

- 训练慢，不稳定
- 高层在训练中，需要不断适应新分布

### 2. 核心思想

在每一层中，对当前 mini-batch 的特征进行标准化，再引入可学习参数恢复表达能力。

即：BN = 强行把每层输入拉回到“统一分布” + 学习缩放平移

---

## 二、数学过程

### 1. 标准化


$$
\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

其中：

- $\epsilon$ 为极小值，防止除零。

### 2. 缩放与平移（可学习参数）

为了保证网络表达能力，引入：

$$
y_i=\gamma \hat{x}_i+\beta
$$

其中：

- $\gamma$：缩放参数  
- $\beta$：平移参数  

---

## 三、模型结构与使用

### 1. BN 在网络中的位置

卷积层：
```
Conv → BN → ReLU
```

- 对当前 mini-batch 该层的每个输入通道单独做 BN

全连接层：
```
Linear → BN → ReLU
```

### 2. 模板

```python
import torch.nn as nn

# Conv-BN-ReLU 块
class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)
    

net = nn.Sequential(
    ConvBNReLU(3, 64, k=3, p=1),
    ConvBNReLU(64, 128, k=3, p=1),
    
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    
    nn.Linear(64, 10)
)
```








