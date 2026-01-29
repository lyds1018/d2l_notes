## 一、经典卷积网络结构

1. 带填充以保持分辨率的卷积层；
    
2. 非线性激活函数，如ReLU；
    
3. 汇聚层，如最大汇聚层。

---

## 二、VGG 网络

### 1. 结构

![[Pasted image 20260127043935.png]]

### 2. 模板

```python
import torch
import torch.nn as nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels  # 后续卷积层的输入通道等于当前的输出通道
    
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

net = nn.Sequential(
    vgg_block(2, 3, 64),   # 2层卷积，输入3通道，输出64通道
    vgg_block(2, 64, 128), 
    nn.Flatten(),
    nn.Linear(128 * 56 * 56, 10) # 假设输入是224x224，经过两次池化变为56x56
)
```

