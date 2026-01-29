### 1. 概念
汇聚层（Pooling Layer）用于**下采样（减少特征图尺寸）**，保留主要特征，同时降低计算量。  

类型：  
- **最大汇聚（Max Pooling）**：取池化窗口内的最大值  
- **平均汇聚（Average Pooling）**：取池化窗口内的平均值  

### 2. 输出尺寸计算


$$
H_{out} = \left\lfloor \frac{H + 2p - k_H}{s} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W + 2p - k_W}{s} \right\rfloor + 1
$$

其中：
- 输入特征图 $X \in \mathbb{R}^{C \times H \times W}$  
- 池化窗口大小 $k_H \times k_W$  
- 步幅 $s$  
- 填充 $p$  

### 3. 多通道处理

汇聚层**不改变通道数**，对每个通道独立进行一次汇聚操作后，直接作为输出通道
