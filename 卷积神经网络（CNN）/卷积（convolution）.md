## 1. 卷积核尺寸

$$
K \in \mathbb{R}^{C_{out} \times C_{in} \times k_H \times k_W}
$$

## 2. 输出尺寸计算

$$
H_{out} = \left\lfloor \frac{H + 2p - k_H}{s} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W + 2p - k_W}{s} \right\rfloor + 1
$$

其中：
- 输入特征图 $X \in \mathbb{R}^{C \times H \times W}$  
- 卷积核大小 $k_H \times k_W$  
- 步幅 $s$  
- 填充 $p$  

## 3. 多通道卷积规则

每个输出通道对各个输入通道做卷积并求和：

$$
Y_c = \sum_{i=0}^{input-1} X_i * K_{c,i}
$$

其中：
- $X_i$ 是输入的第 $i$ 个通道  
- $K_{c,i}$ 是卷积核第 $c$ 个输出通道对应第 $i$ 个输入通道的卷积核  

例如，对某一个输出通道：

![[Pasted image 20260124041403.png]]

## 4. 感受野 $RF$

1. **概念**  
   影响该神经元输出的输入区域大小，随着卷积核大小、步幅和层数增加而扩大。

2. **作用**  
   感受野决定神经元能够捕捉的上下文信息范围：  
   - 小感受野 → 捕捉局部特征  
   - 大感受野 → 捕捉全局特征

3. **计算公式**  
   $$RF_{l} = RF_{l-1} + (k_l - 1) \times S(l-1)$$  
   其中：  
   - $k_l$ 为第 $l$ 层卷积核大小  
   - $S$ 表示前面所有层步幅的累积乘积
