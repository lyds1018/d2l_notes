稠密连接网络（DenseNet）在某种程度上是ResNet的逻辑扩展。

![[Pasted image 20260127215655.png]]

![[Pasted image 20260127215721.png]]

ResNet 和 DenseNet 的关键区别在于，DenseNet 输出是**连接**（将各项作为该层 MLP 的权重），而不是简单相加。

其网络中的映射关系：
$$\mathbf{x} \to [\mathbf{x}, f_1(\mathbf{x}), f_2([\mathbf{x}, f_1(\mathbf{x})]), f_3([\mathbf{x}, f_1(\mathbf{x}), f_2([\mathbf{x}, f_1(\mathbf{x})])]), \dots]$$
