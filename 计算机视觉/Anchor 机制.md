## 一、Anchor-based

### 1. 基本思想

Anchor-based 是一种 **基于先验框（Prior Box）** 的检测机制。

即在特征图中，预先放置若干个固定大小和比例的 Anchor。

### 2. 预测内容

对每个 Anchor 预测：

- 分类：**类别概率**

- 回归：**相对偏移量** $(\Delta x, \Delta y, \Delta w, \Delta h)$

### 3. 代表模型

- SSD
- RetinaNet
- Faster R-CNN（RPN）
- YOLOv2 / YOLOv3 / YOLOv5

---

## 二、Anchor-free 机制

### 1. 基本思想

Anchor-free 是一种 **去先验框（Prior-free）** 的检测机制。

即不再使用 Anchor，直接在特征图位置上预测目标的几何信息。

### 2. 两种形式

### (一) 基于关键点（Keypoint-based）

将目标看作**关键点的组合**。

**常见形式：**

- 中心点 + 宽高 $(x_c, y_c, w, h)$
- 左上角 + 右下角 $(x_{lt},y_{lt})$, $(x_{rb},y_{rb})$

**代表模型：**

- CenterNet
- CornerNet
- ExtremeNet

---

### (二) 基于像素回归（Dense Regression）

把目标检测转化为**逐像素回归问题**。

即预测各个像素点到目标边界的距离：
$$
(l, t, r, b)
$$

**代表模型：**

- FCOS
- YOLOv8 / YOLOX



