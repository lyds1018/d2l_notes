## 一、数据处理

### 1. 自定义数据集（data_set.py）

```python
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

basedir = os.path.dirname(os.path.abspath(__file__))

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_path, transform=None, index=True):
        self.df = pd.read_csv(csv_file)
        self.img_path = img_path
        self.transform = transform  # 是否图像增强
        self.index = index  # 是否需要标签
        
        # label 编码
        if self.index:
            labels = sorted(self.df["label"].unique())
            self.label2idx = {
                label: i for i, label in enumerate(labels)
            }  # 构建字典，i 从 0 开始

            self.df["label_idx"] = self.df["label"].map(self.label2idx)  # 字典映射

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 图片文件夹 + 图片文件名
        img_path = os.path.join(self.img_path, self.df.iloc[idx]["image"])

        if self.index:
            label = self.df.iloc[idx]["label_idx"]
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.long)

        else:
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image
```

### 2. 数据增广

```python
train_transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.randaugment.RandomAugment(num_ops=2, magnitude=9),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])
```

### 3. 创建数据集

```python
dataset = ImageDataset(
    csv_file=os.path.join(basedir, "data/train.csv"),
    img_path=os.path.join(basedir, "data/train"),
)

test_ratio = 0.2
test_size = int(len(dataset) * test_ratio)
train_size = len(dataset) - test_size
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size], generator=generator
)

train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform
```



