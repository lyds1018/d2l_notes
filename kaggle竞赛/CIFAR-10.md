## 一、数据处理

### 1. 自定义数据集（data_set.py）

```python
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


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
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.randaugment.RandomAugment(num_ops=2, magnitude=4),
    T.ToTensor(),
    T.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    ),
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    ),
])
```

### 3. 创建数据集

```python
dataset = ImageDataset(
    csv_file=os.path.join(basedir, "data", "train.csv"),
    img_path=os.path.join(basedir, "data", "train"),
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_idx, test_idx = random_split(
    range(len(dataset)), [train_size, test_size], generator=generator
)

train_dataset = ImageDataset(
    csv_file=os.path.join(basedir, "data", "train.csv"),
    img_path=os.path.join(basedir, "data", "train"),
    transform=train_transform,
)
test_dataset = ImageDataset(
    csv_file=os.path.join(basedir, "data", "train.csv"),
    img_path=os.path.join(basedir, "data", "train"),
    transform=test_transform,
)

train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
test_dataset = torch.utils.data.Subset(test_dataset, test_idx)
```

---

## 二、网络结构

### 1. ResNet 块

```python
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        return F.relu(y)
```

### 2. ResNet18 网络

```python
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CIFAR: 3×32×32
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU()
        )
        
        # 4 stages × 2 blocks
        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        
        self.layer2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        
        self.layer3 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))
        
        self.layer4 = nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512))
        
        self.dropout = nn.Dropout(0.5)
        
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        return self.fc(x)
```

---

## 三、训练循环

```python
def train_model(model, criterion, optimizer, scheduler, batch_size, num_epochs):
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # 训练模型
    device = next(model.parameters()).device  # 获取模型所在设备
    alpha = 0.8  # 测试损失权重
    best_score = float("inf")
    best_epoch = -1
    
    print("开始训练模型...")
    for epoch in range(num_epochs):
        # 训练模型
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()
        running_loss = running_loss / len(train_loader.dataset)
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                
        test_loss = test_loss / len(test_loader.dataset)
        
        print(f"Train Loss: {running_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # 保存目前最优模型
        current_score = alpha * test_loss + (1 - alpha) * running_loss
        if current_score < best_score:
            best_score = current_score
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(), os.path.join(basedir, "model/best_model.pth")
            )

    print(f"最优模型，Epoch: {best_epoch}, Test Loss: {best_score:.4f}")
```

---

## 四、优化器与超参数

```python
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs, lr, batch_size, weight_decay = 200, 1e-4, 128, 1e-5
    num_classes = len(dataset.label2idx)
    
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )
```