# 圆形分割方法说明

## 方法概述

本项目提供了一种基于语义分割的圆形检测方法，与原先的关键点检测方法相比，具有以下优势：

### 关键点检测 vs 语义分割

1. **关键点检测方法**（原方法）：
   - 直接预测圆心坐标
   - 输出是一个点坐标 (x, y)
   - 对遮挡、变形等情况敏感
   - 精度受限于回归坐标的准确性

2. **语义分割方法**（新方法）：
   - 预测整个圆形区域的分割掩码
   - 输出是一个与输入图像相同尺寸的分割图
   - 对遮挡、变形具有更好的鲁棒性
   - 可以通过后处理获得更精确的圆心位置

## 方法优势

### 1. 更好的鲁棒性
- 分割方法考虑了整个圆形区域的信息，而不是单个点
- 对噪声、部分遮挡等情况有更好的容忍度
- 可以处理不完整的圆形

### 2. 更精确的定位
- 通过分割掩码可以计算质心，通常比直接回归坐标更准确
- 可以通过最小外接圆等方法进一步优化圆心位置

### 3. 更丰富的信息
- 分割掩码提供了圆形的完整形状信息
- 可以同时获得圆心位置和半径信息
- 可以处理多个圆形的情况

## 代码结构

```
circle_detection_project/
├── data/
│   └── circle_segmentation_dataset.py  # 圆形分割数据集处理
├── models/
│   └── unet_segmentation.py           # UNet分割模型
├── train_segmentation.py              # 训练脚本
├── predict_segmentation.py            # 预测脚本
└── README_SEGMENTATION.md             # 本说明文件
```

## 使用方法

### 1. 训练模型

```bash
python train_segmentation.py
```

### 2. 预测单张图像

```bash
python predict_segmentation.py --model checkpoints/best_segmentation_model.pth --input test_image.jpg --single
```

### 3. 批量预测

```bash
python predict_segmentation.py --model checkpoints/best_segmentation_model.pth --input datasets/test/images --output results/segmentation
```

## 输出结果

预测结果包括：
1. 原始图像
2. 二值分割掩码
3. 概率分割掩码
4. 可视化结果图像（包含分割区域和检测到的圆心）

## 可能的改进方向

1. **损失函数优化**：使用Dice Loss或IoU Loss替代BCE Loss
2. **模型结构优化**：增加注意力机制或使用更深的网络
3. **后处理优化**：使用椭圆拟合或多圆检测算法
4. **数据增强**：增加更多样的数据增强策略