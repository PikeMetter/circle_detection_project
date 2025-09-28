import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class CircleDataset(Dataset):
    def __init__(self, data_dir, transform=None, heatmap_sigma=10):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.annotations_path = os.path.join(data_dir, 'annotations', 'annotations.json')
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        
        # 加载标注数据
        self.load_annotations()
        
    def load_annotations(self):
        """加载标注文件"""
        if os.path.exists(self.annotations_path):
            with open(self.annotations_path, 'r') as f:
                self.annotations = json.load(f)
        else:
            # 如果是CSV格式
            csv_path = self.annotations_path.replace('.json', '.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                self.annotations = {}
                for _, row in df.iterrows():
                    self.annotations[row['filename']] = {
                        'center_x': row['center_x'],
                        'center_y': row['center_y']
                    }
            else:
                raise FileNotFoundError(f"标注文件不存在: {self.annotations_path}")
        
        # 获取所有图片文件名
        self.image_files = [f for f in os.listdir(self.images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 过滤掉没有标注的图片
        self.image_files = [f for f in self.image_files if f in self.annotations]
        
    def create_heatmap(self, center_x, center_y, img_size, sigma=None):
        """创建高斯热力图"""
        if sigma is None:
            sigma = self.heatmap_sigma
            
        h, w = img_size
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        
        heatmap = np.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))
        return heatmap.astype(np.float32)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图片文件名
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        
        # 加载图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 获取圆心坐标
        annotation = self.annotations[img_file]
        center_x = annotation['center_x']
        center_y = annotation['center_y']
        
        # 创建热力图标签
        heatmap = self.create_heatmap(center_x, center_y, (h, w))
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(
                image=image,
                mask=heatmap,
                keypoints=[(center_x, center_y)]
            )
            image = transformed['image']
            heatmap = transformed['mask']
            if transformed['keypoints']:
                center_x, center_y = transformed['keypoints'][0]
            
            # 更新图像尺寸（如果有resize）
            h, w = image.shape[:2] if isinstance(image, np.ndarray) else image.shape[1:]
        
        # 标准化坐标到 [0, 1]
        norm_x = center_x / w
        norm_y = center_y / h
        
        # 转换为tensor
        if isinstance(image, np.ndarray):
            image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        
        heatmap = torch.FloatTensor(heatmap).unsqueeze(0)
        coords = torch.FloatTensor([norm_x, norm_y])
        
        return {
            'image': image,
            'heatmap': heatmap,
            'coords': coords,
            'filename': img_file
        }
