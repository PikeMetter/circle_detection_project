import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class CircleDataset(Dataset):
    def __init__(self, data_dir=None, transform=None, heatmap_sigma=10):
        # 如果没有指定data_dir，自动构建路径
        if data_dir is None:
            # 获取dataset.py所在目录（data文件夹）
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            # 向上一级到project目录，再进入datasets
            project_dir = os.path.dirname(current_file_dir)
            data_dir = os.path.join(project_dir, 'datasets')
        
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'oriData')
        self.annotations_dir = os.path.join(data_dir, 'annotations')  # 注意是annotations
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        
        print(f"数据目录: {self.data_dir}")
        print(f"图片目录: {self.images_dir}")
        print(f"标注目录: {self.annotations_dir}")
        
        # 检查目录是否存在
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"图片目录不存在: {self.images_dir}")
        if not os.path.exists(self.annotations_dir):
            raise FileNotFoundError(f"标注目录不存在: {self.annotations_dir}")
        
        # 加载标注数据
        self.load_annotations()
        
    def load_annotations(self):
        """加载labelme格式的标注文件"""
        self.annotations = {}
        
        # 获取所有图片文件
        self.image_files = [f for f in os.listdir(self.images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 为每个图片加载对应的json标注
        valid_files = []
        for img_file in self.image_files:
            # 构建对应的json文件名
            json_file = img_file.replace('.jpg', '.json').replace('.jpeg', '.json').replace('.png', '.json')
            json_path = os.path.join(self.annotations_dir, json_file)
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        labelme_data = json.load(f)
                    
                    # 解析labelme格式的标注
                    center_points = []
                    for shape in labelme_data.get('shapes', []):
                        if shape['label'] == 'center' and shape['shape_type'] == 'point':
                            # labelme的point格式：points是一个包含一个坐标的列表
                            if len(shape['points']) > 0:
                                x, y = shape['points'][0]
                                center_points.append((x, y))
                    
                    if center_points:
                        # 如果有多个center点，取第一个
                        center_x, center_y = center_points[0]
                        self.annotations[img_file] = {
                            'center_x': float(center_x),
                            'center_y': float(center_y),
                            'image_width': labelme_data.get('imageWidth', None),
                            'image_height': labelme_data.get('imageHeight', None)
                        }
                        valid_files.append(img_file)
                    else:
                        print(f"Warning: No center point found in {json_file}")
                        
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
            else:
                print(f"Warning: No annotation file found for {img_file}")
        
        self.image_files = valid_files
        print(f"Loaded {len(self.image_files)} images with annotations")
        
    # ... 其余方法保持不变 ...
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
        if image is None:
            raise ValueError(f"Cannot load image: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 获取圆心坐标
        annotation = self.annotations[img_file]
        center_x = annotation['center_x']
        center_y = annotation['center_y']
        
        # 验证坐标是否在图像范围内
        center_x = max(0, min(w-1, center_x))
        center_y = max(0, min(h-1, center_y))
        
        # 创建热力图标签
        heatmap = self.create_heatmap(center_x, center_y, (h, w))
        
        # 应用数据增强
        original_coords = (center_x, center_y)
        if self.transform:
            try:
                transformed = self.transform(
                    image=image,
                    mask=heatmap,
                    keypoints=[(center_x, center_y)]
                )
                image = transformed['image']
                heatmap = transformed['mask']
                
                # 获取变换后的坐标
                if transformed.get('keypoints') and len(transformed['keypoints']) > 0:
                    center_x, center_y = transformed['keypoints'][0]
                else:
                    # 如果keypoints变换失败，使用原始坐标按比例缩放
                    if hasattr(image, 'shape'):
                        new_h, new_w = image.shape[:2] if isinstance(image, np.ndarray) else image.shape[1:]
                        center_x = original_coords[0] * new_w / w
                        center_y = original_coords[1] * new_h / h
                
                # 更新图像尺寸
                if isinstance(image, np.ndarray):
                    h, w = image.shape[:2]
                else:
                    h, w = image.shape[1], image.shape[2]
                    
            except Exception as e:
                print(f"Transform error for {img_file}: {e}")
                # 如果变换失败，使用原始数据
        
        # 标准化坐标到 [0, 1]
        norm_x = center_x / w if w > 0 else 0
        norm_y = center_y / h if h > 0 else 0
        
        # 确保坐标在有效范围内
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        
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

# 使用示例
if __name__ == "__main__":
    # 直接实例化，会自动找到正确的路径
    dataset = CircleDataset()
    
    # 测试加载一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Heatmap shape: {sample['heatmap'].shape}")
        print(f"Coords: {sample['coords']}")
        print(f"Filename: {sample['filename']}")
