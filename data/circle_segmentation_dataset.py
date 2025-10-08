import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

class CircleSegmentationDataset(Dataset):
    def __init__(self, data_dir=None, transform=None, circle_radius=20):
        # 如果没有指定data_dir，自动构建路径
        if data_dir is None:
            # 获取dataset.py所在目录（data文件夹）
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            # 向上一级到project目录，再进入datasets
            project_dir = os.path.dirname(current_file_dir)
            data_dir = os.path.join(project_dir, 'datasets')
        
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'oriData')
        self.annotations_dir = os.path.join(data_dir, 'annotations')
        self.transform = transform
        self.circle_radius = circle_radius
        
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
                    circles = []
                    center_points = []
                    for shape in labelme_data.get('shapes', []):
                        # 处理圆形标注
                        if shape['label'] == 'circle' and shape['shape_type'] == 'circle':
                            # labelme的circle格式：points包含两个坐标点
                            # 第一个点是圆心，第二个点是圆周上的点
                            if len(shape['points']) >= 2:
                                center_x, center_y = shape['points'][0]
                                perimeter_x, perimeter_y = shape['points'][1]
                                # 计算半径
                                radius = np.sqrt((center_x - perimeter_x)**2 + (center_y - perimeter_y)**2)
                                circles.append({
                                    'center_x': float(center_x),
                                    'center_y': float(center_y),
                                    'radius': float(radius)
                                })
                        # 兼容旧的点标注格式
                        elif shape['label'] == 'center' and shape['shape_type'] == 'point':
                            if len(shape['points']) > 0:
                                x, y = shape['points'][0]
                                center_points.append((x, y))
                    
                    # 优先使用圆形标注，如果没有则使用点标注
                    if circles:
                        circle_data = circles[0]  # 如果有多个圆，取第一个
                        self.annotations[img_file] = {
                            'center_x': circle_data['center_x'],
                            'center_y': circle_data['center_y'],
                            'radius': circle_data['radius'],
                            'image_width': labelme_data.get('imageWidth', None),
                            'image_height': labelme_data.get('imageHeight', None)
                        }
                        valid_files.append(img_file)
                    elif center_points:
                        # 如果有多个center点，取第一个
                        center_x, center_y = center_points[0]
                        self.annotations[img_file] = {
                            'center_x': float(center_x),
                            'center_y': float(center_y),
                            'radius': 20.0,  # 默认半径
                            'image_width': labelme_data.get('imageWidth', None),
                            'image_height': labelme_data.get('imageHeight', None)
                        }
                        valid_files.append(img_file)
                    else:
                        print(f"Warning: No circle or center point found in {json_file}")
                        
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
            else:
                print(f"Warning: No annotation file found for {img_file}")
        
        self.image_files = valid_files
        print(f"Loaded {len(self.image_files)} images with annotations")
        
    def create_circle_mask(self, center_x, center_y, radius, image_width, image_height):
        """创建圆形分割掩码"""
        mask = np.zeros((image_height, image_width), dtype=np.float32)
        
        # 创建圆形掩码
        y, x = np.ogrid[:image_height, :image_width]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask[dist_from_center <= radius] = 1.0
        
        return mask
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        
        # 加载图像
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Cannot load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取标注信息
        annotation = self.annotations[img_file]
        center_x = annotation['center_x']
        center_y = annotation['center_y']
        radius = annotation['radius']
        image_width = annotation['image_width'] or image.shape[1]
        image_height = annotation['image_height'] or image.shape[0]
        
        # 创建分割掩码
        mask = self.create_circle_mask(
            center_x, 
            center_y, 
            radius,
            image_width,
            image_height
        )
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask

# 使用示例
if __name__ == "__main__":
    # 实例化数据集
    dataset = CircleSegmentationDataset()
    
    # 测试加载一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Filename: {sample['filename']}")