import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config import Config

class YOLODataset(Dataset):
    """YOLOv1数据集类，用于处理圆形检测任务"""
    def __init__(self, data_dir=None, transform=None, S=7, B=2, C=1):
        # 如果没有指定data_dir，自动构建路径
        if data_dir is None:
            # 获取yolo_dataset.py所在目录（data文件夹）
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            # 向上一级到project目录，再进入datasets_processed
            project_dir = os.path.dirname(current_file_dir)
            data_dir = os.path.join(project_dir, 'datasets_processed')
        
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.annotations_dir = os.path.join(data_dir, 'annotations')
        self.transform = transform
        
        # YOLO参数
        self.S = S  # 网格大小
        self.B = B  # 每个网格预测的边界框数量
        self.C = C  # 类别数量
        
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
                    for shape in labelme_data.get('shapes', []):
                        if shape['label'] == 'circle':
                            # 假设circle形状是用矩形近似的
                            if shape['shape_type'] == 'rectangle':
                                x1, y1 = shape['points'][0]
                                x2, y2 = shape['points'][1]
                                # 计算中心点和半径
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                width = abs(x2 - x1)
                                height = abs(y2 - y1)
                                # 假设宽高近似相等，取平均作为直径
                                radius = (width + height) / 4  # 除以4因为直径是宽或高
                                circles.append((center_x, center_y, radius))
                        elif shape['label'] == 'center' and shape['shape_type'] == 'point':
                            # 如果只有中心点，需要估算半径（这种情况可能需要调整）
                            if len(shape['points']) > 0:
                                center_x, center_y = shape['points'][0]
                                # 这里假设一个默认半径，实际应用中可能需要根据具体情况调整
                                radius = 20  # 示例值，需要根据实际数据调整
                                circles.append((center_x, center_y, radius))
                    
                    if circles:
                        self.annotations[img_file] = {
                            'circles': circles,
                            'image_width': labelme_data.get('imageWidth', None),
                            'image_height': labelme_data.get('imageHeight', None)
                        }
                        valid_files.append(img_file)
                    else:
                        print(f"Warning: No circle found in {json_file}")
                        
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
            else:
                print(f"Warning: No annotation file found for {img_file}")
        
        self.image_files = valid_files
        print(f"Loaded {len(self.image_files)} images with annotations")
    
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
        
        # 获取圆的标注
        annotation = self.annotations[img_file]
        circles = annotation['circles']
        
        # 创建YOLO格式的目标标签
        # 初始化为全0的张量 [S, S, 5 + C]
        # 5表示: x, y, w, h, confidence
        # C表示类别概率
        targets = np.zeros((self.S, self.S, 5 + self.C))
        
        # 处理每个圆
        for (center_x, center_y, radius) in circles:
            # 验证坐标是否在图像范围内
            center_x = max(0, min(w-1, center_x))
            center_y = max(0, min(h-1, center_y))
            
            # 计算边界框的宽和高（直径）
            box_w = 2 * radius
            box_h = 2 * radius
            
            # 标准化坐标到 [0, 1]
            x = center_x / w
            y = center_y / h
            box_w = box_w / w
            box_h = box_h / h
            
            # 计算目标所在的网格
            grid_x = int(x * self.S)
            grid_y = int(y * self.S)
            
            # 计算网格内的相对坐标
            x_grid = x * self.S - grid_x
            y_grid = y * self.S - grid_y
            
            # 设置目标标签
            # 注意：这里假设每个网格最多只有一个物体
            targets[grid_y, grid_x, 0] = x_grid  # x坐标（网格内）
            targets[grid_y, grid_x, 1] = y_grid  # y坐标（网格内）
            targets[grid_y, grid_x, 2] = box_w   # 宽度
            targets[grid_y, grid_x, 3] = box_h   # 高度
            targets[grid_y, grid_x, 4] = 1.0     # 置信度
            
            # 设置类别标签（这里只有一个类别）
            if self.C > 0:
                targets[grid_y, grid_x, 5] = 1.0  # 类别概率
        
        # 应用数据增强
        if self.transform:
            try:
                # 准备要变换的数据
                transformed = self.transform(
                    image=image,
                    # 对于YOLO，我们需要传递边界框信息
                    bboxes=[(center_x, center_y, box_w*w, box_h*h, 0) for (center_x, center_y, radius) in circles]
                )
                image = transformed['image']
                
                # 更新图像尺寸
                if isinstance(image, np.ndarray):
                    h, w = image.shape[:2]
                else:
                    h, w = image.shape[1], image.shape[2]
                    
                # 如果数据增强改变了边界框，需要重新计算targets
                # 这里为了简化，假设数据增强不会改变边界框位置
                # 在实际应用中，应该根据transform的输出重新计算targets
                    
            except Exception as e:
                print(f"Transform error for {img_file}: {e}")
                # 如果变换失败，使用原始数据
        
        # 转换为tensor
        if isinstance(image, np.ndarray):
            image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        
        targets = torch.FloatTensor(targets)
        
        return {
            'image': image,
            'targets': targets,
            'filename': img_file
        }

# 使用示例
if __name__ == "__main__":
    # 直接实例化，会自动找到正确的路径
    dataset = YOLODataset(S=Config.YOLO_S, B=Config.YOLO_B, C=Config.YOLO_C)
    
    # 测试加载一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Targets shape: {sample['targets'].shape}")
        print(f"Filename: {sample['filename']}")
        
        # 查找有物体的网格
        has_object = (sample['targets'][..., 4] == 1).nonzero()
        if len(has_object) > 0:
            print(f"Found {len(has_object)} grids with objects")
            for grid in has_object:
                y, x = grid
                print(f"Grid ({x}, {y}):")
                print(f"  x: {sample['targets'][y, x, 0]}")
                print(f"  y: {sample['targets'][y, x, 1]}")
                print(f"  w: {sample['targets'][y, x, 2]}")
                print(f"  h: {sample['targets'][y, x, 3]}")
                print(f"  conf: {sample['targets'][y, x, 4]}")