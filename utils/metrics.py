import numpy as np
import torch

def euclidean_distance(pred_coords, true_coords):
    """计算欧几里得距离"""
    return torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=1))

def calculate_metrics(pred_coords, true_coords, image_size):
    """计算评估指标"""
    # 反标准化坐标
    pred_coords = pred_coords * torch.tensor([image_size[1], image_size[0]]).to(pred_coords.device)
    true_coords = true_coords * torch.tensor([image_size[1], image_size[0]]).to(true_coords.device)
    
    # 计算距离
    distances = euclidean_distance(pred_coords, true_coords)
    
    # 计算各种指标
    mean_error = torch.mean(distances)
    std_error = torch.std(distances)
    max_error = torch.max(distances)
    
    # 精度阈值 (像素)
    accuracy_thresholds = [1, 2, 5, 10, 20]
    accuracies = {}
    
    for threshold in accuracy_thresholds:
        accuracies[f'acc@{threshold}px'] = torch.sum(distances <= threshold).float() / len(distances)
    
    return {
        'mean_error': mean_error.item(),
        'std_error': std_error.item(),
        'max_error': max_error.item(),
        **accuracies
    }

def find_peak_coordinates(heatmap):
    """从热力图中找到峰值坐标"""
    batch_size, _, h, w = heatmap.shape
    heatmap_flat = heatmap.view(batch_size, -1)
    max_indices = torch.argmax(heatmap_flat, dim=1)
    
    y_coords = (max_indices // w).float() / h
    x_coords = (max_indices % w).float() / w
    
    return torch.stack([x_coords, y_coords], dim=1)
