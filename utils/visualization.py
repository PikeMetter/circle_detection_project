import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_predictions(image, pred_heatmap, pred_coords, true_coords, 
                         save_path=None, show=True):
    """可视化预测结果"""
    # 转换为numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        image = image.permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
    if isinstance(pred_heatmap, torch.Tensor):
        if pred_heatmap.dim() == 4:
            pred_heatmap = pred_heatmap[0, 0]
        pred_heatmap = pred_heatmap.cpu().numpy()
    
    h, w = image.shape[:2]
    
    # 反标准化坐标
    pred_x = int(pred_coords[0] * w)
    pred_y = int(pred_coords[1] * h)
    true_x = int(true_coords[0] * w)
    true_y = int(true_coords[1] * h)
    
    # 创建可视化图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图 + 预测和真实点
    axes[0].imshow(image)
    axes[0].plot(pred_x, pred_y, 'ro', markersize=10, label=f'Pred ({pred_x}, {pred_y})')
    axes[0].plot(true_x, true_y, 'go', markersize=10, label=f'True ({true_x}, {true_y})')
    axes[0].set_title('Predictions')
    axes[0].legend()
    axes[0].axis('off')
    
    # 热力图
    heatmap_resized = cv2.resize(pred_heatmap, (w, h))
    axes[1].imshow(image, alpha=0.7)
    axes[1].imshow(heatmap_resized, alpha=0.5, cmap='hot')
    axes[1].set_title('Heatmap Overlay')
    axes[1].axis('off')
    
    # 纯热力图
    axes[2].imshow(heatmap_resized, cmap='hot')
    axes[2].plot(pred_x, pred_y, 'ro', markersize=8)
    axes[2].set_title('Heatmap')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def create_comparison_plot(results_list, save_path=None):
    """创建多个结果的对比图"""
    n_samples = len(results_list)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results_list):
        image = result['image']
        pred_heatmap = result['pred_heatmap']
        pred_coords = result['pred_coords']
        true_coords = result['true_coords']
        filename = result.get('filename', f'Sample {i+1}')
        
        # 处理图像
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            image = image.permute(1, 2, 0).cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        
        h, w = image.shape[:2]
        pred_x = int(pred_coords[0] * w)
        pred_y = int(pred_coords[1] * h)
        true_x = int(true_coords[0] * w)
        true_y = int(true_coords[1] * h)
        
        # 原图 + 点
        axes[i, 0].imshow(image)
        axes[i, 0].plot(pred_x, pred_y, 'ro', markersize=8, label='Pred')
        axes[i, 0].plot(true_x, true_y, 'go', markersize=8, label='True')
        axes[i, 0].set_title(f'{filename}')
        axes[i, 0].legend()
        axes[i, 0].axis('off')
        
        # 热力图叠加
        if isinstance(pred_heatmap, torch.Tensor):
            pred_heatmap = pred_heatmap[0, 0].cpu().numpy() if pred_heatmap.dim() == 4 else pred_heatmap.cpu().numpy()
        heatmap_resized = cv2.resize(pred_heatmap, (w, h))
        axes[i, 1].imshow(image, alpha=0.7)
        axes[i, 1].imshow(heatmap_resized, alpha=0.5, cmap='hot')
        axes[i, 1].set_title('Heatmap Overlay')
        axes[i, 1].axis('off')
        
        # 误差
        error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
        axes[i, 2].text(0.5, 0.5, f'Error: {error:.1f}px', 
                       transform=axes[i, 2].transAxes, 
                       ha='center', va='center', fontsize=14)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
