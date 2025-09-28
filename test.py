import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from config import Config
from data import CircleDataset, get_transforms
from models import UNetForCircleCenter
from utils import calculate_metrics, visualize_predictions, setup_logger

def test_model(model_path, test_data_dir, save_results=True, visualize=True):
    """测试模型性能"""
    
    # 设置日志
    logger = setup_logger(Config.LOG_DIR, "test")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 加载模型
    model = UNetForCircleCenter(Config.INPUT_CHANNELS, Config.OUTPUT_CHANNELS)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded model from {model_path}')
        if 'val_metrics' in checkpoint:
            logger.info(f'Model validation metrics: {checkpoint["val_metrics"]}')
    else:
        logger.error(f'Model file not found: {model_path}')
        return
    
    model.to(device)
    model.eval()
    
    # 创建测试数据集
    test_transform = get_transforms('test', Config.IMAGE_SIZE)
    test_dataset = CircleDataset(
        test_data_dir, 
        transform=test_transform,
        heatmap_sigma=Config.HEATMAP_SIGMA
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,  # 单个测试便于可视化
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f'Test samples: {len(test_dataset)}')
    
    # 测试结果存储
    results = []
    all_pred_coords = []
    all_true_coords = []
    
    # 创建结果目录
    results_dir = os.path.join(Config.RESULTS_DIR, 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            coords = batch['coords'].to(device)
            filename = batch['filename'][0]
            
            # 预测
            pred_heatmap, pred_coords = model(images)
            
            # 转换为numpy用于保存结果
            pred_coords_np = pred_coords[0].cpu().numpy()
            true_coords_np = coords[0].cpu().numpy()
            
            # 反标准化坐标
            h, w = Config.IMAGE_SIZE
            pred_x = int(pred_coords_np[0] * w)
            pred_y = int(pred_coords_np[1] * h)
            true_x = int(true_coords_np[0] * w)
            true_y = int(true_coords_np[1] * h)
            
            # 计算误差
            error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
            
            # 保存结果
            results.append({
                'filename': filename,
                'true_x': true_x,
                'true_y': true_y,
                'pred_x': pred_x,
                'pred_y': pred_y,
                'error_px': error
            })
            
            all_pred_coords.append(pred_coords.cpu())
            all_true_coords.append(coords.cpu())
            
            # 可视化前几个结果
            if visualize and batch_idx < 10:
                save_path = os.path.join(results_dir, f'test_{filename}')
                visualize_predictions(
                    images[0], 
                    pred_heatmap[0], 
                    pred_coords_np, 
                    true_coords_np,
                    save_path=save_path,
                    show=False
                )
            
            pbar.set_postfix({'Error': f'{error:.1f}px'})
    
    # 计算整体指标
    all_pred_coords = torch.cat(all_pred_coords, dim=0)
    all_true_coords = torch.cat(all_true_coords, dim=0)
    
    metrics = calculate_metrics(all_pred_coords, all_true_coords, Config.IMAGE_SIZE)
    
    # 打印结果
    logger.info("=" * 50)
    logger.info("TEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total samples: {len(results)}")
    logger.info(f"Mean error: {metrics['mean_error']:.2f} ± {metrics['std_error']:.2f} pixels")
    logger.info(f"Max error: {metrics['max_error']:.2f} pixels")
    logger.info(f"Accuracy @ 1px: {metrics['acc@1px']:.1%}")
    logger.info(f"Accuracy @ 2px: {metrics['acc@2px']:.1%}")
    logger.info(f"Accuracy @ 5px: {metrics['acc@5px']:.1%}")
    logger.info(f"Accuracy @ 10px: {metrics['acc@10px']:.1%}")
    logger.info(f"Accuracy @ 20px: {metrics['acc@20px']:.1%}")
    
    # 保存详细结果
    if save_results:
        # 保存CSV
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(results_dir, 'test_results.csv')
        results_df.to_csv(csv_path, index=False)
        logger.info(f'Detailed results saved to {csv_path}')
        
        # 保存指标
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(results_dir, 'test_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f'Metrics saved to {metrics_path}')
        
        # 分析误差分布
        errors = results_df['error_px'].values
        error_stats = {
            'count': len(errors),
            'mean': np.mean(errors),
            'std': np.std(errors),
            'min': np.min(errors),
            'max': np.max(errors),
            'median': np.median(errors),
            'q25': np.percentile(errors, 25),
            'q75': np.percentile(errors, 75),
            'q95': np.percentile(errors, 95),
            'q99': np.percentile(errors, 99)
        }
        
        logger.info("\nError Distribution:")
        for key, value in error_stats.items():
            logger.info(f"{key}: {value:.2f}")
    
    return metrics, results

def main():
    # 测试最佳模型
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_accuracy_model.pth')
    
    if not os.path.exists(best_model_path):
        print(f"Model not found: {best_model_path}")
        print("Available checkpoints:")
        for f in os.listdir(Config.CHECKPOINT_DIR):
            if f.endswith('.pth'):
                print(f"  - {f}")
        return
    
    # 运行测试
    metrics, results = test_model(
        best_model_path, 
        Config.TEST_DATA_DIR,
        save_results=True,
        visualize=True
    )

if __name__ == '__main__':
    main()
