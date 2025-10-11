import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json

from config import Config
from data import CircleDataset, YOLODataset, get_transforms
from models import ModelFactory
from models.unet.unet import UNetForCircleCenter
from models.yolo.yolo import YOLOv1
from utils import calculate_metrics, visualize_predictions, setup_logger, compute_metrics
from torch.cuda.amp import autocast

def test_model(model_path, test_data_dir, save_results=True, visualize=True, model_type=None):
    """测试模型性能"""
    
    # 设置日志
    logger = setup_logger(Config.LOG_DIR, "test")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 使用指定的模型类型或从配置中获取
    model_type = model_type if model_type is not None else Config.MODEL_TYPE
    logger.info(f'Testing with {model_type} model')
    
    # 根据模型类型选择数据集
    if model_type == 'yolo_v1':
        test_transform = get_transforms('test', Config.IMAGE_SIZE)
        test_dataset = YOLODataset(
            test_data_dir,
            transform=test_transform,
            phase='test'
        )
    else:
        # UNet模型使用CircleDataset
        test_transform = get_transforms('test', Config.IMAGE_SIZE)
        test_dataset = CircleDataset(
            test_data_dir, 
            transform=test_transform,
            heatmap_sigma=Config.HEATMAP_SIGMA
        )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,  # 单个测试便于可视化
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f'Test samples: {len(test_dataset)}')
    
    # 使用ModelFactory创建模型
    if model_type == 'yolo_v1':
        model = ModelFactory.create_model(
            'yolo_v1',
            in_channels=Config.INPUT_CHANNELS,
            num_classes=Config.YOLO_C,
            S=Config.YOLO_S,
            B=Config.YOLO_B
        )
    elif model_type == 'unet_segmentation':
        model = ModelFactory.create_model(
            'unet_segmentation',
            in_channels=Config.INPUT_CHANNELS,
            out_channels=Config.OUTPUT_CHANNELS
        )
    else:
        model = ModelFactory.create_model(
            'unet_circle',
            in_channels=Config.INPUT_CHANNELS,
            out_channels=Config.OUTPUT_CHANNELS
        )
    
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
    
    # 测试结果存储
    results = []
    all_pred_coords = []
    all_true_coords = []
    
    # 创建结果目录
    results_dir = os.path.join(Config.RESULTS_DIR, 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'images'), exist_ok=True)
    
    with torch.no_grad():  # 不计算梯度以加速推理
        pbar = tqdm(test_loader, desc='Testing')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device, non_blocking=True)
            filename = batch['filename'][0]
            
            # 混合精度推理
            with autocast(enabled=Config.USE_MIXED_PRECISION):
                if model_type == 'yolo_v1':
                    # YOLOv1模型处理
                    if 'target' in batch:
                        targets = batch['target'].to(device, non_blocking=True)
                    
                    outputs = model(images)
                    
                    # 使用模型的detect方法提取检测结果
                    detections = model.detect(
                        outputs,
                        conf_threshold=Config.YOLO_CONF_THRESHOLD,
                        nms_threshold=Config.YOLO_NMS_THRESHOLD
                    )
                    
                    # 将预测结果移至CPU并转换为numpy数组
                    detections = detections.cpu().numpy()
                    
                    # 处理YOLOv1的检测结果
                    detection = detections[0]
                    
                    # 从target中提取真实中心点
                    if 'target' in batch:
                        target_grid = targets[0].cpu().numpy()
                        # 寻找包含物体的单元格
                        object_mask = target_grid[..., 4] > 0
                        if np.any(object_mask):
                            # 找到第一个包含物体的单元格
                            cell_indices = np.where(object_mask)
                            cell_y, cell_x = cell_indices[0][0], cell_indices[1][0]
                            cell_target = target_grid[cell_y, cell_x]
                            # 计算真实中心点坐标 (归一化)
                            gt_x = (cell_x + cell_target[0]) / Config.YOLO_S
                            gt_y = (cell_y + cell_target[1]) / Config.YOLO_S
                        else:
                            # 如果没有检测到物体，使用默认值
                            gt_x, gt_y = 0.5, 0.5
                    else:
                        # 如果没有target，使用默认值
                        gt_x, gt_y = 0.5, 0.5
                    
                    # 获取预测中心点
                    if len(detection) > 0:
                        # 取置信度最高的检测结果
                        best_idx = np.argmax(detection[:, 4])
                        best_det = detection[best_idx]
                        x1, y1, x2, y2, conf, class_idx = best_det
                        # 计算预测中心点 (归一化)
                        pred_x_norm = (x1 + x2) / 2
                        pred_y_norm = (y1 + y2) / 2
                        # 计算半径
                        radius = max((x2 - x1) / 2, (y2 - y1) / 2)
                        confidence = conf
                    else:
                        # 如果没有预测结果，使用默认值
                        pred_x_norm, pred_y_norm = 0.5, 0.5
                        radius = 0.1
                        confidence = 0.0
                    
                    # 反标准化坐标
                    h, w = Config.IMAGE_SIZE
                    pred_x = int(pred_x_norm * w)
                    pred_y = int(pred_y_norm * h)
                    true_x = int(gt_x * w)
                    true_y = int(gt_y * h)
                    
                    # 计算误差
                    error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)** 2)
                    
                    # 保存结果
                    results.append({
                        'filename': filename,
                        'true_x': true_x,
                        'true_y': true_y,
                        'pred_x': pred_x,
                        'pred_y': pred_y,
                        'error_px': error,
                        'confidence': confidence,
                        'radius': radius * max(w, h)  # 转换半径到像素
                    })
                    
                    # 保存坐标用于整体指标计算
                    all_pred_coords.append(torch.tensor([pred_x_norm, pred_y_norm]))
                    all_true_coords.append(torch.tensor([gt_x, gt_y]))
                    
                    # 可视化前几个结果
                    if visualize and batch_idx < 10:
                        save_path = os.path.join(results_dir, 'images', f'test_{filename}')
                        
                        # 处理图像用于可视化
                        image_np = images[0].cpu().numpy().transpose(1, 2, 0)
                        image_np = (image_np * 255).astype(np.uint8)
                        
                        visualize_predictions(
                            image_np, 
                            None,  # YOLO不使用热力图
                            (pred_x_norm, pred_y_norm), 
                            (gt_x, gt_y),
                            save_path=save_path,
                            show=False,
                            radius=int(radius * max(w, h)),
                            confidence=confidence
                        )
                else:
                    # UNet模型处理
                    coords = batch['coords'].to(device, non_blocking=True)
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
                    error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)** 2)
                    
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
                        save_path = os.path.join(results_dir, 'images', f'test_{filename}')
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
    if all_pred_coords and all_true_coords:
        # 确保所有坐标是相同维度
        if model_type == 'yolo_v1':
            all_pred_coords = torch.stack(all_pred_coords)
            all_true_coords = torch.stack(all_true_coords)
        else:
            all_pred_coords = torch.cat(all_pred_coords, dim=0)
            all_true_coords = torch.cat(all_true_coords, dim=0)
        
        metrics = calculate_metrics(all_pred_coords, all_true_coords, Config.IMAGE_SIZE)
    else:
        metrics = {
            'mean_error': 0.0,
            'std_error': 0.0,
            'max_error': 0.0,
            'acc@1px': 0.0,
            'acc@2px': 0.0,
            'acc@5px': 0.0,
            'acc@10px': 0.0,
            'acc@20px': 0.0
        }
    
    # 打印结果
    logger.info("=" * 50)
    logger.info(f"TEST RESULTS ({model_type})")
    logger.info("=" * 50)
    logger.info(f"Total samples: {len(results)}")
    logger.info(f"Mean error: {metrics['mean_error']:.2f} ± {metrics['std_error']:.2f} pixels")
    logger.info(f"Max error: {metrics['max_error']:.2f} pixels")
    logger.info(f"Accuracy @ 1px: {metrics['acc@1px']:.1%}")
    logger.info(f"Accuracy @ 2px: {metrics['acc@2px']:.1%}")
    logger.info(f"Accuracy @ 5px: {metrics['acc@5px']:.1%}")
    logger.info(f"Accuracy @ 10px: {metrics['acc@10px']:.1%}")
    logger.info(f"Accuracy @ 20px: {metrics['acc@20px']:.1%}")
    
    if model_type == 'yolo_v1' and results:
        # 计算YOLO特有指标
        confidences = [r['confidence'] for r in results]
        mean_confidence = np.mean(confidences)
        logger.info(f"Mean confidence: {mean_confidence:.4f}")
    
    # 保存详细结果
    if save_results:
        # 保存CSV
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(results_dir, f'test_results_{model_type}.csv')
        results_df.to_csv(csv_path, index=False)
        logger.info(f'Detailed results saved to {csv_path}')
        
        # 保存指标
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(results_dir, f'test_metrics_{model_type}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f'Metrics saved to {metrics_path}')
        
        # 保存为JSON格式便于后续分析
        json_results = {
            'model_type': model_type,
            'metrics': metrics,
            'results': results,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        json_path = os.path.join(results_dir, f'test_results_{model_type}.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        logger.info(f'JSON results saved to {json_path}')
        
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
    
    # 运行测试 - 使用配置中的模型类型
    metrics, results = test_model(
        best_model_path, 
        Config.TEST_DATA_DIR,
        save_results=True,
        visualize=True,
        model_type=Config.MODEL_TYPE
    )

if __name__ == '__main__':
    main()
