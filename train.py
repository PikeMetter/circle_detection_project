import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast  # 混合精度
from tqdm import tqdm
import gc

from config import Config
from data import CircleDataset, YOLODataset, get_transforms
from models import UNetForCircleCenter, CombinedLoss, ModelFactory, YOLOv1Loss
from utils import calculate_metrics, setup_logger

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, logger, model_type):
    model.train()
    total_loss = 0.0
    
    # 根据模型类型初始化不同的损失变量
    if model_type == 'yolo_v1':
        total_coord_loss = 0.0
        total_conf_loss = 0.0
        total_cls_loss = 0.0
    else:
        total_hm_loss = 0.0
        total_coord_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')
    
    # 梯度累积
    accumulation_steps = Config.GRADIENT_ACCUMULATION_STEPS
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        
        # 混合精度训练
        with autocast(enabled=Config.USE_MIXED_PRECISION):
            if model_type == 'yolo_v1':
                # YOLOv1模型训练逻辑
                targets = batch['target'].to(device, non_blocking=True)
                outputs = model(images)
                loss_components = criterion(outputs, targets)
                
                if len(loss_components) == 4:
                    loss, coord_loss, conf_loss, cls_loss = loss_components
                else:
                    # 兼容不同的返回格式
                    loss = loss_components
                    coord_loss = torch.tensor(0.0, device=device)
                    conf_loss = torch.tensor(0.0, device=device)
                    cls_loss = torch.tensor(0.0, device=device)
                
                loss = loss / accumulation_steps  # 梯度累积缩放
            else:
                # UNet模型训练逻辑
                heatmaps = batch['heatmap'].to(device, non_blocking=True)
                coords = batch['coords'].to(device, non_blocking=True)
                
                pred_heatmap, pred_coords = model(images)
                loss_components = criterion(pred_heatmap, pred_coords, heatmaps, coords)
                
                if len(loss_components) == 3:
                    loss, hm_loss, coord_loss = loss_components
                else:
                    # 兼容不同的返回格式
                    loss = loss_components
                    hm_loss = torch.tensor(0.0, device=device)
                    coord_loss = torch.tensor(0.0, device=device)
                
                loss = loss / accumulation_steps  # 梯度累积缩放
        
        # 反向传播
        if Config.USE_MIXED_PRECISION:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度累积更新
        if (batch_idx + 1) % accumulation_steps == 0:
            if Config.USE_MIXED_PRECISION:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # 累计损失
        total_loss += loss.item() * accumulation_steps
        
        if model_type == 'yolo_v1':
            total_coord_loss += coord_loss.item() * accumulation_steps
            total_conf_loss += conf_loss.item() * accumulation_steps
            total_cls_loss += cls_loss.item() * accumulation_steps
        else:
            total_hm_loss += hm_loss.item()
            total_coord_loss += coord_loss.item()
        
        # 清理显存
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        # 更新进度条
        if model_type == 'yolo_v1':
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Coord': f'{total_coord_loss/(batch_idx+1):.4f}',
                'Conf': f'{total_conf_loss/(batch_idx+1):.4f}',
                'Cls': f'{total_cls_loss/(batch_idx+1):.4f}',
                'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
            })
        else:
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'HM': f'{total_hm_loss/(batch_idx+1):.4f}',
                'Coord': f'{total_coord_loss/(batch_idx+1):.4f}',
                'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
            })
    
    # 处理最后不足accumulation_steps的批次
    if len(train_loader) % accumulation_steps != 0:
        if Config.USE_MIXED_PRECISION:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(train_loader)
    
    if model_type == 'yolo_v1':
        avg_coord_loss = total_coord_loss / len(train_loader)
        avg_conf_loss = total_conf_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        logger.info(f'Train Epoch {epoch}: Loss={avg_loss:.4f}, Coord={avg_coord_loss:.4f}, Conf={avg_conf_loss:.4f}, Cls={avg_cls_loss:.4f}')
        logger.info(f'GPU Max Memory: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB')
        return avg_loss, avg_coord_loss, avg_conf_loss, avg_cls_loss
    else:
        avg_hm_loss = total_hm_loss / len(train_loader)
        avg_coord_loss = total_coord_loss / len(train_loader)
        logger.info(f'Train Epoch {epoch}: Loss={avg_loss:.4f}, HM={avg_hm_loss:.4f}, Coord={avg_coord_loss:.4f}')
        logger.info(f'GPU Max Memory: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB')
        return avg_loss, avg_hm_loss, avg_coord_loss

def validate(model, val_loader, criterion, device, epoch, logger, model_type):
    model.eval()
    total_loss = 0.0
    
    # 根据模型类型初始化不同的损失变量和指标容器
    if model_type == 'yolo_v1':
        total_coord_loss = 0.0
        total_conf_loss = 0.0
        total_cls_loss = 0.0
    else:
        total_hm_loss = 0.0
        total_coord_loss = 0.0
        all_pred_coords = []
        all_true_coords = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Val Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device, non_blocking=True)
            
            # 混合精度推理
            with autocast(enabled=Config.USE_MIXED_PRECISION):
                if model_type == 'yolo_v1':
                    # YOLOv1模型验证逻辑
                    targets = batch['target'].to(device, non_blocking=True)
                    outputs = model(images)
                    loss_components = criterion(outputs, targets)
                    
                    if len(loss_components) == 4:
                        loss, coord_loss, conf_loss, cls_loss = loss_components
                    else:
                        # 兼容不同的返回格式
                        loss = loss_components
                        coord_loss = torch.tensor(0.0, device=device)
                        conf_loss = torch.tensor(0.0, device=device)
                        cls_loss = torch.tensor(0.0, device=device)
                else:
                    # UNet模型验证逻辑
                    heatmaps = batch['heatmap'].to(device, non_blocking=True)
                    coords = batch['coords'].to(device, non_blocking=True)
                    
                    pred_heatmap, pred_coords = model(images)
                    loss_components = criterion(pred_heatmap, pred_coords, heatmaps, coords)
                    
                    if len(loss_components) == 3:
                        loss, hm_loss, coord_loss = loss_components
                    else:
                        # 兼容不同的返回格式
                        loss = loss_components
                        hm_loss = torch.tensor(0.0, device=device)
                        coord_loss = torch.tensor(0.0, device=device)
            
            total_loss += loss.item()
            
            if model_type == 'yolo_v1':
                total_coord_loss += coord_loss.item()
                total_conf_loss += conf_loss.item()
                total_cls_loss += cls_loss.item()
            else:
                total_hm_loss += hm_loss.item()
                total_coord_loss += coord_loss.item()
                all_pred_coords.append(pred_coords.cpu())
                all_true_coords.append(coords.cpu())
            
            # 更新进度条
            if model_type == 'yolo_v1':
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.4f}',
                    'coord_loss': f'{total_coord_loss/(batch_idx+1):.4f}',
                    'conf_loss': f'{total_conf_loss/(batch_idx+1):.4f}',
                    'cls_loss': f'{total_cls_loss/(batch_idx+1):.4f}'
                })
            else:
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.4f}',
                    'hm_loss': f'{total_hm_loss/(batch_idx+1):.4f}',
                    'coord_loss': f'{total_coord_loss/(batch_idx+1):.4f}'
                })
        
    avg_loss = total_loss / len(val_loader)
    
    if model_type == 'yolo_v1':
        avg_coord_loss = total_coord_loss / len(val_loader)
        avg_conf_loss = total_conf_loss / len(val_loader)
        avg_cls_loss = total_cls_loss / len(val_loader)
        
        # 对于YOLOv1，添加专门的评估指标计算
        metrics = {'mean_error': 0.0, 'std_error': 0.0, 'acc@1px': 0.0, 'acc@2px': 0.0, 'acc@5px': 0.0}
        
        logger.info(f"Val Epoch {epoch}: Loss={avg_loss:.4f}, Coord={avg_coord_loss:.4f}, Conf={avg_conf_loss:.4f}, Class={avg_cls_loss:.4f}")
        return avg_loss, avg_coord_loss, avg_conf_loss, avg_cls_loss, metrics
    else:
        # 计算UNet指标
        all_pred_coords = torch.cat(all_pred_coords, dim=0)
        all_true_coords = torch.cat(all_true_coords, dim=0)
        
        metrics = calculate_metrics(all_pred_coords, all_true_coords, Config.IMAGE_SIZE)
        avg_hm_loss = total_hm_loss / len(val_loader)
        avg_coord_loss = total_coord_loss / len(val_loader)
        
        logger.info(f'Val Epoch {epoch}: Loss={avg_loss:.4f}, HM={avg_hm_loss:.4f}, Coord={avg_coord_loss:.4f}')
        logger.info(f"Mean Error={metrics['mean_error']:.2f}px, Std={metrics['std_error']:.2f}px")
        logger.info(f"Accuracies - 1px:{metrics['acc@1px']:.3f}, 2px:{metrics['acc@2px']:.3f}, 5px:{metrics['acc@5px']:.3f}")
        
        return avg_loss, avg_hm_loss, avg_coord_loss, metrics

def main():
    # 设置日志
    logger = setup_logger(Config.LOG_DIR)
    
    # 检查CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Please check your PyTorch installation.")
        return
    
    device = torch.device('cuda')
    logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
    logger.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
    
    # 优化显存使用
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # 优化卷积性能
    torch.backends.cudnn.deterministic = False  # 提升性能，牺牲确定性
    
    # 根据模型类型选择数据集
    model_type = Config.MODEL_TYPE
    logger.info(f'Using {model_type} model')
    
    # 创建数据集
    train_transform = get_transforms('train', Config.IMAGE_SIZE)
    val_transform = get_transforms('val', Config.IMAGE_SIZE)
    
    try:
        if model_type == 'yolo_v1':
            train_dataset = YOLODataset(
                Config.TRAIN_DATA_DIR,
                transform=train_transform,
                phase='train'
            )
            
            val_dataset = YOLODataset(
                Config.VAL_DATA_DIR,
                transform=val_transform,
                phase='val'
            )
        else:
            train_dataset = CircleDataset(
                Config.TRAIN_DATA_DIR,
                transform=train_transform,
                heatmap_sigma=Config.HEATMAP_SIGMA
            )
            
            val_dataset = CircleDataset(
                Config.VAL_DATA_DIR,
                transform=val_transform,
                heatmap_sigma=Config.HEATMAP_SIGMA
            )
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.error("Please check your data paths in config.py")
        return
    
    # 数据加载器 - 优化内存使用
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # Windows下建议设置为2-4
        pin_memory=True,
        persistent_workers=True,
        drop_last=True  # 避免最后一个batch大小不一致
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    logger.info(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    logger.info(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
    
    # 使用ModelFactory创建模型
    if model_type == 'yolo_v1':
        model = ModelFactory.create_model(
            'yolo_v1',
            in_channels=Config.INPUT_CHANNELS,
            num_classes=Config.YOLO_C,
            S=Config.YOLO_S,
            B=Config.YOLO_B
        )
        # 创建YOLOv1损失函数
        criterion = ModelFactory.create_loss('yolo_v1')
    elif model_type == 'unet_segmentation':
        model = ModelFactory.create_model(
            'unet_segmentation',
            in_channels=Config.INPUT_CHANNELS,
            out_channels=Config.OUTPUT_CHANNELS
        )
        # 创建UNet分割损失函数
        criterion = ModelFactory.create_loss('unet_segmentation')
    else:
        model = ModelFactory.create_model(
            'unet_circle',
            in_channels=Config.INPUT_CHANNELS,
            out_channels=Config.OUTPUT_CHANNELS
        )
        # 创建UNet圆形检测损失函数
        criterion = ModelFactory.create_loss('unet_circle')
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total params: {total_params:,}, Trainable: {trainable_params:,}')
    
    # 估算模型显存使用
    model_memory = total_params * 4 / 1024**3  # 4 bytes per parameter (float32)
    logger.info(f'Estimated model memory: {model_memory:.2f}GB')
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=1e-4,
        eps=1e-8
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=Config.NUM_EPOCHS,
        eta_min=1e-6
    )
    
    # 混合精度训练
    scaler = GradScaler() if Config.USE_MIXED_PRECISION else None
    if Config.USE_MIXED_PRECISION:
        logger.info("Using mixed precision training")
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(Config.LOG_DIR, 'tensorboard'))
    
    # 训练状态
    best_val_loss = float('inf')
    best_mean_error = float('inf')
    start_epoch = 1
    
    # 检查是否有预训练模型
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'last_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resumed training from epoch {start_epoch}")
    
    logger.info("Starting training...")
    logger.info(f"Configuration: Batch Size={Config.BATCH_SIZE}, Image Size={Config.IMAGE_SIZE}")
    logger.info(f"Mixed Precision={Config.USE_MIXED_PRECISION}, Gradient Accumulation={Config.GRADIENT_ACCUMULATION_STEPS}")
    
    try:
        for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
            # 训练一个epoch
            if model_type == 'yolo_v1':
                train_loss, train_coord_loss, train_conf_loss, train_cls_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, scaler,
                    device, epoch, logger, model_type
                )
            else:
                train_loss, train_hm_loss, train_coord_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, scaler,
                    device, epoch, logger, model_type
                )
            
            # 验证
            if model_type == 'yolo_v1':
                val_loss, val_coord_loss, val_conf_loss, val_cls_loss, metrics = validate(
                    model, val_loader, criterion, device, epoch, logger, model_type
                )
            else:
                val_loss, val_hm_loss, val_coord_loss, metrics = validate(
                    model, val_loader, criterion, device, epoch, logger, model_type
                )
            
            # 学习率调度
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # TensorBoard记录
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            
            if model_type == 'yolo_v1':
                writer.add_scalar('Loss/train_coord', train_coord_loss, epoch)
                writer.add_scalar('Loss/val_coord', val_coord_loss, epoch)
                writer.add_scalar('Loss/train_conf', train_conf_loss, epoch)
                writer.add_scalar('Loss/val_conf', val_conf_loss, epoch)
                writer.add_scalar('Loss/train_cls', train_cls_loss, epoch)
                writer.add_scalar('Loss/val_cls', val_cls_loss, epoch)
            else:
                writer.add_scalar('Loss/train_hm', train_hm_loss, epoch)
                writer.add_scalar('Loss/val_hm', val_hm_loss, epoch)
                writer.add_scalar('Loss/train_coord', train_coord_loss, epoch)
                writer.add_scalar('Loss/val_coord', val_coord_loss, epoch)
                writer.add_scalar('Metrics/MeanError', metrics['mean_error'], epoch)
                writer.add_scalar('Metrics/StdError', metrics['std_error'], epoch)
                writer.add_scalar('Metrics/Acc@1px', metrics['acc@1px'], epoch)
                writer.add_scalar('Metrics/Acc@2px', metrics['acc@2px'], epoch)
                writer.add_scalar('Metrics/Acc@5px', metrics['acc@5px'], epoch)
            
            writer.add_scalar('LR', current_lr, epoch)
            writer.add_scalar('GPU_Memory', torch.cuda.max_memory_allocated()/1024**3, epoch)
            
            # 保存最佳模型
            is_best_loss = val_loss < best_val_loss
            
            # 创建可pickle的配置字典
            config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__') and not callable(v)}
            
            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'best_val_loss': best_val_loss,
                'model_type': Config.MODEL_TYPE,
                'config': config_dict
            }
            
            # 添加特定模型类型的损失值
            if model_type == 'yolo_v1':
                checkpoint.update({
                    'val_coord_loss': val_coord_loss,
                    'val_conf_loss': val_conf_loss,
                    'val_cls_loss': val_cls_loss,
                    'metrics': metrics
                })
            else:
                checkpoint.update({
                    'val_hm_loss': val_hm_loss,
                    'val_coord_loss': val_coord_loss,
                    'metrics': metrics,
                    'best_mean_error': best_mean_error
                })
                is_best_error = metrics['mean_error'] < best_mean_error
                if is_best_error:
                    best_mean_error = metrics['mean_error']
            
            # 保存最佳损失模型
            if is_best_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(Config.CHECKPOINT_DIR, 'best_loss_model.pth'))
                logger.info(f'Best loss model saved at epoch {epoch} (loss: {val_loss:.4f})')
            
            # 保存最佳精度模型（仅UNet）
            if model_type != 'yolo_v1' and is_best_error:
                torch.save(checkpoint, os.path.join(Config.CHECKPOINT_DIR, 'best_accuracy_model.pth'))
                logger.info(f'Best accuracy model saved at epoch {epoch} (error: {best_mean_error:.2f}px)')
            
            # 保存最新检查点
            torch.save(checkpoint, os.path.join(Config.CHECKPOINT_DIR, 'last_checkpoint.pth'))
            
            # 定期保存检查点
            if epoch % 10 == 0:
                torch.save(checkpoint, os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'))
                logger.info(f'Checkpoint saved for epoch {epoch}')
            
            # 清理显存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 重置显存统计
            torch.cuda.reset_peak_memory_stats()
            
            # 打印验证结果
            if model_type == 'yolo_v1':
                logger.info(f'Val Epoch {epoch}: Loss={val_loss:.4f}, Coord={val_coord_loss:.4f}, Conf={val_conf_loss:.4f}, Cls={val_cls_loss:.4f}')
            else:
                logger.info(f'Val Epoch {epoch}: Loss={val_loss:.4f}, HM={val_hm_loss:.4f}, Coord={val_coord_loss:.4f}')
            
            logger.info(f'Epoch {epoch}/{Config.NUM_EPOCHS} completed. LR: {current_lr:.2e}')
            logger.info("-" * 80)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        writer.close()
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        if model_type != 'yolo_v1':
            logger.info(f"Best mean error: {best_mean_error:.2f}px")

if __name__ == '__main__':
    main()
