import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

from config import Config
from data.circle_segmentation_dataset import CircleSegmentationDataset
from models.unet_segmentation import UNetSegmentation
from utils import setup_logger

def dice_coefficient(pred, target, smooth=1e-6):
    """计算Dice系数"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice

def dice_loss(pred, target, smooth=1e-6):
    """Dice Loss"""
    return 1 - dice_coefficient(pred, target, smooth)

def train_segmentation_model():
    """训练圆形分割模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化日志记录器
    logger = setup_logger("logs", "segmentation_training")
    logger.info("Starting segmentation training...")
    
    # 创建数据集和数据加载器
    train_dataset = CircleSegmentationDataset()
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0  # 在Windows上可能需要设置为0
    )
    
    # 创建模型
    model = UNetSegmentation(in_channels=3, out_channels=1).to(device)
    
    # 定义损失函数和优化器
    # 可以使用BCE Loss或Dice Loss，或者两者的组合
    criterion = nn.BCELoss()  # 二元交叉熵损失
    # 或者使用Dice Loss: criterion = dice_loss
    # 或者组合损失: criterion = lambda pred, target: nn.BCELoss()(pred, target) + dice_loss(pred, target)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 训练循环
    best_loss = float('inf')
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        total_dice = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS}')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, masks)
            dice = dice_coefficient(outputs, masks)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_dice += dice.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}'
            })
        
        # 计算平均损失和Dice系数
        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        
        # 更新学习率
        scheduler.step()
        
        logger.info(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Dice={avg_dice:.4f}, LR={scheduler.get_last_lr()[0]:.6f}')
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Dice={avg_dice:.4f}, LR={scheduler.get_last_lr()[0]:.6f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'dice': avg_dice
            }, os.path.join(Config.CHECKPOINT_DIR, 'best_segmentation_model.pth'))
            logger.info(f'Best model saved at epoch {epoch+1} (loss: {avg_loss:.4f})')
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'dice': avg_dice
            }, os.path.join(Config.CHECKPOINT_DIR, f'segmentation_checkpoint_epoch_{epoch+1}.pth'))
            logger.info(f'Checkpoint saved for epoch {epoch+1}')
        
        # 显存清理
        torch.cuda.empty_cache()
        gc.collect()
    
    logger.info("Training completed!")
    print("Training completed!")

if __name__ == "__main__":
    train_segmentation_model()