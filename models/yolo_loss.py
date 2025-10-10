import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class YOLOv1Loss(nn.Module):
    """YOLOv1损失函数实现"""
    def __init__(self, S=7, B=2, C=1, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOv1Loss, self).__init__()
        self.S = S  # 网格大小
        self.B = B  # 每个网格预测的边界框数量
        self.C = C  # 类别数量
        self.lambda_coord = lambda_coord  # 坐标损失权重
        self.lambda_noobj = lambda_noobj  # 无物体损失权重
        
    def forward(self, predictions, targets):
        """
        计算YOLOv1损失
        Args:
            predictions: 模型预测，形状为 [batch_size, S, S, B*5 + C]
            targets: 目标标签，形状为 [batch_size, S, S, 5 + C]，其中包含网格中是否有物体、边界框坐标和类别
        Returns:
            total_loss: 总损失
            losses: 各部分损失的字典
        """
        batch_size = predictions.size(0)
        
        # 创建掩码来识别包含物体的网格
        obj_mask = targets[..., 4] == 1  # [batch_size, S, S]
        noobj_mask = targets[..., 4] == 0  # [batch_size, S, S]
        
        # 初始化损失
        coord_loss = 0.0
        conf_loss_obj = 0.0
        conf_loss_noobj = 0.0
        class_loss = 0.0
        
        # 遍历每个样本
        for b in range(batch_size):
            # 遍历每个网格
            for i in range(self.S):
                for j in range(self.S):
                    # 检查该网格是否包含物体
                    if obj_mask[b, i, j]:
                        # 提取目标信息
                        target_box = targets[b, i, j, :5]  # [x, y, w, h, conf]
                        target_class = targets[b, i, j, 5:5+self.C]
                        
                        # 提取所有预测的边界框
                        pred_boxes = predictions[b, i, j, :self.B*5].view(self.B, 5)
                        pred_class = predictions[b, i, j, self.B*5:self.B*5+self.C]
                        
                        # 计算每个预测边界框与目标边界框的IoU
                        ious = []
                        for k in range(self.B):
                            pred_box = pred_boxes[k]
                            iou = self._calculate_iou(pred_box, target_box)
                            ious.append(iou)
                        
                        # 找到IoU最大的边界框
                        ious = torch.tensor(ious, device=predictions.device)
                        best_box_idx = torch.argmax(ious)
                        best_pred_box = pred_boxes[best_box_idx]
                        
                        # 计算坐标损失（只针对最佳边界框）
                        # 中心点坐标损失 (x, y) - 使用MSE
                        coord_loss += self.lambda_coord * F.mse_loss(
                            torch.sigmoid(best_pred_box[:2]), target_box[:2], reduction='sum'
                        )
                        
                        # 宽高损失 (w, h) - 使用MSE和平方根
                        coord_loss += self.lambda_coord * F.mse_loss(
                            torch.sqrt(torch.abs(best_pred_box[2:4]) + 1e-6), 
                            torch.sqrt(target_box[2:4] + 1e-6), 
                            reduction='sum'
                        )
                        
                        # 计算包含物体的置信度损失
                        conf_loss_obj += F.mse_loss(
                            torch.sigmoid(best_pred_box[4]), target_box[4], reduction='sum'
                        )
                        
                        # 计算类别损失
                        if self.C > 0:
                            if self.C == 1:
                                # 二分类情况
                                class_loss += F.binary_cross_entropy_with_logits(
                                    pred_class, target_class, reduction='sum'
                                )
                            else:
                                # 多分类情况
                                class_loss += F.cross_entropy(
                                    pred_class.view(1, -1), 
                                    torch.argmax(target_class).view(1), 
                                    reduction='sum'
                                )
                        
                    # 计算不包含物体的置信度损失
                    if noobj_mask[b, i, j]:
                        # 对所有边界框计算无物体置信度损失
                        for k in range(self.B):
                            conf_loss_noobj += self.lambda_noobj * F.mse_loss(
                                torch.sigmoid(predictions[b, i, j, k*5+4]), 
                                targets[b, i, j, 4], 
                                reduction='sum'
                            )
        
        # 计算总损失
        total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
        
        # 归一化损失
        total_loss /= batch_size
        coord_loss /= batch_size
        conf_loss_obj /= batch_size
        conf_loss_noobj /= batch_size
        class_loss /= batch_size
        
        # 返回损失字典
        losses = {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'conf_loss_obj': conf_loss_obj,
            'conf_loss_noobj': conf_loss_noobj,
            'class_loss': class_loss
        }
        
        return total_loss, losses
    
    def _calculate_iou(self, pred_box, target_box):
        """
        计算预测边界框与目标边界框的IoU
        Args:
            pred_box: 预测边界框，格式为 [x, y, w, h]（归一化坐标）
            target_box: 目标边界框，格式为 [x, y, w, h]（归一化坐标）
        Returns:
            iou: 交并比
        """
        # 转换中心点坐标和宽高为对角坐标 [x1, y1, x2, y2]
        # 预测边界框
        pred_x1 = torch.sigmoid(pred_box[0]) - torch.abs(pred_box[2]) / 2
        pred_y1 = torch.sigmoid(pred_box[1]) - torch.abs(pred_box[3]) / 2
        pred_x2 = torch.sigmoid(pred_box[0]) + torch.abs(pred_box[2]) / 2
        pred_y2 = torch.sigmoid(pred_box[1]) + torch.abs(pred_box[3]) / 2
        
        # 目标边界框
        target_x1 = target_box[0] - target_box[2] / 2
        target_y1 = target_box[1] - target_box[3] / 2
        target_x2 = target_box[0] + target_box[2] / 2
        target_y2 = target_box[1] + target_box[3] / 2
        
        # 计算交集区域
        x1 = torch.max(pred_x1, target_x1)
        y1 = torch.max(pred_y1, target_y1)
        x2 = torch.min(pred_x2, target_x2)
        y2 = torch.min(pred_y2, target_y2)
        
        # 计算交集面积
        intersection = torch.max(torch.tensor(0.0, device=pred_box.device), x2 - x1) * \
                      torch.max(torch.tensor(0.0, device=pred_box.device), y2 - y1)
        
        # 计算并集面积
        pred_area = torch.abs(pred_x2 - pred_x1) * torch.abs(pred_y2 - pred_y1)
        target_area = target_box[2] * target_box[3]
        union = pred_area + target_area - intersection
        
        # 计算IoU
        iou = intersection / (union + 1e-6)  # 添加小值防止除零
        
        return iou