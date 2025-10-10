import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv1(nn.Module):
    """YOLOv1模型实现"""
    def __init__(self, in_channels=3, num_classes=1, S=7, B=2):
        super(YOLOv1, self).__init__()
        self.S = S  # 网格大小
        self.B = B  # 每个网格预测的边界框数量
        self.C = num_classes  # 类别数量
        
        # 特征提取骨干网络 - 类似DarkNet的结构
        self.backbone = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层卷积
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层卷积
            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第四层卷积
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第五层卷积
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        
        # YOLO头 - 预测层
        self.yolo_head = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            # 展平用于全连接
            nn.Flatten(),
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S*S*(B*5 + C))  # 输出形状: S*S*(B*5 + C)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.yolo_head(x)
        # 重塑输出为 [batch_size, S, S, B*5 + C]
        batch_size = x.size(0)
        x = x.view(batch_size, self.S, self.S, self.B*5 + self.C)
        return x

    def detect(self, outputs, conf_threshold=0.5, nms_threshold=0.5):
        """
        从模型输出中提取检测结果
        Args:
            outputs: 模型输出，形状为 [batch_size, S, S, B*5 + C]
            conf_threshold: 置信度阈值
            nms_threshold: 非极大值抑制阈值
        Returns:
            detections: 检测结果列表，每个元素包含 [x1, y1, x2, y2, conf, class_score]
        """
        batch_size = outputs.size(0)
        detections = []
        
        for i in range(batch_size):
            output = outputs[i]
            batch_detections = []
            
            for grid_y in range(self.S):
                for grid_x in range(self.S):
                    cell = output[grid_y, grid_x]
                    
                    for b in range(self.B):
                        # 提取边界框信息
                        tx, ty, tw, th, conf = cell[b*5:b*5+5]
                        
                        # 计算边界框坐标
                        x = (grid_x + F.sigmoid(tx)) / self.S
                        y = (grid_y + F.sigmoid(ty)) / self.S
                        w = torch.exp(tw) * Config.ANCHOR_BOXES[b][0] / self.S if hasattr(Config, 'ANCHOR_BOXES') else torch.exp(tw) / self.S
                        h = torch.exp(th) * Config.ANCHOR_BOXES[b][1] / self.S if hasattr(Config, 'ANCHOR_BOXES') else torch.exp(th) / self.S
                        
                        # 计算类别概率
                        if self.C > 0:
                            class_probs = F.softmax(cell[self.B*5:], dim=0)
                            class_score, class_idx = torch.max(class_probs, dim=0)
                            class_idx = class_idx.item()
                        else:
                            class_score = 1.0
                            class_idx = 0
                        
                        # 计算置信度
                        confidence = F.sigmoid(conf) * class_score
                        
                        # 应用置信度阈值
                        if confidence > conf_threshold:
                            # 转换为对角坐标 [x1, y1, x2, y2]
                            x1 = x - w / 2
                            y1 = y - h / 2
                            x2 = x + w / 2
                            y2 = y + h / 2
                            
                            # 确保坐标在 [0, 1] 范围内
                            x1 = torch.clamp(x1, 0, 1)
                            y1 = torch.clamp(y1, 0, 1)
                            x2 = torch.clamp(x2, 0, 1)
                            y2 = torch.clamp(y2, 0, 1)
                            
                            batch_detections.append([x1, y1, x2, y2, confidence, class_idx])
            
            # 非极大值抑制
            if len(batch_detections) > 0:
                batch_detections = torch.tensor(batch_detections)
                keep = self._nms(batch_detections, nms_threshold)
                detections.append(batch_detections[keep])
            else:
                detections.append(torch.tensor([]))
        
        return detections
    
    def _nms(self, boxes, threshold):
        """非极大值抑制"""
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        
        # 计算面积
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # 按置信度排序
        order = scores.argsort(descending=True)
        
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            
            # 计算交集
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            w = torch.max(torch.tensor(0.0), xx2 - xx1 + 1)
            h = torch.max(torch.tensor(0.0), yy2 - yy1 + 1)
            
            # 计算交并比
            overlap = (w * h) / (areas[i] + areas[order[1:]] - w * h)
            
            # 保留IoU小于阈值的边界框
            indices = torch.where(overlap < threshold)[0]
            order = order[indices + 1]
        
        return keep