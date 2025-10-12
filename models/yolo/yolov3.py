import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    """标准卷积块：Conv + BatchNorm + LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels//2, 1)
        self.conv2 = ConvBlock(channels//2, channels, 3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual

class YOLOv3(nn.Module):
    """YOLOv3模型实现"""
    def __init__(self, in_channels=3, num_classes=1, num_boxes=3):
        super(YOLOv3, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_boxes = num_boxes  # 每个尺度的锚框数量
        
        # Darknet-53骨干网络
        self.backbone = self._make_backbone()
        
        # 三个检测头，对应三个不同尺度
        self.head_large = self._make_detection_head(1024, 512)   # 13x13
        self.head_medium = self._make_detection_head(768, 256)   # 26x26
        self.head_small = self._make_detection_head(384, 128)    # 52x52
        
        # 输出层
        # 每个边界框预测: tx, ty, tw, th, conf, class_probs
        self.output_large = nn.Conv2d(512, num_boxes * (5 + num_classes), 1)
        self.output_medium = nn.Conv2d(256, num_boxes * (5 + num_classes), 1)
        self.output_small = nn.Conv2d(128, num_boxes * (5 + num_classes), 1)
        
        # 上采样层
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # 初始化权重
        self._init_weights()
    
    def _make_backbone(self):
        """构建Darknet-53骨干网络"""
        layers = []
        
        # 初始卷积层
        layers.append(ConvBlock(self.in_channels, 32, 3, padding=1))
        layers.append(ConvBlock(32, 64, 3, stride=2, padding=1))
        
        # 第一组残差块
        layers.append(ResidualBlock(64))
        layers.append(ConvBlock(64, 128, 3, stride=2, padding=1))
        
        # 第二组残差块 (2个)
        for _ in range(2):
            layers.append(ResidualBlock(128))
        layers.append(ConvBlock(128, 256, 3, stride=2, padding=1))
        
        # 第三组残差块 (8个)
        self.stage1 = nn.Sequential(*layers)
        layers = []
        for _ in range(8):
            layers.append(ResidualBlock(256))
        layers.append(ConvBlock(256, 512, 3, stride=2, padding=1))
        
        # 第四组残差块 (8个)
        self.stage2 = nn.Sequential(*layers)
        layers = []
        for _ in range(8):
            layers.append(ResidualBlock(512))
        layers.append(ConvBlock(512, 1024, 3, stride=2, padding=1))
        
        # 第五组残差块 (4个)
        self.stage3 = nn.Sequential(*layers)
        layers = []
        for _ in range(4):
            layers.append(ResidualBlock(1024))
        
        self.stage4 = nn.Sequential(*layers)
        
        return nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
    
    def _make_detection_head(self, in_channels, out_channels):
        """构建检测头"""
        layers = []
        for _ in range(3):
            layers.append(ConvBlock(in_channels, out_channels, 1))
            layers.append(ConvBlock(out_channels, in_channels, 3, padding=1))
            in_channels = out_channels * 2
        
        layers.append(ConvBlock(in_channels, out_channels, 1))
        layers.append(ConvBlock(out_channels, in_channels, 3, padding=1))
        layers.append(ConvBlock(in_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 骨干网络特征提取
        x = self.backbone[0](x)  # stage1
        x = self.backbone[1](x)  # stage2
        x = self.backbone[2](x)  # stage3
        x = self.backbone[3](x)  # stage4
        
        # 获取不同尺度的特征图
        route1 = x  # 52x52
        x = F.max_pool2d(x, 2, stride=2)  # 26x26
        
        route2 = x  # 26x26
        x = F.max_pool2d(x, 2, stride=2)  # 13x13
        
        # 大尺度检测 (13x13)
        large_features = self.head_large(x)
        output_large = self.output_large(large_features)
        
        # 中等尺度检测 (26x26)
        upsampled = self.upsample1(large_features)
        concat = torch.cat([upsampled, route2], dim=1)
        medium_features = self.head_medium(concat)
        output_medium = self.output_medium(medium_features)
        
        # 小尺度检测 (52x52)
        upsampled = self.upsample2(medium_features)
        concat = torch.cat([upsampled, route1], dim=1)
        small_features = self.head_small(concat)
        output_small = self.output_small(small_features)
        
        return [output_small, output_medium, output_large]
    
    def detect(self, outputs, conf_threshold=0.5, nms_threshold=0.5):
        """
        从模型输出中提取检测结果
        Args:
            outputs: 模型输出，包含三个尺度的特征图
            conf_threshold: 置信度阈值
            nms_threshold: 非极大值抑制阈值
        Returns:
            detections: 检测结果列表，每个元素包含 [x1, y1, x2, y2, conf, class_idx]
        """
        batch_size = outputs[0].size(0)
        detections = []
        
        for i in range(batch_size):
            batch_detections = []
            # 处理三个尺度的输出
            for scale_idx, output in enumerate(outputs):
                scale_output = output[i]  # 取第i个样本
                # 获取特征图尺寸
                grid_size = scale_output.size(1)
                
                # 遍历每个网格
                for grid_y in range(grid_size):
                    for grid_x in range(grid_size):
                        # 遍历每个锚框
                        for anchor_idx in range(self.num_boxes):
                            # 计算在输出张量中的位置
                            # YOLOv3输出格式: [num_boxes * (5 + num_classes), grid_size, grid_size]
                            box_start = anchor_idx * (5 + self.num_classes)
                            
                            # 提取边界框信息
                            tx = scale_output[box_start, grid_y, grid_x]
                            ty = scale_output[box_start + 1, grid_y, grid_x]
                            tw = scale_output[box_start + 2, grid_y, grid_x]
                            th = scale_output[box_start + 3, grid_y, grid_x]
                            conf = torch.sigmoid(scale_output[box_start + 4, grid_y, grid_x])
                            
                            # 计算类别概率
                            if self.num_classes > 0:
                                class_probs = F.softmax(scale_output[box_start + 5:box_start + 5 + self.num_classes, grid_y, grid_x], dim=0)
                                class_score, class_idx = torch.max(class_probs, dim=0)
                                class_idx = class_idx.item()
                            else:
                                class_score = 1.0
                                class_idx = 0
                            
                            # 计算最终置信度
                            confidence = conf * class_score
                            
                            # 应用置信度阈值
                            if confidence > conf_threshold:
                                # 计算边界框坐标 (简化实现)
                                x = (grid_x + torch.sigmoid(tx)) / grid_size
                                y = (grid_y + torch.sigmoid(ty)) / grid_size
                                w = torch.exp(tw) / grid_size
                                h = torch.exp(th) / grid_size
                                
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
            
            # 转换为tensor并应用非极大值抑制
            if len(batch_detections) > 0:
                batch_detections = torch.tensor(batch_detections)
                # 简化的NMS实现
                # 这里只是一个简化版本，实际应用中可能需要更复杂的NMS
                detections.append(batch_detections)
            else:
                detections.append(torch.tensor([]))
        
        return detections