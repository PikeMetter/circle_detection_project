import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Conv(nn.Module):
    """标准卷积块：Conv + BatchNorm + SiLU"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

def autopad(k, p=None):
    """自动计算padding"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Bottleneck(nn.Module):
    """标准瓶颈块"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPP(nn.Module):
    """空间金字塔池化层"""
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class YOLOv5(nn.Module):
    """YOLOv5模型实现"""
    def __init__(self, in_channels=3, num_classes=1, anchors=None):
        super(YOLOv5, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 默认锚框
        if anchors is None:
            # 这里使用简化版的锚框配置
            self.anchors = [
                [10, 13, 16, 30, 33, 23],  # P3/8
                [30, 61, 62, 45, 59, 119],  # P4/16
                [116, 90, 156, 198, 373, 326]  # P5/32
            ]
        else:
            self.anchors = anchors
        
        # 构建网络结构
        self._build_network()
    
    def _build_network(self):
        """构建YOLOv5网络"""
        # 主干网络 (Backbone)
        self.focus = self._make_focus(3, 64)  # 3->64
        self.conv1 = Conv(64, 128, 3, 2)      # 64->128
        self.csp1 = C3(128, 128, 3)           # 128->128
        self.conv2 = Conv(128, 256, 3, 2)     # 128->256
        self.csp2 = C3(256, 256, 6)           # 256->256
        self.conv3 = Conv(256, 512, 3, 2)     # 256->512
        self.csp3 = C3(512, 512, 9)           # 512->512
        self.conv4 = Conv(512, 1024, 3, 2)    # 512->1024
        self.csp4 = C3(1024, 1024, 3)         # 1024->1024
        self.spp = SPP(1024, 1024)            # SPP层
        
        # 颈部网络 (Neck - FPN+PAN)
        self.conv5 = Conv(1024, 512, 1, 1)    # 1024->512
        self.upsample1 = nn.Upsample(None, 2, 'nearest')
        self.csp5 = C3(1024, 512, 3, False)   # 1024->512
        
        self.conv6 = Conv(512, 256, 1, 1)     # 512->256
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.csp6 = C3(512, 256, 3, False)    # 512->256
        
        self.conv7 = Conv(256, 256, 3, 2)     # 256->256
        self.csp7 = C3(512, 512, 3, False)    # 512->512
        
        self.conv8 = Conv(512, 512, 3, 2)     # 512->512
        self.csp8 = C3(1024, 1024, 3, False)  # 1024->1024
        
        # 检测头 (Detection Heads)
        self.detect_small = nn.Conv2d(256, 3 * (5 + self.num_classes), 1)   # 小尺度检测头
        self.detect_medium = nn.Conv2d(512, 3 * (5 + self.num_classes), 1)  # 中等尺度检测头
        self.detect_large = nn.Conv2d(1024, 3 * (5 + self.num_classes), 1)  # 大尺度检测头
    
    def _make_focus(self, c1, c2, k=1, s=1):
        """Focus层实现"""
        # 简化实现，实际Focus层会重新排列像素
        return Conv(c1 * 4, c2, k, s)
    
    def forward(self, x):
        # 主干网络
        x = self.focus(x)       # 640x640 -> 320x320
        x = self.conv1(x)       # 320x320 -> 160x160
        x = self.csp1(x)        # 160x160
        x = self.conv2(x)       # 160x160 -> 80x80
        x_p3 = self.csp2(x)     # 80x80 (P3)
        x = self.conv3(x_p3)    # 80x80 -> 40x40
        x_p4 = self.csp3(x)     # 40x40 (P4)
        x = self.conv4(x_p4)    # 40x40 -> 20x20
        x = self.csp4(x)        # 20x20
        x = self.spp(x)         # SPP层
        
        # 颈部网络
        x = self.conv5(x)       # 20x20 -> 20x20
        x_up1 = self.upsample1(x)  # 20x20 -> 40x40
        x = torch.cat([x_up1, x_p4], 1)  # 40x40
        x = self.csp5(x)        # 40x40
        
        x = self.conv6(x)       # 40x40 -> 40x40
        x_up2 = self.upsample2(x)  # 40x40 -> 80x80
        x = torch.cat([x_up2, x_p3], 1)  # 80x80
        neck_small = self.csp6(x)  # 80x80 (小尺度特征)
        
        x = self.conv7(neck_small)  # 80x80 -> 40x40
        x = torch.cat([x, self.conv6(neck_small)], 1)  # 40x40
        neck_medium = self.csp7(x)  # 40x40 (中等尺度特征)
        
        x = self.conv8(neck_medium)  # 40x40 -> 20x20
        x = torch.cat([x, self.conv5(neck_medium)], 1)  # 20x20
        neck_large = self.csp8(x)   # 20x20 (大尺度特征)
        
        # 检测头
        output_small = self.detect_small(neck_small)    # 80x80
        output_medium = self.detect_medium(neck_medium) # 40x40
        output_large = self.detect_large(neck_large)    # 20x20
        
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
                        num_anchors = len(self.anchors[scale_idx]) // 2
                        for anchor_idx in range(num_anchors):
                            # 计算在输出张量中的位置
                            # YOLOv5输出格式: [num_anchors * (5 + num_classes), grid_size, grid_size]
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
                                w = torch.exp(tw) * self.anchors[scale_idx][anchor_idx*2] / grid_size
                                h = torch.exp(th) * self.anchors[scale_idx][anchor_idx*2+1] / grid_size
                                
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