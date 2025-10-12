import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Conv(nn.Module):
    """标准卷积块"""
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
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """C2f模块 (Fasterized CSP Bottleneck)"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(C2f, self).__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """空间金字塔池化快速层"""
    def __init__(self, c1, c2, k=5):
        super(SPPF, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class YOLOv8(nn.Module):
    """YOLOv8模型实现"""
    def __init__(self, in_channels=3, num_classes=1, anchors=None):
        super(YOLOv8, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 默认锚框 (YOLOv8通常使用无锚框设计，但为了兼容性保留此参数)
        if anchors is None:
            # YOLOv8使用无锚框设计，这里仅作示例
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
        """构建YOLOv8网络"""
        # 主干网络 (Backbone)
        self.conv1 = Conv(3, 64, 3, 2)       # 640x640 -> 320x320
        self.conv2 = Conv(64, 128, 3, 2)     # 320x320 -> 160x160
        self.c2f1 = C2f(128, 128, 3, True)   # 160x160
        self.conv3 = Conv(128, 256, 3, 2)    # 160x160 -> 80x80
        self.c2f2 = C2f(256, 256, 6, True)   # 80x80
        self.conv4 = Conv(256, 512, 3, 2)    # 80x80 -> 40x40
        self.c2f3 = C2f(512, 512, 6, True)   # 40x40
        self.conv5 = Conv(512, 1024, 3, 2)   # 40x40 -> 20x20
        self.c2f4 = C2f(1024, 1024, 3, True) # 20x20
        self.sppf = SPPF(1024, 1024, 5)      # SPPF层
        
        # 颈部网络 (Neck - FPN+PAN)
        self.upsample1 = nn.Upsample(None, 2, 'nearest')
        self.c2f5 = C2f(1536, 512, 3, False) # 40x40
        
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.c2f6 = C2f(768, 256, 3, False)  # 80x80
        
        self.conv6 = Conv(256, 256, 3, 2)    # 80x80 -> 40x40
        self.c2f7 = C2f(768, 512, 3, False)  # 40x40
        
        self.conv7 = Conv(512, 512, 3, 2)    # 40x40 -> 20x20
        self.c2f8 = C2f(1536, 1024, 3, False) # 20x20
        
        # 检测头 (Detection Heads)
        self.detect_small = nn.Conv2d(256, 64, 1)    # 小尺度检测头
        self.detect_medium = nn.Conv2d(512, 64, 1)   # 中等尺度检测头
        self.detect_large = nn.Conv2d(1024, 64, 1)   # 大尺度检测头
        
        # 输出层 (通常YOLOv8使用解耦头)
        self.cv2_small = nn.Conv2d(64, 3 * (5 + self.num_classes), 1)   # 小尺度输出
        self.cv2_medium = nn.Conv2d(64, 3 * (5 + self.num_classes), 1)  # 中等尺度输出
        self.cv2_large = nn.Conv2d(64, 3 * (5 + self.num_classes), 1)   # 大尺度输出
        
        self.cv3_small = nn.Conv2d(64, 3 * self.num_classes, 1)         # 小尺度分类输出
        self.cv3_medium = nn.Conv2d(64, 3 * self.num_classes, 1)        # 中等尺度分类输出
        self.cv3_large = nn.Conv2d(64, 3 * self.num_classes, 1)         # 大尺度分类输出
    
    def forward(self, x):
        # 主干网络
        x = self.conv1(x)       # 320x320
        x = self.conv2(x)       # 160x160
        x = self.c2f1(x)        # 160x160
        x_p3 = self.conv3(x)    # 80x80
        x_p3 = self.c2f2(x_p3)  # 80x80 (P3)
        x_p4 = self.conv4(x_p3) # 40x40
        x_p4 = self.c2f3(x_p4)  # 40x40 (P4)
        x_p5 = self.conv5(x_p4) # 20x20
        x_p5 = self.c2f4(x_p5)  # 20x20
        x_p5 = self.sppf(x_p5)  # SPPF层 (20x20)
        
        # 颈部网络 (FPN)
        x_up1 = self.upsample1(x_p5)  # 20x20 -> 40x40
        x = torch.cat([x_up1, x_p4], 1)  # 40x40
        x_p4_neck = self.c2f5(x)  # 40x40
        
        x_up2 = self.upsample2(x_p4_neck)  # 40x40 -> 80x80
        x = torch.cat([x_up2, x_p3], 1)   # 80x80
        x_p3_neck = self.c2f6(x)  # 80x80
        
        # 颈部网络 (PAN)
        x = self.conv6(x_p3_neck)  # 80x80 -> 40x40
        x = torch.cat([x, x_p4_neck], 1)  # 40x40
        x_p4_neck2 = self.c2f7(x)  # 40x40
        
        x = self.conv7(x_p4_neck2)  # 40x40 -> 20x20
        x = torch.cat([x, x_p5], 1)  # 20x20
        x_p5_neck = self.c2f8(x)   # 20x20
        
        # 检测头
        head_small = self.detect_small(x_p3_neck)    # 80x80
        head_medium = self.detect_medium(x_p4_neck2) # 40x40
        head_large = self.detect_large(x_p5_neck)    # 20x20
        
        # 输出层
        output_small = self.cv2_small(head_small)    # 80x80
        output_medium = self.cv2_medium(head_medium) # 40x40
        output_large = self.cv2_large(head_large)    # 20x20
        
        cls_small = self.cv3_small(head_small)       # 80x80
        cls_medium = self.cv3_medium(head_medium)    # 40x40
        cls_large = self.cv3_large(head_large)       # 20x20
        
        return [output_small, output_medium, output_large], [cls_small, cls_medium, cls_large]
    
    def detect(self, outputs, conf_threshold=0.5, nms_threshold=0.5):
        """
        从模型输出中提取检测结果
        Args:
            outputs: 模型输出，包含边界框和分类输出
            conf_threshold: 置信度阈值
            nms_threshold: 非极大值抑制阈值
        Returns:
            detections: 检测结果列表
        """
        # 这里简化实现，实际应用中需要根据具体需求实现
        batch_size = outputs[0][0].size(0)
        detections = []
        
        for i in range(batch_size):
            # 简化处理，实际应该处理三个尺度的输出
            output = outputs[0][0][i]  # 取第一个尺度
            # 转换为检测框格式
            # 这里只是一个示例，实际需要根据YOLOv8的输出格式进行转换
            detections.append(output)
        
        return detections