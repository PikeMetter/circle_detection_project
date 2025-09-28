import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetForCircleCenter(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetForCircleCenter, self).__init__()
        
        # 编码器 (下采样)
        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = self.double_conv(512, 1024)
        
        # 解码器 (上采样)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.double_conv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.double_conv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.double_conv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.double_conv(128, 64)
        
        # 分割头 - 生成热力图
        self.seg_head = nn.Conv2d(64, out_channels, 1)
        
        # 回归头 - 直接预测坐标
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.reg_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # x, y 坐标
        )
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # 瓶颈
        b = self.bottleneck(self.pool(e4))
        
        # 解码
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # 输出
        heatmap = torch.sigmoid(self.seg_head(d1))  # 热力图
        
        # 全局特征用于回归
        global_feat = self.global_pool(d1).flatten(1)
        coords = self.reg_head(global_feat)  # 直接坐标回归
        
        return heatmap, coords
