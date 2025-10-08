import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetSegmentation(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNetSegmentation, self).__init__()
        
        features = init_features
        # 编码器部分
        self.encoder1 = self.double_conv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.double_conv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.double_conv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.double_conv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 编码器底部
        self.bottleneck = self.double_conv(features * 8, features * 16)
        
        # 解码器部分
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self.double_conv((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self.double_conv((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self.double_conv((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self.double_conv(features * 2, features)
        
        # 输出层
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
        
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
        # 编码器路径
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # 编码器底部
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # 解码器路径
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # 输出
        outputs = self.conv(dec1)
        # 使用sigmoid激活函数将输出限制在[0,1]范围内
        outputs = torch.sigmoid(outputs)
        
        return outputs

# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    model = UNetSegmentation(in_channels=3, out_channels=1)
    
    # 创建一个示例输入张量 (batch_size=1, channels=3, height=256, width=256)
    sample_input = torch.randn(1, 3, 256, 256)
    
    # 前向传播
    output = model(sample_input)
    
    # 打印输入和输出的形状
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")