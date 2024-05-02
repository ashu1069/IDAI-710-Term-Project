import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        #downsampling
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 512)

        #upsampling
        self.up1 = DoubleConv(1024, 256)
        self.up2 = DoubleConv(512, 128)
        self.up3 = DoubleConv(256, 64)
        self.up4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        #maxpool to extract significant features
        self.pool = nn.MaxPool2d(2)
        #upscale those significant features using bilinear interpolation (can use bicubic interpolation 
        #too but in case of haze, it may exacerbate noise since it uses 16x16 neighborhood instead of 4x4 by bilinear)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):

        #explained in the architecture diagrams
        x1 = self.inc(x)
        x2 = self.pool(x1)
        x2 = self.down1(x2)
        x3 = self.pool(x2)
        x3 = self.down2(x3)
        x4 = self.pool(x3)
        x4 = self.down3(x4)
        x5 = self.pool(x4)
        x5 = self.down4(x5)

        x = self.upsample(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up1(x)
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up4(x)
        x = self.outc(x)
        return x
