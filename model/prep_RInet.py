import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
from network.single_conv import SingleConv
from network.pure_upsample import PureUpsampling
from network.single_de_conv import SingleDeConv

class UnetInception(nn.Module):
    def __init__(self,config=GlobalConfig(), CoverF=32, WaterF=32):
        super(UnetInception, self).__init__()
        self.config = config
        # input channel: 3, output channel: 96
        """Features with Kernel Size 7---->channel:128 """
        self.downsample_8_Cover = nn.Sequential(
            nn.Conv2d(3, CoverF, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ELU(inplace=True),
            SingleConv(CoverF, out_channels=CoverF, kernel_size=3, stride=1, dilation=1, padding=1),
        )
        self.downsample_8_Secret = nn.Sequential(
            nn.Conv2d(1, WaterF, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ELU(inplace=True),
            SingleConv(WaterF, out_channels=WaterF, kernel_size=3, stride=1, dilation=1, padding=1),
        )
        # 128
        self.downsample_7_Cover = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(CoverF, out_channels=CoverF*2, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF*2, out_channels=CoverF*2, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.downsample_7_Secret = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(WaterF, out_channels=WaterF*2, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(WaterF*2, out_channels=WaterF*2, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 64
        self.downsample_6_Cover = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(CoverF*2, out_channels=CoverF*4, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF*4, out_channels=CoverF*4, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.downsample_6_Secret = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(WaterF*2, out_channels=WaterF*4, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(WaterF*4, out_channels=WaterF*4, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 32
        self.downsample_5_Cover = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(CoverF*4, out_channels=CoverF*8, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF*8, out_channels=CoverF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.downsample_5_Secret = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(WaterF*4, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 16
        self.downsample_4_Cover = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(CoverF*8, out_channels=CoverF*8, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF*8, out_channels=CoverF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.downsample_4_Secret = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 16以下的卷积用4层conv
        self.fullConv = nn.Sequential(
            SingleConv(CoverF*8+WaterF*8, out_channels=CoverF*8+WaterF*8, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(CoverF*8+WaterF*8, out_channels=CoverF*8+WaterF*8, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(CoverF*8+WaterF*8, out_channels=CoverF*8, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(CoverF*8, out_channels=CoverF*8, kernel_size=5, stride=1, dilation=1, padding=2)
        )
        self.pureUpsamle = PureUpsampling(scale=2)
        # 32
        self.upsample4_3 = nn.Sequential(
            SingleConv(CoverF*8*2+WaterF*8, out_channels=CoverF*8, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF*8, out_channels=CoverF*4, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 64
        self.upsample3_3 = nn.Sequential(
            SingleConv(CoverF*4*2+WaterF*4, out_channels=CoverF*4, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF*4, out_channels=CoverF*2, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 128
        self.upsample2_3 = nn.Sequential(
            SingleConv(CoverF*2*2+WaterF*2, out_channels=CoverF*2, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF*2, out_channels=CoverF, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 256
        self.upsample1_3 = nn.Sequential(
            SingleConv(CoverF*2+WaterF, out_channels=CoverF, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF, out_channels=CoverF, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.final256 = nn.Sequential(
            nn.Conv2d(CoverF, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )


    def forward(self, cover, secret):
        # 256
        down8 = self.downsample_8_Cover(cover)
        down8_secret = self.downsample_8_Secret(secret)
        # 128
        down7 = self.downsample_7_Cover(down8)
        down7_secret = self.downsample_7_Secret(down8_secret)
        # 64
        down6 = self.downsample_6_Cover(down7)
        down6_secret = self.downsample_6_Secret(down7_secret)
        # 32
        down5 = self.downsample_5_Cover(down6)
        down5_secret = self.downsample_5_Secret(down6_secret)
        # 16
        down4 = self.downsample_4_Cover(down5)
        down4_secret = self.downsample_4_Secret(down5_secret)
        # 4个Conv
        down4_mix = torch.cat((down4, down4_secret), 1)
        up5 = self.fullConv(down4_mix)
        # 32
        up4_up = self.pureUpsamle(up5)
        up4_cat = torch.cat((down5, down5_secret, up4_up), 1)
        up4 = self.upsample4_3(up4_cat)
        # 64
        up3_up = self.pureUpsamle(up4)
        up3_cat = torch.cat((down6, down6_secret, up3_up), 1)
        up3 = self.upsample3_3(up3_cat)
        # 128
        up2_up = self.pureUpsamle(up3)
        up2_cat = torch.cat((down7, down7_secret, up2_up), 1)
        up2 = self.upsample2_3(up2_cat)
        # 256
        up1_up = self.pureUpsamle(up2)
        up1_cat = torch.cat((down8, down8_secret, up1_up), 1)
        up1 = self.upsample1_3(up1_cat)
        up0 = self.final256(up1)

        return up0
