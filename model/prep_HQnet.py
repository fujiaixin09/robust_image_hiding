import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
from network.single_conv import SingleConv
from network.pure_upsample import PureUpsampling
from network.single_de_conv import SingleDeConv

class pureUnet(nn.Module):
    def __init__(self,config=GlobalConfig(), CoverF=32):
        super(pureUnet, self).__init__()
        self.config = config
        # input channel: 3, output channel: 96
        """Features with Kernel Size 7---->channel:WaterF*2 """
        self.downsample_8 = nn.Sequential(
            nn.Conv2d(3, CoverF, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ELU(inplace=False),
            SingleConv(CoverF, out_channels=CoverF, kernel_size=3, stride=1, dilation=1, padding=1),
        )
        # 128
        self.downsample_7 = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(CoverF, out_channels=CoverF * 2, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF * 2, out_channels=CoverF * 2, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 64
        self.downsample_6 = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(CoverF * 2, out_channels=CoverF * 4, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF * 4, out_channels=CoverF * 4, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 32
        self.downsample_5 = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(CoverF * 4, out_channels=CoverF * 8, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF * 8, out_channels=CoverF * 8, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 16
        self.downsample_4 = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(CoverF * 8, out_channels=CoverF * 8, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF * 8, out_channels=CoverF * 8, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 16以下的卷积用4层conv
        self.fullConv = nn.Sequential(
            SingleConv(CoverF * 8, out_channels=CoverF * 8, kernel_size=5, stride=1,dilation=1, padding=2),
            SingleConv(CoverF * 8, out_channels=CoverF * 8, kernel_size=5, stride=1,dilation=1, padding=2),
            SingleConv(CoverF * 8, out_channels=CoverF * 8, kernel_size=5, stride=1, dilation=1,padding=2),
            SingleConv(CoverF * 8, out_channels=CoverF * 8, kernel_size=5, stride=1, dilation=1, padding=2)
        )
        self.pureUpsamle = PureUpsampling(scale=2)
        # 32
        # self.Upsamle4_3 = nn.Sequential(
        #     PureUpsampling(scale=2),
        #     SingleConv(CoverF * 16, out_channels=CoverF * 8, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        self.upsample4_3 = nn.Sequential(
            SingleConv(CoverF * 8 * 2, out_channels=CoverF * 8, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF * 8, out_channels=CoverF * 4, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 64
        # self.Upsamle3_3 = nn.Sequential(
        #     PureUpsampling(scale=2),
        #     SingleConv(CoverF * 8, out_channels=CoverF * 4, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        self.upsample3_3 = nn.Sequential(
            SingleConv(CoverF * 4 * 2, out_channels=CoverF * 4, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(CoverF * 4, out_channels=CoverF * 2, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.final64 = nn.Sequential(
            nn.Conv2d(CoverF * 2, 1, kernel_size=1, padding=0),
            nn.Tanh()
        )
        # # 128
        # self.Upsamle2_3 = nn.Sequential(
        #     PureUpsampling(scale=2),
        #     SingleConv(CoverF * 4, out_channels=CoverF * 2, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # self.upsample2_3 = nn.Sequential(
        #     SingleConv(CoverF * 2 * 2, out_channels=CoverF * 2, kernel_size=3, stride=1, dilation=1, padding=1),
        #     SingleConv(CoverF * 2, out_channels=CoverF * 2, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # # 256
        # self.Upsamle1_3 = nn.Sequential(
        #     PureUpsampling(scale=2),
        #     SingleConv(CoverF * 2, out_channels=CoverF, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # self.upsample1_3 = nn.Sequential(
        #     SingleConv(CoverF * 2, out_channels=CoverF, kernel_size=3, stride=1, dilation=1, padding=1),
        #     SingleConv(CoverF, out_channels=CoverF, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # self.final256 = nn.Sequential(
        #     nn.Conv2d(CoverF, 3, kernel_size=1, padding=0),
        #     nn.Tanh()
        # )


    def forward(self, p):
        # 256
        down8 = self.downsample_8(p)
        # 128
        down7 = self.downsample_7(down8)
        # 64
        down6 = self.downsample_6(down7)
        # 32
        down5 = self.downsample_5(down6)
        # 16
        down4 = self.downsample_4(down5)
        up5 = self.fullConv(down4)

        # 32
        up4_up = self.pureUpsamle(up5)
        up4_cat = torch.cat((down5, up4_up), 1)
        up4 = self.upsample4_3(up4_cat)
        # 64
        up3_up = self.pureUpsamle(up4)
        up3_cat = torch.cat((down6, up3_up), 1)
        up3 = self.upsample3_3(up3_cat)
        up0 = self.final64(up3)
        # # 128
        # up2_up = self.pureUpsamle(up3)
        # up2_cat = torch.cat((down7, up2_up), 1)
        # up2 = self.upsample2_3(up2_cat)
        # # 256
        # up1_up = self.pureUpsamle(up2)
        # up1_cat = torch.cat((down8, up1_up), 1)
        # up1 = self.upsample1_3(up1_cat)
        # up0 = self.finalWater256(up1)

        return up0
