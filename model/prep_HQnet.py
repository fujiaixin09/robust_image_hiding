import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
from network.single_conv import SingleConv
from network.pure_upsample import PureUpsampling
from network.single_de_conv import SingleDeConv

class pureUnet(nn.Module):
    def __init__(self,config=GlobalConfig(), WaterF=64):
        super(pureUnet, self).__init__()
        self.config = config
        # input channel: 3, output channel: 96
        """Features with Kernel Size 7---->channel:WaterF*2 """
        self.downsample_8 = nn.Sequential(
            nn.Conv2d(3, WaterF, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ELU(inplace=True),
            SingleConv(WaterF, out_channels=WaterF, kernel_size=3, stride=1, dilation=1, padding=1),
        )
        # WaterF*2
        self.downsample_7 = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(WaterF, out_channels=WaterF*2, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(WaterF*2, out_channels=WaterF*2, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # WaterF
        self.downsample_6 = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(WaterF*2, out_channels=WaterF*4, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(WaterF*4, out_channels=WaterF*4, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # WaterF
        self.downsample_5 = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(WaterF*4, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # WaterF
        self.downsample_4 = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # WaterF以下的卷积用4层conv
        self.fullConv = nn.Sequential(
            SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=5, stride=1, dilation=1, padding=2)
        )
        # # 8
        # self.downsample_3 = SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=2, dilation=1, padding=1)
        # # 4
        # self.downsample_2 = SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=2, dilation=1, padding=1)
        # # 2
        # self.downsample_1 = SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=2, dilation=1, padding=1)
        # # 1
        # self.downsample_0 = SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=2, dilation=1, padding=1)
        # # 2
        # self.Up8 = nn.Sequential(
        #     PureUpsampling(scale=2),
        #     SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # self.upsample8_3 = nn.Sequential(
        #     # PureUpsampling(scale=2),
        #     SingleConv(1024, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # self.Up7 = nn.Sequential(
        #     PureUpsampling(scale=2),
        #     SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # # 4
        # self.upsample7_3 = nn.Sequential(
        #     # PureUpsampling(scale=2),
        #     SingleConv(1024, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # # 8
        # self.Up6 = nn.Sequential(
        #     PureUpsampling(scale=2),
        #     SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # self.upsample6_3 = nn.Sequential(
        #     # PureUpsampling(scale=2),
        #     SingleConv(1024, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # # WaterF
        # self.Up5 = nn.Sequential(
        #     PureUpsampling(scale=2),
        #     SingleConv(WaterF*8, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # self.upsample5_3 = nn.Sequential(
        #     # PureUpsampling(scale=2),
        #     SingleConv(1024, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        self.pureUpsamle = PureUpsampling(scale=2)
        # WaterF
        self.upsample4_3 = nn.Sequential(
            SingleConv(WaterF*8*2, out_channels=WaterF*8, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(WaterF*8, out_channels=WaterF*4, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # WaterF
        self.upsample3_3 = nn.Sequential(
            SingleConv(WaterF*8, out_channels=WaterF*4, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(WaterF*4, out_channels=WaterF*2, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # WaterF*2

        self.upsample2_3 = nn.Sequential(
            SingleConv(WaterF*4, out_channels=WaterF*2, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(WaterF*2, out_channels=WaterF, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # WaterF*4

        self.upsample1_3 = nn.Sequential(
            SingleConv(WaterF*2, out_channels=WaterF, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(WaterF, out_channels=WaterF, kernel_size=3, stride=1, dilation=1, padding=1)
        )

        self.finalWater256 = nn.Sequential(
            nn.Conv2d(WaterF, 1, kernel_size=1, padding=0),
            nn.Tanh()
        )


    def forward(self, p):
        # WaterF*4
        down8 = self.downsample_8(p)
        # WaterF*2
        down7 = self.downsample_7(down8)
        # WaterF
        down6 = self.downsample_6(down7)
        # WaterF
        down5 = self.downsample_5(down6)
        # WaterF
        down4 = self.downsample_4(down5)
        up5 = self.fullConv(down4)
        # # 8
        # down3 = self.downsample_3(down4)
        # # 4
        # down2 = self.downsample_2(down3)
        # # 2
        # down1 = self.downsample_1(down2)
        # # 1
        # down0 = self.downsample_0(down1)
        # # 2
        # up8_up = self.Up8(down0)
        # up8_cat = torch.cat((down1, up8_up), 1)
        # up8 = self.upsample8_3(up8_cat)
        # # 4
        # up7_up = self.Up7(up8)
        # up7_cat = torch.cat((down2, up7_up), 1)
        # up7 = self.upsample7_3(up7_cat)
        # # 8
        # up6_up = self.Up6(up7)
        # up6_cat = torch.cat((down3, up6_up), 1)
        # up6 = self.upsample6_3(up6_cat)
        # # WaterF
        # up5_up = self.Up5(up6)
        # up5_cat = torch.cat((down4, up5_up), 1)
        # up5 = self.upsample5_3(up5_cat)
        # WaterF
        up4_up = self.pureUpsamle(up5)
        up4_cat = torch.cat((down5, up4_up), 1)
        up4 = self.upsample4_3(up4_cat)
        # WaterF
        up3_up = self.pureUpsamle(up4)
        up3_cat = torch.cat((down6, up3_up), 1)
        up3 = self.upsample3_3(up3_cat)
        # WaterF*2
        up2_up = self.pureUpsamle(up3)
        up2_cat = torch.cat((down7, up2_up), 1)
        up2 = self.upsample2_3(up2_cat)
        # WaterF*4
        up1_up = self.pureUpsamle(up2)
        up1_cat = torch.cat((down8, up1_up), 1)
        up1 = self.upsample1_3(up1_cat)
        up0 = self.finalWater256(up1)

        return up0
