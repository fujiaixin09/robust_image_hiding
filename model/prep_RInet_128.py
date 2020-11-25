import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
from network.single_conv import SingleConv
from network.pure_upsample import PureUpsampling
from network.single_de_conv import SingleDeConv

class UnetInception(nn.Module):
    def __init__(self,config=GlobalConfig()):
        super(UnetInception, self).__init__()
        self.config = config
        # input channel: 3, output channel: 96
        """Features with Kernel Size 7---->channel:128 """
        self.downsample_8_Cover = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, dilation=1, padding=1), #1
            nn.ELU(inplace=True),
            SingleConv(32, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        # self.downsample_8_Secret = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=5, stride=1, dilation=1, padding=2),
        #     nn.ELU(inplace=True),
        #     SingleConv(16, out_channels=16, kernel_size=5, stride=1, dilation=1, padding=2),
        # )
        # 128
        self.downsample_7_Cover = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(32, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(64, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        # self.pureDownsamle = PureUpsampling(scale=1/2)
        self.downsample_7_Secret = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=1, dilation=1, padding=2),
                nn.ELU(inplace=True),
                SingleConv(32, out_channels=32, kernel_size=5, stride=1, dilation=1, padding=2),
        )
        # self.downsample_7_Secret_added = nn.Sequential(
        #     PureUpsampling(scale=1 / 2),
        #     SingleConv(32, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1),
        #     SingleConv(32, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # 64
        self.downsample_6_Cover = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(64, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(128, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        # self.downsample_6_Secret = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, stride=1, dilation=1, padding=1),
        #     nn.ELU(inplace=True),
        #     SingleConv(32, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1),
        # )
        self.downsample_6_Secret_added = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(32, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2),
        )
        # 32
        self.downsample_5_Cover = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(128, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        self.downsample_5_Secret = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(64, out_channels=128, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(128, out_channels=128, kernel_size=5, stride=1, dilation=1, padding=2),
        )
        # 16
        self.downsample_4_Cover = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        self.downsample_4_Secret = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(128, out_channels=128, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(128, out_channels=128, kernel_size=5, stride=1, dilation=1, padding=2),
        )
        # 16以下的卷积用4层conv
        self.fullConv = nn.Sequential(
            SingleConv(256+128, out_channels=256+128, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(256+128, out_channels=256+128, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(256+128, out_channels=256, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(256, out_channels=256, kernel_size=5, stride=1, dilation=1, padding=2)
        )
        self.pureUpsamle = PureUpsampling(scale=2)
        # 32
        self.upsample4_3 = nn.Sequential(
            SingleConv(256*2+128, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(256, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        # 64
        self.upsample3_3 = nn.Sequential(
            SingleConv(128*2+64, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(128, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        # 128
        # self.upsample2_3 = nn.Sequential(
        #     SingleConv(64*2, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1),
        #     SingleConv(64, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        self.upsample2_3_added = nn.Sequential(
            SingleConv(64*2+32, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(64, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        # 256
        self.upsample1_3 = nn.Sequential(
            SingleConv(32*2, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(32, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        # self.upsample1_3_added = nn.Sequential(
        #     SingleConv(32 * 2 + 32, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1),
        #     SingleConv(32, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        self.final256 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, padding=0),
            # nn.Tanh()
        )


    def forward(self, cover, secret, roundSum=1):
        # 256
        down8 = self.downsample_8_Cover(cover)
        # down8_secret_added = self.downsample_8_Secret(secret)
        # 128
        down7 = self.downsample_7_Cover(down8)
        # down7_secret_original = self.downsample_7_Secret(secret)
        down7_secret_added = self.downsample_7_Secret(secret)
        # down7_secret = down7_secret_original # * roundSum + down7_secret_added * (1 - roundSum)
        # 64
        down6 = self.downsample_6_Cover(down7)
        # down6_secret = self.downsample_6_Secret(self.pureDownsamle(secret))
        down6_secret = self.downsample_6_Secret_added(down7_secret_added)
        # down6_secret = down6_secret_original*roundSum+down6_secret_added*(1-roundSum)
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
        # up2_cat_original = torch.cat((down7, up2_up), 1)
        up2_cat_added = torch.cat((down7, down7_secret_added, up2_up), 1)
        # up2_original = self.upsample2_3(up2_cat_original)
        up2_added = self.upsample2_3_added(up2_cat_added)
        up2 = up2_added
        # 256
        up1_up = self.pureUpsamle(up2)
        up1_cat_original = torch.cat((down8,up1_up), 1)
        up1_original = self.upsample1_3(up1_cat_original)
        # up1_cat_added = torch.cat((down8, down8_secret_added, up1_up), 1)
        # up1_added = self.upsample1_3_added(up1_cat_added)
        up1 = up1_original # * roundSum + up1_added * (1 - roundSum)
        up0 = self.final256(up1)

        return up0
