import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
from network.single_conv import SingleConv
from network.pure_upsample import PureUpsampling
from network.single_de_conv import SingleDeConv

class Prep_pureUnet(nn.Module):
    def __init__(self,config=GlobalConfig()):
        super(Prep_pureUnet, self).__init__()
        self.config = config
        # input channel: 3, output channel: 96
        """Features with Kernel Size 7---->channel:64 """
        self.downsample_8 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, dilation=1, padding=1), #1
            nn.ELU(inplace=True),
            SingleConv(32, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        # 64
        self.downsample_7 = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(32, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(64, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        # 32
        self.downsample_6 = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(64, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(128, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        # 32
        self.downsample_5 = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(128, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        # 16
        self.downsample_4 = nn.Sequential(
            PureUpsampling(scale=1 / 2),
            SingleConv(256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        # 16以下的卷积用4层conv
        self.fullConv = nn.Sequential(
            SingleConv(256, out_channels=256, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(256, out_channels=256, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(256, out_channels=256, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(256, out_channels=256, kernel_size=5, stride=1, dilation=1, padding=2)
        )
        self.pureUpsamle = PureUpsampling(scale=128/64)
        self.pureUpsamle4 = PureUpsampling(scale=128/32)
        # self.downsample = PureUpsampling(scale=32/128)
        # 32
        self.upsample4_3 = nn.Sequential(
            SingleConv(512, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(256, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        # 32
        self.upsample3_3 = nn.Sequential(
            SingleConv(256, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(128, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        self.final64 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )
        # 64
        self.upsample2_3 = nn.Sequential(
            SingleConv(128, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(64, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        self.final128 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )
        # 128
        self.upsample1_3 = nn.Sequential(
            SingleConv(64, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1), #1
            SingleConv(32, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1), #2
        )
        self.final256 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )
        # self.param32 = torch.tensor(0, dtype=torch.float32,requires_grad=True)
        # self.param64 = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        # self.param128 = torch.tensor(1, dtype=torch.float32, requires_grad=True)


    def forward(self, p, roundSum):
        # 128
        down8 = self.downsample_8(p)
        # 64
        down7 = self.downsample_7(down8)
        # 32
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
        # 32
        up3_up = self.pureUpsamle(up4)
        up3_cat = torch.cat((down6, up3_up), 1)
        up3 = self.upsample3_3(up3_cat)
        out_64 = self.final64(up3)
        # 64
        up2_up = self.pureUpsamle(up3)
        up2_cat_original = torch.cat((down7, up2_up), 1)
        up2_original = self.upsample2_3(up2_cat_original)
        out_128 = self.final128(up2_original)

        # 128
        up1_up = self.pureUpsamle(up2_original)
        up1_cat = torch.cat((down8, up1_up), 1)
        up1 = self.upsample1_3(up1_cat)
        # final_out = torch.cat((up1, self.pureUpsamle4(up3),self.pureUpsamle(up2_original)), 1)
        out_256 = self.final256(up1)
        # print(self.param32)
        # print(self.param64)
        # print(self.param128)
        return self.pureUpsamle4(out_64)+self.pureUpsamle(out_128)+out_256
        # return self.pureUpsamle4(out_32)*(1-0.6*(1-roundSum))+self.pureUpsamle(out_64)*(0.3*(1-roundSum))+out_128*(0.3*(1-roundSum))
