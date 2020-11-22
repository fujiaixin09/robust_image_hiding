import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from noise_layers.crop import random_float
from config import GlobalConfig

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, config=GlobalConfig(), resize_ratio_range=(0.5,2), interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.config = config
        self.device = config.device
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method


    def forward(self, noised_image, cover_image=None):
        print("Resize Attack Added")
        self.name = "Resize"
        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        newWidth, newHeight = int(resize_ratio*cover_image.shape[2]), int(resize_ratio*cover_image.shape[3])
        # resize_ratio = 0.5
        # noised_image = noised_and_cover[0]
        out = F.interpolate(
                                    noised_image,
                                    size=[newWidth, newHeight],
                                    recompute_scale_factor=True,
                                    mode=self.interpolation_method)

        recover = F.interpolate(
                                    out,
                                    size=[cover_image.shape[2], cover_image.shape[3]],
                                    recompute_scale_factor=True,
                                    # scale_factor=(1/resize_ratio, 1/resize_ratio),
                                    mode=self.interpolation_method)
        # resize_back = F.interpolate(
        #     noised_image,
        #     size=[cover_image.shape[2], cover_image.shape[3]],
        #     recompute_scale_factor=True,
        #     mode='nearest')
        return recover
