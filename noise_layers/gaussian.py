import torch
import torch.nn as nn
from noise_layers.crop import get_random_rectangle_inside
import matplotlib.pyplot as plt
import numpy as np
from config import GlobalConfig
import math
class Gaussian(nn.Module):
    '''Adds random noise to a tensor.'''

    def __init__(self, config=GlobalConfig()):
        super(Gaussian, self).__init__()
        self.config = config

    def forward(self, tensor, cover_image=None, mean=0, stddev=0.1):
        print("Gaussian Attack Added")
        self.name="Gaussian"
        noise = torch.nn.init.normal_(torch.Tensor(tensor.size()).cuda(), mean, stddev)
        return tensor + noise

