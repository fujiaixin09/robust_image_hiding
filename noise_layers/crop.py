import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import GlobalConfig

def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    """
    Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.
    This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9), then a random number in that range will be chosen
    (say it is 0.75 for illustration), and the image will be cropped such that the remaining height equals 0.75. In fact,
    a random 'starting' position rs will be chosen from (0, 0.25), and the crop will start at rs and end at rs + 0.75. This ensures
    that we crop from top/bottom with equal probability.
    The same logic applies to the width of the image, where width_ratio_range controls the width crop range.
    :param image: The image we want to crop
    :param height_ratio_range: The range of remaining height ratio
    :param width_ratio_range:  The range of remaining width ratio.
    :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
    """
    image_height = image.shape[2]
    image_width = image.shape[3]

    r_float_height, r_float_width = \
        random_float(height_ratio_range[0], height_ratio_range[1]), random_float(width_ratio_range[0], width_ratio_range[1])
    remaining_height = int(np.rint(r_float_height * image_height))
    remaining_width = int(np.rint(r_float_width * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width, r_float_height*r_float_width


class Crop(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
    heigth_ratio_range and width_ratio_range
    """
    def __init__(self, height_ratio_range=(0.5,1), width_ratio_range=(0.5,1), config=GlobalConfig()):
        """

        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Crop, self).__init__()
        self.config = config
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range
        self.h_start, self.h_end, self.w_start, self.w_end = None,None,None,None
        self.bool = False

    def get_random_rectangle_inside(self, noised_image):
        return get_random_rectangle_inside(noised_image, self.height_ratio_range, self.width_ratio_range)


    def forward(self, noised_image, cover_image=None):

        # noised_image = noised_and_cover[0]
        # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)
        if self.h_start is None:
            self.h_start, self.h_end, self.w_start, self.w_end, _ = get_random_rectangle_inside(noised_image, self.height_ratio_range, self.width_ratio_range)
        else:
            self.h_start = int(self.h_start/ (self.config.Width/self.config.Water_Width))
            self.h_end = int(self.h_end / (self.config.Width/self.config.Water_Width))
            self.w_start = int(self.w_start / (self.config.Width/self.config.Water_Width))
            self.w_end = int(self.w_end / (self.config.Width/self.config.Water_Width))

        cropped_image = noised_image[:,:,self.h_start: self.h_end,self.w_start: self.w_end].clone()

        resize_back = F.interpolate(
            cropped_image,
            size=[cover_image.shape[2],cover_image.shape[3]],
            recompute_scale_factor=True,
            mode='nearest')



        print("Crop Attack Added. {0}, Sizee:{1} {2} {3} {4}".format(self.bool,self.h_start,self.h_end,self.w_start,self.w_end))
        self.name = "Crop"
        if self.bool:
            self.h_start, self.h_end, self.w_start, self.w_end = None,None,None,None
        self.bool = not self.bool

        return resize_back
