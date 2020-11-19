import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils



def image_to_tensor(image):
    """
    Transforms a numpy-image into torch tensor
    :param image: (batch_size x height x width x channels) uint8 array
    :return: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    """
    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor


def tensor_to_image(tensor):
    """
    Transforms a torch tensor into numpy uint8 array (image)
    :param tensor: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    :return: (batch_size x height x width x channels) uint8 array
    """
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)

def denormalize(image, std, mean):
    ''' Denormalizes a tensor of images.'''
    for t in range(image.shape[0]):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    return image

def save_images(original_images=None, watermarked_images=None, epoch=None, folder=None, resize_to=None, filename=None, std=None, mean=None):
    # images = original_images[:original_images.shape[0], :, :, :].cpu()
    # watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    # images = (images + 1) / 2
    if std is not None:
        watermarked_images = denormalize(watermarked_images, std, mean)
    else:
        watermarked_images = (watermarked_images + 1) / 2

    if resize_to is not None:
        # images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    stacked_images = watermarked_images # torch.cat([images, watermarked_images], dim=0)
    if filename is None:
        filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    torchvision.utils.save_image(stacked_images, filename, watermarked_images.shape[0], normalize=False)


def denormalize_batch(image, std, mean):
    ''' Denormalizes a tensor of images.'''

    image_denorm = torch.empty_like(image)
    image_denorm[:, 0, :, :] = (image[:, 0, :, :].clone() * std[0]) + mean[0]
    image_denorm[:, 1, :, :] = (image[:, 1, :, :].clone() * std[1]) + mean[1]
    image_denorm[:, 2, :, :] = (image[:, 2, :, :].clone() * std[2]) + mean[2]

    # for t in range(image.shape[1]):
    #     image[:, t, :, :] = (image[:, t, :, :] * std[t]) + mean[t]
    return image_denorm