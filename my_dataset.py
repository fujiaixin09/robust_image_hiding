# Created on 2020-06-23
# Author: fanghan

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, root, mean, std, filename='./val.txt', grayscale=False, size=256):
        # 所有图片的绝对路径
        self.imgs, self.filename = [], filename
        self.mean, self.std = mean, std
        self.grayscale = grayscale
        subfolders = os.listdir(root)
        for subfolder in subfolders:
            subroot = os.path.join(root, subfolder)
            imgs = os.listdir(subroot)
            for img in imgs:
                self.imgs.append(os.path.join(subroot, img))
        if not grayscale:
            self.transforms = transforms.Compose([
                 transforms.Resize(size),
                 transforms.CenterCrop(size),
                 transforms.ToTensor(),
                 transforms.Normalize(
                 mean=self.mean,
                 std=self.std
             )])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean[0],
                    std=self.std[0]
                )])

        if filename is not None:
            #  get groundtruth
            self.ground_truth = {}
            with open(filename, 'r') as f:
                for line in f:
                    keys = line.split(" ")
                    self.ground_truth[keys[0]] = int(keys[1][:-1])


    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        image_name = img_path.split('\\')[-1]
        if self.filename is None:
            return data, None
        else:
            return data, self.ground_truth[image_name]

    def __len__(self):
        return len(self.imgs)


