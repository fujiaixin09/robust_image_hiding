import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import models
from noise_layers.DiffJPEG import DiffJPEG
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.gaussian import Gaussian
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from config import GlobalConfig
from model.prep_HQnet_64 import Prep_pureUnet


class HighQualityNet:
    def __init__(self, config=GlobalConfig()):
        super(HighQualityNet, self).__init__()
        self.config = config
        """ Settings """
        if self.config.architecture == 'AlexNet':
             self.classification_net = models.alexnet(pretrained=True).cuda()
             print(self.classification_net)
        elif self.config.architecture == 'ResNet':
             self.classification_net = models.resnet50(pretrained=True).cuda()
             print(self.classification_net)
        elif self.config.architecture == 'VGG':
             self.classification_net = models.vgg19(pretrained=True).cuda()
             print(self.classification_net)
        elif self.config.architecture == 'DenseNet':
             self.classification_net = models.densenet121(pretrained=True).cuda()
             print(self.classification_net)
        elif self.config.architecture == 'ResNet':
             self.classification_net = models.resnet152(pretrained=True).cuda()
             print(self.classification_net)
        elif self.config.architecture == 'GoogleNet':
             self.classification_net = models.googlenet(pretrained=True).cuda()
             print(self.classification_net)
        else:
             self.classification_net = models.mobilenet_v2(pretrained=True).cuda()
             print(self.classification_net)
        if torch.cuda.device_count() > 1:
            self.classification_net = torch.nn.DataParallel(self.classification_net)
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.encoder = Prep_pureUnet(config=config).cuda()
        if torch.cuda.device_count() > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
        print(self.encoder)
        self.optimizer = torch.optim.Adam(self.encoder.parameters())

        """ Noise Layers """
        self.noise_layers = [Identity()]
        # self.cropout_layer = Cropout(config).cuda()
        self.jpeg_layer_80 = DiffJPEG(256, 256, quality=80, differentiable=True).cuda()
        self.jpeg_layer_90 = DiffJPEG(256, 256, quality=90, differentiable=True).cuda()
        self.jpeg_layer_70 = DiffJPEG(256, 256, quality=70, differentiable=True).cuda()
        self.jpeg_layer_60 = DiffJPEG(256, 256, quality=60, differentiable=True).cuda()
        self.jpeg_layer_50 = DiffJPEG(256, 256, quality=50, differentiable=True).cuda()
        # self.gaussian = Gaussian().cuda()
        # self.dropout = Dropout(self.config,keep_ratio_range=(0.5,0.75)).cuda()
        # self.resize = Resize().cuda()
        # self.crop_layer = Crop((0.2, 0.5), (0.2, 0.5)).cuda()
        self.noise_layers.append(self.jpeg_layer_80)
        self.noise_layers.append(self.jpeg_layer_90)
        self.noise_layers.append(self.jpeg_layer_70)
        self.noise_layers.append(self.jpeg_layer_60)
        self.noise_layers.append(self.jpeg_layer_50)
        # self.noise_layers.append(self.gaussian)
        # self.noise_layers.append(self.resize)

    def train_on_batch(self, Cover, tag):
        """
            训练方法：
            Encoder使用Unet，固定分类网络
            让分类网络在T=1时，分类结果为正常结果
            T=10时，分类结果为水印序列对应结果
        """
        batch_size = Cover.shape[0]
        self.encoder.train()
        self.classification_net.train()
        # for param in self.classification_net.parameters():
        #     param.requires_grad = False
        with torch.enable_grad():
            """ Run, Train the discriminator"""
            self.optimizer.zero_grad()
            Marked = self.encoder(Cover)

            output = self.classification_net(Marked)
            output_original = self.classification_net(Cover)
            # target = torch.zeros_like(output,dtype=torch.long)
            # for i in range(batch_size):
            #     target[i,tag[i]]=1
            # cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1
            # target[tag] = 1.0
            loss_original = self.criterion(output_original, tag)
            loss_class = self.criterion(output, tag)
            loss_water = self.criterion(output/self.config.temperature, tag)
            loss = loss_class + self.config.hyper*loss_water
            loss.backward()
            self.optimizer.step()

            losses = {
                'loss_original': loss_original.item(),
                'loss_class': loss_class.item(),
                'loss_water': loss_water.item(),
                'loss': loss.item()
            }
            print(losses)

            return losses, Marked


    def save_model(self, state, filename='./checkpoint.pth.tar'):
        torch.save(state, filename)
        print("Successfully Saved: " + filename)

    def load_model(self, filename):
        print("Reading From: " + filename)
        checkpoint = torch.load(filename)
        self.encoder.load_state_dict(checkpoint['state_dict'])
        print(self.encoder)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Successfully Loaded: " + filename)

