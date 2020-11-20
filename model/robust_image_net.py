import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from noise_layers.DiffJPEG import DiffJPEG
from noise_layers.identity import Identity
from noise_layers.gaussian import Gaussian
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from config import GlobalConfig
from model.prep_RInet import UnetInception
from model.prep_HQnet import pureUnet
import numpy as np
from model.NLayerDiscriminator import NLayerDiscriminator
from loss.GANloss import GANLoss
from loss.vgg_loss import VGGLoss
import pytorch_ssim
import math
import utils

class RobustImageNet:
    def __init__(self, config=GlobalConfig()):
        super(RobustImageNet, self).__init__()
        self.config = config
        """ Settings """
        self.criterionGAN = GANLoss().cuda()

        self.encoder = UnetInception(config=config).cuda()
        self.decoder = pureUnet(config=config).cuda()
        if torch.cuda.device_count() > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
        if torch.cuda.device_count() > 1:
            self.decoder = torch.nn.DataParallel(self.decoder)
        print(self.encoder)
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters())
        self.optimizer_decoder = torch.optim.Adam(self.decoder.parameters())
        """ Noise Layers """
        self.noise_layers = [Identity()]
        # self.cropout_layer = Cropout(config).cuda()
        self.jpeg_layer_80 = DiffJPEG(256, 256, quality=80, differentiable=True).cuda()
        self.jpeg_layer_90 = DiffJPEG(256, 256, quality=90, differentiable=True).cuda()
        self.jpeg_layer_70 = DiffJPEG(256, 256, quality=70, differentiable=True).cuda()
        self.jpeg_layer_60 = DiffJPEG(256, 256, quality=60, differentiable=True).cuda()
        self.jpeg_layer_50 = DiffJPEG(256, 256, quality=50, differentiable=True).cuda()
        self.gaussian = Gaussian().cuda()
        self.dropout = Dropout(self.config,keep_ratio_range=(0.5,0.75)).cuda()
        self.resize = Resize().cuda()
        # self.crop_layer = Crop((0.2, 0.5), (0.2, 0.5)).cuda()
        self.noise_layers.append(self.jpeg_layer_80)
        self.noise_layers.append(self.jpeg_layer_90)
        self.noise_layers.append(self.jpeg_layer_70)
        self.noise_layers.append(self.jpeg_layer_60)
        self.noise_layers.append(self.jpeg_layer_50)
        self.noise_layers.append(self.gaussian)
        self.noise_layers.append(self.resize)
        self.noise_layers.append(self.dropout)

        self.discriminator_patchHidden = NLayerDiscriminator(input_nc=3).cuda()
        if torch.cuda.device_count() > 1:
            self.discriminator_patchHidden = torch.nn.DataParallel(self.discriminator_patchHidden)
        self.discriminator_patchRecovery = NLayerDiscriminator(input_nc=1).cuda()
        if torch.cuda.device_count() > 1:
            self.discriminator_patchRecovery = torch.nn.DataParallel(self.discriminator_patchRecovery)

        self.optimizer_discrim_patchHiddem = torch.optim.Adam(self.discriminator_patchHidden.parameters())
        self.optimizer_discrim_patchRecovery = torch.optim.Adam(self.discriminator_patchRecovery.parameters())

        self.mse_loss = nn.MSELoss().cuda()
        self.ssim_loss = pytorch_ssim.SSIM().cuda()
        self.vgg_loss = VGGLoss(3, 1, False).cuda()
        if torch.cuda.device_count() > 1:
            self.vgg_loss = torch.nn.DataParallel(self.vgg_loss)


    def train_on_batch(self, Cover, Water):
        """
            训练方法：
            Encoder使用Unet，固定分类网络
            让分类网络在T=1时，分类结果为正常结果
            T=10时，分类结果为水印序列对应结果
        """
        # batch_size = Cover.shape[0]
        self.encoder.train()
        self.decoder.train()
        # for param in self.classification_net.parameters():
        #     param.requires_grad = False
        with torch.enable_grad():
            """ Run, Train the discriminator"""
            self.optimizer_encoder.zero_grad()
            self.optimizer_decoder.zero_grad()
            self.optimizer_discrim_patchHiddem.zero_grad()
            self.optimizer_discrim_patchRecovery.zero_grad()
            Marked = self.encoder(Cover, Water)
            """Attack"""
            random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
            Attacked = random_noise_layer(Marked, cover_image=Cover)
            Extracted = self.decoder(Attacked)

            """Discriminator"""
            loss_D_A = self.backward_D_basic(self.discriminator_patchHidden, Cover, Marked)
            loss_D_A.backward()
            self.optimizer_discrim_patchHiddem.step()
            loss_D_B = self.backward_D_basic(self.discriminator_patchRecovery, Water, Extracted)
            loss_D_B.backward()
            self.optimizer_discrim_patchRecovery.step()

            """Losses"""
            loss_marked = self.mse_loss(Marked, Cover) #self.getVggLoss(Marked, Cover)
            loss_recovery = self.mse_loss(Extracted, Water)
            g_loss_adv_enc = self.criterionGAN(self.discriminator_patchHidden(Marked), True)
            g_loss_adv_recovery = self.criterionGAN(self.discriminator_patchRecovery(Extracted), True)
            loss_enc_dec = (loss_recovery + 5*g_loss_adv_recovery * self.config.hyper_discriminator)
            if loss_marked>1:
                loss_enc_dec += 0.7*(loss_marked + g_loss_adv_enc * self.config.hyper_discriminator)


            loss_enc_dec.backward()
            self.optimizer_encoder.step()
            self.optimizer_decoder.step()

            losses = {
                'loss_marked': loss_marked.item(),
                'loss_recovery': loss_recovery.item(),
                'g_loss_adv_enc': g_loss_adv_enc.item(),
                'g_loss_adv_recovery': g_loss_adv_recovery.item()
            }

            Marked_d = utils.denormalize_batch(Marked, self.config.std, self.config.mean)
            Cover_d = utils.denormalize_batch(Cover, self.config.std, self.config.mean)
            Extracted_d = utils.denormalize_batch(Extracted, self.config.std, self.config.mean)
            Water_d = utils.denormalize_batch(Water, self.config.std, self.config.mean)

            # PSNR
            print("PSNR:(Hidden Cover) {}".format(
                10 * math.log10(255.0 ** 2 / torch.mean((Marked_d * 255 - Cover_d * 255) ** 2))))
            print("PSNR:(Extracted Cover) {}".format(
                10 * math.log10(255.0 ** 2 / torch.mean((Extracted_d * 255 - Cover_d * 255) ** 2))))
            # SSIM
            print("SSIM:(Hidden Cover) {}".format(pytorch_ssim.ssim(Marked_d, Cover_d)))
            print("SSIM:(Recover Cover) {}".format(pytorch_ssim.ssim(Extracted_d, Water_d)))

            return losses, (Marked, Extracted, Water)


    def save_model(self, state, filename='./checkpoint.pth.tar'):
        torch.save(state, filename)
        print("Successfully Saved: " + filename)

    def load_model(self, filename):
        print("Reading From: " + filename)
        checkpoint = torch.load(filename)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(self.encoder)
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(self.encoder)
        print("Successfully Loaded: " + filename)

    def getVggLoss(self, marked, cover):
        vgg_on_cov = self.vgg_loss(cover)
        vgg_on_enc = self.vgg_loss(marked)
        loss = self.mse_loss(vgg_on_cov, vgg_on_enc)
        return loss

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D

