import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torch import utils
from torchvision import transforms

import utils
from network_deepsteg.hiding import HidingNetwork
from network_deepsteg.prepare import PrepNetwork
from network_deepsteg.reveal import RevealNetwork
from config import GlobalConfig
from model.robust_image_net import RobustImageNet


def customized_loss(train_output, train_hidden, train_secrets, train_covers, B):
    ''' Calculates loss specified on the paper.'''
    # train_output, train_hidden, train_secrets, train_covers

    loss_cover = torch.nn.functional.mse_loss(train_hidden, train_covers)
    loss_secret = torch.nn.functional.mse_loss(train_output, train_secrets)
    loss_all = loss_cover + B * loss_secret
    return loss_all, loss_cover, loss_secret


def denormalize(image, std, mean):
    ''' Denormalizes a tensor of images.'''

    for t in range(3):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    return image


# def imshow(img, idx, learning_rate, beta):
#     '''Prints out an image given in tensor format.'''
#
#     img = denormalize(img, std, mean)
#     npimg = img.cpu().numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.title('Example ' + str(idx) + ', lr=' + str(learning_rate) + ', B=' + str(beta)+' 隐藏图像 宿主图像 输出图像 提取得到的图像')
#     plt.show()
#     return





# Join three networks in one module
class Net(nn.Module):
    def gaussian(self, tensor, mean=0, stddev=0.1):
        '''Adds random noise to a tensor.'''

        noise = torch.nn.init.normal_(torch.Tensor(tensor.size()).cuda(), mean, stddev)

        return tensor + noise

    def __init__(self, config=GlobalConfig()):
        super(Net, self).__init__()
        self.config = config
        self.device = config.device
        self.m1 = PrepNetwork().cuda()
        self.m2 = HidingNetwork().cuda()
        self.m3 = RevealNetwork().cuda()

    def forward(self, secret, cover):
        x_1 = self.m1(secret)
        mid = torch.cat((x_1, cover), 1)
        Hidden = self.m2(mid)
        # x_gaussian = self.gaussian(Hidden)
        # x_1_resize = self.resize_layer(x_1_gaussian)
        # x_attack = self.jpeg_layer(x_gaussian)
        Recovery = self.m3(Hidden)
        return Hidden, Recovery


class mainClass:
    def save_model(self, state, filename='./checkpoint.pth.tar'):
        torch.save(state, filename)
        print("Successfully Saved: " + filename)

    def load_model(self, filename, discardRoundCount=False, loadDiscriminator=True):
        print("Reading From: " + filename)
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['state_dict'])
        print("Successfully Loaded All modes in: " + filename)

    def __init__(self):
        self.config = GlobalConfig()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        print("torch.distributed.is_available: "+str(torch.distributed.is_available()))
        print("Device Count: {0}".format(torch.cuda.device_count()))

        # Creates training set
        train_transform = transforms.Compose([
            transforms.Resize(self.config.Width),
            transforms.RandomCrop(self.config.Width),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean,
                                 std=self.config.std)
        ])
        self.train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                self.config.TRAIN_PATH,
                train_transform), batch_size=self.config.train_batch_size, num_workers=4,
            pin_memory=True, shuffle=True, drop_last=True)
        # Creates water set
        train_water_transform = transforms.Compose([
            transforms.Resize(self.config.Width),
            transforms.RandomCrop(self.config.Width),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean[0],
                                 std=self.config.std[0])
        ])
        self.train_water_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                self.config.TRAIN_PATH,
                train_water_transform), batch_size=self.config.train_batch_size, num_workers=4,
            pin_memory=True, shuffle=True, drop_last=True)

        # Creates test set
        test_transform = transforms.Compose([
            transforms.Resize(self.config.Width),
            transforms.RandomCrop(self.config.Width),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean,
                                 std=self.config.std)
        ])
        self.test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                self.config.TEST_PATH,
                test_transform), batch_size=1, num_workers=4,
            pin_memory=True, shuffle=True, drop_last=True)
        # Creates water test set
        test_water_transform = transforms.Compose([
            transforms.Resize(self.config.Water_Width),
            transforms.RandomCrop(self.config.Water_Width),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean[0],
                                 std=self.config.std[0])
        ])
        self.test_water_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                self.config.TEST_PATH,
                test_water_transform), batch_size=1, num_workers=4,
            pin_memory=True, shuffle=True, drop_last=True)

        self.net = Net()
        self.train_cover, self.train_water = None, None
        self.test_cover, self.test_water = None, None

    # ------------------------------------ Begin ---------------------------------------
    # Creates net object
    def run(self,Epoch):
        """Optimizer"""
        name = 'Identity'
        self.optimizer = torch.optim.Adam(self.net.parameters())
        if Epoch==0:
            """pre-trained model"""
            pass
            # self.net.load_model("./checkpoints/Epoch N1 Batch 13311.pth.tar")
        else:
            self.load_model("./checkpoints/Epoch N{0} Batch 14335.pth.tar".format(max(1,Epoch)))
        train_water_iterator = iter(self.train_water_loader)
        test_iterator = iter(self.test_loader)
        test_water_iterator = iter(self.test_water_loader)
        for epoch_1 in range(self.config.num_epochs):
            epoch = epoch_1 + Epoch
            for idx, train_batch in enumerate(self.train_loader):
                self.train_cover, _ = train_batch
                self.train_cover = self.train_cover.cuda()
                self.train_water, _ = train_water_iterator.__next__()
                self.train_water = self.train_water.cuda()
                self.net.train()
                with torch.enable_grad():
                    self.optimizer.zero_grad()
                    marked, extracted = self.net(secret=self.train_water, cover=self.train_cover)
                    Marked_d = utils.denormalize_batch(marked, self.config.std, self.config.mean)
                    Cover_d = utils.denormalize_batch(self.train_cover, self.config.std, self.config.mean)
                    Residual = torch.abs(Marked_d - Cover_d)
                    # Calculate loss and perform backprop
                    train_loss, train_loss_cover, train_loss_secret = customized_loss(extracted, marked,
                                                                                      self.train_water,
                                                                                      self.train_cover, 1)
                    losses = {
                        'train_loss': train_loss.item(),
                        'train_loss_cover': train_loss_cover.item(),
                        'train_loss_secret': train_loss_secret.item()
                    }
                    train_loss.backward()
                    self.optimizer.step()

                str = 'Training Epoch {0}/{1} Batch {2}/{3}. {4}' \
                    .format(epoch, self.config.num_epochs, idx + 1, len(self.train_loader), losses)
                print(str)
                # marked, extracted, Residual, attacked = images
                if idx % 1024 == 1023:
                    self.save_model({
                        'epoch': epoch + 1,
                        'arch': self.config.architecture,
                        'state_dict': self.net.state_dict(),
                        # 'decoder_state_dict': self.net.decoder.state_dict(),
                        # 'discriminator_patchRecovery_state_dict': self.net.discriminator_patchRecovery.state_dict(),
                        # 'discriminator_patchHidden_state_dict': self.net.discriminator_patchHidden.state_dict(),
                    },filename='./checkpoints/Epoch N{0} Batch {1}.pth.tar'.format((epoch + 1),idx))
                if idx % 128 == 127:
                    for i in range(marked.shape[0]):
                        utils.save_images(watermarked_images=marked[i].cpu(),
                                          filename='./network_deepsteg/marked/epoch-{0}-marked-batch-{1}-{2}-{3}.bmp'.format(epoch, idx, i, name),
                                         std=self.config.std,
                                         mean=self.config.mean)
                        utils.save_images(watermarked_images=extracted[i].cpu(),
                                          filename='./network_deepsteg/extracted/epoch-{0}-extracted-batch-{1}-{2}-{3}.bmp'.format(epoch, idx, i, name),
                                          std=self.config.std,
                                          mean=self.config.mean)
                        utils.save_images(watermarked_images=self.train_water[i].cpu(),
                                          filename='./network_deepsteg/water/epoch-{0}-water-batch-{1}-{2}-{3}.bmp'.format(
                                              epoch, idx, i, name),
                                          std=self.config.std,
                                          mean=self.config.mean)
                        utils.save_images(watermarked_images=self.train_cover[i].cpu(),
                                          filename='./network_deepsteg/cover/epoch-{0}-cover-batch-{1}-{2}-{3}.bmp'.format(
                                              epoch, idx, i, name),
                                          std=self.config.std,
                                          mean=self.config.mean)
                        # utils.save_images(watermarked_images=attacked[i].cpu(),
                        #                   filename='./network_deepsteg/attacked/epoch-{0}-attacked-batch-{1}-{2}-{3}.bmp'.format(
                        #                       epoch, idx, i, name),
                        #                   std=self.config.std,
                        #                   mean=self.config.mean)
                        utils.save_images(watermarked_images=Residual[i].cpu(),
                                          filename='./network_deepsteg/residual/epoch-{0}-residual-batch-{1}-{2}-{3}.bmp'.format(
                                              epoch, idx, i, name))
                    print("Saved Images Successfully")

                    # if self.config.conductTest:
                    #     """Test"""
                    #     self.test_cover, _ = test_iterator.__next__()
                    #     self.test_cover = self.test_cover.cuda()
                    #     self.test_water, _ = test_water_iterator.__next__()
                    #     self.test_water = self.test_water.cuda()
                    #     for attack_num in range(len(self.net.noise_layers)):
                    #         losses, images, name = self.net.eval_on_batch(self.test_cover, self.test_water, attack_num=attack_num)
                    #         str = '--Test-- Epoch {0}/{1} Batch {2}/{3}. {4}' \
                    #             .format(epoch, self.config.num_epochs, idx + 1, len(self.test_loader), losses)
                    #         print(str)
                    #         marked, extracted, Residual, attacked = images
                    #         utils.save_images(watermarked_images=marked[0].cpu(),
                    #                           filename='./network_deepsteg/Test/marked/epoch-{0}-marked-batch-{1}-{2}-{3}.bmp'.format(
                    #                               epoch, idx, 0, name),
                    #                           std=self.config.std,
                    #                           mean=self.config.mean)
                    #         utils.save_images(watermarked_images=extracted[0].cpu(),
                    #                           filename='./network_deepsteg/Test/extracted/epoch-{0}-extracted-batch-{1}-{2}-{3}.bmp'.format(
                    #                               epoch, idx, 0, name),
                    #                           std=self.config.std,
                    #                           mean=self.config.mean)
                    #         utils.save_images(watermarked_images=self.test_water[0].cpu(),
                    #                           filename='./network_deepsteg/Test/water/epoch-{0}-water-batch-{1}-{2}-{3}.bmp'.format(
                    #                               epoch, idx, 0, name),
                    #                           std=self.config.std,
                    #                           mean=self.config.mean)
                    #         utils.save_images(watermarked_images=self.test_cover[0].cpu(),
                    #                           filename='./network_deepsteg/Test/cover/epoch-{0}-cover-batch-{1}-{2}-{3}.bmp'.format(
                    #                               epoch, idx, 0, name),
                    #                           std=self.config.std,
                    #                           mean=self.config.mean)
                    #         utils.save_images(watermarked_images=attacked[0].cpu(),
                    #                           filename='./network_deepsteg/Test/attacked/epoch-{0}-attacked-batch-{1}-{2}-{3}.bmp'.format(
                    #                               epoch, idx, 0, name),
                    #                           std=self.config.std,
                    #                           mean=self.config.mean)
                    #         utils.save_images(watermarked_images=Residual[0].cpu(),
                    #                           filename='./network_deepsteg/Test/residual/epoch-{0}-residual-batch-{1}-{2}-{3}.bmp'.format(
                    #                               epoch, idx, 0, name))
                    #
                    #     """Test End"""


if __name__ == '__main__':
    folders = ['./network_deepsteg/','./network_deepsteg/cover/','./network_deepsteg/extracted/','./network_deepsteg/marked/','./network_deepsteg/residual/','./network_deepsteg/water/','./network_deepsteg/attacked/']
    folders += ['./Test/', './Test/cover/', './Test/extracted/', './Test/marked/', './Test/residual/','./Test/water/', './Test/attacked/']
    folders += ['./checkpoints/']
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
            print("Folder Made: "+folder)


    main_class = mainClass()
    for i in range(100):
        main_class.run(i)

