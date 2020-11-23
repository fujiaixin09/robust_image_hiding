import os

import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torch.utils import data
from torchvision import transforms
from config import GlobalConfig
from model.lin_jing_zhi import LinJingZhiNet
from my_dataset import MyDataset
import utils

class mainClass:
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
            transforms.Resize(self.config.Water_Width),
            transforms.RandomCrop(self.config.Water_Width),
            transforms.Grayscale(),
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

        # self.train_dataset = MyDataset(root=self.config.TRAIN_PATH,filename=self.config.TAG_PATH,mean=self.config.mean,std=self.config.std)
        # self.another_dataset = MyDataset(root=self.config.TRAIN_PATH, filename=self.config.TAG_PATH, grayscale=True, size=64,mean=self.config.mean,std=self.config.std)
        # print(len(self.train_dataset))
        # self.train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=self.config.train_batch_size,
        #                                     shuffle=True, num_workers=4)
        # self.another_loader = data.DataLoader(dataset=self.another_dataset, batch_size=self.config.train_batch_size,
        #                                     shuffle=True, num_workers=4)

        self.net = LinJingZhiNet(config=self.config)
        self.train_cover, self.train_water = None, None
        self.test_cover, self.test_water = None, None

    # ------------------------------------ Begin ---------------------------------------
    # Creates net object
    def run(self,Epoch):
        if Epoch==0:
            """pre-trained model"""
            self.net.load_model("./checkpoints/Epoch N1 Batch 13311.pth.tar")
        else:
            self.net.load_model("./checkpoints/Epoch N{0} Batch 14335.pth.tar".format(max(1,Epoch)))
        train_water_iterator = iter(self.train_water_loader)
        test_iterator = iter(self.test_loader)
        test_water_iterator = iter(self.test_water_loader)
        for epoch_1 in range(self.config.num_epochs):
            epoch = epoch_1 + Epoch
            for idx, train_batch in enumerate(self.train_loader):
                self.train_cover, _ = train_batch
                self.train_cover = self.train_cover.cuda()

                """
                说明：水印图像本来是直接由self.train_water_loader读取灰度图像进来，这边现在就是用水印序列经过纠错编码后把它用矩阵形式表示出来
                所以这边需要改一下，水印序列可以用随机0 1来生成，需要找一个比较好的纠错编码
                """
                self.train_water, _ = train_water_iterator.__next__()
                self.train_water = self.train_water.cuda()

                # train_tag = tag.cuda()
                losses, images, name = self.net.train_on_batch(self.train_cover, self.train_water)
                str = 'Training Epoch {0}/{1} Batch {2}/{3}. {4}' \
                    .format(epoch, self.config.num_epochs, idx + 1, len(self.train_loader), losses)
                print(str)
                marked, extracted, Residual, attacked = images
                if idx % 1024 == 1023:
                    self.net.save_model({
                        'epoch': epoch + 1,
                        'arch': self.config.architecture,
                        'encoder_state_dict': self.net.encoder.state_dict(),
                        'decoder_state_dict': self.net.decoder.state_dict(),
                        'discriminator_patchRecovery_state_dict': self.net.discriminator_patchRecovery.state_dict(),
                        'discriminator_patchHidden_state_dict': self.net.discriminator_patchHidden.state_dict(),
                    },filename='./checkpoints/Epoch N{0} Batch {1}.pth.tar'.format((epoch + 1),idx))
                if idx % 128 == 127:
                    for i in range(marked.shape[0]):
                        utils.save_images(watermarked_images=marked[i].cpu(),
                                          filename='./Images/marked/epoch-{0}-marked-batch-{1}-{2}-{3}.bmp'.format(epoch, idx, i, name),
                                         std=self.config.std,
                                         mean=self.config.mean)
                        utils.save_images(watermarked_images=extracted[i].cpu(),
                                          filename='./Images/extracted/epoch-{0}-extracted-batch-{1}-{2}-{3}.bmp'.format(epoch, idx, i, name),
                                          std=self.config.std,
                                          mean=self.config.mean)
                        utils.save_images(watermarked_images=self.train_water[i].cpu(),
                                          filename='./Images/water/epoch-{0}-water-batch-{1}-{2}-{3}.bmp'.format(
                                              epoch, idx, i, name),
                                          std=self.config.std,
                                          mean=self.config.mean)
                        utils.save_images(watermarked_images=self.train_cover[i].cpu(),
                                          filename='./Images/cover/epoch-{0}-cover-batch-{1}-{2}-{3}.bmp'.format(
                                              epoch, idx, i, name),
                                          std=self.config.std,
                                          mean=self.config.mean)
                        utils.save_images(watermarked_images=attacked[i].cpu(),
                                          filename='./Images/attacked/epoch-{0}-attacked-batch-{1}-{2}-{3}.bmp'.format(
                                              epoch, idx, i, name),
                                          std=self.config.std,
                                          mean=self.config.mean)
                        utils.save_images(watermarked_images=Residual[i].cpu(),
                                          filename='./Images/residual/epoch-{0}-residual-batch-{1}-{2}-{3}.bmp'.format(
                                              epoch, idx, i, name))
                    print("Saved Images Successfully")

                    if self.config.conductTest:
                        """Test"""
                        self.test_cover, _ = test_iterator.__next__()
                        self.test_cover = self.test_cover.cuda()
                        self.test_water, _ = test_water_iterator.__next__()
                        self.test_water = self.test_water.cuda()
                        for attack_num in range(len(self.net.noise_layers)):
                            losses, images, name = self.net.eval_on_batch(self.test_cover, self.test_water, attack_num=attack_num)
                            str = '--Test-- Epoch {0}/{1} Batch {2}/{3}. {4}' \
                                .format(epoch, self.config.num_epochs, idx + 1, len(self.test_loader), losses)
                            print(str)
                            marked, extracted, Residual, attacked = images
                            utils.save_images(watermarked_images=marked[0].cpu(),
                                              filename='./Test/marked/epoch-{0}-marked-batch-{1}-{2}-{3}.bmp'.format(
                                                  epoch, idx, 0, name),
                                              std=self.config.std,
                                              mean=self.config.mean)
                            utils.save_images(watermarked_images=extracted[0].cpu(),
                                              filename='./Test/extracted/epoch-{0}-extracted-batch-{1}-{2}-{3}.bmp'.format(
                                                  epoch, idx, 0, name),
                                              std=self.config.std,
                                              mean=self.config.mean)
                            utils.save_images(watermarked_images=self.test_water[0].cpu(),
                                              filename='./Test/water/epoch-{0}-water-batch-{1}-{2}-{3}.bmp'.format(
                                                  epoch, idx, 0, name),
                                              std=self.config.std,
                                              mean=self.config.mean)
                            utils.save_images(watermarked_images=self.test_cover[0].cpu(),
                                              filename='./Test/cover/epoch-{0}-cover-batch-{1}-{2}-{3}.bmp'.format(
                                                  epoch, idx, 0, name),
                                              std=self.config.std,
                                              mean=self.config.mean)
                            utils.save_images(watermarked_images=attacked[0].cpu(),
                                              filename='./Test/attacked/epoch-{0}-attacked-batch-{1}-{2}-{3}.bmp'.format(
                                                  epoch, idx, 0, name),
                                              std=self.config.std,
                                              mean=self.config.mean)
                            utils.save_images(watermarked_images=Residual[0].cpu(),
                                              filename='./Test/residual/epoch-{0}-residual-batch-{1}-{2}-{3}.bmp'.format(
                                                  epoch, idx, 0, name))

                        """Test End"""


if __name__ == '__main__':
    folders = ['./Images/','./Images/cover/','./Images/extracted/','./Images/marked/','./Images/residual/','./Images/water/','./Images/attacked/']
    folders += ['./Test/', './Test/cover/', './Test/extracted/', './Test/marked/', './Test/residual/','./Test/water/', './Test/attacked/']
    folders += ['./checkpoints/']
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
            print("Folder Made: "+folder)


    main_class = mainClass()
    for i in range(100):
        main_class.run(i)

