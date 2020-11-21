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
from model.robust_image_net import RobustImageNet
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
        # Creates another set
        another_transform = transforms.Compose([
            transforms.Resize(self.config.Water_Width),
            transforms.RandomCrop(self.config.Water_Width),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean[0],
                                 std=self.config.std[0])
        ])
        self.another_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                self.config.TRAIN_PATH,
                another_transform), batch_size=self.config.train_batch_size, num_workers=4,
            pin_memory=True, shuffle=True, drop_last=True)

        # self.train_dataset = MyDataset(root=self.config.TRAIN_PATH,filename=self.config.TAG_PATH,mean=self.config.mean,std=self.config.std)
        # self.another_dataset = MyDataset(root=self.config.TRAIN_PATH, filename=self.config.TAG_PATH, grayscale=True, size=64,mean=self.config.mean,std=self.config.std)
        # print(len(self.train_dataset))
        # self.train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=self.config.train_batch_size,
        #                                     shuffle=True, num_workers=4)
        # self.another_loader = data.DataLoader(dataset=self.another_dataset, batch_size=self.config.train_batch_size,
        #                                     shuffle=True, num_workers=4)

        self.net = RobustImageNet(config=self.config)
        self.train_cover, self.another = None, None

    # ------------------------------------ Begin ---------------------------------------
    # Creates net object
    def run(self):
        self.net.load_model("./checkpoints/Epoch N1 Batch 4095.pth.tar")
        for epoch in range(self.config.num_epochs):
            another_iterator = iter(self.another_loader)
            for idx, train_batch in enumerate(self.train_loader):
                self.train_cover, _ = train_batch
                self.train_cover = self.train_cover.cuda()
                self.another, _ = another_iterator.__next__()
                self.another = self.another.cuda()


                # train_tag = tag.cuda()
                losses, images, name = self.net.train_on_batch(self.train_cover, self.another)
                str = 'Net 1 Epoch {0}/{1} Training: Batch {2}/{3}. {4}' \
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
                        utils.save_images(watermarked_images=self.another[i].cpu(),
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


if __name__ == '__main__':
    folders = ['./Images/','./Images/cover/','./Images/extracted/','./Images/marked/','./Images/residual/','./Images/water/','./Images/attacked/']
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
            print("Folder Made: "+folder)


    main_class = mainClass()
    main_class.run()

