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

        transform = transforms.Compose([
            transforms.Resize(self.config.Width),
            transforms.RandomCrop(self.config.Width),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean,
                                 std=self.config.std)
        ])
        # Creates training set
        self.train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                self.config.TRAIN_PATH,
                transform), batch_size=self.config.train_batch_size, num_workers=4,
            pin_memory=True, shuffle=True, drop_last=True)

        self.train_dataset = MyDataset(root='F:\\ILSVRC2012_img_val\\',filename='./val.txt')
        print(len(self.train_dataset))
        self.train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=self.config.train_batch_size,
                                            shuffle=True, num_workers=4)
        self.another_loader = data.DataLoader(dataset=self.train_dataset, batch_size=self.config.train_batch_size,
                                            shuffle=True, num_workers=4)

        self.net = RobustImageNet(config=self.config)
        self.train_cover, self.another = None, None

    # ------------------------------------ Begin ---------------------------------------
    # Creates net object
    def run(self):
        for epoch in range(self.config.num_epochs):
            another_iterator = iter(self.another_loader)
            for idx, train_batch in enumerate(self.train_loader):
                self.train_cover, _ = train_batch
                self.train_cover = self.train_cover.cuda()
                self.another, _ = another_iterator.__next__()
                self.another = self.another.cuda()


                # train_tag = tag.cuda()
                losses, images = self.net.train_on_batch(self.train_cover, self.another)
                str = 'Net 1 Epoch {0}/{1} Training: Batch {2}/{3}. {4}' \
                    .format(epoch, self.config.num_epochs, idx + 1, len(self.train_loader), losses)
                print(str)
                marked, extracted, _ = images
                if idx % 10240 == 10239:
                    self.net.save_model({
                        'epoch': epoch + 1,
                        'arch': self.config.architecture,
                        'encoder_state_dict': self.net.encoder.state_dict(),
                        'decoder_state_dict': self.net.encoder.state_dict(),
                        # 'optimizer_encoder': self.net.optimizer_encoder.state_dict(),
                    },filename='./checkpoints/Epoch N{0} Batch {1}.pth.tar'.format((epoch + 1),idx))
                if idx % 128 == 127:
                    for i in range(marked.shape[0]):
                        utils.save_images(watermarked_images=marked[i].cpu(),
                                          filename='./Images/marked/epoch-{0}-marked-batch-{1}-{2}.bmp'.format(epoch, idx, i),
                                         std=self.config.std,
                                         mean=self.config.mean)
                        utils.save_images(watermarked_images=extracted[i].cpu(),
                                          filename='./Images/extracted/epoch-{0}-extracted-batch-{1}-{2}.bmp'.format(epoch, idx, i),
                                          std=self.config.std,
                                          mean=self.config.mean)
                        utils.save_images(watermarked_images=self.another[i].cpu(),
                                          filename='./Images/water/epoch-{0}-water-batch-{1}-{2}.bmp'.format(
                                              epoch, idx, i),
                                          std=self.config.std,
                                          mean=self.config.mean)
                        utils.save_images(watermarked_images=self.train_cover[i].cpu(),
                                          filename='./Images/cover/epoch-{0}-cover-batch-{1}-{2}.bmp'.format(
                                              epoch, idx, i),
                                          std=self.config.std,
                                          mean=self.config.mean)
                        print("Saved Images Successfully")


if __name__ == '__main__':
    main_class = mainClass()
    main_class.run()

