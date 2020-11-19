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
from model.high_quality_net import HighQualityNet
from my_dataset import MyDataset


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
        self.train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=self.config.train_batch_size, shuffle=True, num_workers=4)

        self.net = HighQualityNet(config=self.config)

    # ------------------------------------ Begin ---------------------------------------
    # Creates net object
    def run(self):
        for epoch in range(self.config.num_epochs):
            # train
            for idx, train_batch in enumerate(self.train_loader):
                data, tag = train_batch
                train_cover = data.cuda()
                train_tag = tag.cuda()
                losses, marked = self.net.train_on_batch(train_cover, train_tag)

                if idx % 10240 == 10239:
                    self.net.save_model({
                        'epoch': epoch + 1,
                        'arch': self.config.architecture,
                        'state_dict': self.net.encoder.state_dict(),
                        'optimizer': self.net.optimizer.state_dict(),
                    })
                # if idx % 128 == 127:
                #     for i in range(marked.shape[0]):
                #
                #         utils.save_images(watermarked_images=marked[i].cpu(),
                #                           filename='./Images/epoch-{0}-residual-batch-{1}-{2}.bmp'.format(epoch, idx, i),
                #                          std=self.config.std,
                #                          mean=self.config.mean)


if __name__ == '__main__':
    main_class = mainClass()
    main_class.run()

