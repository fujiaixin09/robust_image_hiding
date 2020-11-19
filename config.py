import torch

class GlobalConfig():

    def __init__(self):

        self.num_epochs = 100
        self.train_batch_size = 8
        self.test_batch_size = 1
        self.architecture='ResNet'
        self.Height = 256
        self.Width = 256
        self.temperature = 10
        self.hyper = 2
        self.hyper_discriminator = 0.01
        self.attack_portion = 0.2
        self.crop_size = 0.2
        self.encoder_features = 64
        self.water_features = 256
        self.device = torch.device("cuda")
        self.num_classes = 1


        self.use_dataset = 'ImageNet'  # "ImageNet"
        self.MODELS_PATH = './output/models/'
        self.VALID_PATH = './sample/valid_coco/'
        # self.TRAIN_PATH = './sample/Test/'
        self.TRAIN_PATH = 'D:\\SmallImageNet'#'C:\\COCOdataset\\train2017'
        self.TEST_PATH = './sample/Test/'
        self.HIDDEN_PATH = './sample/Test/'
        self.DOODLE_PATH = './sample/Doodle_Test'

        # Discriminator
        self.discriminator_channels = 64
        self.discriminator_blocks = 6


        # Mean and std deviation of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
        if self.use_dataset == 'COCO':
            self.mean = [0.471, 0.448, 0.408]
            self.std = [0.234, 0.239, 0.242]
        else:
            self.std = [0.229, 0.224, 0.225]
            self.mean = [0.485, 0.456, 0.406]
