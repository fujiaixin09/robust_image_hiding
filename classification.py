from torchvision import models
import torch
from torchvision import transforms
import os
from PIL import Image


use = 'Mobile'

transform = transforms.Compose([
     transforms.Resize(128),
     # transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )])

# 读取groundtruth
ground_truth = {}
with open('./val.txt', 'r') as f:
    for line in f:
        keys = line.split(" ")
        ground_truth[keys[0]] = int(keys[1][:-1])

if use == 'AlexNet':
     net = models.alexnet(pretrained=True).cuda()
     print(net)
elif use == 'ResNet':
     net = models.resnet50(pretrained=True).cuda()
     print(net)
elif use == 'VGG':
     net = models.vgg19(pretrained=True).cuda()
     print(net)
elif use == 'DenseNet':
     net = models.densenet121(pretrained=True).cuda()
     print(net)
elif use == 'ResNet':
     net = models.resnet152(pretrained=True).cuda()
     print(net)
elif use == 'GoogleNet':
     net = models.googlenet(pretrained=True).cuda()
     print(net)
else:
     net = models.mobilenet_v2(pretrained=True).cuda()
     print(net)

num, correct = 0, 0
file = "F:\\ILSVRC2012_img_val\\"
for root, dirs, files in os.walk(file):

     # root 表示当前正在访问的文件夹路径
     # dirs 表示该文件夹下的子目录名list
     # files 表示该文件夹下的文件list

     # 遍历文件
     for f in files:
          source_image = os.path.join(root, f)
          image_name = source_image.split('\\')[-1]
          num += 1
          img = Image.open(source_image)
          if img.mode != 'RGB':
               # 单通道，直接跳过
               print("Find Gray-scaled image.Skip...")
               continue
          img_t = transform(img).cuda()
          batch_t = torch.unsqueeze(img_t, 0)

          net.eval()
          out = net(batch_t)
          # print(out.shape)
          # with open('imagenet_classes.txt') as f:
          #   classes = [line.strip() for line in f.readlines()]
          _, index = torch.topk(out, 5)

          percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
          found = False
          if ground_truth[image_name] in index[0]:
              correct+=1
              found = True

          # print(index[0], percentage[index[0]].item())
          print("File Success: {2} Num: {0}, Predicted: {3} G-T: {4} Confidence: {5:.2f} {6} Correct: {1:.6f}"
                .format(num, correct / num, source_image,index[0][0],ground_truth[image_name],percentage[index[0][0]].item(), found))
