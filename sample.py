"""
此文件无用，仅用来测试自己的代码是否正确
"""


import torch.nn.functional as F
from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 池化层
# maxpool = nn.MaxPool2d(3, 1)
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# input = torch.reshape(input, (1, 5, 5))
# output = maxpool(input)
# print(output)


# 卷积层
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]])
# kernel = torch.tensor([[1, 2, 1],
#                        [0, 1, 0],
#                        [2, 1, 0]])
# input = torch.reshape(input, (1, 1, 5, 5))
# kernel = torch.reshape(kernel, (1, 1, 3, 3))
# output1 = F.conv2d(input, kernel, stride=1)
# output2 = F.conv2d(input, kernel, stride=2)
# print(output1)
# print(output2)

# 展平
# input = torch.rand(4, 1, 5, 5)
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]])
# input = torch.reshape(input, (1, 5, 5))
# # print(input)
# flatten = nn.Flatten()
# output = flatten(input)
# print(output)


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.maxpool1(x)
        return x


train_dataset = torchvision.datasets.CIFAR10('./dataset/CIFAR', train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
train_dataloader = DataLoader(train_dataset, batch_size=64)


# writer = SummaryWriter("./logs/conv")
# model = MyModel()
# step = 0
# for data in train_dataloader:
#     print("------第{}步进行中-----".format(step + 1))
#     imgs, targets = data  # torch.Size([64, 3, 32, 32])
#     output = model(imgs)  # torch.Size([64, 6, 30, 30])  32 - 3 + 1 = 30
#     output = torch.reshape(output, (-1, 3, 30, 30))
#     writer.add_images("input", imgs, step)
#     writer.add_images("output", output, step)
#     step += 1
#     break
# writer.close()


# writer = SummaryWriter("./logs/pool")
# model = MyModel()
# step = 0
# for imgs, targets in train_dataloader:
#     print("------第{}步进行中------".format(step + 1))
#     output = model(imgs)  # torch.Size([64, 3, 10, 10])
#     writer.add_images("input", imgs, step)
#     writer.add_images("output", output, step)
#     step += 1
# writer.close()


# writer = SummaryWriter("./logs/pool")
# model = MyModel()
# from PIL import Image
# img = Image.open("../imgs/ship1.png")
# trans_to_tensor = torchvision.transforms.ToTensor()
# input = trans_to_tensor(img)
# writer.add_image("test_in", input)
# output = model(input)
# writer.add_image("test_out", output)
# writer.close()

