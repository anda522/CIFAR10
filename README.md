在对应python环境下，终端进入logs文件夹，进行命令

tensorboard --logdir=scalar

可以查看scalar文件夹下的图像，文件夹下如果有多个文件夹，每个文件夹的图像将用不同颜色显示，且比较具有对比性

logs文件夹代码均为代码生成的图像文件之类的，用处不大。

model68.5.pth 为lr = 0.003的参数文件，轮数大概为一百多轮

model68.2.pth 为lr = 0.001的233轮的参数文件

model67.8.pth 为lr= 0.01的参数文件

学习率的对比可以查看scalar文件夹中的tensorboard文件，发现基本正确率都是在65%-70%附近浮动。
