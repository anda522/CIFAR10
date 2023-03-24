import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import MyModel

# 使用对应设备进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    torchvision.transforms.Resize(40),
    # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
    # 生成一个面积为原始图像面积0.64～1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# 获取每一个 img数据 和 label
# 训练数据集
train_data = torchvision.datasets.CIFAR10("./dataset/CIFAR", train=True, transform=transform_train,
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset/CIFAR", train=False, transform=transform_test,
                                         download=True)


print("训练数据集长度为:{}".format(len(train_data)))
print("测试数据集长度为:{}".format(len(test_data)))

# 加载数据集 打包数据,压缩数据 为网络提供不同的数据形式
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64)

model = MyModel()
model.to(device)

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device)

# 设置优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置网络训练参数
total_train_step = 0
total_test_step = 0
epoch = 400

writer = SummaryWriter("./logs/scalar")

max_acc = 0

for i in range(epoch):
    print("----开始第{}轮训练----".format(i + 1))
    # 训练开始 对指定的层才有作用
    model.train()

    for data in train_dataloader:
        # 获取数据
        imgs, targets = data
        # 转换训练设备
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = model(imgs)

        # 计算loss误差
        loss = loss_fn(outputs, targets)
        # 清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1

        if total_train_step % 100 == 0:
            # 添加item是直接输出tensor对应的数字，不加输出tenso类型的数字
            print("[Train] Step: {}, loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 每次训练一轮后跑一边测试
    # 验证测试
    model.eval()

    total_test_loss = 0
    total_accuracy = 0
    # 网络模型中的梯度都没有， 不发生变化
    with torch.no_grad():
        for img, target in test_data:
            img = torch.reshape(img, (1, 3, 32, 32))
            target = torch.tensor([target])

            img, target = img.to(device), target.to(device)
            # 计算输出
            output = model(img)
            loss = loss_fn(output, target)
            total_test_loss = total_test_loss + loss.item()

            """
            outputs:[0.1, 0.2] 两个类别的概率
                    [0.3, 0.4]
            targets:[0, 1] 目标类别
            predict:[1, 1] 预测类别
            result :[False, True]
            """
            # argmax求横向最大值所在的位置
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy = total_accuracy + accuracy

    print("[Test ] loss: {}".format(total_test_loss))
    print("[Test ] accu: {}".format(total_accuracy / len(test_data)))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / len(test_data), total_test_step)

    total_test_step = total_test_step + 1

    # 保存模型
    if max_acc < total_accuracy / len(test_data):
        max_acc = total_accuracy / len(test_data)
        torch.save(model, "./weights/train/model{}.pth".format(i + 1))
        print("较优模型已保存")

writer.close()
