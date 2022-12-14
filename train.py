import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import MyModel

# 使用对应设备进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取每一个 img数据 和 label
# 训练数据集
train_data = torchvision.datasets.CIFAR10("./dataset/CIFAR", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset/CIFAR", train=False, transform=torchvision.transforms.ToTensor(),
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
epoch = 5

writer = SummaryWriter("./logs/scalar")

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
            print("训练次数：{},loss值：{}".format(total_train_step, loss.item()))  # 添加item是直接输出tensor对应的数字，不加输出tenso类型的数字
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

    print("整体测试集的loss：{}".format(total_test_loss))
    print("整体正确率：{}".format(total_accuracy / len(test_data)))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / len(test_data), total_test_step)

    total_test_step = total_test_step + 1

    # 保存每一轮的模型
    torch.save(model, "./weights/model{}.pth".format(i + 1))
    # 也可以方式二
    print("模型已保存")

writer.close()
