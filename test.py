import torch
import torchvision
from PIL import Image
from model import MyModel
from read_data import MyData
import os

ans = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "D:\\code\\pytorch_learning\\CIFAR10\\dataset\\val"
labels_dir = os.listdir(root_dir)

trans = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                        torchvision.transforms.ToTensor()])

model = torch.load("./weights/3.model75.pth")

tot_num = 0
tot_accuracy = 0
for label in labels_dir:
    cur_accuracy = 0
    myData = MyData(root_dir, label)
    tot_num += len(myData)
    print("----------当前的图片类型为: {} ----------".format(label))

    for img, t_label in myData:
        # 将图片转换为为torch.Size([3, 32, 32])
        img = trans(img)
        # img = img.convert("RGB")
        img = torch.reshape(img, (1, 3, 32, 32))
        img = img.to(device)

        model.eval()
        with torch.no_grad():
            output = model(img)

        predict_result = ans[output.argmax(1).item()]
        cur_accuracy += (predict_result == label)
        print("预测结果: {}  ".format(predict_result), "正确" if predict_result == label else "错误")

    tot_accuracy += cur_accuracy
    print("当前种类预测准确率为: {}".format(cur_accuracy / len(myData)))
print("总预测准确率为: {}".format(tot_accuracy / tot_num))
