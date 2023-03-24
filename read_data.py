import os
from PIL import Image
from torch.utils.data import Dataset


class MyData(Dataset):
    def __init__(self, root_dir, label):
        self.root_dir = root_dir
        self.label = label
        self.path = os.path.join(root_dir, label)
        self.img_names = os.listdir(self.path)

    # 实例对象可迭代
    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path)
        label = self.label
        return img, label

    def __len__(self):
        return len(self.img_names)


if __name__ == '__main__':
    # train data's path
    root_path = "D:\\code\\pytorch_learning\\CIFAR10\\dataset\\val"
    label = "frog"

    data = MyData(root_path, label)
    for i in range(len(data)):
        print(data[i])
