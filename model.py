from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = MyModel()
    print(model)
    input_data = torch.ones((64, 3, 32, 32))

    writer = SummaryWriter("./logs/model")
    writer.add_graph(model, input_data)
    writer.close()

    output_data = model(input_data)
    print(output_data.shape)
