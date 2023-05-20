import torch
from torch import nn



class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layers(x)
        x = torch.softmax(x, dim=1)
        return x


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Linear(32*4*4, 10)

    def forward(self, x):
        x = self.layers(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    m = CNNModel()
    # BxCxHxW
    x = torch.randn(4, 1, 28, 28)
    y = m(x)
    print(y)
    print(y.shape)
