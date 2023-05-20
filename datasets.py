import click
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class MNISTDataset(Dataset):
    def __init__(self, train, ):
        ds = MNIST(
            root='data/MNIST',
            train=train,
            download=True,
        )
        count = len(ds.data)
        self.data = ds.data[:count//10]
        self.labels = ds.targets[:count//10]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = x / 255
        y = self.labels[idx]
        x = x[None, ...]
        return x, y
