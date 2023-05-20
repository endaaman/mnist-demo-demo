import os
import click
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from datasets import MNISTDataset
from models import LinearModel, CNNModel

from tensorboardX import SummaryWriter



ds = MNISTDataset(train=True)


@click.group()
def cli():
    pass

@cli.command()
@click.option('--batch-size', '-B', default=8)
@click.option('--model-name', '-M', default='linear')
@click.option('--lr', default=0.001)
@click.option('--epoch', '-E', default=100)
def train(batch_size, model_name, lr, epoch):
    train_ds = MNISTDataset(train=True)
    val_ds = MNISTDataset(train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    match model_name:
        case 'linear':
            model = LinearModel()
        case 'cnn':
            model = CNNModel()
        case _:
            raise RuntimeError('Invalid model_name:', model_name)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    logdir = f'logs/{model_name}'
    if os.path.isdir(logdir):
        shutil.rmtree(logdir)
    writer = SummaryWriter(logdir=logdir)

    train_losses = []
    val_losses = []
    t = tqdm(range(epoch))
    for i in t:
        losses = []
        t2 = tqdm(train_loader, leave=False)
        for (xx, gts) in t2:
            model.train()
            optimizer.zero_grad()
            preds = model(xx)
            loss = criterion(preds, gts)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            t2.set_description(f'train loss: {loss.item():.3f}')
        train_losses.append(np.mean(losses))
        writer.add_scalar('loss/train', train_losses[-1], i)

        losses = []
        t2 = tqdm(val_loader, leave=False)
        for (xx, gts) in t2:
            model.eval()
            with torch.set_grad_enabled(False):
                preds = model(xx)
            loss = criterion(preds, gts)
            losses.append(loss.item())
            t2.set_description(f'val loss: {loss.item():.3f}')
        val_losses.append(np.mean(losses))
        writer.add_scalar('loss/val', val_losses[-1], i)

        t.set_description(f'[{i+1}/{epoch}] train loss:{train_losses[-1]:.3f} val loss:{val_losses[-1]:.3f}')
        t.refresh()

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.savefig(f'out/{model_name}.png')
    plt.leged()
    plt.show()



# @cli.command()
# def bar():
#     print('bar')


if __name__ == '__main__':
    cli()
