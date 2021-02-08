import numpy as np
import torch

from src.train import metric

import torch.nn as nn
from torch.nn import functional as F


class vggCNN(nn.Module):

    def __init__(self, init=None):
        super().__init__()

        self.cnn_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)
        self.dropout_1 = nn.Dropout(p=0.25)
        self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2)
        self.dropout_2 = nn.Dropout(p=0.25)
        self.cnn_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.cnn_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2)
        self.classifier = nn.Linear(128, 2)
        to_init = [self.cnn_1, self.cnn_2, self.cnn_3, self.cnn_4, self.cnn_5]

        if init is not None:
            for layer in to_init:
                init(layer.weight)

    def forward(self, x):
        x = F.relu(self.cnn_1(x))
        x = self.dropout_1(self.maxpool_1(x))
        x = F.relu(self.cnn_2(x))
        x = F.relu(self.cnn_3(x))
        x = self.dropout_2(self.maxpool_2(x))
        x = F.relu(self.cnn_4(x))
        x = F.relu(self.cnn_5(x))
        x = self.maxpool_3(x)
        return F.softmax(self.classifier(x.squeeze().squeeze()), dim=1)


class resCNN(nn.Module):
    def __init__(self, init=None):
        super().__init__()

        self.cnn_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout_1 = nn.Dropout(p=0.25)
        self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.batch_norm_1 = nn.BatchNorm2d(64, affine=False)
        self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.dropout_2 = nn.Dropout(p=0.25)
        self.cnn_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.batch_norm_2 = nn.BatchNorm2d(128, affine=False)
        self.cnn_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=15, padding=0, ceil_mode=False)
        self.classifier = nn.Linear(128, 2)

        to_init = [self.cnn_1, self.cnn_2, self.cnn_3, self.cnn_4, self.cnn_5]

        if init is not None:
            for layer in to_init:
                init(layer.weight)

    def forward(self, x):
        x = F.relu(self.cnn_1(x))
        x = self.dropout_1(self.maxpool(x))
        x = F.relu(self.batch_norm_1(self.cnn_2(x)))
        x = F.relu(self.cnn_3(x)) + x
        x = self.dropout_2(x)
        x = F.relu(self.batch_norm_2(self.cnn_4(x)))
        x = F.relu(self.cnn_5(x)) + x
        x = self.avgpool(x)
        x = F.softmax(self.classifier(x.squeeze().squeeze()), dim=1)
        return x


class Ensemble(nn.Module):
    def __init__(self, inits, thresh=0.5, lr=1e-5, type="vgg"):
        super().__init__()

        if type is "res":
            self.cnns = [resCNN(init) for init in inits]
        else:
            self.cnns = [vggCNN(init) for init in inits]

        self.thresh = thresh
        self.optimizers = [torch.optim.Adam(cnn.parameters(), lr=lr) for cnn in self.cnns]

    def train_step(self, batch, labels, criterion):
        losses = []
        for cnn, optimizer in zip(self.cnns, self.optimizers):
            optimizer.zero_grad()
            pred_labels = cnn(batch)
            loss = criterion(pred_labels, labels)
            losses.append(loss.item() * len(batch))
            loss.backward()
            optimizer.step()

        return np.mean(losses), np.std(losses)

    def validation_step(self, batch, labels, criterion, metric):
        loss_validation = 0
        predictions = []
        for cnn in self.cnns:
            pred_labels = cnn(batch)
            predictions.append(torch.argmax(pred_labels, dim=1).type(torch.float32))
            loss = criterion(pred_labels, labels)
            loss_validation += loss.item() * len(batch)

        means = torch.mean(torch.stack(predictions, dim=1), dim=1)
        results = (((means >= self.thresh).type(torch.float32) - ((1 - means) >= self.thresh).type(
            torch.float32)) + 1) / 2
        acc_batch = metric(means, labels)

        return loss_validation / len(self.cnns), acc_batch

    def test_step(self, batch, labels):
        predictions = []
        for cnn in self.cnns:
            pred_labels = cnn(batch)
            predictions.append(torch.argmax(pred_labels, dim=1).type(torch.float32))

        means = torch.mean(torch.stack(predictions, dim=1), dim=1)
        results = (((means >= self.thresh).type(torch.float32) - ((1 - means) >= self.thresh).type(
            torch.float32)) + 1) / 2
        acc_batch = metric(results, labels)

        return acc_batch

    def _train(self):
        for cnn in self.cnns:
            cnn.train()

    def eval(self):
        for cnn in self.cnns:
            cnn.eval()
