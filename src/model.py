from random import shuffle

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
        # self.classifier.bias = nn.Parameter(torch.tensor([0.0, 1.0]))

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
        x = torch.flatten(self.maxpool_3(x), start_dim=1)
        return self.classifier(x.squeeze().squeeze())


class Residual(nn.Module):
    """The Residual block of ResNet."""

    def __init__(self, input_channels, num_channels, strides=1, use_1x1conv=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X

        return F.relu(Y)


class resCNN(nn.Module):
    def __init__(self, init=None):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm2d(32), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = Residual(32, 64, 2)
        self.b3 = Residual(64, 128, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(128, 2)
        # self.classifier.bias = nn.Parameter(torch.tensor([1.0]))
        to_init = [self.b1[0], self.b2.conv1, self.b2.conv2, self.b3.conv1, self.b3.conv2]

        if init is not None:
            for layer in to_init:
                init(layer.weight)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.classifier(x)


class CNNsubimage(nn.Module):
    def __init__(self, output_size, init=None):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5),
                                nn.BatchNorm2d(32), nn.ReLU())
        self.b2 = Residual(32, 64, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        to_init = [self.b1[0], self.b2.conv1, self.b2.conv2, self.b2.conv3]

        if init is not None:
            for layer in to_init:
                init(layer.weight)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        return self.avg_pool(x)


class myCNN(nn.Module):
    def __init__(self, init=None):
        super().__init__()

        self.cnn_subimage_1 = CNNsubimage((1, 1), init=init)
        self.cnn_subimage_2 = CNNsubimage((1, 1), init=init)
        self.cnn_subimage_3 = CNNsubimage((1, 1), init=init)
        self.cnn_subimage_4 = CNNsubimage((1, 1), init=init)
        self.cnn_center_image = CNNsubimage((2, 2), init=init)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        sub_image_1 = self.flatten(self.cnn_subimage_1(x[:, :, 0:20, 0:20]))
        sub_image_2 = self.flatten(self.cnn_subimage_2(x[:, :, 0:20, 20:40]))
        sub_image_3 = self.flatten(self.cnn_subimage_3(x[:, :, 20:40, 0:20]))
        sub_image_4 = self.flatten(self.cnn_subimage_4(x[:, :, 20:40, 20:40]))
        sub_image_center = self.flatten(self.cnn_center_image(x[:, :, 10:30, 10:30]))
        x = torch.cat([sub_image_1, sub_image_2, sub_image_3, sub_image_4, sub_image_center], dim=1)
        return self.classifier(x)


class Ensemble(nn.Module):
    def __init__(self, inits, thresh=0.75, lr=1e-3, type="vgg"):
        super().__init__()

        if type is "res":
            self.cnns = [resCNN(init) for init in inits]
        if type is "vgg":
            self.cnns = [vggCNN(init) for init in inits]
        if type is "myCNN":
            self.cnns = [myCNN(init) for init in inits]

        self.thresh = thresh
        self.optimizers = [torch.optim.Adam(cnn.parameters(), lr=lr, weight_decay=1e-5) for cnn in self.cnns]
        self.losses = [torch.nn.CrossEntropyLoss() for _ in self.cnns]

    def train_step(self, batch, labels):
        losses = []

        for cnn, optimizer, loss in zip(self.cnns, self.optimizers, self.losses):
            optimizer.zero_grad()
            pred_labels = cnn(batch)
            loss_cnn = loss(pred_labels, labels)
            loss_cnn.backward()
            optimizer.step()
            losses.append(loss_cnn.item() * len(batch))

        return np.mean(losses), np.std(losses)

    def validation_step(self, batch, labels, metric):
        loss_validation = 0
        predictions = []
        for (cnn, loss) in zip(self.cnns, self.losses):
            pred_labels = cnn(batch)
            predictions.append(torch.argmax(pred_labels, dim=1).type(torch.float32))
            loss_cnn = loss(pred_labels, labels)
            loss_validation += loss_cnn.item() * len(batch)

        means = torch.mean(torch.cat([prediction.unsqueeze(dim=1) for prediction in predictions], dim=1), dim=1)
        results = (((means >= self.thresh).type(torch.float32) - ((1 - means) >= self.thresh).type(
            torch.float32)) + 1) / 2
        answers = results[results != 0.5]
        for i in range(len(results)):
            if results[i] != 0.5:
                results[i] = torch.round(results[i])

        hit_batch = metric(results, labels)
        return loss_validation / len(self.cnns), hit_batch, len(answers)

    def _train(self):
        for cnn in self.cnns:
            cnn.train()

    def _eval(self):
        for cnn in self.cnns:
            cnn.eval()
