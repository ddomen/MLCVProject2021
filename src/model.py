import torch
import torch.nn as nn
from torch.nn import functional as F


class CNN(nn.Module):

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


class Ensemble(nn.Module):
    def __init__(self, inits, thresh=0.5):
        super().__init__()

        self.cnns = [CNN(init) for init in inits]
        self.thresh = thresh

    def forward(self, x):
        results = torch.stack([cnn(x) for cnn in self.cnns], dim=2)
        return torch.mean(results, dim=2)

    def get_parameters(self):
        return [param for cnn in self.cnns for param in cnn.parameters()]
