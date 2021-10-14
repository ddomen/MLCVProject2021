from collections import namedtuple
from random import sample

import torch
from torch.nn import functional as F

from torch import nn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def set_done(self):
        self.memory[self.position - 1] = self.memory[self.position - 1]._replace(next_state=None)


class nonLocalBlock(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.teta = nn.Conv2d(in_channels=input_channels, out_channels=int(input_channels / 2), kernel_size=1)
        self.fi = nn.Conv2d(in_channels=input_channels, out_channels=int(input_channels / 2), kernel_size=1)
        self.gi = nn.Conv2d(in_channels=input_channels, out_channels=int(input_channels / 2), kernel_size=1)
        self.out = nn.Conv2d(in_channels=int(input_channels / 2), out_channels=input_channels, kernel_size=1)
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x):
        x_1 = self.flatten(self.teta(x))
        x_1 = x_1.view(x_1.shape[0], x_1.shape[2], x_1.shape[1])
        x_2 = self.flatten(self.fi(x))
        x_3 = self.flatten(self.gi(x))
        x_3 = x_3.view(x_3.shape[0], x_3.shape[2], x_3.shape[1])
        x_1_2 = F.softmax(torch.matmul(x_1, x_2), dim=1)

        x_1_2_3 = self.out(torch.transpose(torch.matmul(x_1_2, x_3), dim0=1, dim1=2).view(x.shape[0], -1, x.shape[2], x.shape[3]))
        return x + x_1_2_3


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

    def __init__(self, input_channels, num_channels, strides=1, use_1x1conv=True, non_local=False):
        super().__init__()
        self.non_local = non_local
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        if self.non_local:
            self.nl = nonLocalBlock(num_channels)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        Y = F.relu(Y)
        if self.non_local:
            Y = self.nl(Y)
        return Y


class resCNN(nn.Module):
    def __init__(self, init=None, non_local=False):
        super().__init__()
        self.b1 = nn.Sequential(*([
            nn.Conv2d(3, 32, kernel_size=7, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            ] + ([ nonLocalBlock(32) ] if non_local else []))
        )
        self.b2 = Residual(32, 64, 2, non_local=non_local)
        self.b3 = Residual(64, 128, 2, non_local=non_local)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(128, 2)
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
    '''
    Similar to Residual Block but optimized for rhombus cutted images
    (test layer, not used in the final application)
    '''
    def __init__(self, output_size, init=None):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=7),
                                nn.BatchNorm2d(32), nn.ReLU())
        self.b2 = Residual(32, 64, 2)
        self.b3 = Residual(64, 128, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        to_init = [self.b1[0], self.b2.conv1, self.b2.conv2, self.b2.conv3, self.b3.conv1, self.b3.conv2, self.b3.conv3]

        if init is not None:
            for layer in to_init:
                init(layer.weight)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return self.avg_pool(x)


class myCNN(nn.Module):
    '''
    Network: cut rhombus from images and uses a Residual Block
    to compute image features
    '''
    def __init__(self, init=None):
        super().__init__()

        self.cnn_center_image = CNNsubimage((2, 2), init=init)

        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        x = self.flatten(self.cnn_center_image(x[:, :, 10:31, 10:31]))
        return self.classifier(x)


class DQN(nn.Module):

    def __init__(self, inits, non_local=False, type="vgg"):
        super(DQN, self).__init__()

        if type is "res":
            self.cnns = [resCNN(init, non_local=non_local) for init in inits]
        if type is "vgg":
            self.cnns = [vggCNN(init) for init in inits]
        if type is "myCNN":
            self.cnns = [myCNN(init) for init in inits]

        self.batch_norm = nn.BatchNorm1d(2 * len(self.cnns))
        self.output = nn.Linear(2 * len(self.cnns), 3)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def forward(self, x):
        x = torch.flatten(torch.cat([cnn(x).unsqueeze(dim=2) for cnn in self.cnns], dim=2), start_dim=1)
        x = F.relu(self.batch_norm(x))
        x = F.softmax(self.output(x), dim=1)
        return x

    def get_parameters(self):
        return list(cnn.parameters() for cnn in self.cnns) + [self.output.parameters()]