from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import *
from src.model import Ensemble
from src.train import training_loop, metric
import torch.nn as nn


if __name__ == '__main__':
    global_path = ''
    path_close = global_path + 'Data_images/Close'
    transform = transforms.Compose([ConcatenateImgs(),
                                    PermuteImgs()])

    training_set_close = CustomDataSet(path_close + '/train', transform=transform)
    validation_set_close = CustomDataSet(path_close + '/validation', transform=transform)
    test_set_close = CustomDataSet(path_close + '/test', transform=transform)

    train_loader = DataLoader(training_set_close,
                              batch_size=16,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=1)

    validation_loader = DataLoader(validation_set_close,
                                   batch_size=16,
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=1)

    test_loader = DataLoader(test_set_close,
                             batch_size=16,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=1)

    inits = [nn.init.xavier_uniform_,
             nn.init.xavier_uniform_]

    ensemble = Ensemble(inits)
    optimizer = torch.optim.Adam(ensemble.get_parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    training_loop(2, optimizer, ensemble, criterion, metric, train_loader, validation_loader)
