from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import *
from src.initialization import *
from src.model import Ensemble
from src.train import training_loop, metric


if __name__ == '__main__':

    global_path = ''
    path_close = global_path + 'close-open'
    transform = transforms.Compose([ConcatenateImgs(),
                                    PermuteImgs()])

    training_set_close = CustomDataSet(path_close + '/train', transform=transform)

    train_loader = DataLoader(training_set_close,
                              batch_size=40,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=1)

    inits = [torch.nn.init.orthogonal_,
             lecun_uniform(),
             variance_scaling_fan_in(1),
             uniform(-0.05, 0.05),
             normal(0, 0.05),
             truncated_normal_(0, 0.05),
             glorot_uniform(),
             glorot_normal(),
             he_normal(),
             he_uniform(),
             lecun_uniform(seed=42),
             variance_scaling_fan_in(1, seed=42),
             uniform(-0.05, 0.05, seed=42),
             normal(0, 0.05, seed=42),
             truncated_normal_(0, 0.05, seed=42),
             glorot_uniform(seed=42),
             glorot_normal(seed=42),
             he_normal(seed=42),
             he_uniform(seed=42),
             torch.nn.init.orthogonal_
             ]


    ensemble = Ensemble(inits, type="vgg")
    criterion = torch.nn.CrossEntropyLoss()
    len_subset_train = 10
    len_subset_validation = 2
    shift = 2
    times = 3

    for time in range(times):
        print("TIME: ", time)
        training_loop(1, ensemble, criterion, metric, train_loader, len_subset_train, len_subset_validation, shift)
        print()
