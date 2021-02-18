import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import *
from src.initialization import *
from src.model import Ensemble
from src.train import training_loop, metric
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import random


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


def show_first_kernels(state_dict):
    show(torchvision.utils.make_grid(state_dict['cnn_1.weight'].cpu(), normalize=True, scale_each=True))


if __name__ == '__main__':

    global_path = ''
    path_close = global_path + 'archive (2)/close-open'
    transform = transforms.Compose([ConcatenateImgs(),
                                    PermuteImgs()]
                                   )

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

    len_subset_train = 4
    len_subset_validation = 1
    shift = len_subset_train + len_subset_validation
    ids = return_unique_ids(path_close + '/train')
    ensemble = Ensemble(inits, type="myCNN")

    for time in range(50):
        results = dict()
        iters = 0
        print("time", time)
        random.shuffle(ids)
        for id in ids:
            training_set_close = CustomDataSet(path_close + '/train', id=id, transform=transform, rot=True)

            train_loader = DataLoader(training_set_close,
                                      batch_size=30,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=1)
            print("Train/validate on:", id)
            # show_first_kernels(ensemble.cnns[0].state_dict())
            results[id], iter = training_loop(ensemble, metric, train_loader, len_subset_train, len_subset_validation,
                                              shift)
            iters += iter
            print()
        print("valid accuracy (mean):", np.sum([results[i]['accuracy_val'] for i in ids]) / iters)
