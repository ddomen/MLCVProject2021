from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import *

if __name__ == '__main__':
    df = get_dataframe(data_path='Data/ETFs')
    print(df.head(5))

    FEATURES = ["Open", "High", "Low", "Close", "Volume"]
    PERIODS = [1, 2, 3, 5]
    PIXELS = 20
    path = 'Data_images/Close/train'

    training_set = CustomDataSet(path, transform=transforms.Compose([ConcatenateImgs()]))

    train_loader = DataLoader(training_set,
                              batch_size=16,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=1)

    for idx_batch, (imgs, labels) in enumerate(train_loader):
        pass
