import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import cm
from torch.utils.data import Dataset

from src.encoding import gasf, gadf

SCHEMA = {
    'id': 'str',
    'date': 'str',
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'close-open': 'float32',
    'volume': 'uint32',
    'trend': 'uint8',
    'split': 'str'
}


def get_dataframe(data_path, train_val_test_split=None, show_progress=False):
    if train_val_test_split is None:
        train_val_test_split = [1, 0, 0]
    
    appended_data = []

    files = os.listdir(data_path)

    if show_progress:
        import tqdm
        files = tqdm.tqdm(files)

    for file in files:
        try:
            # Read the dataset
            stock_data = pd.read_csv(os.path.join(data_path, file), parse_dates=['Date'])
            # Drop the 'OpenInt' column since it is equal to 0 in each file
            stock_data.drop(columns=['OpenInt'], inplace=True)
            # Lower case the data
            stock_data.rename(columns={column: column.lower() for column in stock_data.columns}, inplace=True)
            # Compute the label and shift
            stock_data['trend'] = (stock_data['close'] > stock_data['open']) * 1
            stock_data['trend'] = stock_data['trend'].shift(periods=-1)
            # Drop the last row since it has not a Trend value
            stock_data.drop(stock_data.tail(1).index, inplace=True)
            # The ID of each intermediate dataframe is the name of the file without '.txt'
            stock_data['id'] = file[:-4]
            stock_data['close-open'] = stock_data['close'] - stock_data['open']
            # Rescale in [0, 1]
            if not stock_data.empty:
                # Assign the split names
                rows = stock_data.shape[0]
                val_split = slice(
                    round(train_val_test_split[0] * rows),
                    round((train_val_test_split[0] + train_val_test_split[1]) * rows)
                )
                train_split = slice(0, val_split.start)
                test_split = slice(val_split.stop, None)

                stock_data['split'] = None
                stock_data.loc[train_split, 'split'] = 'train'
                stock_data.loc[val_split, 'split'] = 'validation'
                stock_data.loc[test_split, 'split'] = 'test'

                for data_key, data_type in SCHEMA.items():
                    stock_data[data_key] = stock_data[data_key].astype(data_type, copy=False)

                # Concatenate the dataframes
                # df = pd.concat([df, stock_data], axis=0)
                appended_data.append(stock_data)
        except KeyboardInterrupt as kex:
            raise kex
        except Exception as ex:
            print(f'Error during the reading of the file: \'{file}\' - {ex}')

    appended_data = pd.concat(appended_data)
    # Reorder the dataframe with the schema
    df = pd.DataFrame()
    for data_key in SCHEMA:
        df[data_key] = appended_data[data_key]
    # for data_key, data_type in SCHEMA.items():
    #     df[data_key] = appended_data[data_key].astype(data_type, copy=False)

    return df


def img_dataset(df, pixels, feature, periods, max_period=None):
    my_cm = cm.get_cmap('Spectral')
    max_period = (max_period or max(periods)) * pixels

    subset_dict = {}

    for period in periods:

        imgs_list = []
        labels_list = []
        split_list = []
        ids_list = []

        for i in range(len(df) - max_period):
            # Compute the temporal window of time
            time_slice = slice(max_period - period * pixels + i, i + max_period, period)

            # Generating and normalizing image
            img = gadf(df[feature][time_slice])
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = my_cm(img)[:, :, :3]

            # Fill the dataset
            imgs_list.append(img)
            labels_list.append(np.array(df['trend'][time_slice.stop]))
            split_list.append(np.array(df['split'][time_slice.stop]))
            ids_list.append(np.array(df['id'][time_slice.stop]))

        subset_dict[period] = (
            np.stack(imgs_list, axis=0),
            np.stack(labels_list, axis=0),
            np.stack(split_list, axis=0),
            np.stack(ids_list, axis=0)
        )

    return subset_dict


def save_dataframe_as_images(path, ids, images, labels, splits, period):
    os.makedirs(f"{path}/train", exist_ok=True)
    os.makedirs(f"{path}/validation", exist_ok=True)
    os.makedirs(f"{path}/test", exist_ok=True)

    cont = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    for id, img, label, split in zip(ids, images, labels, splits):
        file_name = f'{id}_{period}_{cont[split]:04d}_{label:1d}.png'
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.save(os.path.join(path, split, file_name))

        cont[split] += 1


class ConcatenateImgs(object):
    def __call__(self, images):
        raw_1 = torch.cat((images[0], images[1]), dim=0)
        raw_2 = torch.cat((images[2], images[3]), dim=0)

        return torch.cat((raw_1, raw_2), dim=1)


class PermuteImgs(object):
    def __call__(self, images):
        return images.permute(2, 0, 1)


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.images = [x for x in os.listdir(main_dir) if (("aadr.us" in x) or ("aaxj.us" in x))]

    def __len__(self):
        return int(len(self.images) / 4)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.images[idx])
        img_info = img_loc.split(sep="_")
        label = torch.tensor(float(img_info[-1].replace(".png", "")), dtype=torch.int64)
        images = []
        for i in [1, 2, 3, 5]:
            img_info[-3] = str(i)
            img_loc = str.join("_", img_info)
            image = Image.open(img_loc).convert("RGB")
            images.append(torch.from_numpy(np.asarray(image, dtype="float32") / 255))

        tensor_image = self.transform(images)

        return tensor_image, label
