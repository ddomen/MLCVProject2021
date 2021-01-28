import numpy as np
import pandas as pd
import os

import torch
from matplotlib import cm
from PIL import Image

from src.encoding import gasf

from torch.utils.data import Dataset

SCHEMA = {'ID': pd.Series([], dtype='str'),
          'Date': pd.Series([], dtype='str'),
          'Open': pd.Series([], dtype='double'),
          'High': pd.Series([], dtype='double'),
          'Low': pd.Series([], dtype='double'),
          'Close': pd.Series([], dtype='double'),
          'Volume': pd.Series([], dtype='long'),
          'Trend': pd.Series([], dtype='bool'),
          'Split': pd.Series([], dtype='str')
          }


def get_dataframe(data_path,
                  train_val_test_split=None):
    if train_val_test_split is None:
        train_val_test_split = [0.7, 0.1, 0.2]
    df = pd.DataFrame(SCHEMA)
    for file in os.listdir(data_path)[:1]:
        # Read the dataset
        stock_data = pd.read_csv(data_path + '/' + file, parse_dates=['Date'])
        # Drop the 'OpenInt' column since it is equal to 0 in each file
        stock_data = stock_data.drop(columns=['OpenInt'])
        # Compute the label and shift
        stock_data['Trend'] = (stock_data['Close'] > stock_data['Open']) * 1
        stock_data['Trend'] = stock_data['Trend'].shift(periods=-1)
        # Drop the last column since it has not a Trend value
        stock_data = stock_data[:-1]
        # The ID of each intermediate dataframe is the name of the file without '.txt'
        stock_data['ID'] = file[0:(len(file) - 4)]
        # Rescale in [0, 1]
        if not stock_data.empty:
            # Assign the split names
            rows = stock_data.shape[0]
            train_split = [0, round(train_val_test_split[0] * rows)]
            val_split = [round(train_val_test_split[0] * rows),
                         (round((train_val_test_split[0] + train_val_test_split[1]) * rows))]
            test_split = [(round((train_val_test_split[0] + train_val_test_split[1]) * rows)), rows]
            stock_data['Split'] = None
            stock_data.loc[train_split[0]:train_split[1], "Split"] = "train"
            stock_data.loc[val_split[0]:val_split[1], "Split"] = "validation"
            stock_data.loc[test_split[0]:test_split[1], "Split"] = "test"

            # Concatenate the dataframes
            df = pd.concat([df, stock_data], axis=0)

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
            img = gasf(df[feature][max_period - period * pixels + i:i + max_period:period])
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = my_cm(img)
            imgs_list.append(img[:, :, :3])
            labels_list.append(np.array(df['Trend'][i + max_period]))
            split_list.append(np.array(df['Split'][i + max_period]))
            ids_list.append(np.array(df['ID'][i + max_period]))

        subset_dict[period] = np.stack(imgs_list, axis=0), np.stack(labels_list, axis=0), np.stack(split_list,
                                                                                                   axis=0), np.stack(
            ids_list, axis=0)

    return subset_dict


def generate_dataset_imgs(path, df, periods, features, pixels):
    for id in df['ID'].unique():
        df_subset = df[df['ID'] == id]
        for feature in features:
            dict_images = img_dataset(df_subset, pixels, feature, periods)
            for period, (imgs_subset, labels_subset, split_subset, ids_subset) in dict_images.items():
                save_dataframe_as_images(path + feature, ids_subset, imgs_subset, labels_subset, split_subset, period)


def save_dataframe_as_images(path, ids, images, labels, splits, period):
    os.makedirs(f"{path}/train", exist_ok=True)
    os.makedirs(f"{path}/validation", exist_ok=True)
    os.makedirs(f"{path}/test", exist_ok=True)

    cont = {"train": 0,
            "validation": 0,
            "test": 0}

    for id, img, label, split in zip(ids, images, labels, splits):
        file_name = str(id) + "_" + str(period) + "_" + str(cont[split]) + "_" + str(int(label)) + ".png"
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.save(f"{path}/{split}/{file_name}")

        cont[split] += 1


class ConcatenateImgs(object):
    def __call__(self, images):
        raw_1 = torch.cat((images[0], images[1]), dim=0)
        raw_2 = torch.cat((images[2], images[3]), dim=0)

        return torch.cat((raw_1, raw_2), dim=1)


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.images = os.listdir(main_dir)

    def __len__(self):
        return int(len(self.images) / 4)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.images[idx])
        img_info = img_loc.split(sep="_")
        label = int(img_info[-1].replace(".png", ""))
        images = []
        for i in [1, 2, 3, 5]:
            img_info[-3] = str(i)
            img_loc = str.join("_", img_info)
            image = Image.open(img_loc).convert("RGB")
            images.append(torch.from_numpy(np.asarray(image, dtype="float32") / 127.5 - 1))

        tensor_image = self.transform(images)

        return tensor_image, label
