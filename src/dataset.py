import numpy as np
import pandas as pd
import os

from matplotlib import cm

from src.encoding import gasf

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

FEATURE_RESCALING = ["Open", "High", "Low", "Close", "Volume"]


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
            for feature in FEATURE_RESCALING:
                stock_data[feature] = rescaling(stock_data[feature])
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


def rescaling(x):
    return (x - min(x)) / (max(x) - min(x))


def img_dataset(df, pixels, period):
    my_cm = cm.get_cmap('hsv')

    imgs_list = []
    labels_list = []
    split_list = []
    for i in range(0, len(df) - pixels, period):
        img = gasf(df['Close'][i:len(df)], pixels=pixels, period=period)
        img = my_cm(img)
        imgs_list.append(img)
        labels_list.append(np.array(df['Trend'][i + pixels]))
        split_list.append(np.array(df['Split'][i + pixels]))

    return np.stack(imgs_list, axis=0), np.stack(labels_list, axis=0), np.stack(split_list, axis=0)
