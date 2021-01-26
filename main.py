from src.dataset import *


if __name__ == '__main__':
    df = get_dataframe(data_path='Data/ETFs')
    print(df.head(5))

    PIXELS = 20
    PERIOD = 1

    for id in df['ID'].unique():
        df_subset = df[df['ID'] == id]
        imgs_subset, labels_subset, split_subset = img_dataset(df, PIXELS, PERIOD)

        print(len(df))
        print(imgs_subset.shape)
        print(labels_subset.shape)
        print(split_subset.shape)