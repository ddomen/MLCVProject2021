from src.dataset import *


if __name__ == '__main__':
    df = get_dataframe(data_path='Data/ETFs')
    print(df.head(5))

    FEATURES = ["Open", "High", "Low", "Close", "Volume"]
    PERIODS = [1, 2, 3, 5]
    PIXELS = 20

    path = 'Data_images/'
    for id in df['ID'].unique():
        df_subset = df[df['ID'] == id]
        for feature in FEATURES:
            dict_images = img_dataset(df_subset, PIXELS, feature, PERIODS)
            for period, (imgs_subset, labels_subset, split_subset, ids_subset) in dict_images.items():
                save_dataframe_as_images(path + feature, ids_subset, imgs_subset, labels_subset, split_subset, period)