from src.dataset import *


if __name__ == '__main__':
    df = get_dataframe(data_path='Data/ETFs')
    print(df.head(5))