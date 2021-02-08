from src.dataset import *
import os

def generate_data(args):
    index, id = -1, '???'
    try:
        index, id, path, data, features, periods, pixels, max_period = args

        for feature in features:
            dict_images = img_dataset(data, pixels, feature, periods, max_period=max_period)
            for period, (imgs_subset, labels_subset, split_subset, ids_subset) in dict_images.items():
                save_dataframe_as_images(os.path.join(path, feature), ids_subset, imgs_subset, labels_subset, split_subset, period)
        
        return True, index
    except KeyboardInterrupt:
        return False, index
    except Exception as ex:
        print(f'\n[ERROR][{id}]: {ex}')
        return False, index



if __name__ == '__main__':
    import tqdm
    import argparse
    import pandas as pd
    import multiprocessing

    feat_choiches = ( 'open', 'high', 'low', 'close', 'volume' , 'close-open')
    period_choiches = ( 1, 2, 3, 5 )

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--processes', default=None, type=int, help='Number of processes to spawn')
    parser.add_argument('-k', '--key', default='etfs', type=str.lower, choices=('etfs', 'stocks', 'all', '*'), help='Type of data to generate')
    parser.add_argument('-x', '--pixels', default=20, type=int, help='Size of the result image in pixels')
    parser.add_argument('-f', '--features', action='append', default=[], type=str.lower, choices=feat_choiches, help='Features to calculate the images')
    parser.add_argument('-t', '--periods', action='append', default=[], type=int, choices=period_choiches, help='Periods')
    parser.add_argument('-mt', '--max-period', default=None, type=int, help='Max period to calculate')
    parser.add_argument('-i', '--input', default='Data/dataframe.hdf5', type=str, help='File where to load the dataset')
    parser.add_argument('-o', '--output', default='Data/Images/', type=str, help='Folder where to save images')
    parser.add_argument('-s', '--start', default=0, type=int, help='Starting index')
    args = parser.parse_args()

    if args.key in ('all', '*'): args.key = None
    DATASET = None if args.key in ('all', '*') else args.key
    
    FEATURES = args.features if args.features and len(args.features) else [ 'close' ]
    PERIODS = args.periods if args.periods and len(args.periods) else period_choiches
    PIXELS = 20 if (args.pixels or 0) <= 0 else args.pixels
    MAX_PERIOD = args.max_period or None
    OUTPUT = args.output or 'Data/Images/'
    INPUT = args.input or 'Data/dataframe.hdf5'
    PROCESSES = args.processes or multiprocessing.cpu_count()
    START = args.start or 0

    print(f'''Generating Images:
=============================
| Processes:  | {PROCESSES if args.processes is not None else ('Auto (' + str(PROCESSES) + ')')}
| Features:   | {str.join(', ', FEATURES)}
| Periods:    | {str.join(', ', [ str(x) for x in PERIODS ])}
| Pixels:     | {PIXELS}x{PIXELS}
| Max Period: | {MAX_PERIOD or ('Auto (' + str(max(PERIODS)) + ')')}
| Dataset:    | {DATASET}
| Input:      | {INPUT}
| Output:     | {OUTPUT}
| Start:      | {START}
=============================
Loading Dataset...''')
    df = pd.read_hdf(INPUT, key=args.key)

    df['close-open'] = df['close'] - df['open']
    df['split'] = 'train'
    
    print('Dataset Example:')
    print(df.head(5))

    max_index = START
    ids = tuple(df['id'].unique())[START:]
    ids_done = list(range(START + 1))

    def make_desc():
        for min_index in range(max_index, 0, -1):
            if min_index == 1 and 1 not in ids_done:
                break
            correct = True
            for i in range(1, min_index):
                if i not in ids_done:
                    correct = False
                    break
            if correct:
                return f'Factories - Min/Max index {min_index}/{max_index}'
        return f'Factories - Min/Max index 0/{max_index}'
        

    def get_next_sub_dataset():
        index = START
        for id in ids:
            sub_set = df[df['id'] == id]
            index += 1
            yield index, id, OUTPUT, sub_set, FEATURES, PERIODS, PIXELS, MAX_PERIOD


    with multiprocessing.Pool(PROCESSES) as pool, tqdm.tqdm(initial=0, total=len(ids), desc=make_desc(), unit='fact') as global_bar:
        for success, index in pool.imap_unordered(generate_data, iter(get_next_sub_dataset())):
            max_index = max(max_index, index)
            ids_done.append(index)
            global_bar.set_description_str(make_desc())
            global_bar.update()