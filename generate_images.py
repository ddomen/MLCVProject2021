from src.dataset import *
import os

def mean(data, period):
    return sum(data) / period

def generate_gaf(args):
    index, id = -1, '???'
    try:
        index, id, path, data, features, period, pixel, max_period, diff, overwrite = args
        diff = diff == 'gadf'
        for feature in features:
            dict_images = img_dataset(data, pixel, feature, [ period ], max_period=max_period, aggregation=mean, diff=diff)
            for period, (imgs_subset, labels_subset, split_subset, ids_subset) in dict_images.items():
                save_dataframe_as_images(os.path.join(path, str(pixel), feature), ids_subset, imgs_subset, labels_subset, split_subset, period)
        return True, False, index, 0
    except KeyboardInterrupt:
        return False, True, index, 0
    except Exception as ex:
        print(f'\n[ERROR][{id}]: {ex}')
        return False, False, index, 0

def generate_candles(args):
    index, id = -1, '???'
    try:
        index, id, path, data, features, period, pixel, max_period, diff, overwrite = args
        _, n_img = save_candle_dataset(path, data, pixel, period, overwrite)
        return True, False, index, n_img
    except KeyboardInterrupt:
        return False, True, index, 0
    except Exception as ex:
        print(f'\n[ERROR][{id}]: {ex}')
        return False, False, index, 0

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
    parser.add_argument('-x', '--pixels', action='append', default=[], type=int, help='Size of the result image in pixels')
    parser.add_argument('-f', '--features', action='append', default=[], type=str.lower, choices=feat_choiches, help='Features to calculate the images')
    parser.add_argument('-t', '--periods', action='append', default=[], type=int, choices=period_choiches, help='Periods')
    parser.add_argument('-mt', '--max-period', default=None, type=int, help='Max period to calculate')
    parser.add_argument('-i', '--input', default='Data/dataframe.hdf5', type=str, help='File where to load the dataset')
    parser.add_argument('-o', '--output', default=None, type=str, help='Folder where to save images')
    parser.add_argument('-s', '--start', default=0, type=int, help='Starting index')
    parser.add_argument('-q', '--quantity', default=None, type=int, help='Number of factories')
    parser.add_argument('-c', '--type', default='gadf', type=str, choices=('gadf', 'gasf', 'candles'), help='Type of image output')
    parser.add_argument('-w', '--overwrite', nargs='?', default=False, const=True, type=bool, help='Overwrite images (supported just by candles)')
    args = parser.parse_args()

    if args.key in ('all', '*'): args.key = None
    DATASET = None if args.key in ('all', '*') else args.key
    
    TYPE = args.type
    IS_CANDLES = TYPE == 'candles'
    FEATURES = args.features if args.features and len(args.features) else [ 'close' ]
    PERIODS = args.periods if args.periods and len(args.periods) else ((5, 10, 20) if IS_CANDLES else period_choiches)
    BASE_PIXELS = 40 if IS_CANDLES else 20
    PIXELS = [ BASE_PIXELS if (x or 0) <= 0 else x for x in (args.pixels if len(args.pixels) else [ BASE_PIXELS ]) ]
    MAX_PERIOD = args.max_period or None
    INPUT = args.input or 'Data/dataframe.hdf5'
    PROCESSES = args.processes or multiprocessing.cpu_count()
    START = args.start or 0
    QUANTITY = args.quantity
    STOP = None if args.quantity is None else START + QUANTITY
    DEFAULT_OUTPUT = 'Data/Images/candles/' if IS_CANDLES else 'Data/Images/'
    OUTPUT = args.output or DEFAULT_OUTPUT
    OVERWRITE = args.overwrite

    print(f'''Generating Images:
=============================
| Processes:  | {PROCESSES if args.processes is not None else ('Auto (' + str(PROCESSES) + ')')}
| Features:   | {str.join(', ', FEATURES)}
| Periods:    | {str.join(', ', [ str(x) for x in PERIODS ])}
| Pixels:     | {str.join(', ', [ f'{p}x{p}' for p in PIXELS ])}
| Max Period: | {MAX_PERIOD or ('Auto (' + str(max(PERIODS)) + ')')}
| Dataset:    | {DATASET}
| Input:      | {INPUT}
| Output:     | {OUTPUT}
| Start:      | {START}
| Factories:  | {QUANTITY if QUANTITY is not None else 'ALL'}
| Image Type: | {TYPE}
| Overwrite:  | {OVERWRITE}
=============================
Loading Dataset...''')

    if MAX_PERIOD is None: MAX_PERIOD = max(PERIODS)

    df = pd.read_hdf(INPUT, key=args.key)

    df['close-open'] = df['close'] - df['open']
    df['split'] = 'train'

    print('Dataset Example:')
    print(df.head(5))

    max_index = START
    ids = tuple(df['id'].unique())[START:STOP]
    ids_done = list(range(START + 1))
    total_fact = len(ids)
    total_fact_glb = total_fact + (START or 0)

    N_IMG = 0

    glob_min_index = 0


    def make_desc():
        global glob_min_index
        for min_index in range(max_index, 0, -1):
            if min_index == 1 and 1 not in ids_done:
                break
            correct = True
            for i in range(1, min_index):
                if i not in ids_done:
                    correct = False
                    break
            if correct:
                glob_min_index = min_index
                return f'Factories - Min/Max index {min_index} . {max_index} / {total_fact_glb} ({N_IMG})'
        glob_min_index = 0
        return f'Factories - Min/Max index 0 . {max_index} / {total_fact_glb} ({N_IMG})'


    __last_pixel = PIXELS[-1]
    __last_period = PERIODS[-1]
    def next_data():
        index = START
        for id in ids:
            sub_set = df[df['id'] == id]
            index += 1
            for pixel in PIXELS:
                for period in PERIODS:
                    yield index - int(pixel != __last_pixel or period != __last_period), id, OUTPUT, sub_set, FEATURES, period, pixel, MAX_PERIOD, TYPE, OVERWRITE

    generator = generate_candles if TYPE == 'candles' else generate_gaf

    total_length = len(ids) * len(PERIODS) * len(PIXELS)

    with multiprocessing.Pool(PROCESSES) as pool, tqdm.tqdm(initial=0, total=total_length, desc=make_desc(), unit='fact') as global_bar:
        for success, interrupt, index, n_img in pool.imap_unordered(generator, iter(next_data())):
            if interrupt: break
            max_index = max(max_index, index)
            N_IMG += n_img
            ids_done.append(index)
            with open('test.txt', 'w') as f: f.write(str(glob_min_index))
            global_bar.set_description_str(make_desc())
            global_bar.update()
