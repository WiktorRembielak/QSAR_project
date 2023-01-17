from preprocess import Dataset
from tools import find_latest_version
import yaml
import os

'''
The script contains a function which loads raw data, splits, processes and exports it
Usage:
    python3 ./etl.py
'''


def etl(config_file_path: str = './config.yaml'):

    # Config file loading
    with open(config_file_path) as file:
        config = yaml.safe_load(file)['etl']

    # Raw data loading and splitting
    data = Dataset(config['raw_data'])
    data.split_data(class_column=config['class_column'], reg_column=config['reg_column'],
                    classification=config['classification'], regression=config['regression'],
                    test_size=config['test_size'], val_size=config['val_size'])

    # Data processing
    if config['remove_outliers']:
        data.remove_outliers()

    if config['scale_standarization'] or config['scale_minamx']:
        data.scale(standard=config['scale_standarization'], minmax=config['scale_minamx'])

    if config['resample']:
        data.resample()

    if config['adjust_shape']:
        data.adjust_shape()

    # Processed data exporting
    new_data_path = find_latest_version('./data', 'processed_data', next_version=True)
    os.mkdir(new_data_path)
    for subset in data.subsets:
        if eval(f'data.{subset}') is not None:
            eval(f'data.{subset}').to_csv(f'{new_data_path}/{subset}.csv')


if __name__ == '__main__':
    etl()
