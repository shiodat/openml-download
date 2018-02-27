# coding: utf-8

import os

import openml
import pandas as pd
import yaml


with open('openml.yaml') as f:
    config = yaml.load(f)

save_dir = config['save_dir']
if save_dir[-1] != os.sep:
    save_dir += os.sep
os.makedirs(save_dir, exist_ok=True)
data_dir = save_dir + 'data' + os.sep
os.makedirs(data_dir, exist_ok=True)

openml.config.apikey = config['api_key']
openml.config.set_cache_directory(os.path.expanduser(config['cache_dir']))

datasets = pd.read_csv('datasets.csv')
dataset_ids = datasets['id'].tolist()

data_info = openml.datasets.list_datasets()
data_info = pd.DataFrame.from_dict(data_info, orient='index')
data_info = data_info.loc[data_info['did'].isin(dataset_ids)]
data_info.to_csv(save_dir + 'info.csv', index=False)

for i, dataset_id in enumerate(dataset_ids):
    progress = int((i + 1) / len(dataset_ids) * 100)
    print('{0:>3d}% finished.'.format(progress))
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        return_attribute_names=True)
    df = pd.DataFrame(X, columns=attribute_names)
    df['target'] = y
    df.to_csv(data_dir + dataset.name + '.csv', index=False)
