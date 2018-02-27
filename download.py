# coding: utf-8

import os

import openml
import pandas as pd
import yaml

with open('openml.yaml') as f:
    config = yaml.load(f)

openml.config.apikey = config['api_key']
openml.config.set_cache_directory(os.path.expanduser(config['cache_dir']))

datasets = pd.read_csv('datasets.csv')

for dataset_id in datasets['id'].tolist():
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        return_attribute_names=True)
    df = pd.DataFrame(X, columns=attribute_names)
    df['target'] = y
    df.to_csv('{}/{}.csv'.format(
              config['save_dir'], dataset.name), index=False)

