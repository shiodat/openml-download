# coding: utf-8

import os

import numpy as np
import openml
import pandas as pd
import yaml


# load config.
with open('openml.yaml') as f:
    config = yaml.load(f)

# set data directory.
save_dir = os.path.expanduser(config['save_dir'])
if save_dir[-1] != os.sep:
    save_dir = save_dir + os.sep
data_dir = save_dir + 'data' + os.sep
os.makedirs(data_dir, exist_ok=True)

# set OpenML cache.
openml.config.apikey = config['api_key']
openml.config.set_cache_directory(os.path.expanduser(config['cache_dir']))

# save dataset information.
def save_info(dataset_ids):
    data_info = openml.datasets.list_datasets()
    data_info = pd.DataFrame.from_dict(data_info, orient='index')
    data_info = data_info.loc[data_info['did'].isin(dataset_ids)]
    data_info.to_csv(save_dir + 'info.csv', index=False)

# get OpenML dataset list.
datasets = pd.read_csv('datasets.csv')
dataset_ids = datasets['id'].tolist()
save_info(dataset_ids)


def to_csv(dataset_id):
     # get dataset from OpenML repository.
    dataset = openml.datasets.get_dataset(dataset_id)

    # convert dataset to numpy.ndarray format.
    X, categorical_indicator, attribute_names = \
            dataset.get_data(return_categorical_indicator=True,
                             return_attribute_names=True)

    # convert dataset to pandas.DataFrame format.
    df = pd.DataFrame(X, columns=attribute_names)
    for is_category, name in zip(categorical_indicator, attribute_names):
        if not is_category:
            continue
        tokens = dataset.retrieve_class_labels(target_name=name)
        def convert(x):
            if np.isnan(x):
                return x
            else:
                return tokens[int(x)]
        df[name] = df[name].apply(convert)

    # rename target name to 'target'.
    df = df.rename(columns={dataset.default_target_attribute: 'target'})

    # save dataframe to csv file.
    df.to_csv(data_dir + dataset.name + '.csv', index=False)


succeed_dataset_ids = []

for i, dataset_id in enumerate(dataset_ids):
    try:
        to_csv(dataset_id)
    except Exception as e:
        print(e)

    succeed_dataset_ids.append(dataset_id)

    # display progress.
    progress = int((i + 1) / len(dataset_ids) * 100)
    print('{0:>3d}% finished.'.format(progress))

save_info(succeed_dataset_ids)
