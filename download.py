# coding: utf-8

import json
import os

import numpy as np
import openml
import pandas as pd
import yaml


def make_directory(d):
    d = os.path.expanduser(d)
    if d[-1] != os.sep:
        d = d + os.sep
    os.makedirs(d, exist_ok=True)
    return d


class Downloader(object):

    def __init__(self, api_key, cache_dir, save_dir):
        self.api_key = api_key
        self.cache_dir = make_directory(cache_dir)
        self.save_dir = make_directory(save_dir)
        self.info = openml.datasets.list_datasets()
        openml.config.apikey = self.api_key
        openml.config.set_cache_directory(self.cache_dir)

    def get_all(self, dataset_ids):
        saved_dataset_ids = []
        anomaly_labels = []
        feature_types = []
        contains_missings = []
        ml_types = []

        for i, dataset_id in enumerate(dataset_ids):
            try:
                anomaly_label, feature_type, contains_missing, ml_type =\
                    self.get_dataset(dataset_id)
                anomaly_labels.append(anomaly_label)
                feature_types.append(feature_type)
                contains_missings.append(contains_missing)
                ml_types.append(ml_type)
                saved_dataset_ids.append(dataset_id)

                try:
                    self.get_metadata(saved_dataset_ids, anomaly_labels,
                                      feature_types, contains_missings,
                                      ml_types)
                except Exception as e:
                    anomaly_labels = anomaly_labels[:-1]
                    feature_types = feature_types[:-1]
                    contains_missings = contains_missings[:-1]
                    ml_types = ml_types[:-1]
                    saved_dataset_ids = saved_dataset_ids[:-1]
            except Exception as e:
                print('[Exception] id=', dataset_id, e)
            print('{0:>3d}% finished.'.format(
                  int(100 * (i + 1) / len(dataset_ids))))

        self.get_metadata(saved_dataset_ids, anomaly_labels,
                          feature_types, contains_missings, ml_types)

    def get_dataset(self, dataset_id):
        # Set save directory.
        data_dir = self.save_dir + 'data' + os.sep

        # Get dataset from OpenML repository.
        dataset = openml.datasets.get_dataset(dataset_id)

        X, categorical_indicator, attribute_names = \
            dataset.get_data(return_categorical_indicator=True,
                             return_attribute_names=True)
        target_name = dataset.default_target_attribute

        contains_missing_value = np.isnan(X).any()

        # Save dataset to csv file.
        X = X.astype(str)
        anomaly_label = 'none'

        for i, (is_category, name) in enumerate(zip(categorical_indicator,
                                                    attribute_names)):
            if not is_category:
                # In case of float attribute, pass
                continue

            tokens = dataset.retrieve_class_labels(target_name=name)
            for token_id, token in enumerate(tokens):
                X[:, i][np.where(X[:, i] == str(float(token_id)))] = token

            if name == target_name:
                # In case of classification dataset, get anomaly label.
                values, counts = np.unique(X[:, i], return_counts=True)
                anomaly_label = values[np.argmin(counts)]

        df = pd.DataFrame(X, columns=attribute_names)
        df = df.rename(columns={target_name: 'target'})
        df.to_csv(data_dir + dataset.name + '.csv', index=False)

        # Save columns to json file.
        column_hints = []
        num_category = 0
        num_float = 0
        ml_type = ''
        for is_category, name in zip(categorical_indicator, attribute_names):
            if name == target_name:
                name = 'target'
                ml_type = 'classification' if is_category else 'regression'
            else:
                if is_category:
                    num_category += 1
                else:
                    num_float += 1
            hint = {
                'name': name,
                'column_type': 'category' if is_category else 'float'
            }
            column_hints.append(hint)

        if num_category == 0:
            feature_type = 'float'
        elif num_float == 0:
            feature_type = 'category'
        else:
            feature_type = 'mix'

        with open(data_dir + dataset.name + '_columns.csv', 'w') as f:
            json.dump(column_hints, f, indent=4)

        return anomaly_label, feature_type, contains_missing_value, ml_type

    def get_metadata(self, dataset_ids, anomaly_labels,
                     feature_types, contains_missings, ml_types):

        import copy
        info = copy.deepcopy(self.info)
        info = pd.DataFrame.from_dict(info, orient='index')
        info = info.loc[info['did'].isin(dataset_ids)]
        info['AnomalyLabel'] = anomaly_labels
        info['FeatureType'] = feature_types
        info['ContainsMissingValues'] = contains_missings
        info['MLType'] = ml_types
        info.to_csv(self.save_dir + 'info.csv', index=False)


if __name__ == '__main__':
    datasets = pd.read_csv('datasets.csv')
    dataset_ids = datasets['id'].tolist()
    with open('openml.yaml') as f:
        config = yaml.load(f)
    downloader = Downloader(config['api_key'],
                            config['cache_dir'],
                            config['save_dir'])
    downloader.get_all(dataset_ids)
