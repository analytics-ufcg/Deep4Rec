"""Dataset interface for MovieLens 20M dataset.

MovieLens 20M dataset: https://grouplens.org/datasets/movielens/20m/
"""

import os

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from deep4rec.datasets.dataset import Dataset
from deep4rec.datasets import movielens_helper as ml_helper
import deep4rec.utils as utils


class MovieLens20mDataset(Dataset):
    def __init__(self, dataset_name, output_dir, *args, **kwargs):
        super().__init__(dataset_name, output_dir, *args, **kwargs)

    def download(self):
        ml_helper.download(self.dataset_name, self.output_dir)

    def check_downloaded(self):
        return os.path.exists(
            os.path.join(
                self.output_dir, self.dataset_name, "{}.zip".format(self.dataset_name)
            )
        )

    def check_preprocessed(self):
        return False

    def preprocess(self):
        (self.train_wide_data, self.train_embedding_data, self.train_y), (
            self.test_wide_data,
            self.test_embedding_data,
            self.test_y,
        ) = ml_helper.construct_subdatasets(self.dataset_name, self.output_dir)

    @property
    def train_size(self):
        return len(self.train_wide_data)

    @property
    def train_features(self):
        return [self.train_embedding_data, self.train_wide_data]

    @property
    def test_features(self):
        return [self.test_embedding_data, self.test_wide_data]

    @property
    def num_features_one_hot(self):
        return ml_helper.NUM_USER_IDS[self.dataset_name] + ml_helper.NUM_ITEM_IDS

    @property
    def num_features(self):
        return 1
