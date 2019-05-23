"""Dataset interface for MovieLens and Pinterest datasets pre-processed using negative sampling.

We use the data provided by:
Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017).
Neural Collaborative Filtering. In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

At https://github.com/hexiangnan/neural_collaborative_filtering.
"""

import os

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from deep4rec.datasets.dataset import Dataset
from deep4rec.datasets import movielens_helper as ml_helper
import deep4rec.utils as utils


class ClfNegativeSamplingDataset(Dataset):

    url = "https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/master/Data/"

    def __init__(self, dataset_name, output_dir, *args, **kwargs):
        super().__init__(
            dataset_name, output_dir, uses_neg_sampling=True, *args, **kwargs
        )

        self.train_url = os.path.join(
            self.url, "{}.train.rating".format(self.dataset_name)
        )
        self.test_url_rating = os.path.join(
            self.url, "{}.test.rating".format(self.dataset_name)
        )
        self.test_url_negative = os.path.join(
            self.url, "{}.test.negative".format(self.dataset_name)
        )

        self.train_file = os.path.join(
            self.output_dir, "{}.train.rating".format(self.dataset_name)
        )
        self.test_file_rating = os.path.join(
            self.output_dir, "{}.test.rating".format(self.dataset_name)
        )
        self.test_file_negative = os.path.join(
            self.output_dir, "{}.test.negative".format(self.dataset_name)
        )

    def download(self):
        super().download(self.train_url)
        super().download(self.test_url_rating)
        super().download(self.test_url_negative)

    def check_downloaded(self):
        return (
            os.path.exists(self.train_file)
            and os.path.exists(self.test_file_negative)
            and os.path.exists(self.test_file_rating)
        )

    def check_preprocessed(self):
        return False

    def _sample_negative_samples(self, num_negatives: int = 1):
        features, targets = [], []
        for (u, pos_item) in self.train_positive_features:
            features.append([u, pos_item])
            targets.append([1.0])

            for _ in range(num_negatives):
                neg_item = np.random.randint(self.num_items)
                while (u, neg_item) in self.train_map:
                    neg_item = np.random.randint(self.num_items)
                features.append([u, neg_item])
                targets.append([0.0])
        return np.array(features), np.array(targets)

    def _load_positive_train_features(self):
        train_features = []
        num_items, num_users = 0, 0
        train_map = {}
        with open(self.train_file, "r") as f:
            for line in f.readlines():
                if line != None and line != "":
                    user, item, rating, _ = list(map(int, line.split("\t")))
                    if rating:
                        train_features.append([user, item])
                        train_map[(user, item)] = 1
                    num_items = max(num_items, item)
                    num_users = max(num_users, user)
        return np.array(train_features), num_items, num_users, train_map

    def _load_test_features(self):
        def _extract_user(text):
            return int(text.replace("(", "").split(",")[0])

        test_features, test_targets = [], []

        # Negative examples
        with open(self.test_file_negative, "r") as f:
            for line in f.readlines():
                if line != None and line != "":
                    arr = line.split("\t")
                    user, negative_examples = (
                        _extract_user(arr[0]),
                        list(map(int, arr[1:])),
                    )
                    for item in negative_examples:
                        test_features.append([user, item])
                        test_targets.append([0.0])

        # Positive examples
        with open(self.test_file_rating, "r") as f:
            for line in f.readlines():
                if line != None and line != "":
                    user, item, _, _ = list(map(int, line.split("\t")))
                    test_features.append([user, item])
                    test_targets.append([1.0])

        return np.array(test_features), np.array(test_targets)

    def preprocess(self):
        self.train_positive_features, self.num_items, self.num_users, self.train_map = (
            self._load_positive_train_features()
        )
        self.test_data, self.test_y = self._load_test_features()

    @property
    def train_features(self):
        train_features, targets = self._sample_negative_samples()
        self.train_y = targets
        return [train_features]

    @property
    def test_features(self):
        return [self.test_data]

    @property
    def num_features_one_hot(self):
        return self.num_users + self.num_items

    @property
    def num_features(self):
        return 1
