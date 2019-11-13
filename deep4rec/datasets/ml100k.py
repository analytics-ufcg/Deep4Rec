"""Dataset interface for MovieLens 100k dataset.

MovieLens 100k dataset: https://grouplens.org/datasets/movielens/100k/
"""

import os

import numpy as np

import random as rd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import ShuffleSplit

from deep4rec.datasets.dataset import Dataset
from deep4rec.datasets.dataset import DatasetTask
import deep4rec.utils as utils


class MovieLens100kDataset(Dataset):

    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

    def __init__(self, dataset_name, output_dir, *args, **kwargs):
        super(MovieLens100kDataset, self).__init__(
            dataset_name, output_dir, *args, **kwargs
        )
        self.zip_path = os.path.join(
            self.output_dir, "{}.zip".format(self.dataset_name)
        )
        self.preprocessed_path = os.path.join(self.output_dir, self.dataset_name)

        # Used to map users and items to indexes
        self._counter = 0
        self.user_index = {}
        self.item_index = {}
        self.index_user_id = {}
        self.index_item_id = {}

        # Stores users -> items in train data
        self.users_items = {}
        self.users_id_items_id = {}
        # Store `users_items` index in train data
        self._ui_index = {}

    def download(self):
        super(MovieLens100kDataset, self).download()
        utils.maybe_uncompress(self.zip_path)

    def check_downloaded(self):
        return os.path.exists(self.zip_path)

    def check_preprocessed(self):
        return False

    def preprocess(self):
        utils.maybe_uncompress(self.zip_path)
        (
            self.train_data,
            self.train_y,
            self.train_users,
            self.train_items,
        ) = self._load_data("ua.base", is_train=True)
        self.test_data, self.test_y, self.test_users, self.test_items = self._load_data(
            "ua.test", is_train=False
        )

    def _preprocess_target(self, target, th: int = 3):
        if self.task == DatasetTask.REGRESSION:
            return float(target)
        elif self.task == DatasetTask.CLASSIFICATION:
            return target >= th
        else:
            raise NotImplementedError(
                "{} does not support {} task.".format(self.dataset_name, self.task)
            )

    def _load_data(self, filename, is_train):
        data, y = [], []
        users, items = set(), set()
        filepath = os.path.join(self.preprocessed_path, filename)
        with open(filepath) as f:
            for line in f:
                (user_id, movie_id, rating, _) = line.split("\t")
                # Ignore items and users that are not in train_data
                if not is_train and (
                    user_id not in self.train_users or movie_id not in self.train_items
                ):
                    continue

                if user_id not in self.user_index:
                    self.user_index[user_id] = self._counter
                    self.index_user_id[self._counter] = int(user_id)
                    self._counter += 1

                if movie_id not in self.item_index:
                    self.item_index[movie_id] = self._counter
                    self.index_item_id[self._counter] = int(movie_id)
                    self._counter += 1

                if not int(user_id) in self.users_id_items_id:
                    self.users_id_items_id[int(user_id)] = set()

                data.append([self.user_index[user_id], self.item_index[movie_id]])
                y.append(self._preprocess_target(rating))
                users.add(user_id)
                items.add(movie_id)
                self.users_id_items_id[int(user_id)].add(int(movie_id))

        if is_train:
            self._store_users_items(data)

        return (np.array(data), np.array(y), users, items)

    def _store_users_items(self, vect_data):
        for i, (user, item) in enumerate(vect_data):
            if user not in self.users_items:
                self.users_items[user] = set()
            self.users_items[user].add(item)
            self._ui_index[(user, item)] = i

    def kfold_iterator(self, n_splits, test_size=0.1, random_state=0):
        rs = ShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )
        train_splits = [[] for _ in range(n_splits)]
        test_splits = [[] for _ in range(n_splits)]
        for user in self.users_items:
            items = list(self.users_items[user])
            for i, (train_indexes, test_indexes) in enumerate(rs.split(items)):
                train_splits[i].extend(
                    [self._ui_index[(user, items[j])] for j in train_indexes]
                )
                test_splits[i].extend(
                    [self._ui_index[(user, items[j])] for j in test_indexes]
                )

        for train_index, test_index in zip(train_splits, test_splits):
            yield train_index, test_index

    def build_graph(self):
        graph = np.zeros((self.num_users, self.num_items), dtype=np.float32)

        for (user, items) in self.users_id_items_id.items():
            for item in items:
                graph[user, item] = 1

        return graph

    def sample_pos_neg_items(self, users):
        def sample_pos_items_for_u(u, num):
            pos_items = self.users_id_items_id[u]
            if len(pos_items) >= num:
                return rd.sample(pos_items, num)
            else:
                return [rd.choice(pos_items) for _ in range(num)]

        def sample_neg_items_for_u(u, num):
            neg_items = list(
                set(range(self.num_items)) - set(self.users_id_items_id[u])
            )
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    @property
    def train_features(self):
        return [self.train_data]

    @property
    def test_features(self):
        return [self.test_data]

    @property
    def train_size(self):
        return len(self.train_data)

    @property
    def users(self):
        return self.train_users

    @property
    def items(self):
        return self.train_items

    @property
    def num_items(self):
        return 1683

    @property
    def num_users(self):
        return 944

    @property
    def num_features_one_hot(self):
        return len(self.users) + len(self.items)

    @property
    def num_features(self):
        return 2
