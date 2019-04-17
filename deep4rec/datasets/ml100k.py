"""Dataset interface for MovieLens 100k dataset.

MovieLens 100k dataset: https://grouplens.org/datasets/movielens/100k/
"""

import os

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from deep4rec.datasets.dataset import Dataset
import deep4rec.utils as utils


class MovieLens100kDataset(Dataset):

  url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'

  def __init__(self, dataset_name, output_dir, *args, **kwargs):
    super(MovieLens100kDataset, self).__init__(
        dataset_name, output_dir, *args, **kwargs)
    self.zip_path = os.path.join(
        self.output_dir, '{}.zip'.format(self.dataset_name))
    self.preprocessed_path = os.path.join(
        self.output_dir, self.dataset_name)
    self._ord_encoder = OrdinalEncoder()

  def download(self):
    super(MovieLens100kDataset, self).download()
    utils.maybe_unzip(self.zip_path)

  def check_downloaded(self):
    return os.path.exists(self.zip_path)

  def check_preprocessed(self):
    return False

  def preprocess(self):
    utils.maybe_unzip(self.zip_path)
    self.train_data, self.train_y, self.train_users, self.train_items = self._load_data(
      'ua.base', train_data=True)
    self.test_data, self.test_y, self.test_users, self.test_items = self._load_data(
      'ua.test')

  def _load_data(self, filename, train_data=True):
    data, y = [], []
    users, items = set(), set()
    filepath = os.path.join(self.preprocessed_path, filename)
    with open(filepath) as f:
      for line in f:
        (user_id, movie_id, rating, _) = line.split('\t')
        data.append([user_id, movie_id])
        y.append(float(rating))
        users.add(user_id)
        items.add(movie_id)

    if train_data:
      vect_data = self._ord_encoder.fit_transform(data)
    else:
      vect_data = self._ord_encoder.transform(data)

    return (vect_data, np.array(y), users, items)

  @property
  def train(self):
    return (self.train_data, self.train_y, self.train_users, self.train_items)

  @property
  def users(self):
    return self.train_users

  @property
  def items(self):
    return self.train_items

  @property
  def num_features_one_hot(self):
    return len(self.users) + len(self.items)

  @property
  def num_features(self):
    return 2

  @property
  def test(self):
    return (self.test_data, self.test_y, self.test_users, self.test_items)