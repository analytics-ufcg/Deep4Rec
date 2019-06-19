"""A dataset abstraction."""

from abc import ABC
from abc import abstractmethod
from enum import Enum
import logging
from typing import List

import tensorflow as tf

import deep4rec.utils as utils


class DatasetTask(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


class Dataset(ABC):
    """Dataset abstraction.

    This class is responsible for dowloading the data, preprocessing it,
    and serve it using the following attributes:
        - self.train_features, self.train_y
        - self.test_features, self.test_y
    """

    url = None

    def __init__(
        self,
        dataset_name,
        output_dir,
        verbose=True,
        task=DatasetTask.REGRESSION,
        uses_neg_sampling=False,
        *args,
        **kwargs
    ):
        self.dataset_name = dataset_name.replace("-neg", "")
        self.output_dir = output_dir
        self.verbose = verbose
        self.task = task
        self.uses_neg_sampling = uses_neg_sampling

    def _make_tf_dataset(
        self, features, target, shuffle=True, buffer_size=10000, batch_size=32
    ):
        """Make a TensorFlow dataset.

        Args:
            features: A list of numpy array containing features.
            target: A Numpy Array indicating the target.
            shuffle: A boolean indicating if the dataset should shuffled or not.
            buffer_size: An integer indicating the buffer size. Used only when shuffling.
                Default value is 1000.
            batch_size: Batch size. Default value is 32.

        Returns:
            A TensorFlow Dataset instance.
        """
        ds = tf.data.Dataset.from_tensor_slices((*features, target))
        if shuffle:
            ds = ds.shuffle(buffer_size=buffer_size)
        ds = ds.batch(batch_size)
        return ds

    def make_tf_dataset(
        self,
        data_partition: str,
        batch_size: int = 32,
        shuffle: bool = None,
        indexes: List[int] = None,
    ):
        """Make a TensorFlow dataset for a data partition.

        Args:
            data_partition: A string (train | test).
            batch_size: Batch size.
            shuffle: A boolean indicating if the dataset should be shuffled.
            indexes: A list of integer restricting which examples should be used
                to compose the dataset.

        Returns:
            A TensorFlow Dataset instance.

        Raises:
            ValueError if `data_partition` is unknown.
        """
        if data_partition == "train":
            if shuffle is None:
                shuffle = True

            if indexes is not None:
                features = [feature[indexes] for feature in self.train_features]
                targets = self.train_y[indexes]
            else:
                features = self.train_features
                targets = self.train_y

        elif data_partition == "validation":
            if shuffle is None:
                shuffle = False

            features = self.valid_features
            targets = self.valid_y

        elif data_partition == "test":
            if shuffle is None:
                shuffle = False

            if indexes is not None:
                features = [feature[indexes] for feature in self.test_features]
                targets = self.test_y[indexes]
            else:
                features = self.test_features
                targets = self.test_y
        else:
            raise ValueError("Unknown data partition {}".format(data_partition))

        return self._make_tf_dataset(
            features, targets, batch_size=batch_size, shuffle=shuffle
        )

    def download(self, url=None):
        if not url:
            url = self.url
        if self.verbose:
            logging.info(
                "Downloading {} at {}".format(self.dataset_name, self.output_dir)
            )
        utils.download(url, self.output_dir)

    def kfold_iterator(self, n_splits, test_size=0.1, random_state=0):
        """Segregate training dataset in `n_splits`-fold."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError

    @abstractmethod
    def check_downloaded(self):
        """Checks if the downloaded files already exist in `path`."""
        raise NotImplementedError

    @abstractmethod
    def check_preprocessed(self):
        """Checks if the expected preprocessed files exist in `path`."""
        raise NotImplementedError

    def maybe_download(self):
        if not self.check_downloaded():
            self.download()

    def maybe_preprocess(self):
        if not self.check_preprocessed():
            self.preprocess()

    def build(self):
        pass

    @property
    def train_features(self):
        return None

    @property
    def test_features(self):
        return None

    @property
    def valid_features(self):
        return None
