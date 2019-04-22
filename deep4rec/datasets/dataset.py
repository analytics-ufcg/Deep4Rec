"""A dataset abstraction."""

from abc import ABC
from abc import abstractmethod
import logging

import tensorflow as tf

import deep4rec.utils as utils


class Dataset(ABC):
    """Dataset interface."""

    url = None

    def __init__(self, dataset_name, output_dir, verbose=False, *args, **kwargs):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.verbose = verbose

    def _make_tf_dataset(
        self, data, target, shuffle=True, buffer_size=1000, batch_size=32
    ):
        """Make a TensorFlow dataset.

        Args:
            data: A numpy array containing features.
            target: A Numpy Array indicating the target.
            shuffle: A boolean indicating if the dataset should shuffled or not.
            buffer_size: An integer indicating the buffer size. Used only when shuffling.
                Default value is 1000.
            batch_size: Batch size. Default value is 32.

        Returns:
            A TensorFlow Dataset instance.
        """
        ds = tf.data.Dataset.from_tensor_slices((data, target))
        if shuffle:
            ds = ds.shuffle(buffer_size=buffer_size)
        ds = ds.batch(batch_size)
        return ds

    def make_tf_dataset(self, data_partition, batch_size=32):
        """Make a TensorFlow dataset for a data partition.

        Args:
            data_partition: A string (train | valid | test)
            batch_size: Batch size.

        Returns:
            A TensorFlow Dataset instance.
        """
        if data_partition == "train":
            return self._make_tf_dataset(
                self.train_data, self.train_y, batch_size=batch_size
            )
        elif data_partition == "test":
            return self._make_tf_dataset(
                self.test_data, self.test_y, batch_size=batch_size
            )

    def download(self):
        if self.verbose:
            logging.info(
                "Downloading {} at {}".format(self.dataset_name, self.output_dir)
            )
        utils.download(self.url, self.output_dir)

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
