import pytest

from unittest.mock import MagicMock

import tensorflow as tf


@pytest.fixture()
def fake_dataset():
    """A fake dataset used only for testing."""
    fake_ds = MagicMock()
    fake_ds.num_features_one_hot = 3
    fake_ds.num_features = 2
    return fake_ds


@pytest.fixture()
def fake_ds_iterator():
    """A fake dataset iterator used only for testing."""
    fake_iterator = MagicMock()
    fake_iterator.__iter__.return_value = [(tf.to_float([[1, 2]]), tf.to_float([[1]]))]
    return fake_iterator
