import pytest

from deep4rec import datasets


def test_ml100k():
    ds = datasets.build_dataset("ml-100k")

    # Assert number of users and items
    assert 943 == len(ds.users)
    assert 1680 == len(ds.items)
    assert 943 + 1680 == ds.num_features_one_hot

    # Assert train and test size
    assert len(ds.train_data) == 90570
    assert len(ds.test_data) == 9428
