import pytest

from deep4rec import models


def test_fm_train(fake_dataset, fake_ds_iterator):
    model = models.FM(fake_dataset)
    model.train(fake_ds_iterator, epochs=1, loss_function="mse", optimizer="adam")
