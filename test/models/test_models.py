from math import inf
from parameterized import parameterized
import pytest
from unittest.mock import MagicMock

import tensorflow as tf

from deep4rec import models
from test import custom_asserts


class TestModels(custom_asserts.CustomAssertions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ds = self._create_fake_dataset()

    def _create_fake_dataset(self):
        """A fake dataset used only for testing."""

        def _create_fake_data_iterator(data, target):
            fake_data_iterator = MagicMock()
            fake_data_iterator.__iter__.return_value = [(data, target)]
            return fake_data_iterator

        def _create_data(data, *args, **kwargs):
            if data == "train":
                return _create_fake_data_iterator(
                    tf.to_float([[0, 2], [1, 2], [1, 3]]), tf.to_float([[1], [1], [-1]])
                )
            elif data == "test":
                return _create_fake_data_iterator(
                    tf.to_float([[0, 3]]), tf.to_float([[1]])
                )

        fake_ds = MagicMock()
        fake_ds.num_features_one_hot = 4
        fake_ds.num_features = 2

        fake_ds.make_tf_dataset = MagicMock(side_effect=_create_data)
        return fake_ds

    @parameterized.expand([[models.FM], [models.NeuralFM], [models.WideDeep]])
    def test_train_regression(self, model_class):
        model = model_class(self.ds, apply_batchnorm=False)

        # force weights creation
        # TODO: find a way to do this without training
        model.train(
            self.ds, epochs=1, loss_function="mse", optimizer="adam", verbose=False
        )
        weights_after_1_epochs = model.get_weights()

        # train model
        model.train(
            self.ds, epochs=5, loss_function="mse", optimizer="adam", verbose=False
        )
        weights_after_5_epochs = model.get_weights()

        # check if loss decreased
        self.assertTestLossDecreases(model, "mse")

        # check if all weights are being updated
        self.assertModelWeightsChanged(weights_after_1_epochs, weights_after_5_epochs)

    def test_wide_deep_weights(self):
        model = models.WideDeep(self.ds)

        # force weights creation
        # TODO: find a way to do this without training
        model.train(
            self.ds, epochs=1, loss_function="mse", optimizer="adam", verbose=False
        )

        layers = set()
        for layer in model.layers:
            layers.add(layer.name)
            if hasattr(layer, "layers"):
                for inner_layer in layer.layers:
                    layers.add(inner_layer.name)

        assert "deep_dense_0" in layers
        assert "deep_dense_1" in layers
        assert "wide" in layers
