import unittest

import numpy as np


class CustomAssertions(unittest.TestCase):
    def assertArrayIsSorted(self, array, reverse=False):
        if not array:
            raise ValueError("Array is empty.")
        sorted_array = sorted(array, reverse=reverse)
        self.assertSequenceEqual(array, sorted_array)

    def assertTestLossDecreases(self, model, loss_name):
        self.assertArrayIsSorted(
            [loss[loss_name] for loss in model.test_losses], reverse=True
        )

    def assertModelWeightsChanged(self, weights, new_weights):
        for weight, new_weight in zip(weights, new_weights):
            assert weight.shape == new_weight.shape
            assert not np.array_equal(weight, new_weight)
