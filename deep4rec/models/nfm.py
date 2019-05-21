"""
Implementation of Neural Factorization Machines.

Paper:  Neural Factorization Machines for Sparse Predictive Analytics.
In Proceedings of SIGIR '17, Shinjuku, Tokyo, Japan, August 07-11, 2017.

Link: http://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-nfm.pdf

Authors: Xiangnan He and Tat-Seng Chua (2017)
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from deep4rec.models.model import Model


class NeuralFM(Model):
    def __init__(
        self,
        ds,
        num_units=64,
        layers=None,
        drop_prob=None,
        apply_batchnorm=True,
        activation_fn="relu",
        apply_dropout=True,
        l2_regularizer=0.0,
    ):
        super(NeuralFM, self).__init__()
        self._num_weights = ds.num_features_one_hot
        self._num_units = num_units
        self._num_features = ds.num_features

        if layers and drop_prob and apply_dropout:
            assert len(layers) + 1 == len(drop_prob)

        if layers is None:
            layers = [64]

        if drop_prob is None:
            drop_prob = [0.8, 0.5]
        self.drop_prob = drop_prob

        self.apply_batchnorm = apply_batchnorm
        self.apply_dropout = apply_dropout
        self.activation = activation_fn
        self.dense_layers = [
            tf.keras.layers.Dense(units, activation=self.activation) for units in layers
        ]
        self.final_dense_layer = tf.keras.layers.Dense(1)

        if self.apply_batchnorm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()
            self.dense_batch_norm = [
                tf.keras.layers.BatchNormalization() for _ in layers
            ]

        if self.apply_dropout:
            self.fm_dropout = tf.keras.layers.Dropout(self.drop_prob[-1])
            self.dense_dropout = [
                tf.keras.layers.Dropout(self.drop_prob[i])
                for i in range(len(drop_prob) - 1)
            ]

        self.w = tf.keras.layers.Embedding(
            self._num_weights,
            num_units,
            input_length=self._num_features,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.01
            ),
            embeddings_regularizer=tf.keras.regularizers.l2(l2_regularizer),
        )
        self.w0 = tf.keras.layers.Embedding(
            self._num_weights,
            1,
            input_length=self._num_features,
            embeddings_initializer="zeros",
        )
        self.bias = tfe.Variable(tf.constant(0.0))

    def call(self, one_hot_features, training=False, features=None, **kwargs):
        """
        Args:
            one_hot_features: A dense tensor of shape [batch_size, self._num_features]
                that indicates which features are present in this input.
            training: A boolean indicating if is training or not.
            features: A dense tensor of shape [batch_size, self._num_features] that indicates
                the value of each feature.

        Returns:
            Logits.
        """
        # TODO: add support to other features.

        # FM
        weights = self.w(one_hot_features)  # [batch_size, num_features, num_units]

        sum_nzw = tf.reduce_sum(weights, 1)  # [batch_size, num_units]
        squared_sum_nzw = tf.square(sum_nzw)  # [batch_size, num_units]

        squared_nzw = tf.square(weights)  # [batch_size, num_features, num_units]
        sum_squared_nzw = tf.reduce_sum(squared_nzw, 1)  # [batch_size, num_units]

        fm = 0.5 * (squared_sum_nzw - sum_squared_nzw)

        if self.apply_batchnorm:
            fm = self.batch_norm_layer(fm, training=training)

        if self.apply_dropout:
            fm = self.fm_dropout(fm, training=training)

        # Dense layers on top of FM
        for i, layer in enumerate(self.dense_layers):
            fm = layer(fm)
            if self.apply_batchnorm:
                fm = self.dense_batch_norm[i](fm)
            if self.apply_dropout:
                fm = self.dense_dropout[i](fm)

        # Aggregate
        fm = self.final_dense_layer(fm)  # [batch_size, 1]
        bilinear = tf.reduce_sum(fm, 1, keep_dims=True)  # [batch_size, 1]
        weight_bias = tf.reduce_sum(self.w0(one_hot_features), 1)  # [batch_size, 1]
        logits = tf.add_n([bilinear, weight_bias]) + self.bias

        return logits
