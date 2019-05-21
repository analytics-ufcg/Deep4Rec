"""
Implementation of Factorization Machines.

Paper: Factorization Machines
link: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
Authors: Steffen Rendle, Osaka University
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from deep4rec.models.model import Model


class FM(Model):
    def __init__(
        self,
        ds,
        num_units=64,
        apply_batchnorm=True,
        apply_dropout=True,
        dropout_prob=0.5,
        l2_regularizer=0.0,
    ):
        super(FM, self).__init__()

        # total number of features = N weights
        self._num_weights = ds.num_features_one_hot
        # a weight is a vector of size `num_units`
        self._num_units = num_units
        # number of features used
        self._num_features = ds.num_features

        # define weights and biases
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

        self.apply_batchnorm = apply_batchnorm
        if self.apply_batchnorm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()

        self.apply_dropout = apply_dropout
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(dropout_prob)

    def call(self, one_hot_features, training, features=None):
        """Forward pass.

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

        weights = self.w(one_hot_features)  # [batch_size, num_features, num_units]

        sum_nzw = tf.reduce_sum(weights, 1)  # [batch_size, num_units]
        squared_sum_nzw = tf.square(sum_nzw)  # [batch_size, num_units]

        squared_nzw = tf.square(weights)  # [batch_size, num_features, num_units]
        sum_squared_nzw = tf.reduce_sum(squared_nzw, 1)  # [batch_size, num_units]

        fm = 0.5 * (squared_sum_nzw - sum_squared_nzw)

        if self.apply_batchnorm:
            fm = self.batch_norm_layer(fm, training=training)

        if self.apply_dropout:
            fm = self.dropout(fm, training=training)

        bilinear = tf.reduce_sum(fm, 1, keep_dims=True)  # [batch_size, 1]
        weight_bias = tf.reduce_sum(self.w0(one_hot_features), 1)  # [batch_size, 1]
        logits = tf.add_n([bilinear, weight_bias]) + self.bias

        return logits
