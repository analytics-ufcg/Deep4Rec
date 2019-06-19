"""
Implementation of Neural Matrix Factorization (NeuMF) recommender model.

Paper: He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.
Link: https://dl.acm.org/citation.cfm?id=3052569

Authors: Xiangnan He et al.
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from deep4rec.models.model import Model


class NeuralMF(Model):
    def __init__(
        self,
        ds,
        num_units=64,
        layers=None,
        dropout_prob=None,
        apply_batchnorm=True,
        activation_fn="relu",
        apply_dropout=True,
        l2_regularizer=0.0,
    ):
        super(NeuralMF, self).__init__()
        self._num_weights = ds.num_features_one_hot
        self._num_units = num_units
        self._num_features = ds.num_features

        if layers and dropout_prob and apply_dropout:
            assert len(layers) + 1 == len(dropout_prob)

        if layers is None:
            layers = [64]

        if dropout_prob is None:
            dropout_prob = [0.8, 0.5]

        self.dropout_prob = dropout_prob

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
            self.fm_dropout = tf.keras.layers.Dropout(self.dropout_prob[-1])
            self.dense_dropout = [
                tf.keras.layers.Dropout(self.dropout_prob[i])
                for i in range(len(dropout_prob) - 1)
            ]

        self.mf_embedding = tf.keras.layers.Embedding(
            self._num_weights,
            num_units,
            input_length=self._num_features,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.01
            ),
            embeddings_regularizer=tf.keras.regularizers.l2(l2_regularizer),
        )
        self.mlp_embedding = tf.keras.layers.Embedding(
            self._num_weights,
            num_units,
            input_length=self._num_features,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.01
            ),
            embeddings_regularizer=tf.keras.regularizers.l2(l2_regularizer),
        )

        self.flatten = tf.keras.layers.Flatten()

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
        # Matrix Factorization
        mf_latent = self.mf_embedding(
            one_hot_features
        )  # [batch_size, num_features, num_units]
        mf_latent = tf.math.reduce_prod(mf_latent, axis=1)  # [batch_size, num_units]

        # MLP
        mlp_latent = self.flatten(
            self.mlp_embedding(one_hot_features)
        )  # [batch_size, num_features * num_units]
        for i, layer in enumerate(self.dense_layers):
            mlp_latent = layer(mlp_latent)
            if self.apply_batchnorm:
                mlp_latent = self.dense_batch_norm[i](mlp_latent, training=training)
            if self.apply_dropout:
                mlp_latent = self.dense_dropout[i](mlp_latent, training=training)

        # Concatenate MF and MLP
        logits = tf.keras.layers.concatenate([mf_latent, mlp_latent])
        logits = self.final_dense_layer(logits)
        return logits
