"""
Implementation of Factorization Machines.

Paper: Factorization Machines
link: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
Author: Steffen Rendle, Osaka University
"""

import tensorflow as tf

from deep4rec.models.model import Model


class Wide(Model):
    def __init__(self):
        super(Wide, self).__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, wide_data):
        return self.dense(wide_data)


class Deep(Model):
    def __init__(self, hidden_units=None, activation="relu"):
        super(Deep, self).__init__()
        self.hidden_units = hidden_units if hidden_units else [256, 128, 64]
        self.dense_layers = [
            tf.keras.layers.Dense(h, activation=activation) for h in self.hidden_units
        ]

    def call(self, dense_data):
        for layer in self.dense_layers:
            dense_data = layer(dense_data)
        return dense_data


class WideDeep(Model):
    def __init__(self, ds, num_units=8, deep_model=None, wide_model=None):
        super(WideDeep, self).__init__()
        self.deep_model = (
            deep_model if deep_model else Deep(hidden_units=[100, 75, 50, 25])
        )
        self.wide_model = wide_model if wide_model else Wide()

        self._num_weights = ds.num_features_one_hot
        self._num_units = num_units
        self._num_features = ds.num_features

        self.embedding = tf.keras.layers.Embedding(
            self._num_weights,
            num_units,
            input_length=self._num_features,
            embeddings_regularizer=tf.keras.regularizers.l2(),
        )
        self.flat = tf.keras.layers.Flatten()
        self.last_layer = tf.keras.layers.Dense(1)

    def call(
        self,
        one_hot_features=None,
        wide_features=None,
        dense_features=None,
        training: bool = True,
    ):
        """
        Args:
            one_hot_features: A dense tensor of shape [batch_size, self._num_features]
                that indicates which features should be embedded.
            wide_features: A dense tensor of shape [batch_size, None] with wide
                features. If None is not used. Which means there's no wide model.
            dense_features: A dense tensor of shape [batch_size, None] with other
                dense features. If None is not used.
            training: A boolean indicating if is training or not.
        Returns:
            Logits.
        """
        if one_hot_features is not None:
            embeddings = self.embedding(one_hot_features)
            embeddings = tf.cast(self.flat(embeddings), dtype=tf.float32)

        if dense_features is not None:
            dense_features = tf.cast(dense_features, dtype=tf.float32)

        if one_hot_features is not None and dense_features is not None:
            dense_features = tf.keras.layers.concatenate([embeddings, dense_features])
        elif one_hot_features is not None:
            dense_features = embeddings

        if dense_features is not None and wide_features is not None:
            wide_features = tf.cast(wide_features, dtype=tf.float32)
            logits = tf.keras.layers.concatenate(
                [self.deep_model(dense_features), self.wide_model(wide_features)]
            )
        elif dense_features is not None:
            logits = self.deep_model(dense_features)
        elif wide_features is not None:
            wide_features = tf.cast(wide_features, dtype=tf.float32)
            logits = self.wide_model(wide_features)

        logits = self.last_layer(logits)
        return logits

    @property
    def real_variables(self):
        return self.variables + self.deep_model.variables + self.wide_model.variables
