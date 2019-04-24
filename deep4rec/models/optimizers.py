"""TensorFlow optimizers."""

import tensorflow as tf

optimizers = {"adam": tf.train.AdamOptimizer, "adagrad": tf.train.AdagradOptimizer}


def build_optimizer(optimizer_name, *args, **kwargs):
    optimizer_name = optimizer_name.lower()
    if optimizer_name not in optimizers:
        raise ValueError("Unknown optimizer {}".format(optimizer_name))
    else:
        return optimizers[optimizer_name](*args, **kwargs)
