"""TensorFlow loss functions."""

import tensorflow as tf


def mse(real, pred):
    real = tf.reshape(tf.to_float(real), (-1, 1))
    return tf.losses.mean_squared_error(real, pred)


def l2(real, pred):
    real = tf.reshape(tf.to_float(real), (-1, 1))
    return tf.nn.l2_loss(real - pred)


losses = {"mse": mse, "l2": l2}


def get_loss_fn(loss_fn_name):
    loss_fn_name = loss_fn_name.lower()
    if loss_fn_name not in losses:
        raise ValueError("Unknown loss function {}".format(loss_fn_name))
    else:
        return losses[loss_fn_name]
