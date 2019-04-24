"""Loss functions.

Here are defined losses used for training (TensorFlow losses) and losses
used for evaluation.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf


# TensorFlow losses
def tf_mse(target, pred):
    target = tf.reshape(tf.to_float(target), (-1, 1))
    return tf.losses.mean_squared_error(target, pred)


def tf_rmse(target, pred):
    return tf.sqrt(tf_mse(target, pred))


def tf_l2(target, pred):
    target = tf.reshape(tf.to_float(target), (-1, 1))
    return tf.nn.l2_loss(target - pred)


# Evaluation losses
def rmse(target, pred, bound_pred=True):
    if bound_pred:
        pred = np.clip(pred, a_min=min(pred), a_max=max(pred))
    return np.sqrt(mean_squared_error(target, pred))


tf_losses = {"mse": tf_mse, "rmse": tf_rmse, "l2": tf_l2}
eval_losses = {"rmse": rmse}


def get_tf_loss_fn(loss_fn_name):
    loss_fn_name = loss_fn_name.lower()
    if loss_fn_name not in tf_losses:
        raise ValueError("Unknown loss function {}".format(loss_fn_name))
    else:
        return tf_losses[loss_fn_name]


def get_eval_loss_fn(loss_fn_name):
    loss_fn_name = loss_fn_name.lower()
    if loss_fn_name not in eval_losses:
        raise ValueError("Unknown loss function {}".format(loss_fn_name))
    else:
        return eval_losses[loss_fn_name]
