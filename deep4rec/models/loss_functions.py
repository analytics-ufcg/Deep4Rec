"""Loss functions.

Here are defined losses used for training (TensorFlow losses) and losses
used for evaluation.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, log_loss
import tensorflow as tf

from deep4rec import utils


# TensorFlow losses
def tf_mse(target, pred):
    target = tf.reshape(tf.cast(target, dtype=tf.float32), (-1, 1))
    return tf.losses.mean_squared_error(target, pred)


def tf_rmse(target, pred):
    return tf.sqrt(tf_mse(target, pred))


def tf_l2(target, pred):
    target = tf.reshape(tf.cast(target, dtype=tf.float32), (-1, 1))
    return tf.nn.l2_loss(target - pred)


def tf_binary_cross_entropy(target, pred):
    target = tf.reshape(tf.cast(target, dtype=tf.float32), (-1, 1))
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=pred)


# Evaluation losses
def mse(target, pred, bound_pred=True):
    if bound_pred:
        pred = np.clip(pred, a_min=min(pred), a_max=max(pred))
    return mean_squared_error(target, pred)


def rmse(target, pred, bound_pred=True):
    return np.sqrt(mse(target, pred, bound_pred=bound_pred))


def binary_cross_entropy(target, pred):
    assert len(target) == len(pred)
    prob_pred = [[1 - p, p] for p in utils.logits_to_prob(pred)]
    return log_loss(target, prob_pred)


def l2(target, pred, bound_pred=True):
    if bound_pred:
        pred = np.clip(pred, a_min=min(pred), a_max=max(pred))
    return sum((np.array(target) - np.array(pred)) ** 2)


tf_losses = {
    "binary_cross_entropy": tf_binary_cross_entropy,
    "mse": tf_mse,
    "rmse": tf_rmse,
    "l2": tf_l2,
}

eval_losses = {
    "mse": mse,
    "rmse": rmse,
    "binary_cross_entropy": binary_cross_entropy,
    "l2": l2,
}


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
