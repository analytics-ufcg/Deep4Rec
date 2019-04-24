"""Metrics definition."""

import tensorflow as tf


def accuracy(real, pred):
    assert len(real) == len(pred)
    argmax_pred = tf.argmax(pred).numpy()
    return np.sum(argmax_pred == pred) / len(real)


metrics = {"acc": accuracy, "accuracy": accuracy}


def get_metric(metric_name):
    metric_name = metric_name.lower()
    if metric_name not in metrics:
        raise ValueError("Unknown metric {}".format(metric_name))
    else:
        return losses[metric_name]
