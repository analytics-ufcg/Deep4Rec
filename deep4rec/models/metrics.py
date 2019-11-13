"""Metrics definition."""

import numpy as np
from sklearn import metrics as sk_metrics
import tensorflow as tf

from deep4rec import utils


def accuracy(real, pred):
    pred = utils.logits_to_class(pred)
    return sk_metrics.accuracy_score(real, pred)


def auc(real, pred):
    pred = utils.logits_to_prob(pred)
    return sk_metrics.roc_auc_score(real, pred)


def recall(real, pred):
    pred = utils.logits_to_class(pred)
    return sk_metrics.recall_score(real, pred, average='micro')


def precision(real, pred):
    pred = utils.logits_to_class(pred)
    return sk_metrics.precision_score(real, pred, average='micro')


def auc_precision_recall(real, pred):
    pred = utils.logits_to_prob(pred)
    return sk_metrics.average_precision_score(real, pred)


metrics = {
    "acc": accuracy,
    "accuracy": accuracy,
    "auc": auc,
    "auc_precision_recall": auc_precision_recall,
    "precision": precision,
    "recall": recall,
}


def get_metric(metric_name):
    metric_name = metric_name.lower()
    if metric_name not in metrics:
        raise ValueError("Unknown metric {}".format(metric_name))
    else:
        return metrics[metric_name]
