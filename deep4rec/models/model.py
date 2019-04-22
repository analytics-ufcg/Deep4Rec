"""Simple Model abstraction."""
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from deep4rec.models.loss_functions import get_loss_fn
from deep4rec.models.metrics import get_metric
from deep4rec.models.optimizers import build_optimizer
from deep4rec import utils


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

    def train(self, ds, epochs, loss_function, optimizer="adam", verbose=True):
        loss_function = utils.name_to_fn(loss_function, get_loss_fn)
        optimizer = utils.name_to_fn(optimizer, build_optimizer)

        loss_plot = []
        for epoch in tqdm(range(epochs)):
            total_loss, count = 0, 0
            start = time.time()
            for (data, target) in ds:
                count += 1
                with tf.GradientTape() as tape:
                    pred_rating = self.call(data)
                    loss = loss_function(target, pred_rating)

                total_loss += loss.numpy()
                gradients = tape.gradient(loss, self.real_variables)
                optimizer.apply_gradients(
                    zip(gradients, self.real_variables),
                    tf.train.get_or_create_global_step(),
                )

            # storing the epoch end loss value to plot later
            loss_plot.append(total_loss / count)

            if verbose:
                print("Epoch {} Loss {:.6f}".format(epoch + 1, loss_plot[-1]))
                print("1 epoch = {} secs\n".format(time.time() - start))

    def eval(self, ds, loss_functions, metrics=None, verbose=True):
        if not metrics:
            metrics = []

        loss_functions = utils.names_to_fn(loss_functions, get_loss_fn)
        metrics = utils.names_to_fn(metrics, get_metric)

        loss_function_res = [0 for _ in range(len(loss_functions))]
        metrics_res = [0 for _ in range(len(metrics))]
        count_batches = 0
        start = time.time()
        for (data, target) in ds:
            count_batches += 1
            pred_rating = self.call(data)
            for (i, loss_function) in enumerate(loss_functions):
                loss_function_res[i] += loss_function(target, pred_rating).numpy()
            for (i, metric) in enumerate(metrics):
                metric_res[i] += metric(target, pred_rating).numpy()

        loss_function_res = np.array(loss_function_res) / count_batches
        metrics_res = np.array(metrics_res) / count_batches

        if verbose:
            print("Time to evaluate dataset = {} secs\n".format(time.time() - start))

        return loss_function_res, metrics_res

    def call(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def real_variables(self):
        return self.variables
