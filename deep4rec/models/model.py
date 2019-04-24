"""Simple Model abstraction."""
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from deep4rec.models.loss_functions import get_tf_loss_fn
from deep4rec.models.loss_functions import get_eval_loss_fn
from deep4rec.models.metrics import get_metric
from deep4rec.models.optimizers import build_optimizer
from deep4rec import utils


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

    def train(
        self,
        ds,
        epochs,
        loss_function,
        batch_size=128,
        optimizer="adam",
        run_eval=True,
        verbose=True,
        eval_loss_functions=None,
    ):
        if eval_loss_functions is None:
            eval_loss_functions = ["rmse"]

        train_ds = ds.make_tf_dataset("train", batch_size=batch_size)
        test_ds = ds.make_tf_dataset("test", batch_size=batch_size)

        loss_function = utils.name_to_fn(loss_function, get_tf_loss_fn)
        optimizer = utils.name_to_fn(optimizer, build_optimizer)

        for epoch in tqdm(range(epochs)):
            start = time.time()
            for (data, target) in train_ds:
                with tf.GradientTape() as tape:
                    pred_rating = self.call(data, training=True)
                    loss = loss_function(target, pred_rating)
                gradients = tape.gradient(loss, self.real_variables)
                optimizer.apply_gradients(
                    zip(gradients, self.real_variables),
                    tf.train.get_or_create_global_step(),
                )

            if verbose:
                train_losses, _ = self.eval(
                    train_ds, loss_functions=eval_loss_functions
                )
                print(
                    "Epoch {}, Losses {}, Time: {:2f} (s)".format(
                        epoch + 1, train_losses, time.time() - start
                    )
                )

                if run_eval:
                    test_losses, _ = self.eval(
                        test_ds, loss_functions=eval_loss_functions
                    )
                    print("Test Losses {}".format(test_losses))

    def eval(self, ds, loss_functions=[], metrics=None, verbose=False):
        if not metrics:
            metrics = []

        loss_functions = utils.names_to_fn(loss_functions, get_eval_loss_fn)
        metrics = utils.names_to_fn(metrics, get_metric)

        start = time.time()
        predictions, targets = [], []
        for (data, target) in ds:
            pred_rating = self.call(data, training=False).numpy().flatten()
            predictions.extend(list(pred_rating))
            targets.extend(list(target.numpy().flatten()))

        if verbose:
            print("Time to evaluate dataset = {} secs\n".format(time.time() - start))

        loss_function_res = []
        for loss_function in loss_functions:
            loss_function_res.append(loss_function(predictions, targets))

        metrics_res = []
        for metric in metrics:
            metrics_res.append(metric(predictions, targets))

        return loss_function_res, metrics_res

    def call(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def real_variables(self):
        return self.variables
