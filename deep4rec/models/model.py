"""Simple Model abstraction."""
import time
from tqdm import tqdm

import numpy as np
import sklearn.model_selection as sk_model_selection
import tensorflow as tf

from deep4rec.models.loss_functions import get_tf_loss_fn
from deep4rec.models.loss_functions import get_eval_loss_fn
from deep4rec.models.metrics import get_metric
from deep4rec.models.optimizers import build_optimizer
from deep4rec import utils


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

    def _features_dict(self, features):
        features_dict = {}
        for feature_name, feature in zip(
            ["one_hot_features", "wide_features", "dense_features"], features
        ):
            features_dict[feature_name] = feature
        return features_dict

    def kfold_train(
        self,
        ds,
        epochs,
        loss_function,
        n_splits=3,
        batch_size=128,
        optimizer="adam",
        run_eval=True,
        verbose=False,
        eval_metrics=None,
        eval_loss_functions=None,
        early_stop=True,
    ):
        for i, (train_indexes, test_indexes) in enumerate(
            ds.kfold_iterator(n_splits=n_splits)
        ):
            print(
                "{}/{} K-fold execution: train size = {}, test size = {}".format(
                    i + 1, n_splits, len(train_indexes), len(test_indexes)
                )
            )
            self.train(
                ds,
                epochs=epochs,
                loss_function=loss_function,
                batch_size=batch_size,
                optimizer=optimizer,
                run_eval=run_eval,
                verbose=verbose,
                eval_metrics=eval_metrics,
                eval_loss_functions=eval_loss_functions,
                train_indexes=train_indexes,
                test_indexes=test_indexes,
                early_stop=early_stop,
            )

    def train(
        self,
        ds,
        epochs,
        loss_function,
        batch_size=128,
        optimizer="adam",
        run_eval=True,
        verbose=True,
        eval_metrics=None,
        eval_loss_functions=None,
        train_indexes=None,
        test_indexes=None,
        early_stop=True,
    ):
        if eval_loss_functions is None:
            eval_loss_functions = []

        if type(loss_function) == str:
            self.loss_function_name = loss_function
            eval_loss_functions = set(eval_loss_functions + [loss_function])
        else:
            self.loss_function_name = "custom_loss_function"

        if eval_metrics is None:
            eval_metrics = []

        self.train_losses = []
        self.test_losses = []

        self.train_metrics = []
        self.test_metrics = []

        if train_indexes is not None and test_indexes is not None:
            train_ds = ds.make_tf_dataset(
                "train", batch_size=batch_size, indexes=train_indexes
            )
            test_ds = ds.make_tf_dataset(
                "train", batch_size=batch_size, indexes=test_indexes
            )
        else:
            train_ds = ds.make_tf_dataset("train", batch_size=batch_size)
            test_ds = ds.make_tf_dataset("test", batch_size=batch_size)

        loss_function = utils.name_to_fn(loss_function, get_tf_loss_fn)
        optimizer = utils.name_to_fn(optimizer, build_optimizer)

        for epoch in tqdm(range(epochs)):
            start = time.time()
            for (*features, target) in train_ds:
                with tf.GradientTape() as tape:
                    pred_rating = self.call(
                        **self._features_dict(features), training=True
                    )
                    loss = loss_function(target, pred_rating)
                gradients = tape.gradient(loss, self.real_variables)
                optimizer.apply_gradients(
                    zip(gradients, self.real_variables),
                    tf.train.get_or_create_global_step(),
                )

            if verbose:
                print(
                    "Epoch {}, Time: {:2f} (s)".format(epoch + 1, time.time() - start)
                )

            train_losses, train_metrics = self.eval(
                train_ds, loss_functions=eval_loss_functions, metrics=eval_metrics
            )

            if train_losses:
                self.train_losses.append(train_losses)
                if verbose:
                    self._print_res("Train Losses", train_losses)

            if train_metrics:
                self.train_metrics.append(train_metrics)
                if verbose:
                    self._print_res("Train Metrics", train_metrics)

            if run_eval:
                test_losses, test_metrics = self.eval(
                    test_ds, loss_functions=eval_loss_functions, metrics=eval_metrics
                )

                if test_losses:
                    self.test_losses.append(test_losses)
                    if verbose:
                        self._print_res("Test Losses", test_losses)

                if test_metrics:
                    self.test_metrics.append(test_metrics)
                    if verbose:
                        self._print_res("Test Metrics", test_metrics)

            if early_stop and self._eval_early_stop():
                break

    def _eval_early_stop(self):
        if len(self.test_losses) > 3:
            if (
                self.test_losses[-1][self.loss_function_name]
                > self.test_losses[-2][self.loss_function_name]
                and self.test_losses[-2][self.loss_function_name]
                > self.test_losses[-3][self.loss_function_name]
            ):
                return True
        return False

    def eval(self, ds, loss_functions=[], metrics=None):
        if not metrics:
            metrics = []

        loss_functions_fn = utils.names_to_fn(loss_functions, get_eval_loss_fn)
        metrics_fn = utils.names_to_fn(metrics, get_metric)

        predictions, targets = [], []
        for (*features, target) in ds:
            pred_rating = (
                self.call(**self._features_dict(features), training=False)
                .numpy()
                .flatten()
            )
            predictions.extend(list(pred_rating))
            targets.extend(list(target.numpy().flatten()))

        loss_function_res = {}
        for loss_function_name, loss_function_fn in zip(
            loss_functions, loss_functions_fn
        ):
            loss_function_res[loss_function_name] = loss_function_fn(
                targets, predictions
            )

        metrics_res = {}
        for metric_name, metric_fn in zip(metrics, metrics_fn):
            metrics_res[metric_name] = metric_fn(targets, predictions)

        return loss_function_res, metrics_res

    def _print_res(self, res_title, res_dict):
        print("------------ {} ------------".format(res_title))
        for res_name in res_dict:
            print("{}: {:4f}".format(res_name, res_dict[res_name]))

    def call(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def real_variables(self):
        return self.variables
