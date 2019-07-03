"""Compare methods using the given dataset for a regression task."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from deep4rec import datasets
from deep4rec import models
import fire
import tensorflow as tf


def main(ds_name):
    # Dataset
    ds = datasets.build_dataset(ds_name)

    for Model in [models.NeuralMF]:
        for num_units in [16]:
            for layers, dropout_prob in [
                ([32, 16], [0.3, 0.3, 0.8]),
                ([16, 16], [0.8, 0.8, 0.8]),
            ]:
                print()
                print()
                print(
                    "%s, %s, %s, %s" % (Model.__name__, num_units, layers, dropout_prob)
                )
                print()
                print()
                if Model == models.FM:
                    model = Model(
                        ds,
                        num_units=num_units,
                        dropout_prob=dropout_prob[-1],
                        l2_regularizer=0.1,
                    )
                else:
                    model = Model(
                        ds,
                        num_units=num_units,
                        layers=layers,
                        dropout_prob=dropout_prob,
                        l2_regularizer=0.1,
                    )

                model.train(
                    ds,
                    batch_size=16,
                    epochs=50,
                    loss_function="l2",
                    eval_loss_functions=["rmse"],
                    optimizer=tf.train.AdagradOptimizer(learning_rate=0.05),
                    early_stop=True,
                )


if __name__ == "__main__":
    fire.Fire(main)
