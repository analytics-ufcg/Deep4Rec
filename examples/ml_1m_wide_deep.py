from deep4rec import datasets
from deep4rec import models

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# Dataset
ds = datasets.build_dataset("ml-1m")

# Model
model = models.WideDeep(ds)

model.train(
    ds,
    batch_size=40,
    epochs=15,
    loss_function="rmse",
    eval_loss_functions=["rmse"],
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.04),
)
