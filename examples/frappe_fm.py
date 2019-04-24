from deep4rec import datasets
from deep4rec import models

import tensorflow as tf

# Dataset
ds = datasets.build_dataset("frappe")

# Model
model = models.FM(ds)

model.train(
    ds,
    batch_size=128,
    epochs=200,
    loss_function="l2",
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.05),
)
