from deep4rec import datasets
from deep4rec import models

import tensorflow as tf

# Dataset
ds = datasets.build_dataset("frappe")

# Model
model = models.NeuralMF(ds, num_units=128, layers=[128], dropout_prob=[0.5, 0.5])

model.train(
    ds,
    batch_size=128,
    epochs=200,
    loss_function="l2",
    eval_loss_functions=["rmse"],
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.05),
)
