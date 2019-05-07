from deep4rec import datasets
from deep4rec import models

import tensorflow as tf

# Dataset
ds = datasets.build_dataset("census")

# Model
model = models.WideDeep(ds)

model.train(
    ds,
    batch_size=40,
    epochs=15,
    loss_function="binary_cross_entropy",
    eval_loss_functions=["binary_cross_entropy"],
    eval_metrics=["auc", "accuracy", "auc_precision_recall", "precision", "recall"],
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.04),
)
