import os
# Ignore some tensorflow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from deep4rec import datasets
from deep4rec import models

ds = datasets.build_dataset("ml-100k")

spfc = models.SpectralCF(ds)

spfc.train(
    ds,
    batch_size=1024,
    epochs=200,
    loss_function="rmse",
    eval_metrics=["recall"]
)
