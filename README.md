# Deep4Rec

Popular Deep Learning based recommendation algorithms built on top of TensorFlow served on a simple API.

```python

from deep4rec import datasets
from deep4rec import models

import tensorflow as tf

# Dataset
ds = datasets.build_dataset("ml-100k")

# Model
model = models.FM(ds)

model.train(
    ds,
    batch_size=128,
    epochs=200,
    loss_function="l2",
    eval_loss_functions=["rmse"],
    optimizer="adam",
)
```

## What is Deep4Rec ?

Deep4Rec is a high level API that serves popular Deep Learning based recommendation algorithms as black box models. The models are built on top of TensorFlow >= 1.13.1 using Eager mode.

## What is not Deep4Rec?

Deep4Rec is not a general purposes framework for trying new approaches and models to recommender systems.

Deep4Rec is not affiliated to the official TensorFlow project in any way.

## Why Deep4Rec?

Deep4Rec was developed mainly to accelerate research by providing a way for researchers to use previously proposed models as black boxes.

## Contributing

[Here is all you need to get started](CONTRIBUTE.md).
