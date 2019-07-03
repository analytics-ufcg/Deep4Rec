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

Deep4Rec was mainly developed to accelerate research by providing a way for researchers to use previously proposed models as black boxes implemented using the same framework.

Deep4Rec is also a great tool for teaching and learning Recommendation Systems methods based on Deep Learning.

## Supported models and datasets

### Models

| Model name  | Paper                                                                                                                        | Authors                       | Implementation                                             | Example                                                                  |
|-------------|------------------------------------------------------------------------------------------------------------------------------|-------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------|
| FM          | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)                                         | Steffen Rendle                | [deep4rec/models/fm.py](deep4rec/models/fm.py)             | [examples/frappe_fm.py](examples/frappe_fm.py)                               |
| NeuralFM    | [Neural Factorization Machines for Sparse Predictive Analytics](http://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-nfm.pdf) | Xiangnan He and Tat-Seng Chua | [deep4rec/models/nfm.py](deep4rec/models/nfm.py)           | [examples/frappe_nfm.py](examples/frappe_nfm.py)                             |
| Wide & Deep | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)                                             | Heng-Tze Cheng et al          | [deep4rec/models/widedeep.py](deep4rec/models/widedeep.py) | [examples/census_dataset_wide_deep.py](examples/census_dataset_wide_deep.py) |

### Datasets

| Dataset Name          | Original Source                                                                                                              | Reference                                                                                                                    | Implementation                                           | Use example                                                                  |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|------------------------------------------------------------------------------|
| Census Income Dataset (WIP) | [UCI Machine Learning directory](https://archive.ics.uci.edu/ml/machine-learning-databases/adult)                            | https://archive.ics.uci.edu/ml/citation_policy.html                                                                          | [deep4rec/dataset/census.py](deep4rec/dataset/census.py) | [examples/census_dataset_wide_deep.py](examples/census_dataset_wide_deep.py) |
| Frappe Dataset        | [Neural Factorization Machines for Sparse Predictive Analytics](http://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-nfm.pdf) | [Neural Factorization Machines for Sparse Predictive Analytics](http://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-nfm.pdf) | [deep4rec/dataset/frappe.py](deep4rec/dataset/frappe.py) | [examples/frappe_nfm.py](examples/frappe_nfm.py)                             |
| MovieLens 100k        | [Grouplens MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)                                                   | [Grouplens MovieLens 100k ]( https://grouplens.org/datasets/movielens/100k/)                                                | [deep4rec/dataset/ml100k.py](deep4rec/dataset/ml100k.py) | [examples/ml_100k_fm.py](examples/ml_100k_fm.py)                              |
| MovieLens 1M (WIP)          | [Grouplens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)                                                       | [Grouplens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)                                                       | [deep4rec/dataset/ml.py](deep4rec/dataset/ml.py)        | [examples/ml_1m_wide_deep.py](examples/ml_1m_wide_deep.py)                  |
| MovieLens 20M (WIP)         | [Grouplens MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)                                                     | [Grouplens MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)                                                     | [deep4rec/dataset/ml.py](deep4rec/dataset/ml.py)       |                                                                              |



## Contributing

[Here is all you need to get started](CONTRIBUTE.md).
