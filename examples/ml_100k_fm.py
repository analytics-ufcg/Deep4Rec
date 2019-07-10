import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from fastFM.mcmc import FMRegression as FM_mcmc
from fastFM.sgd import FMRegression as FM_sgd

from deep4rec import datasets
from deep4rec import models

import pandas
import numpy
import tensorflow as tf
import time

# Dataset
ds = datasets.build_dataset("ml-100k")


# FastFM
def load_problem_movielens_100k():
    ratings = pandas.read_csv(
        os.path.join("/tmp/deep4rec_data/ml-100k", "ml-100k", "u.data"),
        sep="\t",
        names=["user", "movie", "rating", "timestamp"],
        header=None,
    )
    ratings = ratings.drop("timestamp", axis=1)

    answers = ratings["rating"].values
    ratings = ratings.drop("rating", axis=1)

    for feature in ratings.columns:
        _, ratings[feature] = numpy.unique(ratings[feature], return_inverse=True)

    trainX, testX, trainY, testY = train_test_split(
        ratings, answers, train_size=0.75, random_state=42
    )
    return trainX, testX, trainY, testY


trainX, testX, trainY, testY = load_problem_movielens_100k()


def fitpredict_mcmc(trainX, trainY, testX, rank=8, n_iter=100):
    encoder = OneHotEncoder(handle_unknown="ignore").fit(trainX)
    trainX = encoder.transform(trainX)
    testX = encoder.transform(testX)
    clf = FM_mcmc(rank=rank, n_iter=n_iter)
    pred = clf.fit_predict(trainX, trainY, testX)
    return None, pred


def fitpredict_sgd(trainX, trainY, testX, rank=2, encode=True, epochs=10):
    n_iter = epochs * len(trainX)
    encoder = OneHotEncoder(handle_unknown="ignore").fit(trainX)
    trainX = encoder.transform(trainX)
    testX = encoder.transform(testX)
    clf = FM_sgd(rank=rank, n_iter=n_iter, l2_reg_w=0, step_size=0.01, l2_reg_V=0.1)
    clf = clf.fit(trainX, trainY)
    return clf.predict(trainX), clf.predict(testX)


def test_on_dataset(trainX, testX, trainY, testY, encode=True):
    results = {}
    start = time.time()
    pred_train, pred_test = fitpredict_sgd(trainX, trainY, testX, encode=encode)
    spent_time = time.time() - start
    results["time"] = spent_time
    results["train_RMSE"] = (
        None if pred_train is None else numpy.mean((trainY - pred_train) ** 2) ** 0.5
    )
    results["test_RMSE"] = numpy.mean((testY - pred_test) ** 2) ** 0.5
    return results


print(test_on_dataset(trainX, testX, trainY, testY))
print(test_on_dataset(ds.train_data, ds.test_data, ds.train_y, ds.test_y))

# Model
model = models.FM(ds, num_units=2, l2_regularizer=0.1)
model.train(
    ds,
    batch_size=128,
    epochs=10,
    loss_function="rmse",
    optimizer=tf.train.GradientDescentOptimizer(0.01),
)
