# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Dataset interface for Census dataset.

Census dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/adult
"""

import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

from deep4rec.datasets.dataset import Dataset
import deep4rec.utils as utils


_CSV_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income_bracket",
]

_CSV_COLUMN_DEFAULTS = [
    [0],
    [""],
    [0],
    [""],
    [0],
    [""],
    [""],
    [""],
    [""],
    [""],
    [0],
    [0],
    [0],
    [""],
    [""],
]


class CensusDataset(Dataset):

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"

    def __init__(self, dataset_name, output_dir, *args, **kwargs):
        super().__init__(dataset_name, output_dir, *args, **kwargs)

        self.train_filename = "adult.data"
        self.test_filename = "adult.test"

        self.train_url = os.path.join(self.url, self.train_filename)
        self.train_path = os.path.join(self.output_dir, self.train_filename)

        self.test_url = os.path.join(self.url, self.test_filename)
        self.test_path = os.path.join(self.output_dir, self.test_filename)

        self.preprocessed_path = os.path.join(self.output_dir, self.dataset_name)
        self._ord_encoder = OrdinalEncoder()
        self._occupation_ord_encoder = OrdinalEncoder()
        self._one_hot_encoder = OneHotEncoder(sparse=False)

    def _download_and_clean_file(self, url, filename):
        """Downloads data from url, and makes changes to match the CSV format."""
        temp_file, _ = urllib.request.urlretrieve(url)
        with tf.gfile.Open(temp_file, "r") as temp_eval_file:
            with tf.gfile.Open(filename, "w") as eval_file:
                for line in temp_eval_file:
                    line = line.strip()
                    line = line.replace(", ", ",")
                    if not line or "," not in line:
                        continue
                    if line[-1] == ".":
                        line = line[:-1]
                    line += "\n"
                    eval_file.write(line)
        tf.gfile.Remove(temp_file)

    def download(self):
        self._download_and_clean_file(self.train_url, self.train_path)
        self._download_and_clean_file(self.test_url, self.test_path)

    def check_downloaded(self):
        return os.path.exists(self.train_path) and os.path.exists(self.test_path)

    def check_preprocessed(self):
        return False

    def _preprocess(self, filename, train_data=False):
        df = pd.read_csv(filename, names=_CSV_COLUMNS)

        # Categorical columns
        df_base_columns = df[
            ["education", "marital_status", "relationship", "workclass"]
        ]
        if train_data:
            base_columns = self._ord_encoder.fit_transform(df_base_columns.values)
            occupation_column = self._occupation_ord_encoder.fit_transform(
                df["occupation"].values.reshape(-1, 1)
            )
            one_hot_base_columns = self._one_hot_encoder.fit_transform(
                df_base_columns.values
            )
        else:
            base_columns = self._ord_encoder.transform(df_base_columns.values)
            occupation_column = self._occupation_ord_encoder.transform(
                df["occupation"].values.reshape(-1, 1)
            )
            one_hot_base_columns = self._one_hot_encoder.transform(
                df_base_columns.values
            )

        # Age buckets
        buckets = [0, 18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 200]
        age_buckets = np.array(
            pd.cut(df["age"], buckets, labels=range(len(buckets) - 1)).values
        )

        wide_columns = np.concatenate(
            (base_columns, age_buckets.reshape(-1, 1)), axis=1
        )

        numerical_columns = df[
            ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
        ].values
        deep_columns = np.concatenate((one_hot_base_columns, numerical_columns), axis=1)

        labels = np.where(df["income_bracket"].values == ">50K", 1, 0)

        if train_data:
            self.train_wide_data = wide_columns
            self.train_deep_data = deep_columns
            self.train_embedding_data = occupation_column
            self.train_y = labels
        else:
            self.test_wide_data = wide_columns
            self.test_deep_data = deep_columns
            self.test_embedding_data = occupation_column
            self.test_y = labels

    def preprocess(self):
        self._preprocess(self.train_path, train_data=True)
        self._preprocess(self.test_path)

    @property
    def train_size(self):
        return len(self.train_wide_data)

    @property
    def train_features(self):
        return [self.train_wide_data, self.train_deep_data, self.train_embedding_data]

    @property
    def test_features(self):
        return [self.test_wide_data, self.test_deep_data, self.test_embedding_data]

    @property
    def num_features_one_hot(self):
        return len(np.unique(self.train_embedding_data))

    @property
    def num_features(self):
        return 1
