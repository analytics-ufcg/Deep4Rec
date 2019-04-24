"""Dataset interface for the version of the Frappe dataset made available by:
Xiangnan He (xiangnanhe@gmail.com)
Lizi Liao (liaolizi.llz@gmail.com)

At the work Neural Factorization Machines for Sparse Predictive Analytics
https://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-nfm.pdf
"""
import numpy as np
import os

from deep4rec.datasets.dataset import Dataset
import deep4rec.utils as utils


class FrappeDataset(Dataset):

    url = "https://raw.githubusercontent.com/hexiangnan/neural_factorization_machine/master/data/frappe/"

    # Three files are needed in the path
    def __init__(self, dataset_name, output_dir, *args, **kwargs):
        super(FrappeDataset, self).__init__(dataset_name, output_dir, *args, **kwargs)

        self.train_url = self.url + "frappe.train.libfm"
        self.test_url = self.url + "frappe.test.libfm"

        self.train_file = os.path.join(self.output_dir, "frappe.train.libfm")
        self.test_file = os.path.join(self.output_dir, "frappe.test.libfm")

    def download(self):
        super(FrappeDataset, self).download(self.train_url)
        super(FrappeDataset, self).download(self.test_url)

    def check_downloaded(self):
        return os.path.exists(self.train_file) and os.path.exists(self.test_file)

    def preprocess(self):
        self.features_M = self.map_features()
        self.construct_data("otherloss")

    def check_preprocessed(self):
        return False

    def map_features(self):
        self.features = {}
        self.read_features(self.train_file)
        self.read_features(self.test_file)
        return len(self.features)

    def read_features(self, file):
        with open(file) as f:
            line = f.readline()
            last_index = len(self.features)
            while line:
                items = line.strip().split(" ")
                for item in items[1:]:
                    if item not in self.features:
                        self.features[item] = i
                        last_index += 1
                line = f.readline()

    def construct_data(self, loss_type):
        self.train_data, self.train_y = self.read_data(self.train_file)
        self.test_data, self.test_y = self.read_data(self.test_file)

    def read_data(self, file_name):
        with open(file_name) as f:
            x, y = [], []
            line = f.readline()
            while line:
                items = line.strip().split(" ")
                y.append(1.0 * float(items[0]))
                x.append([self.features[item] for item in items[1:]])
                line = f.readline()
        return x, y

    @property
    def train_size(self):
        return len(self.train_data)

    @property
    def train(self):
        return (self.train_data, self.train_y)

    @property
    def num_features_one_hot(self):
        return self.features_M

    @property
    def num_features(self):
        return len(self.train_data[0])

    @property
    def test(self):
        return (self.test_data, self.test_y)
