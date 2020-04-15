"""
Converts the dataset into X(features) and Y(labels: numerical values) format.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data import ROOT_DIR

# all names of the datasets
DATASET_NAMES = ["ecoli", "glass", "letter", "optdigits", "page-blocks", "pendigits", "satimage", "vehicle", "yeast"]


class Data(object):
    def __init__(self, if_noisy=False):
        self.x = None
        self.y = None
        self.y_onehot = None
        self.load_data()
        self.preprocess()
        self.relabelling()
        self.one_hot_vectorise()
        if if_noisy:
            self.add_noise()

    def load_data(self):
        raise NotImplementedError()

    def preprocess(self):
        self.x = StandardScaler().fit_transform(X=self.x)

    def relabelling(self):
        """ Since in some dataset the label is discontinuous(missing some values), we manually relabel them
            ranging from 0 to the number of labels
        """
        num_label = len(np.unique(self.y))
        if not np.alltrue(np.unique(self.y) == np.arange(num_label)):
            master_table = {key: value for key, value in zip(np.unique(self.y), np.arange(num_label))}
            for i in range(self.y.shape[0]):
                self.y[i] = master_table[self.y[i]]

    def one_hot_vectorise(self):
        num_label = len(np.unique(self.y))
        self.y_onehot = np.eye(num_label)[self.y]

    def add_noise(self):
        """ In addition to this deterministic reward model, we also consider
            a noisy reward model for each data set, which reveals the correct reward
            with probability 0.5 and outputs a random coin toss otherwise.
            Theoretically, this should lead to bigger std and larger variance in all estimators.
        """
        mask = np.random.random(size=self.y.shape[0]) > 0.5
        self.y_onehot[~mask] = np.random.binomial(n=1, p=0.5, size=self.y_onehot[~mask].shape)

    @property
    def num_label(self):
        if self.y_onehot is not None:
            return self.y_onehot.shape[1]
        else:
            return self.y.shape[1]


class load_ecoli(Data):
    def load_data(self):
        data = pd.read_csv(ROOT_DIR + "/data/ecoli/ecoli.data", header=None, delim_whitespace=True)
        data.iloc[:, -1] = pd.factorize(data.iloc[:, -1])[0]  # convert the string labels into integers
        self.x = data.iloc[:, :-1].to_numpy()
        self.y = data.iloc[:, -1].to_numpy()


class load_glass(Data):
    def load_data(self):
        data = pd.read_csv(ROOT_DIR + "/data/glass/glass.data", header=None, sep=",")
        self.x = data.iloc[:, 1:-1].to_numpy()
        self.y = data.iloc[:, -1].to_numpy()


class load_letter(Data):
    def load_data(self):
        data = pd.read_csv(ROOT_DIR + "/data/letter/letter-recognition.data", header=None, sep=",")
        data.iloc[:, 0] = pd.factorize(data.iloc[:, 0])[0]  # convert the string labels into integers
        self.x = data.iloc[:, 1:].to_numpy()
        self.y = data.iloc[:, 0].to_numpy()


class load_optdigits(Data):
    def load_data(self):
        data = pd.read_csv(ROOT_DIR + "/data/optdigits/data.txt", header=None, sep=",")
        self.x = data.iloc[:, :-1].to_numpy()
        self.y = data.iloc[:, -1].to_numpy()


class load_page_blocks(Data):
    def load_data(self):
        data = pd.read_csv(ROOT_DIR + "/data/page-blocks/page-blocks.data", header=None, delim_whitespace=True)
        self.x = data.iloc[:, :-1].to_numpy()
        self.y = data.iloc[:, -1].to_numpy()


class load_pendigits(Data):
    def load_data(self):
        data = pd.read_csv(ROOT_DIR + "/data/pendigits/data.txt", header=None, sep=",")
        self.x = data.iloc[:, :-1].to_numpy()
        self.y = data.iloc[:, -1].to_numpy()


class load_satimage(Data):
    def load_data(self):
        data = pd.read_csv(ROOT_DIR + "/data/satimage/data.txt", header=None, sep=" ")
        self.x = data.iloc[:, :-1].to_numpy()
        self.y = data.iloc[:, -1].to_numpy()


class load_vehicle(Data):
    def load_data(self):
        data = pd.read_csv(ROOT_DIR + "/data/vehicle/data.txt", header=None, delim_whitespace=True)
        data.iloc[:, -1] = pd.factorize(data.iloc[:, -1])[0]  # convert the string labels into integers
        self.x = data.iloc[:, :-1].to_numpy()
        self.y = data.iloc[:, -1].to_numpy()


class load_yeast(Data):
    def load_data(self):
        data = pd.read_csv(ROOT_DIR + "/data/yeast/yeast.data", header=None, delim_whitespace=True)
        data.iloc[:, -1] = pd.factorize(data.iloc[:, -1])[0]  # convert the string labels into integers
        self.x = data.iloc[:, 1:-1].to_numpy()
        self.y = data.iloc[:, -1].to_numpy()


if __name__ == '__main__':
    print("=== Without Noise ===")
    for name in DATASET_NAMES:
        if name == "page-blocks":
            name = "page_blocks"
        data = eval("load_{}()".format(name))
        assert np.alltrue(data.y == np.argmax(data.y_onehot, axis=-1))
        assert data.y.min() == 0
        print("[{}] x: {} y: {} num_label: {}".format(name, data.x.shape, data.y.shape, data.num_label))

    print("=== With Noise ===")
    for name in DATASET_NAMES:
        if name == "page-blocks":
            name = "page_blocks"
        data = eval("load_{}(if_noisy=True)".format(name))
        print("[{}] x: {} y: {} num_label: {}".format(name, data.x.shape, data.y.shape, data.num_label))
