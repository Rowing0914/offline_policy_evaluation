"""
Converts the dataset into X(features) and Y(labels: numerical values) format.
"""

import pandas as pd
import numpy as np

from data import ROOT_DIR

# all names of the datasets
DATASET_NAMES = ["ecoli", "glass", "letter", "optdigits", "page-blocks", "pendigits", "satimage", "vehicle", "yeast"]


class Data(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.y_onehot = None
        self.load_data()
        self.relabelling()
        self.one_hot_vectorise()

    def load_data(self):
        raise NotImplementedError()

    def relabelling(self):
        """ Since in some dataset the labels are skipping some values, we manually relabel them """
        num_label = len(np.unique(self.y))
        if not np.alltrue(self.unique_label == np.arange(num_label)):
            master_table = {key: value for key, value in zip(self.unique_label, np.arange(num_label))}
            for i in range(self.y.shape[0]):
                self.y[i] = master_table[self.y[i]]

    def one_hot_vectorise(self):
        num_label = len(np.unique(self.y))
        self.y_onehot = np.eye(num_label)[self.y - 1]

    @property
    def num_label(self):
        if self.y_onehot is not None:
            return self.y_onehot.shape[1]
        else:
            return self.y.shape[1]

    @property
    def unique_label(self):
        if self.y_onehot is not None:
            return np.unique(self.y_onehot)
        else:
            return np.unique(self.y)


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
    for name in DATASET_NAMES:
        if name == "page-blocks":
            name = "page_blocks"
        data = eval("load_{}()".format(name))
        print("[{}] x: {} y: {} num_label: {}".format(name, data.x.shape, data.y.shape, data.num_label))
