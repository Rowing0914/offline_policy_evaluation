import random
import numpy as np
import pandas as pd
from tabulate import tabulate


def rmse(a, b):
    """ Rooted Mean Squared Error """
    return np.sqrt(np.mean(np.square(a - b)))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def normalise(x):
    """Compute normalised values for each sets of scores in x."""
    e_x = x - np.max(x)
    return e_x / e_x.sum(axis=0)


def aggregator(buffer):
    """ Aggregate the buffer which stores all the resulting variables for experiments

    :param buffer: a list of dictionaries of matrices(e.g., estimated rewards)
    :return temp: the aggregated dictionary along with keys(e.g., estimator names)
    """
    temp = dict()
    for run in buffer:
        for key, value in run.items():
            if key not in temp.keys():
                temp[key] = value
            else:
                temp[key] += value
    return temp


def twoD_gather(array, indices):
    """ Gather items according to the specified indices at each row in 2D array """
    temp = list()
    for row, i in zip(array, indices):
        temp.append(row[i])
    return np.asarray(temp)


def prep_for_visualisation(results, data_names, est_names):
    """ Extracts items from the resulting nested dict and summarise them on Pandas Dataframe """
    _bias, _rmse = list(), list()
    for data_name, value in results.items():
        _dict_bias, _dict_rmse = value
        _bias.append(list(_dict_bias.values()))
        _rmse.append(list(_dict_rmse.values()))

    df_bias = pd.DataFrame(np.asarray(_bias).T, columns=data_names)
    df_rmse = pd.DataFrame(np.asarray(_rmse).T, columns=data_names)
    df_bias["algo"] = df_rmse["algo"] = est_names
    return df_bias, df_rmse


def summary_in_txt(df, _metric_name="bias"):
    """ Summarise the result in a text file given the resultant dataframe """
    with open("./results/result_{}.txt".format(_metric_name), "w") as f:
        f.write(tabulate(df, tablefmt="pipe", headers="keys"))


def train_test_split(data, test_size=0.5):
    """ Split the data into train/test

        Note that if you use sklearn's train_test_split it doesn't account for the entropy of data in terms classes
        So that sometimes there are missing classes in train/test and that leads us to the situation below.

        In IPS, the basic continuity assumption doesn't hold anymore
        meaning that the p_prod(a|x) != 0 doesn't necessarily mean p_targ(a|x) != 0
        and you'd get error during the computation of the importance weight because both have the different shape!!
    """
    indices = np.arange(data.y.shape[0])
    train, test = list(), list()
    for cls in np.unique(data.y):
        subset = data.y[data.y == cls]
        subset_id = indices[data.y == cls]
        if subset.shape[0] == 1:
            subset_train, subset_test = [subset_id], [subset_id]
        elif subset.shape[0] == 2:
            subset_train, subset_test = [subset_id[0]], [subset_id[1]]
        else:
            mid_point = subset_id.shape[0] // 2
            random.shuffle(subset_id)
            subset_train, subset_test = subset_id[:mid_point].tolist(), subset_id[mid_point:].tolist()
        assert type(subset_train) == list and type(subset_test) == list
        train += subset_train
        test += subset_test

    train, test = np.asarray(train), np.asarray(test)

    # Bandit Feedback in Cost Sensitive format
    x_train, x_test, y_train, y_test = data.x[train], data.x[test], data.y_onehot[train], data.y_onehot[test]
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    from data.data_manager import load_ecoli
    data = load_ecoli()
    x_train, x_test, y_train, y_test = train_test_split(data=data)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
