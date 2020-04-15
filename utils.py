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

    df_bias = pd.DataFrame(_bias, columns=data_names)
    df_rmse = pd.DataFrame(_rmse, columns=data_names)
    df_bias["algo"] = df_rmse["algo"] = est_names
    return df_bias, df_rmse


def summary_in_txt(df, _metric_name="bias"):
    """ Summarise the result in a text file given the resultant dataframe """
    with open("./results/result_{}.txt".format(_metric_name), "w") as f:
        f.write(tabulate(df, tablefmt="pipe", headers="keys"))
