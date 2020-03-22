import numpy as np
import pandas as pd
import tensorflow as tf
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


def prep_for_visualisation(_dict, _metric_name="bias"):
    """ Extracts items from the resulting nested dict and summarise them on Pandas Dataframe """
    temp = list()
    for data_name, __dict in _dict.items():
        for metric_name, ___dict in __dict.items():
            if metric_name == _metric_name:
                temp.append(list(___dict.values()))

    # Convert the list into Dataframe
    df = pd.DataFrame(np.asarray(temp).T, columns=list(_dict.keys()))

    # set the column names and exchange the columns
    df["algo"] = list(___dict.keys())
    columns_titles = ["algo"] + list(_dict.keys())
    df = df.reindex(columns=columns_titles)
    df.set_index("algo", inplace=True)
    return df


def summary_in_txt(df, _metric_name="bias"):
    """ Summarise the result in a text file given the resultant dataframe """
    with open("./results/result_{}.txt".format(_metric_name), "w") as f:
        f.write(tabulate(df, tablefmt="pipe", headers="keys"))


def eager_setup():
    """
    it enables an eager execution in tensorflow with config that allows us to flexibly access to a GPU
    from multiple python scripts
    """

    # === before TF 2.0 ===
    # config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
    #                                   intra_op_parallelism_threads=1,
    #                                   inter_op_parallelism_threads=1)
    # config.gpu_options.allow_growth = True
    # tf.compat.v1.enable_eager_execution(config=config)
    # tf.compat.v1.enable_resource_variables()

    # === For TF 2.0 ===
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.InteractiveSession(config=config)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # TODO: if you don't need it, remove!!


def create_checkpoint(model, optimizer, model_dir, verbose=False):
    """ Create a checkpoint for managing a model

    :param model: TF Neural Network
    :param optimizer: TF optimisers
    :param model_dir: a directory to save the optimised weights and other checkpoints
    :return manager: a manager to control the save timing
    """
    checkpoint_dir = model_dir
    check_point = tf.train.Checkpoint(optimizer=optimizer,
                                      model=model,
                                      optimizer_step=tf.compat.v1.train.get_or_create_global_step())
    manager = tf.train.CheckpointManager(check_point, checkpoint_dir, max_to_keep=3)

    if verbose:
        # try re-loading the previous training progress!
        try:
            print("Try loading the previous training progress")
            check_point.restore(manager.latest_checkpoint)
            print("===================================================\n")
            print("Restored the model from {}".format(checkpoint_dir))
            print("Currently we are on Epoch: {}".format(tf.compat.v1.train.get_global_step().numpy()))
            print("\n===================================================")
        except:
            print("===================================================\n")
            print("Previous Training files are not found in Directory: {}".format(checkpoint_dir))
            print("\n===================================================")
    return manager