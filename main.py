import argparse
from functools import partial

from data.data_manager import *
from estimator import DM, IPS, DR
from poilcy import UniformPolicy, DeterministicPolicy, DeterministicPolicy2
from utils import rmse, aggregator, twoD_gather, prep_for_visualisation, summary_in_txt, train_test_split
from plot import plot_bar_chart
from multi_thread import RunInParallel
from tf_utils import eager_setup


def _train_policy(policy, x_train, y_train, x_test=None, y_test=None):
    policy.update(x=x_train, y=y_train, epochs=1000, batch_size=64, verbose=False)
    if x_test is not None and y_test is not None:
        action, score = policy.select_action(context=x_test)
        label = np.argmax(y_test, axis=-1)
        print("Accuracy: {}".format(np.mean(action == label)))
    return policy


def single_run(estimators, data_name="ecoli", test_size=0.5):
    """ See Sec 5.1.2 in the paper

    :param data_name: a name of a dataset
    :return reward_est: a dict of estimated rewards by the Estimators of interest
    :return reward_true: a vector of true rewards
    """
    # load the dataset
    data = eval("load_{}()".format(data_name))

    # (Acronym) prod: Production, targ: Target
    x_train, x_test, y_train, y_test = train_test_split(data=data, test_size=test_size)

    # Instantiate and train the prod/targ policies on the training set
    prod_policy = UniformPolicy(num_action=data.num_label)
    # prod_policy = DeterministicPolicy2(num_action=data.num_label)
    # prod_policy = _train_policy(policy=prod_policy, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    # targ_policy = UniformPolicy(num_action=data.num_label)
    targ_policy = DeterministicPolicy2(num_action=data.num_label)
    targ_policy = _train_policy(policy=targ_policy, x_train=x_train, y_train=y_train)

    # let the policies predict on the test set
    prod_a_tr, prod_score_tr = prod_policy.select_action(context=x_train)
    prod_a_te, prod_score_te = prod_policy.select_action(context=x_test)
    targ_a_te, targ_score_te = targ_policy.select_action(context=x_test)
    prod_r_te = twoD_gather(y_test, prod_a_te)
    # reward_true = twoD_gather(y_test, targ_a_te)
    reward_true = 1 - np.mean(targ_a_te == np.argmax(y_test, axis=-1))

    reward_est = dict()

    for name, estimator in estimators.items():
        estimator.train(context=x_train, action=prod_a_tr, reward=y_train)
        _reward_est = estimator.estimate(context=x_test,
                                         prod_r_te=prod_r_te,
                                         prod_a_te=prod_a_te,
                                         targ_a_te=targ_a_te,
                                         prod_score_te=prod_score_te,
                                         targ_score_te=targ_score_te)
        # construct a dict of the estimated rewards
        reward_est[name] = _reward_est
    return reward_est, reward_true


def _exp(num_episodes, estimators, data_name, verbose=0):
    # Loop for a number of experiments on the single dataset
    rmse_buffer, reward_est_buffer = list(), list()
    for episode in range(num_episodes):
        print("=== {} ===".format(data_name))
        reward_est, reward_true = single_run(estimators=estimators, data_name=data_name)
        _rmse = {key: rmse(a=np.mean(value), b=reward_true) for key, value in reward_est.items()}
        rmse_buffer.append(_rmse)
        reward_est_buffer.append(reward_est)

    """ Compute overall Bias and RMSE """
    # aggregate all the results over all the epochs
    _bias_ = aggregator(buffer=reward_est_buffer)
    _rmse_ = aggregator(buffer=rmse_buffer)

    # run one more experiment to compute the bias
    reward_est, _ = single_run(estimators=estimators, data_name=data_name)
    dict_bias = {key: np.mean((value / num_episodes) - _value)
                 for (key, value), (_, _value) in zip(_bias_.items(), reward_est.items())}
    dict_rmse = {key: value / num_episodes for key, value in _rmse_.items()}

    if verbose:
        for (key, value_bias), (_, value_rmse) in zip(dict_bias.items(), dict_rmse.items()):
            print("[{}: {}] RMSE over {}-run: {}".format(data_name, key, num_episodes, value_rmse))
            print("[{}: {}] Bias over {}-run: {}".format(data_name, key, num_episodes, value_bias))

    return dict_bias, dict_rmse


def exp(estimators, num_episodes=500, verbose=0):
    """ conducts the whole experiments """

    fn_names, fns = list(), list()

    # Loop for all experiments on all datasets
    for data_name in DATASET_NAMES:
        if data_name == "page-blocks": data_name = "page_blocks"

        def _fn(_data_name):
            dict_bias, dict_rmse = _exp(num_episodes=num_episodes,
                                        estimators=estimators,
                                        data_name=_data_name,
                                        verbose=verbose)
            return dict_bias, dict_rmse

        fn_names.append(data_name)
        _fn = partial(_fn, _data_name=data_name)
        fns.append(_fn)

    # Run the estimators in parallel
    results = RunInParallel(fn_names=fn_names, fns=fns)

    # Summarise the results
    df_bias, df_rmse = prep_for_visualisation(results=results, data_names=DATASET_NAMES, est_names=estimators.keys())
    summary_in_txt(df=df_bias, _metric_name="bias")
    summary_in_txt(df=df_rmse, _metric_name="rmse")
    plot_bar_chart(df=df_bias, plot_name="Bias", fig_name="./results/bias.png")
    plot_bar_chart(df=df_rmse, plot_name="RMSE", fig_name="./results/rmse.png")


def main(num_episodes=500, verbose=0):
    """ Intermediate fun to coordinate the experiments """
    eager_setup()

    # Prepare all the estimators
    estimators = {
        "DM": DM(model_type="ridge"),
        "IPS": IPS(),
        "CIPS": IPS(if_cap=True, _min=0, _max=10),
        "NIPS": IPS(if_normalise=True),
        "NCIPS": IPS(if_cap=True, _min=0, _max=10, if_normalise=True),
        "Pointwise-NCIPS": IPS(if_cap=True, _min=0, _max=10, if_normalise=True, if_pointwise=True),
        "DR-IPS": DR(ips=IPS(), dm=DM(model_type="ridge")),
        "DR-CIPS": DR(ips=IPS(if_cap=True, _min=0, _max=10), dm=DM(model_type="ridge")),
        "DR-NCIPS": DR(ips=IPS(if_cap=True, _min=0, _max=10, if_normalise=True), dm=DM(model_type="ridge")),
        "SWITCH-IPS": DR(ips=IPS(), dm=DM(model_type="ridge"), switch_tau=2.0, switch_flg="IPS"),
        "SWITCH-DR": DR(ips=IPS(), dm=DM(model_type="ridge"), switch_tau=2.0, switch_flg="DR"),
        "CAB": DR(ips=IPS(), dm=DM(model_type="ridge"), cab_coeff=2.0, cab_flg=""),
        "CAB-DR": DR(ips=IPS(), dm=DM(model_type="ridge"), cab_coeff=2.0, cab_flg="DR")
    }

    # run the whole experiment
    np.random.seed(1)
    exp(num_episodes=num_episodes, estimators=estimators, verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", default=1, help="number of runs of experiments")
    parser.add_argument("--verbose", default=0, help="verbose of training log")
    params = parser.parse_args()
    main(num_episodes=params.num_episodes, verbose=params.verbose)
