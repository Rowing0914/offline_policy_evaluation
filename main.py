import argparse

from sklearn.model_selection import train_test_split

from data.data_manager import *
from estimator import DirectMethod, InversePropensityScore, DoublyRobustEstimator
from poilcy import UniformPolicy, DeterministicPolicy
from utils import rmse, aggregator, twoD_gather, prep_for_visualisation, summary_in_txt, eager_setup


def _train_policy(policy, x_prod, y_prod, x_targ=None, y_targ=None):
    policy.update(x=x_prod, y=y_prod, epochs=1000, batch_size=64, verbose=False)
    if x_targ and y_targ:
        action, score = policy.select_action(context=x_targ)
        pred = np.argmax(score, axis=1)
        label = np.argmax(y_targ, axis=1)
        print("Accuracy: {}".format(np.mean(pred == label)))
    return policy


def single_run(estimators, data_name="ecoli", test_size=0.5):
    """ See Sec 5.1.2 in the paper

    :param data_name: a name of a dataset
    :param model_type: a name of a base algorithm for a policy
    :return reward_est: a dict of estimated rewards by the Estimators of interest
    :return reward_true: a vector of true rewards
    """

    # load the dataset
    data = eval("load_{}()".format(data_name))

    # (Acronym) prod: Production, targ: Target
    x_prod, x_targ, y_prod, y_targ = train_test_split(data.x, data.y, test_size=test_size)

    # instantiate and train the prod/targ policies on the training set
    # TODO: take this part outside and automate the process of experiments
    prod_policy = UniformPolicy(num_action=data.num_label)
    targ_policy = UniformPolicy(num_action=data.num_label)
    # prod_policy = DeterministicPolicy(num_action=data.num_label, weight_path="./model/{}".format(data_name))
    # targ_policy = DeterministicPolicy(num_action=data.num_label, weight_path="./model/{}".format(data_name))

    prod_policy = _train_policy(policy=prod_policy, x_prod=x_prod, y_prod=y_prod)
    targ_policy = _train_policy(policy=targ_policy, x_prod=x_prod, y_prod=y_prod)

    # let the policies predict on the test set
    prod_a_tr, prod_score_tr = prod_policy.select_action(context=x_prod)
    prod_a_te, prod_score_te = prod_policy.select_action(context=x_targ)
    targ_a_te, targ_score_te = targ_policy.select_action(context=x_targ)
    prod_r_te = twoD_gather(y_targ, np.argmax(prod_a_te, axis=-1))
    reward_true = twoD_gather(y_targ, np.argmax(targ_a_te, axis=-1))

    reward_est = dict()

    for name, estimator in estimators.items():
        estimator.train(context=x_prod, action=prod_a_tr, reward=y_prod)
        _reward_est = estimator.estimate(context=x_targ,
                                         prod_r_te=prod_r_te,
                                         prod_a_te=prod_a_te,
                                         targ_a_te=targ_a_te,
                                         prod_score_te=prod_score_te,
                                         targ_score_te=targ_score_te)
        reward_est[name] = _reward_est

    # construct a dict of the estimated rewards
    return reward_est, reward_true


def exp(estimators, num_episodes=500, verbose=0):
    """ conducts the whole experiments """

    result_dict = dict()

    # Loop for all experiments on all datasets
    for data_name in names:
        if data_name == "page-blocks": data_name = "page_blocks"

        # Loop for a number of experiments on the single dataset
        rmse_buffer, reward_est_buffer = list(), list()
        for episode in range(num_episodes):
            reward_est, reward_true = single_run(estimators=estimators, data_name=data_name)
            _rmse = {key: rmse(a=value, b=reward_true) for key, value in reward_est.items()}
            rmse_buffer.append(_rmse)
            reward_est_buffer.append(reward_est)

            if verbose:
                for key, value in reward_est.items():
                    print("[{}] RMSE: {}".format(key, rmse(a=value, b=reward_true)))

        """ Compute overall bias and RMSE """
        # aggregate all the results
        _bias_ = aggregator(buffer=reward_est_buffer)
        _rmse_ = aggregator(buffer=rmse_buffer)

        # run one more experiment to compute the bias
        reward_est, _ = single_run(estimators=estimators, data_name=data_name)
        bias_dict = {key: np.mean((value / num_episodes) - _value)
                     for (key, value), (_, _value) in zip(_bias_.items(), reward_est.items())}
        rmse_dict = {key: value / num_episodes for key, value in _rmse_.items()}

        for (key, value_bias), (_, value_rmse) in zip(bias_dict.items(), rmse_dict.items()):
            print("[{}: {}] RMSE over {}-run: {}".format(data_name, key, num_episodes, value_rmse))
            print("[{}: {}] Bias over {}-run: {}".format(data_name, key, num_episodes, value_bias))

        result_dict[data_name] = {"bias": bias_dict, "rmse": rmse_dict}

    # Summarise the results
    bias_df = prep_for_visualisation(_dict=result_dict, _metric_name="bias")
    rmse_df = prep_for_visualisation(_dict=result_dict, _metric_name="rmse")
    summary_in_txt(df=bias_df, _metric_name="bias")
    summary_in_txt(df=rmse_df, _metric_name="rmse")


def main(num_episodes=500, verbose=0):
    """ Intermediate fun to coordinate the experiments """
    eager_setup()

    # Prepare all the estimators
    estimators = {
        "DM": DirectMethod(model_type="ridge"),
        "IPS": InversePropensityScore(),
        "CIPS": InversePropensityScore(cap=10),
        "NIPS": InversePropensityScore(if_normalise=True),
        "NCIPS": InversePropensityScore(cap=10, if_normalise=True),
        "DR_IPS": DoublyRobustEstimator(ips_estimator=InversePropensityScore(),
                                        dm_estimator=DirectMethod(model_type="ridge")),
        "DR_CIPS": DoublyRobustEstimator(ips_estimator=InversePropensityScore(cap=10),
                                         dm_estimator=DirectMethod(model_type="ridge")),
        "DR_NIPS": DoublyRobustEstimator(ips_estimator=InversePropensityScore(if_normalise=True),
                                         dm_estimator=DirectMethod(model_type="ridge")),
        "DR_NCIPS": DoublyRobustEstimator(ips_estimator=InversePropensityScore(cap=10, if_normalise=True),
                                          dm_estimator=DirectMethod(model_type="ridge")),
    }

    # run the whole experiment
    exp(num_episodes=num_episodes, estimators=estimators, verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", default=1, help="number of runs of experiments")
    parser.add_argument("--verbose", default=0, help="verbose of training log")
    params = parser.parse_args()
    main(num_episodes=params.num_episodes, verbose=params.verbose)
