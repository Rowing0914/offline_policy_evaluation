import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV

from utils import twoD_gather


class BaseEstimator(object):
    def __init__(self):
        pass

    def train(self, context=None, action=None, reward=None):
        """ (Optional) Training the estimator

        :param context: (num_sample, num_feature)
        :param action: (num_sample, num_action)
        :param reward: (num_sample, num_action)
        """
        del context, action, reward
        pass

    def estimate(self, context=None, prod_r_te=None, prod_a_te=None,
                 targ_a_te=None, prod_score_te=None, targ_score_te=None):
        """ Estimate a reward using the inverse propensity score

        :param context: context(features)
        :param prod_r_te: the observed rewards given the actions by prod policy on the test set
        :param prod_a_te: the selected actions by prod policy on the test set
        :param targ_a_te: the selected actions by targ policy on the test set
        :param targ_score_te: the computed scores for each targ policy's arm on the test set
        :return: reward_est: the estimated rewards on the test set
        """
        raise NotImplementedError


class DirectMethod(BaseEstimator):
    def __init__(self, model_type="ridge"):
        """ Direct Method(Reward Prediction)

        :param model_type: type of base algorithm to deal with contexts
        """
        super(DirectMethod, self).__init__()
        self._model_type = model_type
        self._alpha = np.logspace(-3, 2, num=6, base=5)

    def train(self, context=None, action=None, reward=None):
        """ Train the model parameters given contexts and taken actions by the prod policy """
        # Rewards represent the cost of taking actions(e.g., Cost-Sensitive Classification)
        # So, we need to compute y given the taken action by prod policy and the contexts
        feedback = twoD_gather(reward, action)

        if self._model_type == 'ridge':
            self._clf = RidgeCV(alphas=self._alpha, fit_intercept=True, cv=5)
        elif self._model_type == 'lasso':
            self._clf = LassoCV(alphas=self._alpha, tol=1e-3, cv=5, fit_intercept=True)
        self._clf.fit(context, feedback)

    def estimate(self, context=None, prod_r_te=None, prod_a_te=None,
                 targ_a_te=None, prod_score_te=None, targ_score_te=None):
        """ Estimate a reward given a context """
        reward_est = self._clf.predict(X=context)
        return reward_est.flatten()


class InversePropensityScore(BaseEstimator):
    def __init__(self, cap=None, if_normalise=False):
        """ Inverse Propensity Score (IPS) / Importance Sampling(IS)

        This function is able to act as variants of IPS.
        1. Vanilla IPS(Horvitz & Thompson, 1952): cap=None, if_normalise=False
        2. Capped IPS(Léon Bottou and Jonas Peters. 2013): cap=scalar_value, if_normalise=False
        3. Normalised IPS(A. Swaminathan & T. Joachims., 2016): cap=None, if_normalise=True
        4. Normalised Capped IPS: cap=scalar_value, if_normalise=True
        """
        super(InversePropensityScore, self).__init__()
        self._cap = cap
        self._if_normalise = if_normalise

    def estimate(self, context=None, prod_r_te=None, prod_a_te=None,
                 targ_a_te=None, prod_score_te=None, targ_score_te=None):
        """ Estimate a reward using the inverse propensity score """
        # Apply indicator function
        bool_mat = np.asarray(prod_a_te == targ_a_te).astype(np.float32)

        # take the score only for the taken action
        targ_score = twoD_gather(targ_score_te, targ_a_te)
        prod_score = twoD_gather(prod_score_te, targ_a_te)

        # Avoid the division by Zero error
        targ_score[targ_score == 0.0] = np.spacing(1)
        prod_score[prod_score == 0.0] = np.spacing(1)

        # compute the importance weight
        imp_weight = targ_score / prod_score

        if self._cap:
            # See Sec4.2 -> https://arxiv.org/pdf/1801.07030.pdf
            imp_weight = np.minimum(self._cap, imp_weight)

        ips = bool_mat * imp_weight

        # replace the infinity with the extremely small value
        ips[ips == np.inf] = np.spacing(1)

        if self._if_normalise:
            # self normalisation part(Eq.7 on the paper)
            est = (prod_r_te * ips) / np.mean(ips)
        else:
            est = prod_r_te * ips
        return est


class DoublyRobustEstimator(BaseEstimator):
    def __init__(self, ips_estimator, dm_estimator, switch_tau=0.23, switch_flg=None):
        """ Doubly Robust Estimator(DR) by Miroslav Dudik et al., 2011

        :param ips_estimator: Inverse Propensity Scoring
        :param dm_estimator: Direct Method
        :param switch_tau: SWITCH-DR
        """
        super(DoublyRobustEstimator, self).__init__()
        self.ips_estimator = ips_estimator
        self.dm_estimator = dm_estimator
        self._switch_tau = switch_tau
        self._switch_flg = switch_flg

    def train(self, context=None, action=None, reward=None):
        """ Train the model parameters given contexts and taken actions by the prod policy """
        self.dm_estimator.train(context=context, action=action, reward=reward)

    def estimate(self, context=None, prod_r_te=None, prod_a_te=None,
                 targ_a_te=None, prod_score_te=None, targ_score_te=None):
        """ Estimate a reward using the inverse propensity score and Direct Method"""
        dm_est = self.dm_estimator.estimate(context=context)
        reward_adv = prod_r_te - dm_est  # subtract the baseline to get the advantage
        ips_est = self.ips_estimator.estimate(prod_r_te=reward_adv,
                                              prod_a_te=prod_a_te,
                                              targ_a_te=targ_a_te,
                                              prod_score_te=prod_score_te,
                                              targ_score_te=targ_score_te)
        return ips_est + dm_est


if __name__ == '__main__':
    from main import _train_policy
    from utils import rmse, train_test_split
    from data.data_manager import load_ecoli
    from poilcy import UniformPolicy, DeterministicPolicy2

    # load a dataset
    data = load_ecoli()
    x_train, x_test, y_train, y_test = train_test_split(data=data, test_size=0.5)

    # define a prod and a targ policies
    prod_policy = UniformPolicy(num_action=data.num_label)
    # prod_policy = DeterministicPolicy2(num_action=data.num_label)
    # prod_policy = _train_policy(policy=prod_policy, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    # targ_policy = UniformPolicy(num_action=data.num_label)
    targ_policy = DeterministicPolicy2(num_action=data.num_label)
    targ_policy = _train_policy(policy=targ_policy, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    # get dummy actions
    prod_a_tr, prod_score_tr = prod_policy.select_action(context=x_train)
    prod_a_te, prod_score_te = prod_policy.select_action(context=x_test)
    targ_a_te, targ_score_te = targ_policy.select_action(context=x_test)
    prod_r_te = twoD_gather(y_test, prod_a_te)

    # test the estimator
    dm = DirectMethod(model_type="ridge")
    dm.train(context=x_train, action=prod_a_tr, reward=y_train)
    dm_est = dm.estimate(context=x_test,
                         prod_r_te=prod_r_te,
                         prod_a_te=prod_a_te,
                         targ_a_te=targ_a_te,
                         prod_score_te=prod_score_te,
                         targ_score_te=targ_score_te)
    ground_truth = 1 - np.mean(targ_a_te == np.argmax(y_test, axis=-1))
    print("[DM] RMSE: {}".format(rmse(a=np.mean(dm_est), b=ground_truth)))

    bool_mat = np.asarray(prod_a_te == targ_a_te).astype(np.float32)
    # ips_est = prod_r_te * (bool_mat / twoD_gather(prod_score_te, prod_a_te))
    # ips_est = prod_r_te * (bool_mat / twoD_gather(prod_score_te, targ_a_te))
    imp_weight = (twoD_gather(targ_score_te, targ_a_te) / twoD_gather(prod_score_te, targ_a_te))
    ips_est = prod_r_te * (bool_mat * imp_weight)
    ips_est = np.mean(ips_est)
    print("[IPS] RMSE: {}".format(rmse(a=np.mean(ips_est), b=ground_truth)))

    r = (prod_r_te - dm_est)
    # ips_est = r * (bool_mat / twoD_gather(prod_score_te, prod_a_te))
    # ips_est = r * (bool_mat / twoD_gather(prod_score_te, targ_a_te))
    imp_weight = (twoD_gather(targ_score_te, targ_a_te) / twoD_gather(prod_score_te, targ_a_te))
    ips_est = r * (bool_mat * imp_weight)
    dr_est = ips_est + dm_est
    print("[DR] RMSE: {}".format(rmse(a=np.mean(dr_est), b=ground_truth)))