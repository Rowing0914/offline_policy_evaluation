import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV

from data.data_manager import load_ecoli
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
        feedback = twoD_gather(reward, np.argmax(action, axis=-1))

        if self._model_type == 'ridge':
            self._clf = RidgeCV(alphas=self._alpha, fit_intercept=False, cv=5)
        elif self._model_type == 'lasso':
            self._clf = LassoCV(alphas=self._alpha, tol=1e-3, cv=5, fit_intercept=False)
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
        bool_mat = np.asarray(np.argmax(prod_a_te, axis=-1) == np.argmax(targ_a_te, axis=-1)).astype(np.float32)

        # take the score only for the taken action
        targ_score = twoD_gather(targ_score_te, np.argmax(targ_a_te, axis=-1))
        prod_score = twoD_gather(prod_score_te, np.argmax(targ_a_te, axis=-1))

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
    def __init__(self, ips_estimator: InversePropensityScore, dm_estimator: DirectMethod):
        """ Doubly Robust Estimator(DR) by Miroslav Dudik et al., 2011 """
        super(DoublyRobustEstimator, self).__init__()
        self.ips_estimator = ips_estimator
        self.dm_estimator = dm_estimator

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
    # load a dataset
    data = load_ecoli()

    # define a prod and a targ policies
    prod_policy = lambda x: np.eye(data.num_label)[int(np.random.uniform(low=0, high=data.num_label - 1))]
    targ_policy = lambda x: np.eye(data.num_label)[int(np.random.uniform(low=0, high=data.num_label - 1))]

    # get dummy actions
    a = np.asarray([prod_policy(row) for row in data.x])

    # test the estimator
    dm = DirectMethod(x=data.x, a=a, r=data.y, model_type="ridge")
    sample_x = data.x[0]
    sample_y = targ_policy(x=sample_x[:, None])
    reward_est = dm.estimate(context=sample_x[None, :])
    print("R Est: {}, R True: {}".format(reward_est, data.y[0][np.argmax(sample_y)]))
