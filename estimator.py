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


class DM(BaseEstimator):
    def __init__(self, model_type="ridge"):
        """ Direct Method(Reward Prediction)

        :param model_type: type of base algorithm to deal with contexts
        """
        super(DM, self).__init__()
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
        """ This is the part described by the DR paper(Sec 2.1) as follows
             > A problem with this method is that the estimate is formed without the knowledge of a policy
        """
        self._clf.fit(context, feedback)

    def estimate(self, context=None, prod_r_te=None, prod_a_te=None,
                 targ_a_te=None, prod_score_te=None, targ_score_te=None):
        """ Estimate a reward given a context
            
            This is the part described by the DR paper(Sec 2.1) as follows
             > A problem with this method is that the estimate is formed without the knowledge of a policy
        """
        reward_est = self._clf.predict(X=context)
        return reward_est.flatten()


class IPS(BaseEstimator):
    def __init__(self, _min=0, _max=10, if_cap=False, if_normalise=False, if_pointwise=False):
        """ Inverse Propensity Score (IPS) / Importance Sampling(IS)

        Variants are implemented as follows
        1. Vanilla IPS(Horvitz & Thompson, 1952): cap=None, if_normalise=False
        2. Capped IPS(Léon Bottou and Jonas Peters. 2013): cap=scalar_value, if_normalise=False
        3. Normalised IPS(A. Swaminathan & T. Joachims., 2016): cap=None, if_normalise=True
        4. Normalised Capped IPS: cap=scalar_value, if_normalise=True
        4. Pointwise IPS: if_pointwise=True

        :param _min: CIPS lower bound
        :param _max: CIPS upper bound
        :param if_cap: flag if use CIPS
        :param if_normalise: flag if use NIPS, NCIPS
        :param if_pointwise: flag if use Pointwise IPS
        """
        super(IPS, self).__init__()
        self._if_cap = if_cap
        self._if_normalise = if_normalise
        self._if_pointwise = if_pointwise
        self._min = _min
        self._max = _max

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
        self.imp_weight = targ_score / prod_score

        if self._if_cap:
            # See Sec4.2 -> https://arxiv.org/pdf/1801.07030.pdf
            self.imp_weight = np.clip(self.imp_weight, a_min=self._min, a_max=self._max)

        # replace the infinity with the extremely small value
        self.imp_weight[self.imp_weight == np.inf] = np.spacing(1)
        ips = bool_mat * self.imp_weight

        # replace the infinity with the extremely small value
        ips[ips == np.inf] = np.spacing(1)

        if self._if_normalise:
            # self normalised IPS

            if self._if_pointwise:
                """ Midzuno-Sen Rejection Sampling Method

                    Under this system of selection of probabilities, the unit in the first draw is selected with
                    unequal probabilities of selection and remaining all the units are selected 
                    with simple random sampling without replacement at all subsequent draws.

                    [Ref]
                        Midzuno, H. (1951). On the sampling system with probability proportional
                        to sum of sizes. Ann. Inst. Stat. Math., 3:99–107.
                """
                # 1. Only first unit is selected with unequal probability
                dummy_imp_weight = self.imp_weight.copy()
                u = np.random.uniform(low=0.0, high=1.0)  # TODO: I guess we should use max of imp_weight!
                for _id, x in enumerate(dummy_imp_weight):
                    if u < np.mean(x):
                        first_unit = x
                        break

                # 2. For remaining units, we use Simple Random Sampling
                dummy_imp_weight = dummy_imp_weight[_id:]
                size = dummy_imp_weight.shape[0]
                mask = np.random.binomial(1, p=1 / size, size=size)
                samples = [first_unit] + dummy_imp_weight[mask].tolist()
                norm = np.mean(samples)
            else:
                norm = np.mean(self.imp_weight, axis=0)
        else:
            norm = np.ones(self.imp_weight.shape[-1]).astype(np.float32)

        # estimate the feedback based on the importance sampling
        est = (self.imp_weight * prod_r_te) / norm
        return est


class DR(BaseEstimator):
    def __init__(self, ips, dm, switch_tau=0.23, switch_flg="", cab_coeff=None, cab_flg=""):
        """ Doubly Robust Estimator(DR)

        Variants are implemented as follows
        1. Doubly Robust Estimator(DR)(Miroslav Dudik et al., 2011)
        2. SWITCH(YX Wang et al., ‎2016)
        3. CAB[Continuous Adaptive Blending](Y Su et al., ‎2019)

        :param ips: Inverse Propensity Scoring
        :param dm: Direct Method
        :param switch_tau: threshold for SWITCH, pls refer to the paper for more detail!
        :param switch_flg: SWITCH-DR or SWITCH-IPS. (Supported Options); 'ips' or 'dr'
        :param cab_coeff: CAB's coeff, pls refer to the paper for more detail!
        :param cab_flg: CAB or CAB-DR. (Supported Options); '' or 'dr'
        """
        super(DR, self).__init__()
        self.ips = ips
        self.dm = dm
        self._switch_tau = switch_tau
        self._switch_flg = switch_flg.lower()
        self._cab_coeff = cab_coeff
        self._cab_flg = cab_flg.lower()

    def train(self, context=None, action=None, reward=None):
        """ Train the model parameters given contexts and taken actions by the prod policy """
        self.dm.train(context=context, action=action, reward=reward)

    def estimate(self, context=None, prod_r_te=None, prod_a_te=None,
                 targ_a_te=None, prod_score_te=None, targ_score_te=None):
        """ Estimate a reward using the inverse propensity score and Direct Method"""
        dm_est = self.dm.estimate(context=context)
        reward_adv = prod_r_te - dm_est  # subtract the baseline to get the advantage
        ips_est = self.ips.estimate(prod_r_te=reward_adv,
                                    prod_a_te=prod_a_te,
                                    targ_a_te=targ_a_te,
                                    prod_score_te=prod_score_te,
                                    targ_score_te=targ_score_te)

        if self._cab_coeff is not None:
            """ See Sec 3 and Sec 3.2 of CAB(Y Su et al., 2019)

                This method is to adaptively blend(differentiable) DM and IPS,
                whereas in SWITCH we employ the hard switching mechanism which is not differentiable.
            """
            inv_imp_weight = (1 / self.ips.imp_weight)
            if self._cab_flg == "":
                ips_coeff = np.minimum(inv_imp_weight * self._cab_coeff, 1)
                dm_coeff = 1 - ips_coeff
                dr_est = ips_coeff * ips_est + dm_coeff * dm_est
                return dr_est
            elif self._cab_flg == "dr":
                ips_coeff = dr_coeff = np.minimum(inv_imp_weight * self._cab_coeff, 1)
                reward_adv = prod_r_te - dm_est * dr_coeff
                ips_est = self.ips.estimate(prod_r_te=reward_adv,
                                            prod_a_te=prod_a_te,
                                            targ_a_te=targ_a_te,
                                            prod_score_te=prod_score_te,
                                            targ_score_te=targ_score_te)
                dr_est = ips_coeff * ips_est + dm_est
                return dr_est

        # Define the Vanilla DR estimate
        dr_est = ips_est + dm_est

        if self._switch_tau is not None:
            """ See Sec 4.1 of SWITCH(YX Wang et al., ‎2016)

                When importance weights are small, we continue to use IPS, but when it's large,
                then switch to directly applying the (potentially biased) reward model(DM) on actions.
                Here, `small` and `large` are defined via a threshold parameter
            """
            switch_mask = self.ips.imp_weight <= self._switch_tau
            dr_est[~switch_mask] = dm_est[~switch_mask]
            if self._switch_flg == "ips":
                dr_est[switch_mask] = ips_est[switch_mask]
            elif self._switch_flg == "dr":
                # For clarity of the logic, this part is being preserved
                pass
        return dr_est


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
    dm = DM(model_type="ridge")
    dm.train(context=x_train, action=prod_a_tr, reward=y_train)
    dm_est = dm.estimate(context=x_test)
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

    ips = IPS(if_cap=True, _min=0, _max=10, if_normalise=True, if_pointwise=True)
    ips.estimate(context=x_test,
                 prod_r_te=prod_r_te,
                 prod_a_te=prod_a_te,
                 targ_a_te=targ_a_te,
                 prod_score_te=prod_score_te,
                 targ_score_te=targ_score_te)
