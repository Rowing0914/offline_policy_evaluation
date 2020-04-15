import numpy as np
from data.data_manager import load_ecoli
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from utils import twoD_gather


def get_ground_truth(data):
    """ See 5.1.2. POLICY EVALUATION """
    # 1. We randomly split data into training and test sets of (roughly) the same size;
    x_train, x_teste, y_train, y_test = train_test_split(data.x, data.y, test_size=0.5)

    # 2. Train the policy
    model = LogisticRegression(fit_intercept=False, multi_class="multinomial")
    model.fit(x_train, y_train)

    # 3. Get the ground truth on the fully revealed test data
    baseline = model.score(x_teste, y_test)
    return baseline


def get_partially_labelled_data(data):
    """ To construct a partially labelled dataset, exactly one loss
        component for each example is observed, following the approach
        of Beygelzimer & Langford (2009).
    """
    # Uniformly sample the random action for each sample
    random_action = np.random.randint(low=0, high=data.num_label - 1, size=data.x.shape[0])
    masked_y = twoD_gather(array=data.y_onehot, indices=random_action)
    return masked_y


if __name__ == '__main__':
    data = load_ecoli()
    np.random.seed(1)
    # baseline = get_ground_truth(data=data)
    # print(baseline)
    masked_y = get_partially_labelled_data(data=data)
    print(masked_y.shape)
