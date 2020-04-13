""" Mix and Rank Framework: Multi Criteria Decision Analysis

- Ref: https://ieeexplore.ieee.org/document/9006199
"""

import numpy as np
from sklearn.preprocessing import minmax_scale


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def calc_distance(model_1, model_2):
    """ Distance function: See eq(1) in the paper """
    return model_1 - model_2


def preference_fn_factory(_type="usual"):
    """ Preference Computation: See eq(2) in the paper

    - Variants of Preference functions being implemented
    Ref: https://en.wikipedia.org/wiki/Preference_ranking_organization_method_for_enrichment_evaluation
    """
    if _type == "usual":
        def fn(model_1, model_2, distance):
            return 1.0 if distance > 0.0 else 0.0
    elif _type == "u-shape":
        def fn(model_1, model_2, distance):
            return 1.0 if np.abs(distance) > model_1 else 0.0
    elif _type == "v-shape":
        def fn(model_1, model_2, distance):
            return 1.0 if np.abs(distance) > model_2 else np.abs(distance) / model_2
    elif _type == "level":
        def fn(model_1, model_2, distance):
            if np.abs(distance) <= model_1:
                return 0.0
            elif model_2 >= np.abs(distance) > model_1:
                return 1 / 2
            else:
                return 1.0
    elif _type == "linear":
        def fn(model_1, model_2, distance):
            if np.abs(distance) <= model_1:
                return 0.0
            elif model_2 >= np.abs(distance) > model_1:
                return (np.abs(distance) - model_1) / (model_2 - model_1)
            else:
                return 1.0
    else:
        raise ValueError
    return fn


def mix_and_rank(decision_matrix, preference_fn):
    """ Mix And Rank Framework, B.Paudel et al., 2019
        Ref: https://ieeexplore.ieee.org/document/9006199

    :param decision_matrix: num_model x num_metric
    :param preference_fn: fn to compute the preference
    """
    num_model, num_metric = decision_matrix.shape

    # To prepare the weighting function, we follow Sec.3 ~ A. Entropy-based weighting ~
    # Normalise the decision matrix: See eq(6) & eq(7)
    decision_matrix_norm = minmax_scale(X=decision_matrix, feature_range=(0, 1))
    D = np.sum(decision_matrix_norm, axis=0)  # See eq(8)
    K = 1 / (np.exp(0.5) - num_model)  # See eq(10)
    W = lambda x: x * np.exp(1 - x) + (1 - x) * np.exp(x) - 1 # See eq(11)
    entropy = K * np.sum(W(decision_matrix_norm / D), axis=0)  # See eq(9)

    # Compute the weights: See eq(12)
    E = entropy.sum()
    numerator = (1 / (num_model - E)) * (1 - entropy)
    denominator = numerator.sum()
    weight_table = numerator / denominator

    # Prepare the preference/distance tables for pairwise comparison of all combinations of models
    model_ids = np.arange(num_model)
    combinations = [(model_1, model_2) for model_1 in model_ids for model_2 in model_ids]
    pref_table = np.zeros((len(combinations), num_metric))
    dist_table = np.zeros((len(combinations), num_metric))

    # 1. Pairwise comparison of models along with each metric
    for metric_id in range(num_metric - 1):
        for (model_id_1, model_id_2) in combinations:
            model_1, model_2 = decision_matrix[model_id_1, metric_id], decision_matrix[model_id_2, metric_id]
            dist = calc_distance(model_1=model_1, model_2=model_2)
            pref = preference_fn(model_1=model_1, model_2=model_2, distance=dist)
            pref_table[model_id_1, metric_id] = pref
            dist_table[model_id_1, metric_id] = dist

    # 2. Weighted average of each metric along with models
    scores = list()
    for model_pair_id in range(pref_table.shape[0]):
        score = 0
        for metric_id in range(pref_table.shape[1] - 1):
            score += weight_table[metric_id] * pref_table[model_pair_id, metric_id]
        scores.append(score / pref_table.shape[1])

    # 3. Aggregate the weighted average over all models
    # A list to a square matrix conversion: shape -> (num_model x num_model)
    model_comparison_table = np.asarray(np.split(np.asarray(scores), num_model))
    _row, _col = model_comparison_table.shape
    assert _row == _col

    # Get the Lower/Upper triangle of the model comparison square matrix
    # See eq(3), eq(4) as well
    net_positive, net_negative = np.tril(model_comparison_table), np.triu(model_comparison_table)
    net_positive, net_negative = net_positive.mean(axis=0), net_negative.mean(axis=0)
    model_scores = net_positive - net_negative
    model_scores_norm = softmax(model_scores)
    return model_scores, model_scores_norm


def _test_mix_and_rank():
    num_models = 6
    num_decision_matrix = 10

    decision_matrix = np.random.randn(num_models, num_decision_matrix)
    preference_fn = preference_fn_factory(_type="usual")
    print("Decision Matrix: ", decision_matrix.shape)

    model_scores, model_scores_norm = mix_and_rank(decision_matrix=decision_matrix, preference_fn=preference_fn)
    print(model_scores, model_scores_norm)


if __name__ == '__main__':
    _test_mix_and_rank()