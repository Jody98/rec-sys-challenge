from itertools import product

import numpy as np
import scipy.sparse as sps
from tqdm import tqdm

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender
from challenge.utils.functions import read_data, generate_submission_csv


def custom_tail_boost(URM, users_list, items_list, step=1, lastN=40):
    for user_index in tqdm(range(URM.shape[0])):
        if user_index in users_list:
            sorted_items = get_sorted_items_for_user(users_list, user_index, items_list)
            user_items = URM[user_index].indices
            len_items = len(user_items)

            for i in range(len_items):
                index_of_item, = np.where(sorted_items == user_items[i])
                if len_items - index_of_item <= lastN:
                    additive_score = ((lastN + 1) - (len_items - index_of_item)) * step
                    URM.data[URM.indptr[user_index] + i] += additive_score

    return URM


def get_sorted_items_for_user(users_list, user_id, items_list):
    index_list = np.where(users_list == user_id)
    return items_list[index_list]


def fine_tune_parameters(URM, users_list, items_list, parameter_grid):
    best_map = 0.0
    best_parameters = {}

    for parameters in product(*parameter_grid.values()):
        step, lastN = parameters
        URM_boosted = custom_tail_boost(URM.copy(), users_list, items_list, step=step, lastN=lastN)

        URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_boosted, train_percentage=0.80)

        recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
        recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)

        evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
        results, _ = evaluator.evaluateRecommender(recommender)

        map_10 = results.loc[10]["MAP"]

        print("Parameters: {}, MAP: {}".format(parameters, map_10))

        if map_10 > best_map:
            best_map = map_10
            best_parameters = {'step': step, 'lastN': lastN}

    return best_parameters


def __main__():
    cutoff_list = [10]
    alpha = 0.05
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    items_list = URM_all_dataframe['ItemID'].values

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    # Define the parameter grid for fine-tuning
    parameter_grid = {'step': [0.1, 0.5, 1, 2, 5, 10], 'lastN': [3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

    # Fine-tune the parameters
    best_params = fine_tune_parameters(URM_all, users_list, items_list, parameter_grid)
    print("Best Parameters:", best_params)

    # Use the best parameters to boost URM
    URM_boosted = custom_tail_boost(URM_all.copy(), users_list, items_list, step=best_params['step'],
                                    lastN=best_params['lastN'])

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_boosted, train_percentage=0.80)

    recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                    implicit=True, normalize_similarity=True)

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/SLIMElasticNetSubmission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                    implicit=True, normalize_similarity=True)

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/SLIMElasticNetSubmission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
