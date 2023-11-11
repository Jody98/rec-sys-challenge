import time

import numpy as np
import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from recommenders.collaborative_filtering_recommender import ItemKNNCFRecommender
from utils.functions import read_data, evaluate_algorithm, generate_submission_csv
from Recommenders.SLIM import SLIM_BPR_Python


def __main__():
    URM_all_dataframe, users_list = read_data()

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    recommender = ItemKNNCFRecommender(URM_train)
    # il miglior fit è dato da topK=10, shrink=10.0 e similarity='cosine' (default)
    recommender.fit(shrink=10.0, topK=10, similarity='cosine')

    recommendations = []

    for user_id in users_list:
        recommendation = recommender.recommend(user_id, at=10)[0]
        recommendations.append(recommendation)

    generate_submission_csv("output_files/IBCF_submission.csv", recommendations)

    evaluate_algorithm(URM_test, recommender, at=10)

    recommender = SLIM_BPR_Python.SLIM_BPR_Python(URM_train)
    recommender.fit()

    recommendations = []

    for user_id in users_list:
        recommendation = recommender.recommend(user_id, at=10)[1]
        recommendations.append(recommendation)
        print(user_id)

    generate_submission_csv("output_files/SLIM_BPR_Python_submission.csv", recommendations)


if __name__ == '__main__':
    __main__()
