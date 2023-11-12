import time

import numpy as np
import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from recommenders.collaborative_filtering_recommender import ItemKNNCFRecommender
from utils.functions import read_data, evaluate_algorithm, generate_submission_csv
from Recommenders.SLIM import SLIM_BPR_Python


def __main__():
    data_file_path = 'input_files/data_train.csv'
    users_file_path = 'input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    topk = [5, 10, 20, 50, 100, 200]
    shrink = [0, 10, 20, 50, 100, 200, 500]

    for k in topk:
        for s in shrink:
            recommender = ItemKNNCFRecommender(URM_train)
            recommender.fit(shrink=s, topK=k, similarity='cosine')

            recommendations = []

            for user_id in users_list:
                recommendation = recommender.recommend(user_id, at=10)[0]
                recommendations.append(recommendation)

            generate_submission_csv("output_files/IBCF_submission.csv", recommendations)
            print("k: {}, s: {}".format(k, s))
            evaluate_algorithm(URM_test, recommender, at=10)


if __name__ == '__main__':
    __main__()
