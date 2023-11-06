import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import scipy.sparse as sps

from utils.functions import evaluate_algorithm, generate_submission_csv, read_data, split
from recommenders.random_recommender import RandomRecommender
from recommenders.top_pop_recommender import TopPopRecommender
from recommenders.pure_SVD_recommender import SVDBasedRecommender


def __main__():
    URM_all_dataframe, users_list = read_data()

    print("Number of items\t {}, Number of users\t {}".format(len(URM_all_dataframe['ItemID'].unique()),
                                                              len(URM_all_dataframe['UserID'].unique())))
    print("Number of interactions\t {}".format(URM_all_dataframe['Data'].count()))
    print("Max value of UserID\t {}, Max value of ItemID\t {}".format(URM_all_dataframe['UserID'].max(),
                                                                      URM_all_dataframe['ItemID'].max()))

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train = sps.load_npz("input_files/URM_train.npz")
    URM_test = sps.load_npz("input_files/URM_test.npz")

    top_pop_recommender = TopPopRecommender()
    top_pop_recommender.fit(URM_train)

    recommendations = []

    for user_id in users_list:
        recommendation = top_pop_recommender.recommend(user_id, at=10)[0]
        recommendations.append(recommendation)

    generate_submission_csv("output_files/top_pop_submission.csv", recommendations)

    evaluate_algorithm(URM_test, top_pop_recommender, at=10)

    svd_recommender = SVDBasedRecommender()
    svd_recommender.fit(URM_train)

    recommendations = []

    for user_id in users_list:
        recommendation = svd_recommender.recommend(user_id, at=10)[0]
        recommendations.append(recommendation)

    generate_submission_csv("output_files/svd_submission.csv", recommendations)

    evaluate_algorithm(URM_test, svd_recommender, at=10)


if __name__ == '__main__':
    __main__()
