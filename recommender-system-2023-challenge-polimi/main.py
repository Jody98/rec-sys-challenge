import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import scipy.sparse as sps

from utils.functions import evaluate_algorithm, generate_submission_csv, read_data
from utils.random_recommender import RandomRecommender
from utils.top_pop_recommender import TopPopRecommender


def __main__():
    URM_all_dataframe, users_list = read_data()

    print("Number of items\t {}, Number of users\t {}".format(len(URM_all_dataframe['ItemID'].unique()),
                                                              len(URM_all_dataframe['UserID'].unique())))
    print("Number of interactions\t {}".format(URM_all_dataframe['Data'].count()))
    print("Max value of UserID\t {}, Max value of ItemID\t {}".format(URM_all_dataframe['UserID'].max(),
                                                                      URM_all_dataframe['ItemID'].max()))

    mapped_id, original_id = pd.factorize(URM_all_dataframe['UserID'].unique())
    user_original_ID_to_index = pd.Series(data=mapped_id, index=original_id)

    mapped_id, original_id = pd.factorize(URM_all_dataframe['ItemID'].unique())
    item_original_ID_to_index = pd.Series(data=mapped_id, index=original_id)

    URM_all_dataframe['UserID'] = URM_all_dataframe['UserID'].map(user_original_ID_to_index)
    URM_all_dataframe['ItemID'] = URM_all_dataframe['ItemID'].map(item_original_ID_to_index)

    userID_unique = URM_all_dataframe['UserID'].unique()
    itemID_unique = URM_all_dataframe['ItemID'].unique()

    n_users = len(userID_unique)
    n_items = len(itemID_unique)
    n_interactions = len(URM_all_dataframe)

    print("Number of items\t {}, Number of users\t {}".format(n_items, n_users))
    print("Max value of UserID\t {}, Max value of ItemID\t {}".format(userID_unique.max(), itemID_unique.max()))
    print("Number of interactions\t {}".format(n_interactions))

    print("Average interactions per user {:.2f}\nAverage interactions per item {:.2f}".format(n_interactions / n_users,
                                                                                              n_interactions / n_items))

    print("Sparsity {:.5f} %".format((1 - n_interactions / (n_users * n_items)) * 100))

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    item_popularity = np.ediff1d(URM_all.tocsc().indptr)
    item_popularity = np.sort(item_popularity)

    ten_percent = int(n_items / 10)

    print("Average per-item interactions over the whole dataset {:.2f}\n".format(item_popularity.mean()))
    print("Average per-item interactions for the top 10% popular items {:.2f}\n".format(
        item_popularity[-ten_percent:].mean()))
    print("Average per-item interactions for the least 10% popular items {:.2f}\n".format(
        item_popularity[:ten_percent].mean()))
    print("Average per-item interactions for the median 10% popular items {:.2f}\n".format(
        item_popularity[int(n_items / 2 - ten_percent / 2):int(n_items / 2 + ten_percent / 2)].mean()))

    user_activity = np.ediff1d(URM_all.tocsr().indptr)
    user_activity = np.sort(user_activity)

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


if __name__ == '__main__':
    __main__()
