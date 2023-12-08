import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.model_selection import train_test_split

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.KNN import ItemKNNCFRecommender
from challenge.utils.functions import read_data, evaluate_algorithm, dataset_splits, write_submission, \
    preprocess_data, prepare_submission, generate_submission_csv


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz("../input_files/URM_train_plus_validation.npz")
    URM_test = sps.load_npz("../input_files/URM_test.npz")

    topk = [10, 50, 100, 200, 500, 1000]

    for k in topk:
        recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
        recommender.fit(topK=k, shrink=19, similarity='jaccard', normalize=False, feature_weighting="TF-IDF")

        recommended_items = recommender.recommend(users_list, cutoff=10)
        recommendations = []
        for i in zip(users_list, recommended_items):
            recommendation = {"user_id": i[0], "item_list": i[1]}
            recommendations.append(recommendation)

        generate_submission_csv("../output_files/P3alphaSubmission.csv", recommendations)

        evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
        results, _ = evaluator.evaluateRecommender(recommender)

        print("TopK: {}".format(k))
        print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
