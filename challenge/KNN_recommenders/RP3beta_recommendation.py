import numpy as np
import scipy.sparse as sps
from tqdm import tqdm

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender
from challenge.utils.functions import read_data, generate_submission_csv, evaluate_algorithm


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


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz("../input_files/URM_train_plus_validation.npz")
    URM_test = sps.load_npz("../input_files/URM_test.npz")

    topK = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 750, 1000]

    for topk in topK:
        recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
        recommender.fit(topK=topk, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)

        recommended_items = recommender.recommend(users_list, cutoff=10)
        recommendations = []
        for i in zip(users_list, recommended_items):
            recommendation = {"user_id": i[0], "item_list": i[1]}
            recommendations.append(recommendation)

        generate_submission_csv("../output_files/RP3betaSubmission.csv", recommendations)

        evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
        results, _ = evaluator.evaluateRecommender(recommender)

        print("TopK: {}".format(topk))
        print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
