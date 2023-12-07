import numpy as np
import scipy.sparse as sps
from tqdm import tqdm

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender
from Recommenders.Hybrid.HybridDifferentLoss import DifferentLossScoresHybridRecommender
from Recommenders.SLIM import SLIMElasticNetRecommender
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

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)
    RP3_Wsparse = RP3_recommender.W_sparse

    SLIM_recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train)
    SLIM_recommender.fit(topK=46, l1_ratio=0.005997129498003861, alpha=0.004503120402472538, positive_only=True)
    SLIM_Wsparse = SLIM_recommender.W_sparse

    recommender = DifferentLossScoresHybridRecommender(URM_train, RP3_recommender, SLIM_recommender)
    recommender.fit(norm=2, alpha=0.4304989217384739)

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/RP3betaSubmission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    evaluate_algorithm(URM_test, recommender, cutoff_list[0])

    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
