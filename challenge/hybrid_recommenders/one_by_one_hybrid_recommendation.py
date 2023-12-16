import numpy as np
import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.Hybrid.HybridDifferentLoss import DifferentLossScoresHybridRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNSimilarityTripleHybridRecommender import ItemKNNSimilarityTripleHybridRecommender
from Recommenders.EASE_R import EASE_R_Recommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    folder_path = "../result_experiments/"
    filename80 = "EASE_R_Recommender_best_model80.zip"
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    p3alpha = P3alphaRecommender.P3alphaRecommender(URM_train)
    p3alpha.fit(topK=40, alpha=0.3119217553589628, min_rating=0.01, implicit=True,
                normalize_similarity=True)
    p3alpha_Wsparse = p3alpha.W_sparse

    results, _ = evaluator.evaluateRecommender(p3alpha)
    print("P3alphaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item_recommender.fit(topK=9, shrink=13, similarity='tversky', tversky_alpha=0.036424892090848766,
                         tversky_beta=0.9961018325655608)
    item_Wsparse = item_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)
    RP3_Wsparse = RP3_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(RP3_recommender)
    print("RP3betaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    hybrid_recommender = ItemKNNSimilarityTripleHybridRecommender(URM_train, p3alpha_Wsparse, item_Wsparse, RP3_Wsparse)
    hybrid_recommender.fit(topK=225, alpha=0.4976629488640914, beta=0.13017801200221196)

    results, _ = evaluator.evaluateRecommender(hybrid_recommender)
    print("ItemKNNSimilarityTripleHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    EASE_R = EASE_R_Recommender.EASE_R_Recommender(URM_train)
    EASE_R.load_model(folder_path, filename80)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(EASE_R)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender = DifferentLossScoresHybridRecommender(URM_train, hybrid_recommender, EASE_R)
    recommender.fit(norm=1, alpha=0.70918052775199)

    results, _ = evaluator.evaluateRecommender(recommender)
    print("DifferentLossScoresHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/OneByOneSubmission.csv", recommendations)


if __name__ == '__main__':
    __main__()
