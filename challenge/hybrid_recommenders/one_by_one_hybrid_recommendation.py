import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender
from Recommenders.Hybrid.HybridDifferentLoss import DifferentLossScoresHybridRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from Recommenders.SLIM import SLIMElasticNetRecommender
from Recommenders.EASE_R import EASE_R_Recommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_all)
    item_recommender.fit(topK=9, shrink=13, similarity='tversky', tversky_alpha=0.03642489209084876,
                         tversky_beta=0.9961018325655608)
    item_Wsparse = item_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_all)
    RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)
    RP3_Wsparse = RP3_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(RP3_recommender)
    print("RP3betaRecommender MAP: {}".format(results.loc[10]["MAP"]))

    SLIM_recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_all)
    SLIM_recommender.fit(topK=216, l1_ratio=0.0032465600313226354, alpha=0.002589066655986645, positive_only=True)
    SLIM_Wsparse = SLIM_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIM_recommender)
    print("SLIM MAP: {}".format(results.loc[10]["MAP"]))

    SLIMRP3 = ItemKNNSimilarityHybridRecommender(URM_all, RP3_Wsparse, SLIM_Wsparse)
    SLIMRP3.fit(alpha=0.5364079633111103, topK=468)
    SLIMRP3_Wsparse = SLIMRP3.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIMRP3)
    print("SLIMRP3")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender = DifferentLossScoresHybridRecommender(URM_all, item_recommender, SLIMRP3)
    recommender.fit(norm=1, alpha=0.16987693419766395)

    results, _ = evaluator.evaluateRecommender(recommender)
    print("DifferentLossScoresHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    EASE_R = EASE_R_Recommender.EASE_R_Recommender(URM_all)
    EASE_R.fit(topK=32, l2_norm=38, normalize_matrix=False)
    EASE_R_Wsparse = sps.csr_matrix(EASE_R.W_sparse)

    results, _ = evaluator.evaluateRecommender(EASE_R)
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender = DifferentLossScoresHybridRecommender(URM_all, EASE_R, recommender)
    recommender.fit(norm=1, alpha=0.16987693419766395)

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
