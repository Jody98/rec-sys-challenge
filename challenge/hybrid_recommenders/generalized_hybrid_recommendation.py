import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.Hybrid.GeneralizedLinearHybridRecommender import GeneralizedLinearHybridRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.Hybrid.HybridDifferentLoss import DifferentLossScoresHybridRecommender
from Recommenders.MatrixFactorization import IALSRecommender
from Recommenders.SLIM import SLIMElasticNetRecommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    ials_recommender = IALSRecommender.IALSRecommender(URM_train)
    ials_recommender.fit(epochs=100, num_factors=92, confidence_scaling="linear", alpha=2.5431444656816597,
                         epsilon=0.035779451402656745,
                         reg=1.5, init_mean=0.0, init_std=0.1)

    results, _ = evaluator.evaluateRecommender(ials_recommender)
    print("IALSRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item_recommender.fit(topK=10, shrink=19, similarity='jaccard', normalize=False,
                         feature_weighting="TF-IDF")
    item_Wsparse = item_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    EASE_recommender = EASE_R_Recommender.EASE_R_Recommender(URM_train)
    EASE_recommender.fit(topK=10, l2_norm=101, normalize_matrix=False)
    EASE_R_Wsparse = sps.csr_matrix(EASE_recommender.W_sparse)

    results, _ = evaluator.evaluateRecommender(EASE_recommender)
    print("EASE_R_Recommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    P3_recommender = P3alphaRecommender.P3alphaRecommender(URM_train)
    P3_recommender.fit(topK=64, alpha=0.35496275558011753, min_rating=0.1, implicit=True,
                       normalize_similarity=True)
    P3_Wsparse = P3_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(P3_recommender)
    print("P3alphaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)
    RP3_Wsparse = RP3_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(RP3_recommender)
    print("RP3betaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIM_recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train)
    SLIM_recommender.fit(l1_ratio=0.005997129498003861, alpha=0.004503120402472539,
                         positive_only=True, topK=45)
    SLIM_Wsparse = SLIM_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIM_recommender)
    print("SLIMElasticNetRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIMRP3 = DifferentLossScoresHybridRecommender(URM_train, RP3_recommender, SLIM_recommender)
    SLIMRP3.fit(norm=2, alpha=0.4304989217384739)

    results, _ = evaluator.evaluateRecommender(SLIMRP3)
    print("SLIMRP3")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommenders = [ials_recommender, item_recommender, EASE_recommender, P3_recommender, SLIMRP3]
    alpha = 0.222222
    beta = 1.8161240086943757
    gamma = 1.7842468494380264
    delta = 2.8553272775373806
    epsilon = 3.5

    recommender_object = GeneralizedLinearHybridRecommender(URM_train, recommenders=recommenders)
    recommender_object.fit(alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon)

    recommended_items = recommender_object.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/HybridSubmission.csv", recommendations)

    results, _ = evaluator.evaluateRecommender(recommender_object)
    print("GeneralizedLinearHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
