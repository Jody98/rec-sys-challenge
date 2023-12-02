import scipy.sparse as sps

from Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.Hybrid.GeneralizedLinearHybridRecommender import GeneralizedLinearHybridRecommender
from Recommenders.KNN import ItemKNNCFRecommender
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

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item_recommender.fit(topK=10, shrink=19, similarity='jaccard', normalize=False,
                         feature_weighting="TF-IDF")
    item_Wsparse = item_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender = EASE_R_Recommender.EASE_R_Recommender(URM_train)
    recommender.fit(topK=10, l2_norm=101, normalize_matrix=False)
    EASE_R_Wsparse = sps.csr_matrix(recommender.W_sparse)

    results, _ = evaluator.evaluateRecommender(recommender)
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

    SLIMRP3 = ItemKNNSimilarityHybridRecommender(URM_train, RP3_Wsparse, SLIM_Wsparse)
    SLIMRP3.fit(alpha=0.5153665793050106, topK=48)
    SLIMRP3_Wsparse = SLIMRP3.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIMRP3)
    print("SLIMRP3")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIMP3 = ItemKNNSimilarityHybridRecommender(URM_train, P3_Wsparse, SLIM_Wsparse)
    SLIMP3.fit(alpha=0.5153665793050106, topK=48)  # finetunare alpha e topK
    SLIMP3_Wsparse = SLIMP3.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIMP3)
    print("SLIMP3")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIMitem = ItemKNNSimilarityHybridRecommender(URM_train, item_Wsparse, SLIM_Wsparse)
    SLIMitem.fit(alpha=0.5153665793050106, topK=48)  # finetunare alpha e topK
    SLIMitem_Wsparse = SLIMitem.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIMitem)
    print("SLIMitem")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIMEASE_R = ItemKNNSimilarityHybridRecommender(URM_train, EASE_R_Wsparse, SLIM_Wsparse)
    SLIMEASE_R.fit(alpha=0.5153665793050106, topK=48)  # finetunare alpha e topK
    SLIMEASE_R_Wsparse = SLIMEASE_R.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIMEASE_R)
    print("SLIMEASE_R")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIMRP3P3 = ItemKNNSimilarityHybridRecommender(URM_train, RP3_Wsparse, SLIMP3_Wsparse)
    SLIMRP3P3.fit(alpha=0.2153665793050106, topK=48)  # finetunare alpha e topK
    SLIMRP3P3_Wsparse = SLIMRP3P3.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIMRP3P3)
    print("SLIMRP3P3")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIMRP3item = ItemKNNSimilarityHybridRecommender(URM_train, RP3_Wsparse, SLIMitem_Wsparse)
    SLIMRP3item.fit(alpha=0.2153665793050106, topK=48)  # finetunare alpha e topK
    SLIMRP3item_Wsparse = SLIMRP3item.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIMRP3item)
    print("SLIMRP3item")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIMRP3EASE_R = ItemKNNSimilarityHybridRecommender(URM_train, RP3_Wsparse, SLIMEASE_R_Wsparse)
    SLIMRP3EASE_R.fit(alpha=0.2153665793050106, topK=48)  # finetunare alpha e topK
    SLIMRP3EASE_R_Wsparse = SLIMRP3EASE_R.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIMRP3EASE_R)
    print("SLIMRP3EASE_R")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIMP3item = ItemKNNSimilarityHybridRecommender(URM_train, P3_Wsparse, SLIMitem_Wsparse)
    SLIMP3item.fit(alpha=0.2153665793050106, topK=48)  # finetunare alpha e topK
    SLIMP3item_Wsparse = SLIMP3item.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIMP3item)
    print("SLIMP3item")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommenders = [item_recommender, P3_recommender, RP3_recommender, SLIM_recommender]
    alphas = [0.25, 0.5, 0.75, 1.5]

    recommender_object = GeneralizedLinearHybridRecommender(URM_train, recommenders=recommenders)
    recommender_object.fit(alphas=alphas)

    recommended_items = recommender_object.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/HybridSubmission.csv", recommendations)

    results, _ = evaluator.evaluateRecommender(recommender_object)
    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
