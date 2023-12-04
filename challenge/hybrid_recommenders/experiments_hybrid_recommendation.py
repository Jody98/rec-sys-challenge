import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.Hybrid.GeneralizedLinearHybridRecommender import GeneralizedLinearHybridRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from Recommenders.MatrixFactorization import IALSRecommender
from Recommenders.SLIM import SLIMElasticNetRecommender
from challenge.utils.functions import read_data


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    ials_recommender = IALSRecommender.IALSRecommender(URM_train)
    ials_recommender.fit(epochs=100, num_factors=100, confidence_scaling="linear", alpha=0.5,
                         epsilon=0.026681180348966625,
                         reg=0.01, init_mean=0.0, init_std=0.1)

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
    SLIM_recommender.fit(topK=46, l1_ratio=0.005997129498003861, alpha=0.004503120402472538, positive_only=True)
    SLIM_Wsparse = SLIM_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIM_recommender)
    print("SLIMElasticNetRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommenders = [ials_recommender, SLIM_recommender]
    alpha = 0.25
    beta = 0.75

    recommender_object = GeneralizedLinearHybridRecommender(URM_train, recommenders=recommenders)
    recommender_object.fit(alpha=alpha, beta=beta)

    results, _ = evaluator.evaluateRecommender(recommender_object)
    print("SLIMiALS")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIMRP3 = ItemKNNSimilarityHybridRecommender(URM_train, RP3_Wsparse, SLIM_Wsparse)
    SLIMRP3.fit(alpha=0.5153665793050106, topK=48)
    SLIMRP3_Wsparse = SLIMRP3.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIMRP3)
    print("SLIMRP3")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIMRP3iALS = GeneralizedLinearHybridRecommender(URM_train, recommenders=[ials_recommender, SLIMRP3])
    SLIMRP3iALS.fit(alpha=0.25, beta=0.75)

    results, _ = evaluator.evaluateRecommender(SLIMRP3iALS)
    print("SLIMRP3iALS")
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

    SLIMEASE_R = ItemKNNSimilarityHybridRecommender(URM_train, item_Wsparse, SLIM_Wsparse)
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


if __name__ == '__main__':
    __main__()