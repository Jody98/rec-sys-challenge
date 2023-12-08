import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.Hybrid.HybridLinear import HybridLinear
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from Recommenders.SLIM import SLIMElasticNetRecommender
from Recommenders.MatrixFactorization import IALSRecommender
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

    ials_recommender = P3alphaRecommender.P3alphaRecommender(URM_train)
    ials_recommender.fit(topK=64, alpha=0.35496275558011753, min_rating=0.1, implicit=True,
                         normalize_similarity=True)

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item_recommender.fit(topK=10, shrink=19, similarity='jaccard', normalize=False,
                         feature_weighting="TF-IDF")
    item_Wsparse = item_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    EASE_R_recommender = P3alphaRecommender.P3alphaRecommender(URM_train)
    EASE_R_recommender.fit(topK=64, alpha=0.35496275558011753, min_rating=0.1, implicit=True,
                           normalize_similarity=True)
    EASE_R_Wsparse = sps.csr_matrix(EASE_R_recommender.W_sparse)

    results, _ = evaluator.evaluateRecommender(EASE_R_recommender)
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

    recommenders = {
        "iALS": ials_recommender,
        "ItemKNN": item_recommender,
        "EASE_R": EASE_R_recommender,
        "P3alpha": P3_recommender,
        "RP3beta": RP3_recommender,
        "SLIM": SLIM_recommender
    }

    all_recommender = HybridLinear(URM_train, recommenders)
    all_recommender.fit(iALS=0.0, ItemKNN=0.3677964803954745,
                        EASE_R=0.0, P3alpha=0.7239455833579226,
                        RP3beta=0.4439258978922839, SLIM=0.55520329895492982)

    results, _ = evaluator.evaluateRecommender(all_recommender)
    print("HybridLinear")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    all_recommender = HybridLinear(URM_train, recommenders)
    all_recommender.fit(iALS=0.8063775585626619, ItemKNN=0.32792294569765507,
                        EASE_R=0.2233733817355404, P3alpha=0.9098345552342707,
                        RP3beta=0.342017078321237, SLIM=0.85520329895492982)

    results, _ = evaluator.evaluateRecommender(all_recommender)
    print("HybridLinear")
    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
