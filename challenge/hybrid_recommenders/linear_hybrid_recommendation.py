import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.Hybrid.HybridLinear import HybridLinear
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNSimilarityTripleHybridRecommender import  ItemKNNSimilarityTripleHybridRecommender
from Recommenders.SLIM import SLIMElasticNetRecommender
from Recommenders.MatrixFactorization import IALSRecommender, PureSVDRecommender
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

    ials_recommender = IALSRecommender.IALSRecommender(URM_train)
    ials_recommender.fit(epochs=100, num_factors=92, confidence_scaling="linear", alpha=2.5431444656816597,
                         epsilon=0.035779451402656745,
                         reg=1.5, init_mean=0.0, init_std=0.1)

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item_recommender.fit(topK=9, shrink=13, similarity='tversky', tversky_alpha=0.03642489209084876,
                         tversky_beta=0.9961018325655608)
    item_Wsparse = item_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    EASE_R = EASE_R_Recommender.EASE_R_Recommender(URM_train)
    EASE_R.fit(topK=59, l2_norm=29.792347118106623, normalize_matrix=False)
    EASE_R_Wsparse = sps.csr_matrix(EASE_R.W_sparse)

    results, _ = evaluator.evaluateRecommender(EASE_R)
    print("EASE_R_Recommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    P3_recommender = P3alphaRecommender.P3alphaRecommender(URM_train)
    P3_recommender.fit(topK=40, alpha=0.3119217553589628, min_rating=0.01, implicit=True, normalize_similarity=True)
    p3alpha_Wsparse = P3_recommender.W_sparse

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
    SLIM_recommender.fit(topK=216, l1_ratio=0.0032465600313226354, alpha=0.002589066655986645, positive_only=True)
    SLIM_Wsparse = SLIM_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIM_recommender)
    print("SLIMElasticNetRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    hybrid_recommender = ItemKNNSimilarityTripleHybridRecommender(URM_train, p3alpha_Wsparse, item_Wsparse, RP3_Wsparse)
    hybrid_recommender.fit(topK=225, alpha=0.4976629488640914, beta=0.13017801200221196)

    results, _ = evaluator.evaluateRecommender(hybrid_recommender)
    print("ItemKNNSimilarityTripleHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    pureSVDitem = PureSVDRecommender.PureSVDItemRecommender(URM_train)
    pureSVDitem.fit(num_factors=145, topK=28)

    results, _ = evaluator.evaluateRecommender(pureSVDitem)
    print("PureSVDItem MAP: {}".format(results.loc[10]["MAP"]))

    recommenders = {
        "iALS": ials_recommender,
        "PureSVD": pureSVDitem,
        "EASE_R": EASE_R,
        "Hybrid": hybrid_recommender,
        "SLIM": SLIM_recommender
    }

    all_recommender = HybridLinear(URM_train, recommenders)
    all_recommender.fit(iALS=0.9674103723952641, PureSVD=0.9885478633045848,
                        EASE_R=0.02470494636545416, Hybrid=1.0, SLIM=1.9018687827525076)

    results, _ = evaluator.evaluateRecommender(all_recommender)
    print("HybridLinear")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommended_items = all_recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/LinearHybridSubmission.csv", recommendations)


if __name__ == '__main__':
    __main__()
