import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.Hybrid.HybridLinear import HybridLinear
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNSimilarityTripleHybridRecommender import ItemKNNSimilarityTripleHybridRecommender
from Recommenders.MatrixFactorization import ALSRecommender, IALSRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender_PyTorch_OptimizerMask
from Recommenders.SLIM import SLIMElasticNetRecommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    folder_path = "../result_experiments/"
    EASE80 = "EASE_R_Recommender_best_model100.zip"
    SLIM80 = "SLIMElasticNetRecommender_best_model100.zip"
    MultVAE80 = "MultVAERecommender_best_model100.zip"
    ALS80 = "ALSRecommender_best_model100.zip"
    IALS80 = "IALSRecommender_best_model100.zip"
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    evaluator_train = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_all)
    item_recommender.fit(topK=9, shrink=13, similarity='tversky', tversky_alpha=0.03642489209084876,
                         tversky_beta=0.9961018325655608)
    item_Wsparse = item_recommender.W_sparse

    results, _ = evaluator_train.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    P3_recommender = P3alphaRecommender.P3alphaRecommender(URM_all)
    P3_recommender.fit(topK=40, alpha=0.3119217553589628, min_rating=0.01, implicit=True, normalize_similarity=True)
    p3alpha_Wsparse = P3_recommender.W_sparse

    results, _ = evaluator_train.evaluateRecommender(P3_recommender)
    print("P3alphaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_all)
    RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)
    RP3_Wsparse = RP3_recommender.W_sparse

    results, _ = evaluator_train.evaluateRecommender(RP3_recommender)
    print("RP3betaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    hybrid_recommender = ItemKNNSimilarityTripleHybridRecommender(URM_all, p3alpha_Wsparse, item_Wsparse, RP3_Wsparse)
    hybrid_recommender.fit(topK=225, alpha=0.4976629488640914, beta=0.13017801200221196)

    results, _ = evaluator_train.evaluateRecommender(hybrid_recommender)
    print("ItemKNNSimilarityTripleHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    EASE_R = EASE_R_Recommender.EASE_R_Recommender(URM_all)
    EASE_R.load_model(folder_path, EASE80)
    EASE_R_Wsparse = sps.csr_matrix(EASE_R.W_sparse)

    results, _ = evaluator_train.evaluateRecommender(EASE_R)
    print("EASE_R_Recommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIM_recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_all)
    SLIM_recommender.load_model(folder_path, SLIM80)
    SLIM_Wsparse = SLIM_recommender.W_sparse

    results, _ = evaluator_train.evaluateRecommender(SLIM_recommender)
    print("SLIMElasticNetRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    ALS = ALSRecommender.ALS(URM_all)
    ALS.load_model(folder_path, ALS80)

    results, _ = evaluator_train.evaluateRecommender(ALS)
    print("ALSRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    MultVAE = MultVAERecommender_PyTorch_OptimizerMask(URM_all)
    MultVAE.load_model(folder_path, MultVAE80)

    results, _ = evaluator_train.evaluateRecommender(MultVAE)
    print("MultVAE MAP: {}".format(results.loc[10]["MAP"]))

    IALS = IALSRecommender.IALSRecommender(URM_all)
    IALS.load_model(folder_path, IALS80)

    results, _ = evaluator_train.evaluateRecommender(IALS)
    print("IALSRecommender MAP: {}".format(results.loc[10]["MAP"]))

    recommenders = {
        "MultVAE": MultVAE,
        "ALS": IALS,
        "EASE_R": EASE_R,
        "Hybrid": RP3_recommender,
        "SLIM": SLIM_recommender,
        "Item": item_recommender
    }

    all_recommender = HybridLinear(URM_all, recommenders)
    all_recommender.fit(MultVAE=18.234181652023064, ALS=0.9683608848343739, EASE_R=-0.5361755922412472,
                        Hybrid=2.733233072276764, SLIM=2.369827855774208, Item=1.9690164229453446)

    results, _ = evaluator_train.evaluateRecommender(all_recommender)
    print("HybridLinear")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommended_items = all_recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/LinearHybridBIGSubmission.csv", recommendations)


if __name__ == '__main__':
    __main__()
