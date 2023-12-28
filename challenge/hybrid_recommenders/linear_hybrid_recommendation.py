import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.Hybrid.HybridLinear import HybridLinear
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNSimilarityTripleHybridRecommender import ItemKNNSimilarityTripleHybridRecommender
from Recommenders.MatrixFactorization import IALSRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender_PyTorch_OptimizerMask
from Recommenders.SLIM import SLIMElasticNetRecommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    folder_path = "../result_experiments/"
    SLIM80 = "SLIMElasticNetRecommender_best_model100.zip"
    MultVAE80Best = "MultVAERecommender_best_model100.zip"
    MultVAE80 = "Mult_VAE_Recommender_best_model100.zip"
    IALS80 = "IALSRecommender_best_model100.zip"
    EASE80 = "EASE_R_Recommender_best_model100.zip"
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_train = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_train = sps.load_npz('../input_files/URM_all.npz')

    evaluator_train = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item_recommender.fit(topK=9, shrink=13, similarity='tversky', tversky_alpha=0.03642489209084876,
                         tversky_beta=0.9961018325655608)
    item_Wsparse = item_recommender.W_sparse

    results, _ = evaluator_train.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    P3_recommender = P3alphaRecommender.P3alphaRecommender(URM_train)
    P3_recommender.fit(topK=40, alpha=0.3119217553589628, min_rating=0.01, implicit=True, normalize_similarity=True)
    p3alpha_Wsparse = P3_recommender.W_sparse

    results, _ = evaluator_train.evaluateRecommender(P3_recommender)
    print("P3alphaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)
    RP3_Wsparse = RP3_recommender.W_sparse

    results, _ = evaluator_train.evaluateRecommender(RP3_recommender)
    print("RP3betaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    hybrid_recommender = ItemKNNSimilarityTripleHybridRecommender(URM_train, p3alpha_Wsparse, item_Wsparse, RP3_Wsparse)
    hybrid_recommender.fit(topK=75, alpha=0.4976629488640914, beta=0.13017801200221196)

    results, _ = evaluator_train.evaluateRecommender(hybrid_recommender)
    print("ItemKNNSimilarityTripleHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIM_recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train)
    SLIM_recommender.load_model(folder_path, SLIM80)

    results, _ = evaluator_train.evaluateRecommender(SLIM_recommender)
    print("SLIMElasticNetRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    MultVAE = MultVAERecommender_PyTorch_OptimizerMask(URM_train)
    MultVAE.load_model(folder_path, MultVAE80)

    results, _ = evaluator_train.evaluateRecommender(MultVAE)
    print("MultVAE MAP: {}".format(results.loc[10]["MAP"]))

    MultVAEBest = MultVAERecommender_PyTorch_OptimizerMask(URM_train)
    MultVAEBest.load_model(folder_path, MultVAE80Best)

    results, _ = evaluator_train.evaluateRecommender(MultVAEBest)
    print("MultVAE MAP: {}".format(results.loc[10]["MAP"]))

    IALS = IALSRecommender.IALSRecommender(URM_train)
    IALS.load_model(folder_path, IALS80)

    results, _ = evaluator_train.evaluateRecommender(IALS)
    print("IALSRecommender MAP: {}".format(results.loc[10]["MAP"]))

    EASE_R = EASE_R_Recommender.EASE_R_Recommender(URM_train)
    EASE_R.load_model(folder_path, EASE80)
    EASE_R_Wsparse = sps.csr_matrix(EASE_R.W_sparse)

    results, _ = evaluator_train.evaluateRecommender(EASE_R)
    print("EASE_R_Recommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommenders = {
        "MultVAE": MultVAE,
        "ALS": IALS,
        "Hybrid": RP3_recommender,
        "SLIM": SLIM_recommender,
        "Item": item_recommender,
        "P3": P3_recommender
    }

    all_recommender = HybridLinear(URM_train, recommenders)
    all_recommender.fit(MultVAE=15.180249222221073, ALS=-0.38442274063330273, P3=1.060407131177933,
                        Hybrid=4.060407131177933, SLIM=5.945116702486108, Item=0.9737256690221096)

    results, _ = evaluator_train.evaluateRecommender(all_recommender)
    print("BEST\n")
    print("MAP: {}".format(results.loc[10]["MAP"]))
    print("RECALL: {}".format(results.loc[10]["RECALL"]))

    recommended_items = all_recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/LinearHybridBIGSubmission.csv", recommendations)

    all_recommender = HybridLinear(URM_train, recommenders)
    all_recommender.fit(MultVAE=24.064552469359455, ALS=-0.9864760541829147, P3=3.037541814942154,
                        Hybrid=9.735222755831965, SLIM=9.071681040796111, Item=2.966590102582196)

    results, _ = evaluator_train.evaluateRecommender(all_recommender)
    print("MAP: {}".format(results.loc[10]["MAP"]))
    print("RECALL: {}".format(results.loc[10]["RECALL"]))

    recommended_items = all_recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/LinearHybridBIG2Submission.csv", recommendations)


if __name__ == '__main__':
    __main__()
