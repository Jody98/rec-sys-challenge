import scipy.sparse as sps
from tqdm import tqdm

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender
from Recommenders.Hybrid.HybridLinear import HybridLinear
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender_PyTorch_OptimizerMask
from Recommenders.SLIM import SLIMElasticNetRecommender
from Recommenders.MatrixFactorization import IALSRecommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    folder_path = "../result_experiments/"
    SLIM80 = "SLIMElasticNetRecommender_best_model100.zip"
    MultVAE80 = "Mult_VAE_Recommender_best_model100.zip"
    IALS80 = "IALSRecommender_best_model100.zip"
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_train = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_train = sps.load_npz('../input_files/URM_all.npz')

    evaluator_train = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    ials_recommender = IALSRecommender.IALSRecommender(URM_train)
    ials_recommender.load_model(folder_path, IALS80)

    results, _ = evaluator_train.evaluateRecommender(ials_recommender)
    print("IALSRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item_recommender.fit(topK=9, shrink=13, similarity='tversky', tversky_alpha=0.03642489209084876,
                         tversky_beta=0.9961018325655608)
    item_Wsparse = item_recommender.W_sparse

    results, _ = evaluator_train.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)
    RP3_Wsparse = RP3_recommender.W_sparse

    results, _ = evaluator_train.evaluateRecommender(RP3_recommender)
    print("RP3betaRecommender")
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

    recommenders = {
        "SLIM": SLIM_recommender,
        "MultVAE": MultVAE,
        "RP3": RP3_recommender,
        "Item": item_recommender,
        "IALS": ials_recommender,
    }

    all_recommender = HybridLinear(URM_train, recommenders)
    all_recommender.fit(eta=15.180249222221073, gamma=-0.68442274063330273,
                        alpha=3.060407131177933, beta=2.995116702486108, zeta=0.9737256690221096)

    results, _ = evaluator_train.evaluateRecommender(all_recommender)
    print("BEST")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommended_items = all_recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in tqdm(zip(users_list, recommended_items)):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/LinearHybridBIGSubmission.csv", recommendations)


if __name__ == '__main__':
    __main__()
