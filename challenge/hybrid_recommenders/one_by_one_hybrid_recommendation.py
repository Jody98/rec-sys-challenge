import numpy as np
import scipy.sparse as sps
from tqdm import tqdm

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.Hybrid.HybridDifferentLoss import DifferentLossScoresHybridRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender_PyTorch_OptimizerMask
from Recommenders.SLIM import SLIMElasticNetRecommender
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.MatrixFactorization import IALSRecommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    folder_path = "../result_experiments/"
    SLIM80 = "new_SLIMElasticNetRecommender_best_model100.zip"
    MultVAE80 = "new_MultVAERecommender_best_model100.zip"
    EASE80 = "EASE_R_Recommender_best_model80.zip"
    IALS80 = "new_IALSRecommender_best_model80.zip"
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz('../input_files/new_URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/new_URM_test.npz')
    URM_all = sps.load_npz('../input_files/new_URM_all.npz')

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    ials = IALSRecommender.IALSRecommender(URM_train)
    ials.load_model(folder_path, IALS80)

    results, _ = evaluator.evaluateRecommender(ials)
    print("IALSRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    multvae = MultVAERecommender_PyTorch_OptimizerMask(URM_train)
    multvae.load_model(folder_path, MultVAE80)

    results, _ = evaluator.evaluateRecommender(multvae)
    print("MultVAERecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    p3alpha = P3alphaRecommender.P3alphaRecommender(URM_train)
    p3alpha.fit(topK=40, alpha=0.3119217553589628, min_rating=0.01, implicit=True,
                normalize_similarity=True)
    p3alpha_Wsparse = p3alpha.W_sparse

    results, _ = evaluator.evaluateRecommender(p3alpha)
    print("P3alphaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    ease = EASE_R_Recommender.EASE_R_Recommender(URM_train)
    ease.load_model(folder_path, EASE80)

    results, _ = evaluator.evaluateRecommender(ease)
    print("EASE_R_Recommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    item = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item.fit(topK=9, shrink=13, similarity='tversky', tversky_alpha=0.036424892090848766,
             tversky_beta=0.9961018325655608)
    item_Wsparse = item.W_sparse

    results, _ = evaluator.evaluateRecommender(item)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    rp3beta = RP3betaRecommender.RP3betaRecommender(URM_train)
    rp3beta.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                implicit=True, normalize_similarity=True)
    RP3_Wsparse = rp3beta.W_sparse

    results, _ = evaluator.evaluateRecommender(rp3beta)
    print("RP3betaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    slim = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train)
    slim.load_model(folder_path, SLIM80)

    results, _ = evaluator.evaluateRecommender(slim)
    print("SLIM MAP: {}".format(results.loc[10]["MAP"]))

    SLIMRP3 = DifferentLossScoresHybridRecommender(URM_train, slim, rp3beta)
    SLIMRP3.fit(norm=np.inf, alpha=0.6787901569849559)

    results, _ = evaluator.evaluateRecommender(SLIMRP3)
    print("SLIMRP3 MAP: {}".format(results.loc[10]["MAP"]))

    SLIMRP3Item = DifferentLossScoresHybridRecommender(URM_train, SLIMRP3, item)
    SLIMRP3Item.fit(norm=2, alpha=0.8828064753046293)

    results, _ = evaluator.evaluateRecommender(SLIMRP3Item)
    print("SLIMRP3Item MAP: {}".format(results.loc[10]["MAP"]))

    SLIMRP3ItemMult = DifferentLossScoresHybridRecommender(URM_train, SLIMRP3Item, multvae)
    SLIMRP3ItemMult.fit(norm=2, alpha=0.2759969848752059)

    results, _ = evaluator.evaluateRecommender(SLIMRP3ItemMult)
    print("SLIMRP3ItemMult MAP: {}".format(results.loc[10]["MAP"]))

    recommended_items = SLIMRP3ItemMult.recommend(users_list, cutoff=10)
    recommendations = []
    for i in tqdm(zip(users_list, recommended_items)):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/OneByOneSubmission.csv", recommendations)


if __name__ == '__main__':
    __main__()
