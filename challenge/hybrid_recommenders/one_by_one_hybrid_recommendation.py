import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.Hybrid.HybridDifferentLoss import DifferentLossScoresHybridRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender_PyTorch_OptimizerMask
from Recommenders.SLIM import SLIMElasticNetRecommender
from challenge.utils.functions import read_data


def __main__():
    cutoff_list = [10]
    folder_path = "../result_experiments/"
    SLIM80 = "SLIMElasticNetRecommender_best_model80.zip"
    MultVAE80 = "Mult_VAE_Recommender_best_model80.zip"
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

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
    SLIMRP3.fit(norm=2, alpha=0.5489413475116747)

    results, _ = evaluator.evaluateRecommender(SLIMRP3)
    print("DifferentLossScoresHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
