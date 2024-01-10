import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender
from Recommenders.Hybrid.GeneralizedLinearHybridRecommender import GeneralizedLinearHybridRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.MatrixFactorization import IALSRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender_PyTorch_OptimizerMask
from Recommenders.SLIM import SLIMElasticNetRecommender
from challenge.utils.functions import read_data


def __main__():
    cutoff_list = [10]
    folder_path = "../result_experiments/"
    SLIM64 = "SLIMElasticNetRecommender_best_model64.zip"
    MultVAE64 = "MultVAERecommender_best_model64.zip"
    IALS64 = "IALSRecommender_best_model64.zip"
    SLIM80 = "SLIMElasticNetRecommender_best_model80.zip"
    MultVAE80 = "Mult_VAE_Recommender_best_model80.zip"
    IALS80 = "IALSRecommender_best_model80.zip"
    SLIM100 = "SLIMElasticNetRecommender_best_model100.zip"
    MultVAE100 = "Mult_VAE_Recommender_best_model100.zip"
    IALS100 = "IALSRecommender_best_model100.zip"

    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_train = sps.load_npz('../input_files/URM_train.npz')
    URM_train_validation = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    evaluator_train = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train_validation)
    item_recommender.fit(topK=9, shrink=13, similarity='tversky', tversky_alpha=0.03642489209084876,
                         tversky_beta=0.9961018325655608)
    item_Wsparse = item_recommender.W_sparse

    results, _ = evaluator_train.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_train_validation)
    RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)
    RP3_Wsparse = RP3_recommender.W_sparse

    results, _ = evaluator_train.evaluateRecommender(RP3_recommender)
    print("RP3betaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIM_recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train_validation)
    SLIM_recommender.load_model(folder_path, SLIM80)

    results, _ = evaluator_train.evaluateRecommender(SLIM_recommender)
    print("SLIMElasticNetRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    MultVAE = MultVAERecommender_PyTorch_OptimizerMask(URM_train_validation)
    MultVAE.load_model(folder_path, MultVAE80)

    results, _ = evaluator_train.evaluateRecommender(MultVAE)
    print("MultVAE MAP: {}".format(results.loc[10]["MAP"]))

    ials_recommender = IALSRecommender.IALSRecommender(URM_train_validation)
    ials_recommender.load_model(folder_path, IALS80)

    results, _ = evaluator_train.evaluateRecommender(ials_recommender)
    print("IALSRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    best_parameters = {'alpha': 12.005267608140038, 'beta': 2.3349650422172834, 'gamma': 1.4514107511320142,
                       'delta': 17.874066643406685, 'epsilon': 14.627521502682754}

    recommenders = [item_recommender, MultVAE, ials_recommender, RP3_recommender, SLIM_recommender]

    recommender_object = GeneralizedLinearHybridRecommender(URM_train_validation, recommenders=recommenders)
    recommender_object.fit(**best_parameters)

    results, _ = evaluator_train.evaluateRecommender(recommender_object)
    print("GeneralizedLinearHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender_object.save_model(folder_path, "Hybrid_80.zip")

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
    SLIM_recommender.load_model(folder_path, SLIM64)

    results, _ = evaluator_train.evaluateRecommender(SLIM_recommender)
    print("SLIMElasticNetRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    MultVAE = MultVAERecommender_PyTorch_OptimizerMask(URM_train)
    MultVAE.load_model(folder_path, MultVAE64)

    results, _ = evaluator_train.evaluateRecommender(MultVAE)
    print("MultVAE MAP: {}".format(results.loc[10]["MAP"]))

    ials_recommender = IALSRecommender.IALSRecommender(URM_train)
    ials_recommender.load_model(folder_path, IALS64)

    results, _ = evaluator_train.evaluateRecommender(ials_recommender)
    print("IALSRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    best_parameters = {'alpha': 12.005267608140038, 'beta': 2.3349650422172834, 'gamma': 1.4514107511320142,
                       'delta': 17.874066643406685, 'epsilon': 14.627521502682754}

    recommenders = [item_recommender, MultVAE, ials_recommender, RP3_recommender, SLIM_recommender]

    recommender_object = GeneralizedLinearHybridRecommender(URM_train, recommenders=recommenders)
    recommender_object.fit(**best_parameters)

    results, _ = evaluator_train.evaluateRecommender(recommender_object)
    print("GeneralizedLinearHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender_object.save_model(folder_path, "Hybrid_64.zip")

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_all)
    item_recommender.fit(topK=9, shrink=13, similarity='tversky', tversky_alpha=0.03642489209084876,
                         tversky_beta=0.9961018325655608)
    item_Wsparse = item_recommender.W_sparse

    results, _ = evaluator_train.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_all)
    RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)
    RP3_Wsparse = RP3_recommender.W_sparse

    results, _ = evaluator_train.evaluateRecommender(RP3_recommender)
    print("RP3betaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIM_recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_all)
    SLIM_recommender.load_model(folder_path, SLIM100)

    results, _ = evaluator_train.evaluateRecommender(SLIM_recommender)
    print("SLIMElasticNetRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    MultVAE = MultVAERecommender_PyTorch_OptimizerMask(URM_all)
    MultVAE.load_model(folder_path, MultVAE100)

    results, _ = evaluator_train.evaluateRecommender(MultVAE)
    print("MultVAE MAP: {}".format(results.loc[10]["MAP"]))

    ials_recommender = IALSRecommender.IALSRecommender(URM_all)
    ials_recommender.load_model(folder_path, IALS100)

    results, _ = evaluator_train.evaluateRecommender(ials_recommender)
    print("IALSRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    best_parameters = {'alpha': 12.005267608140038, 'beta': 2.3349650422172834, 'gamma': 1.4514107511320142,
                       'delta': 17.874066643406685, 'epsilon': 14.627521502682754}

    recommenders = [item_recommender, MultVAE, ials_recommender, RP3_recommender, SLIM_recommender]

    recommender_object = GeneralizedLinearHybridRecommender(URM_all, recommenders=recommenders)
    recommender_object.fit(**best_parameters)

    results, _ = evaluator_train.evaluateRecommender(recommender_object)
    print("GeneralizedLinearHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender_object.save_model(folder_path, "Hybrid_100.zip")


if __name__ == '__main__':
    __main__()
