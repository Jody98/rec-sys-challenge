import os

import numpy as np
import scipy.sparse as sps
from skopt.space import Real, Categorical

from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.DataIO import DataIO
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.Hybrid.HybridDifferentLoss import DifferentLossScoresHybridRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender_PyTorch_OptimizerMask
from Recommenders.SLIM import SLIMElasticNetRecommender
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.MatrixFactorization import IALSRecommender


def __main__():
    cutoff_list = [10]
    folder_path = "../result_experiments/"
    SLIM64 = "SLIMElasticNetRecommender_best_model64.zip"
    MultVAE64 = "Mult_VAE_Recommender_best_model64.zip"
    EASE64 = "EASE_R_Recommender_best_model64.zip"
    IALS64 = "IALSRecommender_best_model64.zip"
    URM_train_validation = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_train = sps.load_npz('../input_files/URM_train.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')

    evaluator_validation = EvaluatorHoldout(URM_train_validation, cutoff_list=cutoff_list)
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    ials = IALSRecommender.IALSRecommender(URM_train)
    ials.load_model(folder_path, IALS64)

    results, _ = evaluator.evaluateRecommender(ials)
    print("IALSRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    multvae = MultVAERecommender_PyTorch_OptimizerMask(URM_train)
    multvae.load_model(folder_path, MultVAE64)

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
    ease.load_model(folder_path, EASE64)

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
    slim.load_model(folder_path, SLIM64)

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

    SLIMRP3ItemEASE = DifferentLossScoresHybridRecommender(URM_train, SLIMRP3Item, ease)
    SLIMRP3ItemEASE.fit(norm=np.inf, alpha=0.9936820574807692)

    results, _ = evaluator.evaluateRecommender(SLIMRP3ItemEASE)
    print("SLIMRP3ItemEASE MAP: {}".format(results.loc[10]["MAP"]))

    SLIMRP3ItemMult = DifferentLossScoresHybridRecommender(URM_train, SLIMRP3Item, multvae)
    SLIMRP3ItemMult.fit(norm=2, alpha=0.2759969848752059)

    results, _ = evaluator.evaluateRecommender(SLIMRP3ItemMult)
    print("SLIMRP3ItemMult MAP: {}".format(results.loc[10]["MAP"]))

    hyperparameters_range_dictionary = {
        "norm": Categorical([1, 2, np.inf]),
        "alpha": Real(low=0.9, high=1.0, prior="uniform"),
    }

    recommender_class = DifferentLossScoresHybridRecommender

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, SLIMRP3ItemMult, p3alpha],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={
            "validation_every_n": 1,
            "stop_on_validation": True,
            "evaluator_object": evaluator_validation,
            "lower_validations_allowed": 5,
            "validation_metric": "MAP",
        },
    )

    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_validation, SLIMRP3ItemMult, p3alpha],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={
            "validation_every_n": 1,
            "stop_on_validation": True,
            "evaluator_object": evaluator_validation,
            "lower_validations_allowed": 5,
            "validation_metric": "MAP",
        },
    )

    output_folder_path = "../result_experiments/"

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    n_cases = 1000
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    hyperparameterSearch.search(recommender_input_args,
                                recommender_input_args_last_test=recommender_input_args_last_test,
                                hyperparameter_search_space=hyperparameters_range_dictionary,
                                n_cases=n_cases,
                                n_random_starts=n_random_starts,
                                save_model="last",
                                output_folder_path=output_folder_path,
                                output_file_name_root=recommender_class.RECOMMENDER_NAME,
                                metric_to_optimize=metric_to_optimize,
                                cutoff_to_optimize=cutoff_to_optimize,
                                )

    data_loader = DataIO(folder_path=output_folder_path)
    search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

    search_metadata.keys()

    hyperparameters_df = search_metadata["hyperparameters_df"]
    print(hyperparameters_df)
    result_on_validation_df = search_metadata["result_on_validation_df"]
    print(result_on_validation_df)
    result_best_on_test = search_metadata["result_on_last"]
    print(result_best_on_test)
    best_hyperparameters = search_metadata["hyperparameters_best"]
    print(best_hyperparameters)
    time_df = search_metadata["time_df"]
    print(time_df)


if __name__ == '__main__':
    __main__()
