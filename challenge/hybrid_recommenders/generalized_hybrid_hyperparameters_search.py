import os

import scipy.sparse as sps
from skopt.space import Real

from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.DataIO import DataIO
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.Hybrid.GeneralizedLinearHybridRecommender import GeneralizedLinearHybridRecommender
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.SLIM import SLIMElasticNetRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.MatrixFactorization import IALSRecommender
from Recommenders.Hybrid.HybridDifferentLoss import DifferentLossScoresHybridRecommender
from challenge.utils.functions import read_data


def __main__():
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train_validation = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_validation = sps.load_npz('../input_files/URM_validation.npz')
    URM_train = sps.load_npz('../input_files/URM_train.npz')

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])

    ials_recommender = IALSRecommender.IALSRecommender(URM_train)
    ials_recommender.fit(epochs=100, num_factors=92, confidence_scaling="linear", alpha=2.5431444656816597,
                         epsilon=0.035779451402656745,
                         reg=1.5, init_mean=0.0, init_std=0.1)

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

    EASE_recommender = EASE_R_Recommender.EASE_R_Recommender(URM_train)
    EASE_recommender.fit(topK=10, l2_norm=101, normalize_matrix=False)
    EASE_R_Wsparse = sps.csr_matrix(EASE_recommender.W_sparse)

    results, _ = evaluator.evaluateRecommender(EASE_recommender)
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

    SLIMRP3 = DifferentLossScoresHybridRecommender(URM_train, RP3_recommender, SLIM_recommender)
    SLIMRP3.fit(norm=2, alpha=0.4304989217384739)

    results, _ = evaluator.evaluateRecommender(SLIMRP3)
    print("SLIMRP3")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommenders = [ials_recommender, item_recommender, EASE_recommender, P3_recommender, SLIMRP3]

    hyperparameters_range_dictionary = {
        "alpha": Real(0, 0.2),
        "beta": Real(0, 0.1),
        "gamma": Real(1, 2),
        "delta": Real(3, 4),
        "epsilon": Real(3, 4),
    }

    recommender_class = GeneralizedLinearHybridRecommender

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, recommenders],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_validation, recommenders],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    output_folder_path = "../result_experiments/"

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    n_cases = 10
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
