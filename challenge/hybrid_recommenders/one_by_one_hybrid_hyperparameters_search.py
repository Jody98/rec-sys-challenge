import os
import numpy as np

import scipy.sparse as sps
from skopt.space import Real, Categorical

from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.DataIO import DataIO
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.Hybrid.HybridLinear import HybridLinear
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.SLIM import SLIMElasticNetRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.Hybrid.HybridDifferentLoss import DifferentLossScoresHybridRecommender
from Recommenders.KNN.ItemKNNSimilarityTripleHybridRecommender import ItemKNNSimilarityTripleHybridRecommender
from challenge.utils.functions import read_data
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.8)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])

    SLIM_recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train)
    SLIM_recommender.fit(topK=216, l1_ratio=0.0032465600313226354, alpha=0.002589066655986645, positive_only=True)
    SLIM_Wsparse = SLIM_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIM_recommender)
    print("SLIM MAP: {}".format(results.loc[10]["MAP"]))

    p3alpha = P3alphaRecommender.P3alphaRecommender(URM_train)
    p3alpha.fit(topK=40, alpha=0.3119217553589628, min_rating=0.01, implicit=True,
                normalize_similarity=True)
    p3alpha_Wsparse = p3alpha.W_sparse

    results, _ = evaluator.evaluateRecommender(p3alpha)
    print("P3alphaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item_recommender.fit(topK=9, shrink=13, similarity='tversky', tversky_alpha=0.036424892090848766,
                         tversky_beta=0.9961018325655608)
    item_Wsparse = item_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)
    RP3_Wsparse = RP3_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(RP3_recommender)
    print("RP3betaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    hybrid_recommender = ItemKNNSimilarityTripleHybridRecommender(URM_train, p3alpha_Wsparse, item_Wsparse, RP3_Wsparse)
    hybrid_recommender.fit(topK=225, alpha=0.4976629488640914, beta=0.13017801200221196)

    results, _ = evaluator.evaluateRecommender(hybrid_recommender)
    print("ItemKNNSimilarityTripleHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    hyperparameters_range_dictionary = {
        "norm": Categorical([1, 2, np.inf]),
        "alpha": Real(low=0, high=1, prior="uniform"),
    }

    recommender_class = DifferentLossScoresHybridRecommender

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, item_recommender, SLIM_recommender],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_validation, item_recommender, SLIM_recommender],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    output_folder_path = "../result_experiments/"

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    n_cases = 100
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
