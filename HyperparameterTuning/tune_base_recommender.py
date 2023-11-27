import os
from typing import Dict, Any

import scipy.sparse as sps
from skopt.space import Real, Integer, Categorical

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.DataIO import DataIO
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from challenge.utils.functions import read_data


def _get_hyperparameters_range(recommender_class: BaseRecommender) -> Dict[str, Any] | ValueError:
    if recommender_class == EASE_R_Recommender:
        hyperparameters_range_dictionary = {
            "topK": Categorical([None]),
            "normalize_matrix": Categorical([False]),
            "l2_norm": Real(low=1e0, high=1e7, prior="log-uniform"),
        }
    elif recommender_class == IALSRecommender:
        hyperparameters_range_dictionary = {
            "num_factors": Integer(1, 500),
            "epochs": Integer(100, 200),
            "alpha": Real(low=1e-3, high=50.0, prior="log-uniform"),
            "reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
        }
    elif recommender_class == ItemKNNCFRecommender:
        hyperparameters_range_dictionary = {
            "topK": Integer(5, 20),
            "shrink": Integer(0, 20),
            "similarity": Categorical(["cosine", "jaccard"]),
            "normalize": Categorical([True]),
        }
    elif recommender_class == RP3betaRecommender:
        hyperparameters_range_dictionary = {
            "topK": Integer(5, 1000),
            "alpha": Real(low=0, high=2, prior="uniform"),
            "beta": Real(low=0, high=2, prior="uniform"),
            "normalize_similarity": Categorical([True, False]),
        }
    elif recommender_class == SLIMElasticNetRecommender:
        hyperparameters_range_dictionary = {
            "topK": Integer(500, 2000),
            "l1_ratio": Real(low=1e-4, high=0.1, prior="log-uniform"),
            "alpha": Real(low=1e-4, high=0.1, prior="uniform"),
        }
    else:
        return ValueError(f"The recommender class {recommender_class} is not supported")
    return hyperparameters_range_dictionary


def tune_base_recommender(
        recommender_class: BaseRecommender,
        output_folder: str,
        n: int = 100,
        save_trained_on_all: bool = True,
):
    if not os.path.exists("results_experiments"):
        os.makedirs("result_experiments")

    full_output_path = "results_experiments/" + output_folder
    if not os.path.exists(full_output_path):
        os.makedirs(full_output_path)

    cutoff_list = [10]
    data_file_path = '../challenge/input_files/data_train.csv'
    users_file_path = '../challenge/input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    evaluator = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)

    hyperparameter_search = SearchBayesianSkopt(
        recommender_class,
        evaluator_validation=evaluator,
    )
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[
            URM_train,
        ],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )
    hyperparameters_range_dictionary = _get_hyperparameters_range(recommender_class)

    hyperparameter_search.search(
        recommender_input_args,
        hyperparameter_search_space=hyperparameters_range_dictionary,
        n_cases=n,
        n_random_starts=int(n * 0.3),
        save_model="best",
        output_folder_path=full_output_path,
        output_file_name_root=recommender_class.RECOMMENDER_NAME,
        metric_to_optimize="MAP",
        cutoff_to_optimize=10,
    )

    data_loader = DataIO(folder_path=full_output_path)
    search_metadata = data_loader.load_data(
        recommender_class.RECOMMENDER_NAME + "_metadata.zip"
    )
    best_hyperparameters = search_metadata["hyperparameters_best"]
    print("best_param", best_hyperparameters)

    if save_trained_on_all:
        recommender: BaseRecommender = recommender_class(URM_train + URM_validation)
        recommender.fit(**best_hyperparameters)
        recommender.save_model(
            folder_path=full_output_path,
            file_name=recommender_class.RECOMMENDER_NAME
                      + "_best_model_trained_on_everything.zip",
        )
