import os
from pathlib import Path
from skopt.space import Real
import scipy.sparse as sps

from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.EASE_R.EASE_R_Recommender import (
    EASE_R_Recommender,
)
from Recommenders.Hybrid.HybridLinear import HybridLinear
from Recommenders.DataIO import DataIO
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from challenge.utils.functions import read_data

base_recommenders = {
    "KNN": (ItemKNNCFRecommender, Path("result_experiments/KNN")),
    "RP3beta": (RP3betaRecommender, Path("result_experiments/RP3beta")),
    "iALS": (IALSRecommender, Path("result_experiments/iALS")),
    "EASE_R": (EASE_R_Recommender, Path("result_experiments/EASE_R")),
    "SLIM": (SLIMElasticNetRecommender, Path("result_experiments/SLIM")),
}

hyperparameters_range_dictionary = {
    "KNN": Real(0.0, 0.7),
    "RP3beta": Real(0.2, 1.0),
    "IALS": Real(0.0, 0.8),
    "EASE_R": Real(0.3, 1.0),
    "SLIMElasticNet": Real(0.3, 1.0),
}

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

loaded_recommenders = {}
for recommender_id, (recommender_class, folder) in base_recommenders.items():
    URM_train_file = folder / "tuned_URM/URM_train.npz"
    if URM_train_file.exists():
        recommender_URM_train = sps.load_npz(URM_train_file)
        recommender_obj = recommender_class(recommender_URM_train)
        recommender_obj.load_model(
            str(folder / "tuned_URM"),
            (recommender_class.RECOMMENDER_NAME + "_best_model.zip"),
        )
    else:
        print(f"WARNING: Using implicit URM for {recommender_id}")
        recommender_obj = recommender_class(URM_train)
        recommender_obj.load_model(
            str(folder),
            (recommender_class.RECOMMENDER_NAME + "_best_model.zip"),
        )

    loaded_recommenders[recommender_id] = recommender_obj

output_folder_path = "result_experiments/Hybrid/"
recommender_class = HybridLinear
n_cases = 100
n_random_starts = int(n_cases * 0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

hyperparameter_search = SearchBayesianSkopt(
    recommender_class,
    evaluator_validation=evaluator,
)

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, loaded_recommenders],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={},
)

hyperparameter_search.search(
    recommender_input_args,
    hyperparameter_search_space=hyperparameters_range_dictionary,
    n_cases=n_cases,
    n_random_starts=n_random_starts,
    save_model="no",
    output_folder_path=output_folder_path,  # Where to save the results
    output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
    metric_to_optimize=metric_to_optimize,
    cutoff_to_optimize=cutoff_to_optimize,
)

data_loader = DataIO(folder_path=output_folder_path)
search_metadata = data_loader.load_data(
    recommender_class.RECOMMENDER_NAME + "_metadata.zip"
)

result_on_validation_df = search_metadata["result_on_validation_df"]
print(result_on_validation_df)
best_hyperparameters = search_metadata["hyperparameters_best"]
print(best_hyperparameters)