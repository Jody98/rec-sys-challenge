import os
from pathlib import Path
from skopt.space import Real
import scipy.sparse as sps

from Recommenders.Hybrid.HybridLinear import HybridLinear
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from utils.load_best_hyperparameters import load_best_hyperparameters
from utils.create_submission import create_submission

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
    "alpha": Real(-4, 1),
    "beta": Real(-4, 4)
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

best_weights = load_best_hyperparameters(Path("result_experiments/Hybrid"))

loaded_recommenders = {}
for recommender_id, (recommender_class, folder) in base_recommenders.items():
    URM_train_file = folder / "tuned_URM/URM_train.npz"
    if URM_train_file.exists():
        recommender_URM_train = sps.load_npz(URM_train_file)
        recommender_URM_val = sps.load_npz(folder / "tuned_URM/URM_val.npz")
        recommender_obj = recommender_class(recommender_URM_train + recommender_URM_val)
        recommender_obj.load_model(
            str(folder / "tuned_URM"),
            (recommender_class.RECOMMENDER_NAME + "_best_model_trained_on_everything.zip"),
        )
    else:
        print(f"WARNING: Using implicit URM for {recommender_id}")
        recommender_obj = recommender_class(URM_train + URM_validation)
        recommender_obj.load_model(
            str(folder),
            (recommender_class.RECOMMENDER_NAME + "_best_model_trained_on_everything.zip"),
        )

    loaded_recommenders[recommender_id] = recommender_obj

output_folder_path = "result_experiments/Hybrid/"
recommender_class = HybridLinear

recommender = recommender_class(URM_train, URM_validation, cutoff_list=cutoff_list)
recommender.fit(
    **best_weights
)
create_submission(recommender, users_list, output_folder_path, recommender.RECOMMENDER_NAME)


