import numpy as np
import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.BaseRecommender import BaseRecommender
from challenge.utils.functions import read_data
from hybridization.utils.load_best_hyperparameters import load_best_hyperparameters


def tune_URM(recommender_class: BaseRecommender, recommender_folder: str, n: int = 100):
    cutoff_list = [10]
    data_file_path = '../challenge/input_files/data_train.csv'
    users_file_path = '../challenge/input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    best_hyperparameters = load_best_hyperparameters(recommender_folder)

    best_map = 0
    best_base = 0
    best_views_weight = 0
    best_details_weight = 0
    for i in range(n):
        print(f"\nIter {i}")
        random_base = np.random.uniform(2, 100)
        random_details_weight = np.random.uniform(0, 100)
        random_views_weight = np.random.uniform(random_details_weight, 100)
        print(f"base={random_base}, views_weight={random_views_weight}, details_weight={random_details_weight}")

        evaluator = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)

        print("Fitting")
        rec: BaseRecommender = recommender_class(URM_train)
        rec.fit(**best_hyperparameters)
        print("Evaluation")
        results_df, result_str = evaluator.evaluateRecommender(rec)
        print(result_str)

        result_map = list(results_df["MAP"])[0]
        if result_map > best_map:
            best_map = result_map
            best_base = random_base
            best_views_weight = random_views_weight
            best_details_weight = random_details_weight

        print(f"\nBest MAP so far: {best_map}")
        print(f"base={best_base}, views_weight={best_views_weight}, details_weight={best_details_weight}")
