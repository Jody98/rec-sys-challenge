import os
import traceback

import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.Recommender_import_list import *
from challenge.utils import functions as f


def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):
    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)

    return recommender_object


if __name__ == '__main__':
    data_file_path = '../challenge/input_files/data_train.csv'
    users_file_path = '../challenge/input_files/data_target_users_test.csv'

    URM_all_dataframe, users_list = f.read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    ICM_all = None

    UCM_all = None

    recommender_class_list = [
        # Random,
        # TopPop,
        # GlobalEffects,
        # SLIMElasticNetRecommender,
        UserKNNCFRecommender,  # 0.0288819
        IALSRecommender,  # 0.0396401
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # MatrixFactorization_AsySVD_Cython,
        EASE_R_Recommender,  # 0.0473539
        ItemKNNCFRecommender,  # 0.0505696
        P3alphaRecommender,  # 0.0185236
        # SLIM_BPR_Cython,
        RP3betaRecommender,  # 0.0146424
        PureSVDRecommender,  # 0.0378801
        # NMFRecommender,
        # UserKNNCBFRecommender,
        # ItemKNNCBFRecommender,
        # UserKNN_CFCBF_Hybrid_Recommender,
        # ItemKNN_CFCBF_Hybrid_Recommender,
        LightFMCFRecommender,  # 0.0251791
        # LightFMUserHybridRecommender,
        # LightFMItemHybridRecommender,
    ]

    evaluator = EvaluatorHoldout(URM_test, [5, 20], exclude_seen=True)

    # from MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch

    earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": EvaluatorHoldout(URM_validation, [20], exclude_seen=True),
                              "lower_validations_allowed": 5,
                              "validation_metric": "MAP",
                              }

    output_root_path = "./result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    logFile = open(output_root_path + "result_all_algorithms.txt", "a")

    for recommender_class in recommender_class_list:

        try:

            print("Algorithm: {}".format(recommender_class))

            recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)

            if isinstance(recommender_object, Incremental_Training_Early_Stopping):
                fit_params = {"epochs": 15, **earlystopping_keywargs}
            else:
                fit_params = {}

            recommender_object.fit(**fit_params)

            results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender_object)

            recommender_object.save_model(output_root_path, file_name="temp_model.zip")

            recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)
            recommender_object.load_model(output_root_path, file_name="temp_model.zip")

            os.remove(output_root_path + "temp_model.zip")

            results_run_2, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)

            if recommender_class not in [Random]:
                assert results_run_1.equals(results_run_2)

            print("Algorithm: {}, results: \n{}".format(recommender_class, results_run_string_1))
            logFile.write("Algorithm: {}, results: \n{}\n".format(recommender_class, results_run_string_1))
            logFile.flush()


        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()
