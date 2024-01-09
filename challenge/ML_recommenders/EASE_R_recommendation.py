import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R import EASE_R_Recommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    folder_path = "../result_experiments/"
    filename80 = "EASE_R_Recommender_best_model80.zip"
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train_xgboost = sps.load_npz("../input_files/URM_train_xgboost.npz")
    URM_test_xgboost = sps.load_npz("../input_files/URM_validation_xgboost.npz")

    EASE_R = EASE_R_Recommender.EASE_R_Recommender(URM_train_xgboost)
    EASE_R.fit(topK=32, l2_norm=38, normalize_matrix=False)

    EASE_R.save_model(folder_path=folder_path, file_name='EASE_xgboost.zip')

    evaluator = EvaluatorHoldout(URM_test_xgboost, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(EASE_R)

    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
