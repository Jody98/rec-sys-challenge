import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R import EASE_R_Recommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    folder_path = "../result_experiments/"
    filename64 = "EASE_R_Recommender_best_model64.zip"
    filename80 = "EASE_R_Recommender_best_model80.zip"
    filename100 = "EASE_R_Recommender_best_model100.zip"
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train_plus_validation = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    EASE_R = EASE_R_Recommender.EASE_R_Recommender(URM_all)
    EASE_R.fit(topK=32, l2_norm=38, normalize_matrix=False)
    EASE_R_Wsparse = sps.csr_matrix(EASE_R.W_sparse)

    EASE_R.save_model(folder_path=folder_path, file_name=filename100)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    results, _ = evaluator.evaluateRecommender(EASE_R)
    print("MAP: {}".format(results.loc[10]["MAP"]))

    URM_train = sps.load_npz('../input_files/URM_train.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    EASE_R = EASE_R_Recommender.EASE_R_Recommender(URM_train)
    EASE_R.fit(topK=32, l2_norm=38, normalize_matrix=False)
    EASE_R_Wsparse = sps.csr_matrix(EASE_R.W_sparse)

    EASE_R.save_model(folder_path=folder_path, file_name=filename64)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    results, _ = evaluator.evaluateRecommender(EASE_R)
    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
