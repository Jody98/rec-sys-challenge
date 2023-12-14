import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R import EASE_R_Recommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    folder_path = "../result_experiments/"
    filename = "EASE_R_Recommender_best_model.zip"
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    tops = [60, 80, 100, 150]
    l2_norms = [10, 20, 25, 30, 40, 50, 75, 100]

    for topk in tops:
        for l2_norm in l2_norms:
            EASE_R = EASE_R_Recommender.EASE_R_Recommender(URM_train)
            # EASE_R.fit(topK=59, l2_norm=29.792347118106623, normalize_matrix=False)
            EASE_R.fit(topK=topk, l2_norm=l2_norm, normalize_matrix=False)
            EASE_R_Wsparse = sps.csr_matrix(EASE_R.W_sparse)

            evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

            results, _ = evaluator.evaluateRecommender(EASE_R)
            print("topk: {}".format(topk))
            print("l2_norm: {}".format(l2_norm))
            print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
