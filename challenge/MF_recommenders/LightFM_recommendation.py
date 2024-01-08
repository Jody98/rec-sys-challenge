import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.FactorizationMachines import LightFMRecommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train_plus_validation = sps.load_npz('../input_files/new_URM_train_plus_validation.npz')
    URM_train = sps.load_npz('../input_files/new_URM_train.npz')
    URM_test = sps.load_npz('../input_files/new_URM_test.npz')
    URM_all = sps.load_npz('../input_files/new_URM_all.npz')

    recommender = LightFMRecommender.LightFMCFRecommender(URM_train)
    recommender.fit(loss="bpr", sgd_mode="adagrad", n_components=10,
                    item_alpha=0.0, user_alpha=0.0,
                    learning_rate=0.05, num_threads=4)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
