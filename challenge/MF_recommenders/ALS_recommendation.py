import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.MatrixFactorization import ALSRecommender
from challenge.utils.functions import read_data


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train_plus_validation = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')
    URM_train = sps.load_npz('../input_files/URM_train.npz')

    recommender = ALSRecommender.ALS(URM_train_plus_validation)
    recommender.fit(factors=116, regularization=0.0003763107182807487, iterations=28, alpha=3.879694465934832)

    recommender.save_model(folder_path="../result_experiments/", file_name="ALSRecommender_best_model80.zip")

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender = ALSRecommender.ALS(URM_all)
    recommender.fit(factors=116, regularization=0.0003763107182807487, iterations=28, alpha=3.879694465934832)

    recommender.save_model(folder_path="../result_experiments/", file_name="ALSRecommender_best_model100.zip")

    recommender = ALSRecommender.ALS(URM_train)
    recommender.fit(factors=116, regularization=0.0003763107182807487, iterations=28, alpha=3.879694465934832)

    recommender.save_model(folder_path="../result_experiments/", file_name="ALSRecommender_best_model64.zip")


if __name__ == '__main__':
    __main__()
