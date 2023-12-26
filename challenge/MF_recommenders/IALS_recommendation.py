import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.MatrixFactorization import IALSRecommender
from challenge.utils.functions import read_data


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train_plus_validation = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_train = sps.load_npz('../input_files/URM_train.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    recommender = IALSRecommender.IALSRecommender(URM_train_plus_validation)
    recommender.fit(epochs=83, num_factors=181, confidence_scaling="linear", alpha=3.363978058152649,
                    epsilon=0.08380486656074528,
                    reg=0.00016114721471727515, init_mean=0.5621008271466322, init_std=0.9665159175706768)

    recommender.save_model(folder_path="../result_experiments/", file_name="IALSRecommender_best_model80.zip")

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender = IALSRecommender.IALSRecommender(URM_train)
    recommender.fit(epochs=83, num_factors=181, confidence_scaling="linear", alpha=3.363978058152649,
                    epsilon=0.08380486656074528,
                    reg=0.00016114721471727515, init_mean=0.5621008271466322, init_std=0.9665159175706768)

    recommender.save_model(folder_path="../result_experiments/", file_name="IALSRecommender_best_model64.zip")

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender = IALSRecommender.IALSRecommender(URM_all)
    recommender.fit(epochs=83, num_factors=181, confidence_scaling="linear", alpha=3.363978058152649,
                    epsilon=0.08380486656074528,
                    reg=0.00016114721471727515, init_mean=0.5621008271466322, init_std=0.9665159175706768)

    recommender.save_model(folder_path="../result_experiments/", file_name="IALSRecommender_best_model100.zip")

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
