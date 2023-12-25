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
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    recommender = IALSRecommender.IALSRecommender(URM_train_plus_validation)
    recommender.fit(epochs=27, num_factors=157, confidence_scaling="linear", alpha=3.200536137225281,
                    epsilon=0.0020667154823803733,
                    reg=0.3275956463846864, init_mean=0.0, init_std=0.1)

    recommender.save_model(folder_path="../result_experiments/", file_name="IALSRecommender_best_model80.zip")

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender = IALSRecommender.IALSRecommender(URM_all)
    recommender.fit(epochs=100, num_factors=173, confidence_scaling="linear", alpha=7.31319386499139,
                    epsilon=0.0019197416753549824,
                    reg=0.6581722208487086, init_mean=0.0, init_std=0.1)

    recommender.save_model(folder_path="../result_experiments/", file_name="IALSRecommender_best_model100.zip")

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    URM_train = sps.load_npz('../input_files/URM_train.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    recommender = IALSRecommender.IALSRecommender(URM_train)
    recommender.fit(epochs=100, num_factors=173, confidence_scaling="linear", alpha=7.31319386499139,
                    epsilon=0.0019197416753549824,
                    reg=0.6581722208487086, init_mean=0.0, init_std=0.1)

    recommender.save_model(folder_path="../result_experiments/", file_name="IALSRecommender_best_model64.zip")

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
