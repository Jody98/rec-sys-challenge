import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.SLIM import SLIMElasticNetRecommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train_plus_validation = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train_plus_validation)
    recommender.fit(topK=80, l1_ratio=0.02515808645930968, alpha=0.0020708761507802933, positive_only=True)

    recommender.save_model(folder_path="../result_experiments/", file_name="SLIMElasticNetRecommender_best_model80.zip")

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_all)
    recommender.fit(topK=80, l1_ratio=0.02515808645930968, alpha=0.0020708761507802933, positive_only=True)

    recommender.save_model(folder_path="../result_experiments/", file_name="SLIMElasticNetRecommender_best_model100.zip")

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    URM_train = sps.load_npz('../input_files/URM_train.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train)
    recommender.fit(topK=80, l1_ratio=0.02515808645930968, alpha=0.0020708761507802933, positive_only=True)

    recommender.save_model(folder_path="../result_experiments/", file_name="SLIMElasticNetRecommender_best_model64.zip")

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))



if __name__ == '__main__':
    __main__()
