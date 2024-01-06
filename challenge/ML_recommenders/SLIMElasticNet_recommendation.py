import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.SLIM import SLIMElasticNetRecommender
from challenge.utils.functions import read_data


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz('../input_files/new_URM_train.npz')
    URM_test = sps.load_npz('../input_files/new_URM_test.npz')
    URM_all = sps.load_npz('../input_files/new_URM_all.npz')
    URM_train_validation = sps.load_npz('../input_files/new_URM_train_plus_validation.npz')

    recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train)
    recommender.fit(topK=354, l1_ratio=0.05656717667595227, alpha=0.001337477108476206, positive_only=True)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender.save_model("../result_experiments/", "new_SLIMElasticNetRecommender_best_model64.zip")

    recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train_validation)
    recommender.fit(topK=354, l1_ratio=0.05656717667595227, alpha=0.001337477108476206, positive_only=True)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender.save_model("../result_experiments/", "new_SLIMElasticNetRecommender_best_model80.zip")

    recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_all)
    recommender.fit(topK=354, l1_ratio=0.05656717667595227, alpha=0.001337477108476206, positive_only=True)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender.save_model("../result_experiments/", "new_SLIMElasticNetRecommender_best_model100.zip")


if __name__ == '__main__':
    __main__()
