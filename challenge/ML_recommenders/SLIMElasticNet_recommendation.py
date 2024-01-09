import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.SLIM import SLIMElasticNetRecommender
from challenge.utils.functions import read_data


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train_xgboost = sps.load_npz("../input_files/URM_train_xgboost.npz")
    URM_test_xgboost = sps.load_npz("../input_files/URM_validation_xgboost.npz")

    recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train_xgboost)
    recommender.fit(topK=354, l1_ratio=0.05656717667595227, alpha=0.001337477108476206, positive_only=True)

    evaluator = EvaluatorHoldout(URM_test_xgboost, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender.save_model("../result_experiments/", "SLIM_xgboost.zip")


if __name__ == '__main__':
    __main__()
